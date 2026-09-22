from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Dataset, HeteroData

from axialnet.utils.registry import DATASETS


def _to_tensor_2d(value, dtype=torch.float32):
	if value is None:
		return None
	if isinstance(value, torch.Tensor):
		out = value.to(dtype=dtype)
	elif hasattr(value, "shape"):
		out = torch.as_tensor(value, dtype=dtype)
	else:
		out = torch.tensor(value, dtype=dtype)
	if out.dim() == 1:
		out = out.unsqueeze(-1)
	return out


def _normalize_names(file_list: Optional[List[str]]) -> Optional[List[str]]:
	if file_list is None:
		return None
	result = []
	for name in file_list:
		name = name.strip()
		if not name:
			continue
		if name.endswith(".pt"):
			name = name[:-3]
		if name.endswith("_pixel"):
			name = name[:-6]
		if name.endswith("_net"):
			name = name[:-4]
		result.append(name)
	return result


class DualPretrainGraphDataset(Dataset):
	"""
	Dual dataset that merges two processed pt files into one HeteroData sample.
	- pixel file: `*_pixel.pt`
	- net file: `*_net.pt`
	"""

	def __init__(
		self,
		root_dir: str,
		file_list: Optional[List[str]] = None,
		pixel_root_dir: Optional[str] = None,
		net_root_dir: Optional[str] = None,
		pixel_suffix: str = "_pixel.pt",
		net_suffix: str = "_net.pt",
		transform=None,
		pre_transform=None,
		pre_filter=None,
	):
		self.root_dir = Path(root_dir)
		if not self.root_dir.exists():
			raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

		self.pixel_suffix = pixel_suffix
		self.net_suffix = net_suffix

		self.pixel_root_dir = Path(pixel_root_dir) if pixel_root_dir else self.root_dir
		if net_root_dir:
			self.net_root_dir = Path(net_root_dir)
		else:
			default_net_root = self.root_dir.parent / "processed_net_data"
			self.net_root_dir = default_net_root if default_net_root.exists() else self.root_dir

		if not self.pixel_root_dir.exists():
			raise FileNotFoundError(f"Pixel root not found: {self.pixel_root_dir}")
		if not self.net_root_dir.exists():
			raise FileNotFoundError(f"Net root not found: {self.net_root_dir}")

		if file_list is None:
			self.sample_ids = self._discover_sample_ids()
		else:
			self.sample_ids = _normalize_names(file_list)

		if not self.sample_ids:
			raise FileNotFoundError(
				f"No valid dual samples found under pixel={self.pixel_root_dir}, net={self.net_root_dir}"
			)

		super().__init__(str(self.root_dir), transform, pre_transform, pre_filter)

	def _discover_sample_ids(self) -> List[str]:
		pixel_ids = {
			path.name[: -len(self.pixel_suffix)]
			for path in self.pixel_root_dir.glob(f"*{self.pixel_suffix}")
		}
		net_ids = {
			path.name[: -len(self.net_suffix)]
			for path in self.net_root_dir.glob(f"*{self.net_suffix}")
		}
		return sorted(pixel_ids & net_ids)

	def _paths(self, sample_id: str):
		pixel_path = self.pixel_root_dir / f"{sample_id}{self.pixel_suffix}"
		net_path = self.net_root_dir / f"{sample_id}{self.net_suffix}"
		if not pixel_path.exists():
			raise FileNotFoundError(f"Pixel file not found: {pixel_path}")
		if not net_path.exists():
			raise FileNotFoundError(f"Net file not found: {net_path}")
		return pixel_path, net_path

	def len(self) -> int:
		return len(self.sample_ids)

	@staticmethod
	def add_reverse_edge(data: HeteroData, src_id: str, rel: str, dst_id: str) -> HeteroData:
		if (src_id, rel, dst_id) not in data.edge_types:
			return data

		src, dst = data[src_id, rel, dst_id].edge_index
		data[dst_id, rel, src_id].edge_index = torch.stack([dst, src], dim=0)
		if hasattr(data[src_id, rel, dst_id], "edge_attr"):
			data[dst_id, rel, src_id].edge_attr = data[src_id, rel, dst_id].edge_attr
		return data

	def _build_hetero(self, pixel_data: Dict, net_data: Dict) -> HeteroData:
		data = HeteroData()

		pixel_x = torch.as_tensor(pixel_data["x"], dtype=torch.float32)
		if pixel_x.dim() == 3:
			height, width, channels = pixel_x.shape
			grid_x = pixel_x.view(height * width, channels)
		elif pixel_x.dim() == 2:
			grid_x = pixel_x
			num_pixels = int(net_data.get("num_pixels", grid_x.size(0)))
			if grid_x.size(0) != num_pixels:
				raise ValueError(
					f"Pixel feature shape mismatch: got {grid_x.size(0)} pixels, expected {num_pixels}"
				)
			width = int(pixel_data.get("layout_size", [num_pixels, 1])[1]) if "layout_size" in pixel_data else 1
			height = num_pixels // max(width, 1)
		else:
			raise ValueError(f"Unsupported pixel feature shape: {tuple(pixel_x.shape)}")

		data["grid_l1"].x = grid_x
		
		data["grid_l1"].size = torch.tensor([height, width], dtype=torch.long)

		y_h = _to_tensor_2d(pixel_data.get("y_h"))
		y_v = _to_tensor_2d(pixel_data.get("y_v"))
		if y_h is not None:
			data["grid_l1"].y_h = y_h.view(-1, y_h.size(-1))
		if y_v is not None:
			data["grid_l1"].y_v = y_v.view(-1, y_v.size(-1))

		if y_h is not None and y_v is not None:
			data["grid_l1"].congestion_label = (data["grid_l1"].y_h + data["grid_l1"].y_v)
                  
		# rpa_target = grid_x[:, 2:3] if grid_x.size(-1) >= 3 else torch.zeros((grid_x.size(0), 1), dtype=torch.float32)
		# data["grid_l1"].rpa_target = rpa_target

		net_x = torch.as_tensor(net_data["net_x"], dtype=torch.float32)
		data["net"].x = net_x

		if "edge_index_net_to_pixel_cell" in net_data:
			edge_index = torch.as_tensor(net_data["edge_index_net_to_pixel_cell"], dtype=torch.long)
			data["net", "to", "grid_l1"].edge_index = edge_index
			if "edge_attr_net_to_pixel_cell" in net_data:
				edge_attr = torch.as_tensor(net_data["edge_attr_net_to_pixel_cell"], dtype=torch.float32)
				data["net", "to", "grid_l1"].edge_attr = edge_attr

		if "edge_index_net_to_pixel_bbox" in net_data:
			edge_index_bbox = torch.as_tensor(net_data["edge_index_net_to_pixel_bbox"], dtype=torch.long)
			data["net", "bbox_to", "grid_l1"].edge_index = edge_index_bbox

		num_grid_nodes = data["grid_l1"].x.size(0)
		safe_width = max(int(width), 1)
		grid_idx = torch.arange(num_grid_nodes, dtype=torch.long)
		x_pos = (grid_idx % safe_width).to(torch.float32)
		y_pos = torch.div(grid_idx, safe_width, rounding_mode="floor").to(torch.float32)

		if ("net", "to", "grid_l1") in data.edge_types:
			grid_degree = torch.bincount(
				data["net", "to", "grid_l1"].edge_index[1], minlength=num_grid_nodes
			).to(torch.float32)
		else:
			raise ValueError("Missing edge_index for ('net', 'to', 'grid_l1'), cannot compute grid degree")
			grid_degree = torch.zeros(num_grid_nodes, dtype=torch.float32)

		data["grid_l1"].x_pos_and_d = torch.stack([x_pos, y_pos, grid_degree], dim=-1)

		data = self.add_reverse_edge(data, "net", "to", "grid_l1")
		data = self.add_reverse_edge(data, "net", "bbox_to", "grid_l1")
		return data

	def get(self, idx: int):
		sample_id = self.sample_ids[idx]
		pixel_path, net_path = self._paths(sample_id)

		pixel_data = torch.load(pixel_path, weights_only=False)
		net_data = torch.load(net_path, weights_only=False)

		data = self._build_hetero(pixel_data=pixel_data, net_data=net_data)
		data.sample_id = sample_id
		return data


@DATASETS.register
class DualPretrainMaskDataset(DualPretrainGraphDataset):
	def __init__(
		self,
		root_dir: str,
		file_list: Optional[List[str]] = None,
		masking: Optional[Dict] = None,
		coord_indices: Optional[List[int]] = None,
		transform=None,
		pre_transform=None,
		pre_filter=None,
		pixel_root_dir: Optional[str] = None,
		net_root_dir: Optional[str] = None,
		pixel_suffix: str = "_pixel.pt",
		net_suffix: str = "_net.pt",
	):
		super().__init__(
			root_dir=root_dir,
			file_list=_normalize_names(file_list),
			pixel_root_dir=pixel_root_dir,
			net_root_dir=net_root_dir,
			pixel_suffix=pixel_suffix,
			net_suffix=net_suffix,
			transform=transform,
			pre_transform=pre_transform,
			pre_filter=pre_filter,
		)
		self.masking = masking or {}
		self.mask_ratio = float(self.masking.get("ratio", 0.5))
		self.coord_indices = coord_indices or [0, 1]

	def _build_grid_mask(self, data: HeteroData) -> torch.Tensor:
		num_grid = data["grid_l1"].num_nodes
		device = data["grid_l1"].x.device
		mask = torch.zeros(num_grid, dtype=torch.bool, device=device)

		if ("net", "to", "grid_l1") in data.edge_types:
			non_empty = torch.unique(data["net", "to", "grid_l1"].edge_index[1])
		else:
			non_empty = torch.arange(num_grid, device=device)

		n_count = non_empty.size(0)
		if n_count == 0:
			return mask

		mask_num = max(1, min(int(round(self.mask_ratio * n_count)), max(n_count - 1, 1)))
		perm = torch.randperm(n_count, device=device)
		mask[non_empty[perm[:mask_num]]] = True
		return mask

	def get(self, idx: int):
		data = super().get(idx)
		if not hasattr(data["grid_l1"], "rpa_target"):
			data["grid_l1"].rpa_target = torch.zeros((data["grid_l1"].num_nodes, 1), dtype=torch.float32)
		data["grid_l1"].mask = self._build_grid_mask(data)
		return data


@DATASETS.register
class DualFinetuneDataset(DualPretrainGraphDataset):
	def __init__(
		self,
		root_dir: str,
		file_list: Optional[List[str]] = None,
		transform=None,
		pre_transform=None,
		pre_filter=None,
		pixel_root_dir: Optional[str] = None,
		net_root_dir: Optional[str] = None,
		pixel_suffix: str = "_pixel.pt",
		net_suffix: str = "_net.pt",
	):
		super().__init__(
			root_dir=root_dir,
			file_list=_normalize_names(file_list),
			pixel_root_dir=pixel_root_dir,
			net_root_dir=net_root_dir,
			pixel_suffix=pixel_suffix,
			net_suffix=net_suffix,
			transform=transform,
			pre_transform=pre_transform,
			pre_filter=pre_filter,
		)

	def get(self, idx: int):
		data = super().get(idx)
		congestion = _to_tensor_2d(getattr(data["grid_l1"], "congestion_label", None))
		if congestion is None:
			raise ValueError("Missing `grid_l1.congestion_label` in finetune sample")
		data["grid_l1"].congestion_label = congestion
		return data


@DATASETS.register
class DualDataset(DualPretrainGraphDataset):
	pass
