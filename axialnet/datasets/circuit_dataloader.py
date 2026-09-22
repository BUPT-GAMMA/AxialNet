from pathlib import Path
import random
import time
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from axialnet.utils.registry import DATALOADERS, DATASETS


class EpochFixedDataLoader:
	def __init__(self, dataloader):
		self._dataloader = dataloader
		self.iter_loader = iter(self._dataloader)

	def __next__(self):
		try:
			data = next(self.iter_loader)
		except StopIteration:
			time.sleep(2)
			self.iter_loader = iter(self._dataloader)
			data = next(self.iter_loader)
		return data

	def __len__(self):
		return len(self._dataloader)

	def __iter__(self):
		return self


def _read_split_file(path: Path) -> List[str]:
	if not path.exists():
		return []
	lines = [line.strip() for line in path.read_text().splitlines()]
	return [ln for ln in lines if ln]


def _ensure_pt_suffix(names: List[str]) -> List[str]:
	out = []
	for name in names:
		if not name.endswith(".pt"):
			out.append(f"{name}.pt")
		else:
			out.append(name)
	return out


# def _pad_rpa_and_mask(data_list):
# 	max_h = 0
# 	max_w = 0
# 	for data in data_list:
# 		rpa = data.rpa_map
# 		max_h = max(max_h, rpa.size(-2))
# 		max_w = max(max_w, rpa.size(-1))

# 	for data in data_list:
# 		rpa = data.rpa_map
# 		grid_mask = data.grid_mask
# 		pad_h = max_h - rpa.size(-2)
# 		pad_w = max_w - rpa.size(-1)
# 		if pad_h > 0 or pad_w > 0:
# 			data.rpa_map = F.pad(rpa, (0, pad_w, 0, pad_h), value=0.0)
# 			data.grid_mask = F.pad(grid_mask, (0, pad_w, 0, pad_h), value=0).bool()
# 	return data_list


# def _collate_with_padding(data_list):
# 	data_list = _pad_rpa_and_mask(data_list)
# 	return Batch.from_data_list(data_list)


def _split_files(
	root_dir: Path,
	split_dir: Optional[Path],
	split_ratio: Tuple[float, float, float],
	seed: int,
) -> Tuple[List[str], List[str], List[str]]:
	if split_dir and split_dir.exists():
		train_list = _ensure_pt_suffix(_read_split_file(split_dir / "train.txt"))
		val_list = _ensure_pt_suffix(_read_split_file(split_dir / "val.txt"))
		test_list = _ensure_pt_suffix(_read_split_file(split_dir / "test.txt"))
		if train_list or val_list or test_list:
			return train_list, val_list, test_list

	files = sorted([p.name for p in root_dir.glob("*.pt")])
	if not files:
		raise FileNotFoundError(f"No .pt files found in {root_dir}")

	rng = random.Random(seed)
	rng.shuffle(files)

	n_total = len(files)
	n_train = int(n_total * split_ratio[0])
	n_val = int(n_total * split_ratio[1])
	train_files = files[:n_train]
	val_files = files[n_train:n_train + n_val]
	test_files = files[n_train + n_val:]
	return train_files, val_files, test_files


def _write_split_file(path: Path, names: List[str]):
	path.parent.mkdir(parents=True, exist_ok=True)
	base_names = [Path(name).stem for name in names]
	path.write_text("\n".join(base_names) + ("\n" if base_names else ""))


def _resolve_count(count: Optional[int], ratio: float, total: int) -> int:
	if count is None:
		return int(total * ratio)
	return max(0, int(count))


def _split_files_by_count(
	root_dir: Path,
	split_dir: Optional[Path],
	seed: int,
	train_count: Optional[int] = None,
	val_count: Optional[int] = None,
	test_count: Optional[int] = None,
	split_ratio: Optional[Tuple[float, float, float]] = None,
	regen_split: bool = False,
):
	if split_dir and split_dir.exists() and not regen_split:
		train_list = _ensure_pt_suffix(_read_split_file(split_dir / "train.txt"))
		val_list = _ensure_pt_suffix(_read_split_file(split_dir / "val.txt"))
		test_list = _ensure_pt_suffix(_read_split_file(split_dir / "test.txt"))
		if train_list or val_list or test_list:
			return train_list, val_list, test_list

	files = sorted([p.name for p in root_dir.glob("*.pt")])
	if not files:
		raise FileNotFoundError(f"No .pt files found in {root_dir}")

	rng = random.Random(seed)
	rng.shuffle(files)

	total = len(files)
	ratio = split_ratio or (0.8, 0.1, 0.1)
	n_train = _resolve_count(train_count, ratio[0], total)
	n_val = _resolve_count(val_count, ratio[1], total)
	n_test = _resolve_count(test_count, ratio[2], total)

	if n_train + n_val + n_test > total:
		raise ValueError(
			f"Requested split counts exceed dataset size: "
			f"train={n_train}, val={n_val}, test={n_test}, total={total}"
		)

	train_files = files[:n_train]
	val_files = files[n_train:n_train + n_val]
	test_files = files[n_train + n_val:n_train + n_val + n_test]

	if split_dir:
		split_dir.mkdir(parents=True, exist_ok=True)
		_write_split_file(split_dir / "train.txt", train_files)
		_write_split_file(split_dir / "val.txt", val_files)
		_write_split_file(split_dir / "test.txt", test_files)

	return train_files, val_files, test_files


def _build_dataset(dataset_cfg: Dict[str, Any], file_list: Optional[List[str]]):
	allowed_keys = {
		"root_dir",
		"transform",
		"pre_transform",
		"pre_filter",
		"masking",
		"coord_indices",
		"pixel_root_dir",
		"net_root_dir",
		"pixel_suffix",
		"net_suffix",
	}
	clean_cfg = {k: v for k, v in dataset_cfg.items() if k in allowed_keys}
	clean_cfg["file_list"] = file_list
	if "type" in dataset_cfg:
		clean_cfg["type"] = dataset_cfg["type"]
		return DATASETS.build(clean_cfg)
	return DATASETS.build({"type": "CircuitDataset", **clean_cfg})
 

@DATALOADERS.register
def CircuitDataLoader(
	dataset: Dict[str, Any],
	loader: Optional[Dict[str, Any]] = None,
	split_dir: Optional[str] = None,
	split_ratio: Optional[List[float]] = None,
	seed: int = 42,
	cfg=None,
):
	loader = loader or {}
	root_dir = Path(dataset.get("root_dir"))
	split_dir = Path(split_dir) if split_dir else None

	if split_ratio is None:
		split_ratio = [0.8, 0.1, 0.1]
	if len(split_ratio) != 3:
		raise ValueError("split_ratio must have 3 values: [train, val, test]")
	train_files, val_files, test_files = _split_files(
		root_dir,
		split_dir,
		(split_ratio[0], split_ratio[1], split_ratio[2]),
		int(seed),
	)

	train_dataset = _build_dataset(dataset, train_files)
	val_dataset = _build_dataset(dataset, val_files) if val_files else None
	test_dataset = _build_dataset(dataset, test_files) if test_files else None

	batch_size = int(loader.get("batch_size", 1))
	num_workers = int(loader.get("num_workers", 0))
	pin_memory = bool(loader.get("pin_memory", False))
	shuffle = bool(loader.get("shuffle", True))
	samples_per_epoch = loader.get("samples_per_epoch")

	# collate_fn = _collate_with_padding if batch_size > 1 else None

	sampler = None
	if samples_per_epoch is not None:
		sampler = RandomSampler(
			train_dataset,
			replacement=True,
			num_samples=int(samples_per_epoch),
		)
		shuffle = False

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		sampler=sampler,
		num_workers=num_workers,
		pin_memory=pin_memory,
		# collate_fn=collate_fn,
	)
	if samples_per_epoch is not None:
		train_loader = EpochFixedDataLoader(train_loader)

	val_loader = None
	if val_dataset is not None:
		val_loader = DataLoader(
			val_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=pin_memory,
			# collate_fn=collate_fn,
		)

	test_loader = None
	if test_dataset is not None:
		test_loader = DataLoader(
			test_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=pin_memory,
			# collate_fn=collate_fn,
		)

	return {
		"train": train_loader,
		"val": val_loader,
		"test": test_loader,
	}


def _build_standard_loaders(
	dataset_cfg: Dict[str, Any],
	loader_cfg: Dict[str, Any],
	train_files: List[str],
	val_files: List[str],
	test_files: List[str],
):
	train_dataset = _build_dataset(dataset_cfg, train_files)
	val_dataset = _build_dataset(dataset_cfg, val_files) if val_files else None
	test_dataset = _build_dataset(dataset_cfg, test_files) if test_files else None

	batch_size = int(loader_cfg.get("batch_size", 1))
	num_workers = int(loader_cfg.get("num_workers", 0))
	pin_memory = bool(loader_cfg.get("pin_memory", False))
	shuffle = bool(loader_cfg.get("shuffle", True))

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)

	val_loader = None
	if val_dataset is not None:
		val_loader = DataLoader(
			val_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)

	test_loader = None
	if test_dataset is not None:
		test_loader = DataLoader(
			test_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)

	return {
		"train": train_loader,
		"val": val_loader,
		"test": test_loader,
	}


@DATALOADERS.register
def CircuitPretrainDataLoader(
	dataset: Dict[str, Any],
	loader: Optional[Dict[str, Any]] = None,
	split_dir: Optional[str] = None,
	split_ratio: Optional[List[float]] = None,
	train_count: Optional[int] = None,
	val_count: Optional[int] = None,
	test_count: Optional[int] = None,
	seed: int = 42,
	regen_split: bool = False,
	cfg=None,
):
	loader = loader or {}
	root_dir = Path(dataset.get("root_dir"))
	split_dir_path = Path(split_dir) if split_dir else None

	if split_ratio is None:
		split_ratio = [0.8, 0.1, 0.1]
	if len(split_ratio) != 3:
		raise ValueError("split_ratio must have 3 values: [train, val, test]")

	train_files, val_files, test_files = _split_files_by_count(
		root_dir=root_dir,
		split_dir=split_dir_path,
		seed=int(seed),
		train_count=train_count,
		val_count=val_count,
		test_count=test_count,
		split_ratio=(split_ratio[0], split_ratio[1], split_ratio[2]),
		regen_split=bool(regen_split),
	)

	return _build_standard_loaders(dataset, loader, train_files, val_files, test_files)


@DATALOADERS.register
def CircuitFinetuneDataLoader(
	dataset: Dict[str, Any],
	loader: Optional[Dict[str, Any]] = None,
	split_dir: Optional[str] = None,
	split_ratio: Optional[List[float]] = None,
	train_count: Optional[int] = None,
	val_count: Optional[int] = None,
	test_count: Optional[int] = None,
	seed: int = 42,
	regen_split: bool = False,
	cfg=None,
):
	return CircuitPretrainDataLoader(
		dataset=dataset,
		loader=loader,
		split_dir=split_dir,
		split_ratio=split_ratio,
		train_count=train_count,
		val_count=val_count,
		test_count=test_count,
		seed=seed,
		regen_split=regen_split,
		cfg=cfg,
	)
