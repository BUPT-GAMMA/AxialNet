import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import (
	mean_absolute_error,
	mean_squared_error,
	pearson_corrcoef,
	spearman_corrcoef,
	structural_similarity_index_measure,
)

from .base_solver import BaseSolver
from axialnet.utils.registry import SOLVERS


@SOLVERS.register
class FinetuneSolver(BaseSolver):
	def __init__(self, cfg, model, dataloaders, logger, **kwargs):
		super().__init__(cfg, model, dataloaders, logger, **kwargs)
		pretrained_ckpt = self.cfg.solver.get('pretrained_ckpt')
		if pretrained_ckpt:
			self.logger.info(f"Loading pretrained checkpoint: {pretrained_ckpt}")
			self.load_weights_only(pretrained_ckpt)

		self.freeze_backbone_epochs = int(self.cfg.solver.get('freeze_backbone_epochs', 0))
		self.swap_direction_labels = bool(self.cfg.solver.get('swap_direction_labels', False))
		self.ssim_loss_weight = float(self.cfg.solver.get('ssim_loss_weight', 0.0))
		if not 0.0 <= self.ssim_loss_weight <= 1.0:
			raise ValueError('`solver.ssim_loss_weight` must be in [0, 1]')

	def before_train_epoch(self, epoch):
		freeze_flag = epoch < self.freeze_backbone_epochs
		backbone = getattr(self.model, 'backbone', None)
		if backbone is None:
			return
		for param in backbone.parameters():
			param.requires_grad = not freeze_flag

	def run_train_step(self, batch, batch_idx):
		pred_pack = self._collect_predictions(batch)
		target = self._normalize_label(self._get_congestion_target(batch, pred_pack['pred_total'].device))
		loss = self._compute_total_loss(pred_pack, batch, target)
		metrics = compute_metrics(
			pred_pack['pred_total'],
			target,
			batch_sizes=batch['grid_l1'].size,
			ptr=batch['grid_l1'].ptr,
		)
		return {
			'loss': loss,
			**metrics,
		}

	def run_test_step(self, batch, batch_idx):
		pred_pack = self._collect_predictions(batch)
		target = self._normalize_label(self._get_congestion_target(batch, pred_pack['pred_total'].device))
		loss = self._compute_total_loss(pred_pack, batch, target)
		metrics = compute_metrics(
			pred_pack['pred_total'],
			target,
			batch_sizes=batch['grid_l1'].size,
			ptr=batch['grid_l1'].ptr,
		)
		return {
			'loss': loss,
			**metrics,
		}

	@staticmethod
	def _normalize_label(label):
		if label.dim() == 1:
			return label.view(-1, 1)
		if label.dim() > 2:
			return label.view(-1, 1)
		if label.dim() == 2 and label.size(1) > 1:
			return label.view(-1, 1)
		return label

	def _collect_predictions(self, batch):
		outputs = self.model(batch)
		prediction_type = str(self.cfg.solver.get('prediction_type', 'auto')).lower()

		def _as_col(tensor):
			if tensor.dim() == 1:
				return tensor.view(-1, 1)
			if tensor.dim() > 2:
				return tensor.view(-1, 1)
			if tensor.dim() == 2 and tensor.size(1) > 1:
				return tensor.sum(dim=-1, keepdim=True)
			return tensor

		pred_h = None
		pred_v = None
		pred_total = None
		is_separate = False

		if isinstance(outputs, dict):
			if 'pred_h' in outputs and 'pred_v' in outputs:
				pred_h = _as_col(outputs['pred_h'])
				pred_v = _as_col(outputs['pred_v'])
				pred_total = _as_col(outputs.get('pred_congestion', pred_h + pred_v))
				is_separate = True
			elif 'pred_congestion' in outputs:
				pred_total = _as_col(outputs['pred_congestion'])
			elif 'pred_rpa' in outputs:
				pred_total = _as_col(outputs['pred_rpa'])
			elif 'grid_l1' in outputs:
				pred_total = _as_col(outputs['grid_l1'])
			if pred_total is None:
				raise KeyError('Model output dict must contain `pred_congestion`/`pred_h`+`pred_v`/`pred_rpa`/`grid_l1` for finetune')
		else:
			pred_total = _as_col(outputs)

		if prediction_type == 'separate' and (pred_h is None or pred_v is None):
			raise ValueError('`solver.prediction_type=separate` requires model output keys: `pred_h` and `pred_v`')

		return {
			'pred_total': pred_total,
			'pred_h': pred_h,
			'pred_v': pred_v,
			'is_separate': is_separate,
		}

	def _compute_total_loss(self, pred_pack, batch, target_total):
		prediction_type = str(self.cfg.solver.get('prediction_type', 'auto')).lower()
		use_separate = prediction_type == 'separate' or (
			prediction_type == 'auto' and pred_pack['is_separate']
		)

		if use_separate:
			target_h, target_v = self._get_direction_targets(batch, target_total.device)
			if self.swap_direction_labels:
				target_h, target_v = target_v, target_h
			target_h = self._normalize_label(target_h)
			target_v = self._normalize_label(target_v)
			loss_h = self._compute_loss(pred_pack['pred_h'], target_h)
			loss_v = self._compute_loss(pred_pack['pred_v'], target_v)
			base_loss = loss_h + loss_v
		else:
			base_loss = self._compute_loss(pred_pack['pred_total'], target_total)

		if self.ssim_loss_weight <= 0.0:
			return base_loss

		ssim_loss = self._compute_ssim_loss(
			pred_pack['pred_total'],
			target_total,
			batch_sizes=batch['grid_l1'].size,
			ptr=batch['grid_l1'].ptr,
		)
		return (1.0 - self.ssim_loss_weight) * base_loss + self.ssim_loss_weight * ssim_loss

	@staticmethod
	def _compute_ssim_loss(preds, target, batch_sizes=None, ptr=None):
		preds_f = preds.float()
		target_f = target.float()
		device = preds_f.device

		if len(batch_sizes) == 2:
			batch_sizes = [batch_sizes]

		if batch_sizes is None or ptr is None or len(batch_sizes) == 0:
			return torch.tensor(0.0, device=device, dtype=preds_f.dtype)

		ssim_losses = []
		for i in range(len(batch_sizes)):
			start_idx = ptr[i].item()
			end_idx = ptr[i + 1].item()
			h, w = batch_sizes[i]
			sample_pred = preds_f[start_idx:end_idx]
			sample_target = target_f[start_idx:end_idx]
			if sample_pred.numel() != h * w:
				continue
			sample_pred = sample_pred.reshape(1, 1, h, w)
			sample_target = sample_target.reshape(1, 1, h, w)
			data_range = (sample_target.max() - sample_target.min()).clamp(min=1e-6)
			ssim_score = structural_similarity_index_measure(
				sample_pred,
				sample_target,
				data_range=data_range,
			)
			if torch.isfinite(ssim_score):
				ssim_losses.append(1.0 - ssim_score)

		if not ssim_losses:
			return torch.tensor(0.0, device=device, dtype=preds_f.dtype)

		return torch.stack(ssim_losses).mean()

	def _compute_loss(self, pred, target):
		pred = pred.view(-1)
		target = target.view(-1)
		loss_type = self.cfg.solver.get('loss_type', 'mse')
		if loss_type == 'mse':
			return F.mse_loss(pred, target)
		if loss_type == 'mae':
			return F.l1_loss(pred, target)
		raise ValueError(f'Unsupported loss type: {loss_type}')

	@staticmethod
	def _get_congestion_target(batch, device):
		label = batch['grid_l1'].congestion_label
		if isinstance(label, torch.Tensor):
			return label.to(device=device, dtype=torch.float32)
		if isinstance(label, np.ndarray):
			return torch.from_numpy(label).to(device=device, dtype=torch.float32)
		if isinstance(label, list):
			if len(label) == 0:
				raise ValueError('Empty congestion label list')
			if isinstance(label[0], torch.Tensor):
				return torch.cat([x.float() for x in label], dim=0).to(device)
			return torch.from_numpy(np.vstack(label)).to(device=device, dtype=torch.float32)
		raise TypeError(f'Unsupported congestion label type: {type(label)}')

	@staticmethod
	def _to_float_tensor(value, device):
		if isinstance(value, torch.Tensor):
			return value.to(device=device, dtype=torch.float32)
		if isinstance(value, np.ndarray):
			return torch.from_numpy(value).to(device=device, dtype=torch.float32)
		if isinstance(value, list):
			if len(value) == 0:
				raise ValueError('Empty direction label list')
			if isinstance(value[0], torch.Tensor):
				return torch.cat([x.float() for x in value], dim=0).to(device)
			return torch.from_numpy(np.vstack(value)).to(device=device, dtype=torch.float32)
		raise TypeError(f'Unsupported direction label type: {type(value)}')

	def _get_direction_targets(self, batch, device):
		grid = batch['grid_l1']
		if hasattr(grid, 'y_h') and hasattr(grid, 'y_v'):
			return self._to_float_tensor(grid.y_h, device), self._to_float_tensor(grid.y_v, device)
		if hasattr(grid, 'congestion_h') and hasattr(grid, 'congestion_v'):
			return self._to_float_tensor(grid.congestion_h, device), self._to_float_tensor(grid.congestion_v, device)
		raise ValueError('Separate prediction requires directional labels: `grid_l1.y_h/y_v` or `grid_l1.congestion_h/congestion_v`')


def compute_metrics(preds: torch.Tensor, target: torch.Tensor, batch_sizes=None, ptr=None):
	preds_f = preds.float()
	target_f = target.float()
	device = preds_f.device

	flat_preds = preds_f.reshape(-1)
	flat_target = target_f.reshape(-1)

	if not torch.isfinite(flat_preds).all() or not torch.isfinite(flat_target).all():
		return {k: torch.tensor(float('nan'), device=device) for k in ['pearson', 'spearman', 'mae', 'rmse', 'nrms', 'ssim']}

	preds_std = flat_preds.std(unbiased=True)
	target_std = flat_target.std(unbiased=True)
	if preds_std < 1e-8 or target_std < 1e-8:
		pearson = torch.tensor(float(-1), device=device)
		spearman = torch.tensor(float('nan'), device=device)
	else:
		pearson = pearson_corrcoef(flat_preds.double(), flat_target.double())
		spearman = spearman_corrcoef(flat_preds.double(), flat_target.double())

	mae = mean_absolute_error(flat_preds, flat_target)
	rmse = torch.sqrt(mean_squared_error(flat_preds, flat_target))
	target_range = (flat_target.max() - flat_target.min()).clamp(min=1e-6)
	nrms = rmse / target_range

	ssim = torch.tensor(float('nan'), device=device)
	if len(batch_sizes) == 2:
		batch_sizes = [batch_sizes]
	if batch_sizes is not None and ptr is not None and len(batch_sizes) > 0:
		ssim_list = []
		for i in range(len(batch_sizes)):
			start_idx = ptr[i].item()
			end_idx = ptr[i + 1].item()
			h, w = batch_sizes[i]
			sample_pred = preds_f[start_idx:end_idx]
			sample_target = target_f[start_idx:end_idx]
			if sample_pred.numel() != h * w:
				continue
			sample_pred = sample_pred.reshape(1, 1, h, w)
			sample_target = sample_target.reshape(1, 1, h, w)
			data_range = (sample_target.max() - sample_target.min()).clamp(min=1e-6)
			ssim_list.append(
				structural_similarity_index_measure(sample_pred, sample_target, data_range=data_range)
			)
		if ssim_list:
			ssim = torch.stack(ssim_list).mean()

	return {
		'pearson': pearson,
		'spearman': spearman,
		'mae': mae,
		'rmse': rmse,
		'nrms': nrms,
		'ssim': ssim,
	}
