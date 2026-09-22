import os
import time
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class BaseSolver(ABC):
    def __init__(self, cfg, model, dataloaders, logger, **kwargs):
        """
        Args:
            cfg: EasyDict 配置
            model: PyTorch Model
            dataloaders: dict {'train': loader, 'val': loader, 'test': loader}
            logger: 已经初始化好的 logger 实例
        """
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.experiment.device if torch.cuda.is_available() else 'cpu')
        
        # 1. 组件
        self.model = model.to(self.device)
        self.train_loader = dataloaders.get('train')
        self.val_loader = dataloaders.get('val')
        self.test_loader = dataloaders.get('test')
        
        # 2. 优化器 (由子类调用 build_optimizer 或直接在此处通用构建)
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        # 3. 记录器
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'tb_logs'))
        
        # 4. 状态追踪
        self.start_epoch = 0
        self.early_stop_cfg = self.cfg.solver.get('early_stop', {})
        self.early_stop_enabled = bool(self.early_stop_cfg.get('enable', False))
        self.early_stop_monitor = self.early_stop_cfg.get('monitor', 'loss')
        self.early_stop_mode = str(self.early_stop_cfg.get('mode', 'min')).lower()
        self.early_stop_patience = int(self.early_stop_cfg.get('patience', 10))
        self.early_stop_min_delta = float(self.early_stop_cfg.get('min_delta', 0.0))
        self.early_stop_bad_epochs = 0
        self.early_stop_best = None
        self.checkpoint_monitor = self.cfg.solver.get(
            'checkpoint_monitor',
            self.early_stop_monitor if self.early_stop_enabled else 'loss'
        )
        self.checkpoint_mode = str(
            self.cfg.solver.get(
                'checkpoint_mode',
                self.early_stop_mode if self.early_stop_enabled else 'min'
            )
        ).lower()
        self.best_metric = self._get_initial_best_value(self.checkpoint_mode)
        self.best_model_path = None      # 记录训练过程中产生的最优模型路径
        self.timing_stats = {
            'train': {'total_time': 0.0, 'total_samples': 0},
            'infer': {'total_time': 0.0, 'total_samples': 0},
        }
        self.use_tqdm = bool(self.cfg.solver.get('use_tqdm', True))
        self.grad_accum_steps = max(1, int(self.cfg.solver.get('grad_accum_steps', 1)))
        
        # 5. 自动 Resume
        if cfg.solver.resume_path:
            self.load_checkpoint(cfg.solver.resume_path)

    # ================= 核心接口 =================
    
    def train(self):
        """ 训练主循环 """
        if not self.train_loader:
            self.logger.warning("No train_loader provided. Skipping training.")
            return

        self.logger.info(f"Start training from epoch {self.start_epoch}")
        
        for epoch in range(self.start_epoch, self.cfg.solver.max_epochs):
            self.before_train_epoch(epoch)
            
            # 1. Train One Epoch
            train_metrics = self.run_epoch(epoch)
            self.log_metrics(train_metrics, 'train', epoch)
            val_metrics = None
            
            # 2. Validation
            if epoch % self.cfg.solver.val_freq == 0 and self.val_loader:
                val_metrics = self.validate(epoch)
                self.log_metrics(val_metrics, 'val', epoch)
                
                # Check Best
                current_metric = val_metrics.get(self.checkpoint_monitor)
                if current_metric is None:
                    self.logger.warning(
                        f"Checkpoint monitor '{self.checkpoint_monitor}' not in val_metrics. Fallback to 'loss'."
                    )
                    current_metric = val_metrics['loss']
                is_best = self._is_improved(current_metric, self.best_metric, self.checkpoint_mode, 0.0)
                if is_best:
                    self.best_metric = current_metric
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.save_checkpoint(epoch, is_best=False)

                if self.early_stop_enabled:
                    should_stop = self.update_early_stop(val_metrics)
                    if should_stop:
                        self.logger.info("Early stopping triggered. Stopping training.")
                        break

            # 3. Scheduler Step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_cfg = self.cfg.solver.get('scheduler', {})
                    monitor_name = scheduler_cfg.get(
                        'monitor',
                        self.checkpoint_monitor if val_metrics is not None else 'loss'
                    )
                    metric_source = val_metrics if val_metrics is not None else train_metrics
                    metric_value = metric_source.get(monitor_name)
                    if metric_value is None:
                        metric_value = metric_source.get('loss')

                    if metric_value is None:
                        self.logger.warning(
                            f"ReduceLROnPlateau monitor '{monitor_name}' and fallback 'loss' not found. Skip scheduler.step()."
                        )
                    else:
                        self.scheduler.step(metric_value)
                else:
                    self.scheduler.step()
                
            self.after_train_epoch(epoch)

        self.log_global_timing_stats()
        self.logger.info("Training finished.")

    def test(self, checkpoint_path=None):
        """ 
        测试接口 
        Args:
            checkpoint_path: 指定权重路径。如果不指定，优先加载 Train 过程中保存的 Best，
                             其次尝试加载 Config 中的 test_weights
        """
        if not self.test_loader:
            self.logger.warning("No test_loader provided. Skipping testing.")
            return

        # 1. 确定加载哪个权重
        ckpt_to_load = checkpoint_path or self.best_model_path or self.cfg.solver.get('test_weights')
        
        if ckpt_to_load:
            self.logger.info(f"Testing using checkpoint: {ckpt_to_load}")
            # 注意：测试时通常只需加载权重，不需要加载 Optimizer 状态
            self.load_weights_only(ckpt_to_load)
        else:
            self.logger.warning("No checkpoint provided for testing. Using current model weights.")

        self.model.eval()
        test_metrics = {}
        # 可以用一个 List 收集所有预测结果用于后续分析
        all_preds = [] 
        
        with torch.no_grad():
            test_iter = self.test_loader
            if self.use_tqdm:
                test_iter = tqdm(test_iter, desc="Test", leave=False)
            for batch_idx, batch in enumerate(test_iter):
                batch = batch.to(self.device)
                batch_samples = self.get_batch_sample_count(batch)
                self._sync_device()
                start_time = time.perf_counter()
                metrics = self.run_test_step(batch, batch_idx)
                self._sync_device()
                elapsed = time.perf_counter() - start_time
                self.update_timing_stats('infer', elapsed, batch_samples)
                metrics['infer_time_per_sample_ms'] = torch.tensor(elapsed * 1000.0 / max(batch_samples, 1))
                
                # 聚合 Metrics
                for k, v in metrics.items():
                    test_metrics[k] = test_metrics.get(k, 0.0) + v.item()

        # 平均化
        for k in test_metrics:
            test_metrics[k] /= len(self.test_loader)
            
        self.log_global_timing_stats()
        self.logger.info(f"Test Results: {test_metrics}")
        return test_metrics

    # ================= 抽象方法 (子类实现) =================

    @abstractmethod
    def run_train_step(self, batch, batch_idx):
        """ 返回 dict: {'loss': tensor, ...} """
        pass

    @abstractmethod
    def run_test_step(self, batch, batch_idx):
        """ 返回 dict: {'mae': tensor, ...} """
        pass

    # ================= 内部逻辑 =================

    def run_epoch(self, epoch):
        self.model.train()
        epoch_metrics = {}

        samples_per_epoch = None
        if hasattr(self.cfg, "data") and hasattr(self.cfg.data, "loader"):
            samples_per_epoch = self.cfg.data.loader.get("samples_per_epoch")
        steps_per_epoch = int(samples_per_epoch) if samples_per_epoch else len(self.train_loader)

        train_iter = iter(self.train_loader)
        step_iter = range(steps_per_epoch)
        if self.use_tqdm:
            step_iter = tqdm(step_iter, desc=f"Train {epoch}", leave=False, total=steps_per_epoch)

        self.optimizer.zero_grad()

        for batch_idx in step_iter:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            batch = batch.to(self.device)
            batch_samples = self.get_batch_sample_count(batch)
            self._sync_device()
            start_time = time.perf_counter()
            
            step_metrics = self.run_train_step(batch, batch_idx)
            
            loss = step_metrics['loss']
            (loss / self.grad_accum_steps).backward()

            should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or ((batch_idx + 1) == steps_per_epoch)
            if should_step:
                if self.cfg.solver.get('clip_grad'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.solver.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self._sync_device()
            elapsed = time.perf_counter() - start_time
            self.update_timing_stats('train', elapsed, batch_samples)
            step_metrics['train_time_per_sample_ms'] = loss.new_tensor(elapsed * 1000.0 / max(batch_samples, 1))
            
            # 累加 Epoch Metrics
            for k, v in step_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v.item()
                
            # Step Log
            if batch_idx % self.cfg.solver.log_interval == 0:
                log_msg = f"Epoch [{epoch}][{batch_idx}/{steps_per_epoch}] Loss: {loss.item():.4f}"
                if self.use_tqdm:
                    step_iter.set_postfix(loss=f"{loss.item():.4f}")
                # self.logger.info(log_msg)

        # Average
        for k in epoch_metrics:
            epoch_metrics[k] /= steps_per_epoch
            
        return epoch_metrics

    def validate(self, epoch):
        self.model.eval()
        val_metrics = {}
        with torch.no_grad():
            val_iter = self.val_loader
            if self.use_tqdm:
                val_iter = tqdm(val_iter, desc=f"Val {epoch}", leave=False)
            for batch_idx, batch in enumerate(val_iter):
                batch = batch.to(self.device)
                batch_samples = self.get_batch_sample_count(batch)
                self._sync_device()
                start_time = time.perf_counter()
                # 复用 run_test_step，或者单独定义 run_val_step
                metrics = self.run_test_step(batch, batch_idx) 
                self._sync_device()
                elapsed = time.perf_counter() - start_time
                self.update_timing_stats('infer', elapsed, batch_samples)
                metrics['infer_time_per_sample_ms'] = torch.tensor(elapsed * 1000.0 / max(batch_samples, 1))
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v.item()
        
        for k in val_metrics:
            val_metrics[k] /= len(self.val_loader)
        return val_metrics

    # ================= 辅助方法 =================
    
    def build_optimizer(self):
        # 简单实现，支持 AdamW
        return torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.cfg.solver.optimizer.lr,
            weight_decay=self.cfg.solver.optimizer.weight_decay
        )
    
    def build_scheduler(self):
        if 'scheduler' not in self.cfg.solver:
            return None

        scheduler_cfg = self.cfg.solver.scheduler
        scheduler_type = str(scheduler_cfg.get('type', 'StepLR'))

        if scheduler_type == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_cfg.step_size,
                gamma=scheduler_cfg.gamma
            )

        if scheduler_type == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_cfg.get('mode', 'min'),
                factor=scheduler_cfg.get('factor', 0.1),
                patience=scheduler_cfg.get('patience', 10),
                threshold=scheduler_cfg.get('threshold', 1.0e-4),
                threshold_mode=scheduler_cfg.get('threshold_mode', 'rel'),
                cooldown=scheduler_cfg.get('cooldown', 0),
                min_lr=scheduler_cfg.get('min_lr', 0.0),
                eps=scheduler_cfg.get('eps', 1.0e-8),
            )

        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'early_stop_best': self.early_stop_best,
            'early_stop_bad_epochs': self.early_stop_bad_epochs,
            'timing_stats': self.timing_stats,
        }
        filename = os.path.join(self.cfg.work_dir, 'checkpoints', f'epoch_{epoch}.pth')
        torch.save(state, filename)
        
        if is_best:
            best_path = os.path.join(self.cfg.work_dir, 'checkpoints', 'best_model.pth')
            torch.save(state, best_path)
            self.best_model_path = best_path # 更新内存中的最佳路径
            self.logger.info(f"Saved Best Model: {best_path}")

    def load_checkpoint(self, path):
        """ 恢复所有训练状态 """
        self.logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_metric = ckpt.get('best_metric', self._get_initial_best_value(self.checkpoint_mode))
        self.early_stop_best = ckpt.get('early_stop_best', self.early_stop_best)
        self.early_stop_bad_epochs = ckpt.get('early_stop_bad_epochs', self.early_stop_bad_epochs)
        timing_stats = ckpt.get('timing_stats')
        if timing_stats is not None:
            self.timing_stats = timing_stats

    def load_weights_only(self, path):
        """ 仅加载权重 (用于 Test 或 Finetune 初始化) """
        self.logger.info(f"Loading weights from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # 兼容 state_dict 在 key 'state_dict' 下或者直接是 dict 的情况
        state_dict = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(state_dict, strict=False)

    def log_metrics(self, metrics, prefix, step):
        for k, v in metrics.items():
            self.writer.add_scalar(f'{prefix}/{k}', v, step)
        self.logger.info(f"[{prefix}] Step {step} | {metrics}")

    def update_early_stop(self, val_metrics):
        monitor_value = val_metrics.get(self.early_stop_monitor)
        if monitor_value is None:
            self.logger.warning(
                f"Early stop monitor '{self.early_stop_monitor}' not in val_metrics. Skipping early stop check."
            )
            return False

        if self.early_stop_best is None:
            self.early_stop_best = monitor_value
            self.early_stop_bad_epochs = 0
            return False

        improved = self._is_improved(
            monitor_value,
            self.early_stop_best,
            self.early_stop_mode,
            self.early_stop_min_delta,
        )

        if improved:
            self.early_stop_best = monitor_value
            self.early_stop_bad_epochs = 0
        else:
            self.early_stop_bad_epochs += 1

        if self.early_stop_bad_epochs >= self.early_stop_patience:
            return True
        return False

    def update_timing_stats(self, phase, elapsed_time, sample_count):
        if phase not in self.timing_stats:
            self.timing_stats[phase] = {'total_time': 0.0, 'total_samples': 0}
        self.timing_stats[phase]['total_time'] += float(elapsed_time)
        self.timing_stats[phase]['total_samples'] += int(sample_count)

    def log_global_timing_stats(self):
        train_avg_ms = self.get_global_average_time_ms('train')
        infer_avg_ms = self.get_global_average_time_ms('infer')
        if train_avg_ms is not None:
            self.logger.info(f"Global average train time per sample: {train_avg_ms:.4f} ms")
        if infer_avg_ms is not None:
            self.logger.info(f"Global average inference time per sample: {infer_avg_ms:.4f} ms")

    def get_global_average_time_ms(self, phase):
        stats = self.timing_stats.get(phase)
        if not stats or stats['total_samples'] <= 0:
            return None
        return stats['total_time'] * 1000.0 / stats['total_samples']

    def get_batch_sample_count(self, batch):
        grid = None
        try:
            grid = batch['grid_l1']
        except Exception:
            grid = None

        if grid is not None:
            ptr = getattr(grid, 'ptr', None)
            if ptr is not None and len(ptr) >= 2:
                return int(ptr.numel() - 1)
            num_graphs = getattr(grid, 'num_graphs', None)
            if num_graphs is not None:
                return int(num_graphs)
            batch_attr = getattr(grid, 'batch', None)
            if batch_attr is not None and batch_attr.numel() > 0:
                return int(batch_attr.max().item() + 1)

        batch_size = getattr(batch, 'batch_size', None)
        if batch_size is not None:
            return int(batch_size)
        return 1

    def _sync_device(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _is_improved(current_value, best_value, mode, min_delta):
        if mode == 'max':
            return (current_value - best_value) > min_delta
        return (best_value - current_value) > min_delta

    @staticmethod
    def _get_initial_best_value(mode):
        return float('-inf') if mode == 'max' else float('inf')
        
    def before_train_epoch(self, epoch): pass
    def after_train_epoch(self, epoch): pass