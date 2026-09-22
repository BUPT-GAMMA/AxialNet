import os
import yaml
import shutil
import time
from pathlib import Path
from easydict import EasyDict

class ConfigManager:
    def __init__(self, config_path, override_args=None):
        self.config_path = config_path
        self.raw_cfg = self._load_yaml(config_path)
        
        # 允许命令行参数覆盖 Config (可选)
        if override_args:
            self._override_config(self.raw_cfg, override_args)

        # 1. 准备实验目录
        self.work_dir = self._prepare_work_dir()
        self.raw_cfg['work_dir'] = self.work_dir
        
        # 2. 转换为 EasyDict (方便点访问)
        self.cfg = EasyDict(self.raw_cfg)
        
        # 3. 备份 Config
        self._backup_config()

    def _load_yaml(self, path):
        config_path = Path(path).resolve()
        project_root = config_path.parent.parent
        text = config_path.read_text(encoding='utf-8')
        text = text.replace('${PROJECT_ROOT}', str(project_root))
        text = os.path.expandvars(text)
        return yaml.safe_load(text)

    def _prepare_work_dir(self):
        # 结构: experiments/{exp_name}/{timestamp}/
        exp_name = self.raw_cfg.get('experiment', {}).get('name', 'default_exp')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        root = self.raw_cfg.get('solver', {}).get('output_root', './experiments')
        
        work_dir = os.path.join(root, exp_name, timestamp)
        
        # 创建子目录
        os.makedirs(os.path.join(work_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'vis'), exist_ok=True) # 可视化结果
        
        return work_dir

    def _backup_config(self):
        dst = os.path.join(self.work_dir, 'config_backup.yaml')
        shutil.copy(self.config_path, dst)

    @property
    def config(self):
        return self.cfg

def load_config(path, args=None):
    return ConfigManager(path, args).config
