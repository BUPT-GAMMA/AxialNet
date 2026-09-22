import argparse
import random

import numpy as np
import torch

from axialnet.utils.config import load_config
from axialnet.utils.logger import setup_logger
from axialnet.utils.registry import DATALOADERS, SOLVERS

# Import packages to populate the registries.
import axialnet.datasets  # noqa: F401
import axialnet.models
import axialnet.solvers  # noqa: F401


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # 从命令行获取配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_congestion.yaml')
    parser.add_argument('--mode', type=str, default='train_test', choices=['train', 'test', 'train_test'])
    parser.add_argument('--pixel-root', type=str, help='Override data.dataset.pixel_root_dir')
    parser.add_argument('--net-root', type=str, help='Override data.dataset.net_root_dir/root_dir')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint used in test mode')
    parser.add_argument('--device', type=str, help='Device override, e.g. cuda:0 or cpu')
    args = parser.parse_args()

    # 1. 加载配置文件，设置日志
    cfg = load_config(args.config)
    if args.pixel_root:
        cfg.data.dataset.pixel_root_dir = args.pixel_root
    if args.net_root:
        cfg.data.dataset.net_root_dir = args.net_root
        cfg.data.dataset.root_dir = args.net_root
    if args.checkpoint:
        cfg.solver.test_weights = args.checkpoint
    if args.device:
        cfg.experiment.device = args.device

    seed_everything(int(cfg.experiment.get('seed', 42)))
    logger = setup_logger(cfg.work_dir) # 初始化一次
    logger.info(f"Config loaded from {args.config}")

    # 2. 配置数据集
    # 这里假设 build_dataloader 封装了 Dataset 实例化和 DataLoader 包装
    # 实际项目中可能写在 Builder 里
    logger.info("Building dataloaders")
    dataloaders = DATALOADERS.build(
        cfg.data,
    )
    # dataloaders = {
    #     'train': ..., 
    #     'val': ...,
    #     'test': ...
    # }

    # 3. 配置模型
    # model_cfg 中包含 backbone 和 head 的定义
    logger.info(f"Building model: {cfg.model.backbone.type}")
    # model = MODELS.build(cfg.model) # 这里需要在 model 内部处理 backbone+head 的组装
    model = axialnet.models.build_model(cfg)

    # 4. 配置求解器
    # 关键点：直接把 logger 传进去
    logger.info(f"Building solver: {cfg.solver.type}")
    solver = SOLVERS.build(
        cfg.solver, 
        cfg=cfg, 
        model=model, 
        dataloaders=dataloaders, 
        logger=logger
    )

    # 5. 执行
    if 'train' in args.mode:
        solver.train()
    
    if 'test' in args.mode:
        # 如果刚跑完 train，solver 内部已经记录了 best_model_path
        # 如果是纯 test 模式，solver 会去读 cfg 中的 test_weights
        solver.test()

if __name__ == '__main__':
    main()
