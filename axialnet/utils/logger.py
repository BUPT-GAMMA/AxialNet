import logging
import os
import sys

def setup_logger(work_dir=None, name="DeepLayout", level=logging.INFO):
    """
    初始化 Logger。
    如果在 main 中调用过一次，后续 solver 中可以通过 logging.getLogger(name) 获取同一个实例。
    但也支持直接传参。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # 避免重复添加 Handler (例如在 Notebook 或多次调用时)
    if logger.handlers:
        return logger

    # 1. 格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 2. Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 3. File Handler
    if work_dir:
        log_path = os.path.join(work_dir, 'logs', 'running.log')
        fh = logging.FileHandler(log_path, mode='a') # 追加模式
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger