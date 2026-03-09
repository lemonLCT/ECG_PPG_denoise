"""日志工具。

该模块提供统一 logger 构建方法，原理是集中日志格式配置，避免各脚本重复。
"""

import logging


def build_logger(name: str = "ecg_ppg_denoise") -> logging.Logger:
    """创建并返回项目 logger。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger

