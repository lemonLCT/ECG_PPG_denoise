"""日志工具。"""

from __future__ import annotations

import logging
from typing import Optional


def build_logger(name: str = "ecg_ppg_denoise", level: int = logging.INFO) -> logging.Logger:
    """构建标准输出 logger。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
