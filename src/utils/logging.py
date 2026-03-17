"""日志工具。"""

from __future__ import annotations

import logging
from pathlib import Path


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _ensure_file_handler(logger: logging.Logger, log_path: str | Path) -> None:
    target = Path(log_path).resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == target:
            return

    target.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setFormatter(_build_formatter())
    logger.addHandler(file_handler)


def build_logger(name: str = "ecg_ppg_denoise", level: int = logging.INFO, log_path: str | Path | None = None) -> logging.Logger:
    """构建控制台 logger，并可选同步写入日志文件。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        if log_path is not None:
            _ensure_file_handler(logger, log_path)
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_build_formatter())
    logger.addHandler(console_handler)
    if log_path is not None:
        _ensure_file_handler(logger, log_path)
    logger.propagate = False
    return logger
