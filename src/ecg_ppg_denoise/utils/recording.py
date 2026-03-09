"""结果记录骨架。

该模块定义结果落盘接口，原理是统一管理配置与指标输出格式，
方便后续实验对比与追踪。
"""

from pathlib import Path
from typing import Any


class ResultRecorder:
    """实验产物记录器。"""

    def __init__(self, exp_dir: Path) -> None:
        self.exp_dir = exp_dir

    def save_config(self, raw_config: dict[str, Any]) -> None:
        """保存原始配置字典。"""
        _ = raw_config
        # TODO: 将配置保存为 YAML 或 JSON 文件
        raise NotImplementedError("TODO: 实现配置落盘逻辑")

    def save_metrics(self, metrics: dict[str, float]) -> None:
        """保存评估指标字典。"""
        _ = metrics
        # TODO: 将指标保存为 CSV 或 JSON 文件
        raise NotImplementedError("TODO: 实现指标落盘逻辑")

