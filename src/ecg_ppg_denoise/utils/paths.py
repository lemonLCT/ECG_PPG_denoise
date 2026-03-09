"""路径工具。

该模块集中管理实验输出目录构建，原理是把路径规则从业务逻辑中分离，
确保不同脚本的输出位置一致。
"""

from pathlib import Path

from ..config.schema import RuntimeConfig


def build_experiment_dir(runtime: RuntimeConfig) -> Path:
    """根据运行配置构建实验输出目录。"""
    exp_dir = runtime.output_root / runtime.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

