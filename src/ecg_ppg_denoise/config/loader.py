"""配置加载入口。

该模块负责把外部配置文件解析为统一 dataclass，原理是集中处理
配置来源，避免业务代码直接依赖文件格式细节。
"""

from pathlib import Path

from .schema import ExperimentConfig


def load_experiment_config(config_path: Path | None = None) -> ExperimentConfig:
    """加载实验配置。

    当未传入配置路径时，返回默认配置对象。
    """
    if config_path is None:
        return ExperimentConfig()

    # TODO: 解析 YAML/JSON 并映射到 ExperimentConfig
    raise NotImplementedError("TODO: 实现配置文件解析逻辑")

