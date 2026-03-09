"""配置数据结构定义。

该模块使用 dataclass 显式描述实验参数，原理是通过强类型结构
统一配置入口，减少脚本参数分散导致的维护成本。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """数据配置：描述数据路径和基础采样属性。"""

    dataset_root: Path = Path("./data")
    sample_rate: int = 0  # TODO: 按实际数据集填写采样率


@dataclass
class ModelConfig:
    """模型配置：描述模型注册名和输入输出维度。"""

    name: str = "placeholder_model"
    in_channels: int = 2  # ECG + PPG
    out_channels: int = 2


@dataclass
class TrainConfig:
    """训练配置：描述训练过程中的基础超参数。"""

    epochs: int = 0  # TODO: 按实验需求设置
    batch_size: int = 0  # TODO: 按实验需求设置
    seed: int = 42


@dataclass
class RuntimeConfig:
    """运行时配置：描述实验命名和输出路径。"""

    experiment_name: str = "exp_skeleton"
    output_root: Path = Path("./outputs")


@dataclass
class ExperimentConfig:
    """顶层配置：聚合 runtime/data/model/train 子配置。"""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

