"""模型工厂骨架。

该模块集中管理模型构建逻辑，原理是把“按配置选择模型”的决策
从训练脚本中抽离出来，降低入口脚本复杂度。
"""

from typing import Any

from ..config.schema import ModelConfig
from .base import BaseDenoiseModel


class PlaceholderModel(BaseDenoiseModel):
    """占位模型：仅用于流程连通性。"""

    def forward(self, noisy_signal: Any) -> Any:
        # TODO: 替换为真实模型前向逻辑
        raise NotImplementedError("TODO: 实现具体模型前向")


def build_model(config: ModelConfig) -> BaseDenoiseModel:
    """根据配置创建模型实例。"""
    # TODO: 增加模型注册表，根据 config.name 返回不同模型
    _ = config
    return PlaceholderModel()

