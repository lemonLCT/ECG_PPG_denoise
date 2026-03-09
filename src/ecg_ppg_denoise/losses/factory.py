"""损失工厂骨架。

该模块统一损失创建入口，原理是通过名称到实例的映射来降低
训练脚本中的条件分支数量。
"""

from typing import Any

from .base import BaseLoss


class PlaceholderLoss(BaseLoss):
    """占位损失：仅用于骨架占位。"""

    def forward(self, pred: Any, target: Any) -> float:
        # TODO: 替换为真实损失计算逻辑
        _ = (pred, target)
        raise NotImplementedError("TODO: 实现具体损失逻辑")


def build_loss(name: str) -> BaseLoss:
    """按名称构建损失函数。"""
    # TODO: 增加损失注册表，根据 name 返回不同损失
    _ = name
    return PlaceholderLoss()

