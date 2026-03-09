"""损失函数抽象定义。

该模块定义统一损失接口，原理是将训练目标计算与训练循环解耦，
便于在不同实验间快速切换损失组合。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseLoss(ABC):
    """损失函数统一接口。"""

    @abstractmethod
    def forward(self, pred: Any, target: Any) -> float:
        """计算损失值并返回标量结果。"""
        raise NotImplementedError

