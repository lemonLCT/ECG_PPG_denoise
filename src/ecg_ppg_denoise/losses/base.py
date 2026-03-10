"""损失函数基类定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLoss(ABC):
    """损失函数基类。"""

    @abstractmethod
    def __call__(self, prediction: Any, target: Any) -> Any:
        """计算损失值。"""
