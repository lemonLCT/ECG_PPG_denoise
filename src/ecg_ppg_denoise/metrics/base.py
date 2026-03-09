"""指标抽象接口。

该模块定义指标生命周期接口，原理是将“批次更新-最终汇总-状态重置”
这一通用过程标准化，便于复用。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """实验指标统一接口。"""

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """接收一个批次结果并更新内部统计。"""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        """计算最终指标值。"""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """重置内部状态，准备下轮统计。"""
        raise NotImplementedError

