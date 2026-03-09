"""预处理步骤抽象定义。

该模块约束单步变换接口，原理是通过统一可调用对象协议，
让不同预处理步骤可以被流水线自由组合。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTransform(ABC):
    """单个预处理步骤接口。"""

    @abstractmethod
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """输入单样本字典并返回处理后的样本。"""
        raise NotImplementedError

