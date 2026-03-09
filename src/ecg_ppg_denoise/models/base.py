"""模型抽象定义。

该模块定义去噪模型最小接口，原理是通过统一 `forward` 签名，
使训练器可在不感知具体模型细节的前提下完成调用。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseDenoiseModel(ABC):
    """去噪模型统一接口。"""

    @abstractmethod
    def forward(self, noisy_signal: Any) -> Any:
        """前向推理接口。"""
        raise NotImplementedError

