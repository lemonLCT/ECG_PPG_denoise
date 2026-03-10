"""模型基类定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDenoiseModel(ABC):
    """去噪模型基类。"""

    @abstractmethod
    def forward(self, noisy_signal: Any) -> Any:
        """前向推理接口。"""

    def __call__(self, noisy_signal: Any) -> Any:
        return self.forward(noisy_signal)
