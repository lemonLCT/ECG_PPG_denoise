"""训练器抽象定义。

该模块定义训练流程标准入口，原理是通过统一接口隔离不同训练策略
（监督、自监督、多任务等）的实现细节。
"""

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """训练流程统一接口。"""

    @abstractmethod
    def fit(self) -> None:
        """执行训练主流程。"""
        raise NotImplementedError

