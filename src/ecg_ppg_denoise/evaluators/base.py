"""评估器抽象定义。

该模块统一评估入口，原理是把评估流程封装成独立对象，
让训练脚本和评估脚本共享一致调用方式。
"""

from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """离线/在线评估统一接口。"""

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """执行评估并返回指标字典。"""
        raise NotImplementedError

