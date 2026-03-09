"""指标管理器骨架。

该模块负责批量管理多个指标对象，原理是统一调度 `update/compute/reset`
减少训练器中重复样板代码。
"""

from .base import BaseMetric


class MetricManager:
    """统一管理多个指标实例。"""

    def __init__(self, metrics: dict[str, BaseMetric]) -> None:
        self.metrics = metrics

    def reset_all(self) -> None:
        """重置所有指标状态。"""
        for metric in self.metrics.values():
            metric.reset()

