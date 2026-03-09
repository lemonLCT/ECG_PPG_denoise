"""预处理流水线骨架。

该模块按顺序执行多个变换，原理是将复杂预处理拆分为可复用的小步骤，
并通过有序组合实现可配置的数据处理流程。
"""

from typing import Any

from .base import BaseTransform


class PreprocessPipeline:
    """顺序执行预处理步骤的容器。"""

    def __init__(self, transforms: list[BaseTransform]) -> None:
        self.transforms = transforms

    def run(self, sample: dict[str, Any]) -> dict[str, Any]:
        """运行流水线。

        该方法只负责调度，不包含任何具体信号处理算法。
        """
        output = sample
        for transform in self.transforms:
            output = transform(output)
        return output

