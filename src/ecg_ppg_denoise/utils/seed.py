"""随机种子工具。

该模块统一设置常见随机源，原理是固定随机状态以提高实验可复现性。
"""

import random

try:
    # 可选导入 NumPy；未安装时自动降级到 Python 随机源。
    import numpy as np
except ImportError:  # pragma: no cover - 依赖缺失时的兜底分支
    np = None


def set_global_seed(seed: int) -> None:
    """设置 Python 和 NumPy 的随机种子。"""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
