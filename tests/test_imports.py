"""导入级别烟雾测试。

原理：通过核心模块导入验证工程骨架是否存在循环依赖或路径问题。
"""


def test_import_core_modules() -> None:
    """核心模块应可正常导入。"""
    import ecg_ppg_denoise.config.schema  # noqa: F401
    import ecg_ppg_denoise.models  # noqa: F401
    import ecg_ppg_denoise.losses  # noqa: F401
    import ecg_ppg_denoise.trainers.runner  # noqa: F401
