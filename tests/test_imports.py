"""导入级别烟雾测试。"""


def test_import_core_modules() -> None:
    import config  # noqa: F401
    import losses  # noqa: F401
    import models  # noqa: F401
    import trainers.runner  # noqa: F401
