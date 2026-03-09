"""配置结构测试。

原理：通过默认配置对象断言关键字段存在，保证配置层基本可用。
"""

from ecg_ppg_denoise.config.schema import ExperimentConfig


def test_default_config_has_runtime_name() -> None:
    """默认配置应包含非空实验名。"""
    cfg = ExperimentConfig()
    assert isinstance(cfg.runtime.experiment_name, str)
    assert cfg.runtime.experiment_name != ""

