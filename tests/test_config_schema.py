"""配置结构测试。"""

from ecg_ppg_denoise.config.loader import load_experiment_config
from ecg_ppg_denoise.config.schema import ExperimentConfig


def test_default_config_has_runtime_name() -> None:
    """默认配置对象应包含非空实验名。"""
    cfg = ExperimentConfig()
    assert isinstance(cfg.runtime.experiment_name, str)
    assert cfg.runtime.experiment_name != ""


def test_yaml_loader_selects_model_and_stage() -> None:
    """YAML loader 应能按模型名和阶段名选择配置。"""
    cfg = load_experiment_config(model_name="modality_flexible_conditional_diffusion_debug", stage="joint")
    assert cfg.model.name == "modality_flexible_conditional_diffusion_debug"
    assert cfg.model.base_channels == 32
    assert cfg.train.stage_name == "joint"
    assert cfg.train.max_steps_per_epoch == 2500
