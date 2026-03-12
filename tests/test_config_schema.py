"""配置结构测试。"""

from ecg_ppg_denoise.config.loader import load_experiment_config
from ecg_ppg_denoise.config.schema import ExperimentConfig


def test_default_config_has_runtime_name() -> None:
    """默认配置对象应包含非空实验名。"""
    cfg = ExperimentConfig()
    assert isinstance(cfg.runtime.experiment_name, str)
    assert cfg.runtime.experiment_name != ""
    assert cfg.path.train_output_dir != ""
    assert cfg.model.cond_channels == 96


def test_yaml_loader_selects_model_and_stage() -> None:
    """YAML loader 应能按模型名和阶段名选择配置。"""
    cfg = load_experiment_config(model_name="modality_flexible_conditional_diffusion_debug", stage="joint")
    assert cfg.model.name == "modality_flexible_conditional_diffusion_debug"
    assert cfg.model.base_channels == 32
    assert cfg.model.cond_channels == 96
    assert cfg.model.ecg_encoder.branch_channels == 16
    assert cfg.model.ecg_encoder.use_projection is False
    assert cfg.model.ecg_encoder.output_channels == 96
    assert cfg.train.stage_name == "joint"
    assert cfg.train.max_steps_per_epoch == 2500
    assert cfg.train.batch_size == 32
