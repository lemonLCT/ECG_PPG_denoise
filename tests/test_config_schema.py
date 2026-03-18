from __future__ import annotations

from config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_exists_and_has_required_sections() -> None:
    cfg = load_config()
    assert DEFAULT_CONFIG_PATH.exists()
    assert set(cfg.keys()) == {"runtime", "data", "path", "bidmc", "loss", "model", "train"}
    assert cfg["runtime"]["experiment_name"] != ""
    assert cfg["path"]["train_output_dir"] != ""
    assert cfg["bidmc"]["path"]["bidmc_root"] != ""


def test_default_encoder_and_loss_settings() -> None:
    cfg = load_config()
    assert cfg["model"]["ecg_encoder"]["branch_channels"] == 16
    assert cfg["model"]["ecg_encoder"]["use_projection"] is False
    assert cfg["model"]["ppg_encoder"]["branch_channels"] == 16
    assert cfg["model"]["conditional_model"]["cond_channels"] == 96
    assert cfg["loss"]["diffusion_weight"] == 1.0
    assert cfg["loss"]["derivative_weight"] == 0.0
    assert cfg["bidmc"]["data"]["window_length"] == 500
    assert cfg["bidmc"]["data"]["sampling_rate_hz"] == 125.0
