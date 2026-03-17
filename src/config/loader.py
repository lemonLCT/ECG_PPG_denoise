from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "base.yaml"
REQUIRED_TOP_LEVEL_KEYS = ("runtime", "data", "path", "loss", "model", "train")
REQUIRED_MODEL_KEYS = ("main_model", "ecg_encoder", "ppg_encoder", "conditional_model", "diffusion")


def _require_positive_int(value: Any, dotted_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{dotted_name}` 必须是正整数，实际为 {value!r}")
    return value


def _require_non_negative_int(value: Any, dotted_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"`{dotted_name}` 必须是非负整数，实际为 {value!r}")
    return value


def _require_positive_float(value: Any, dotted_name: str) -> float:
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ValueError(f"`{dotted_name}` 必须是正数，实际为 {value!r}")
    return float(value)


def _validate_config(payload: dict[str, Any]) -> None:
    runtime_cfg = payload["runtime"]
    data_cfg = payload["data"]
    path_cfg = payload["path"]
    loss_cfg = payload["loss"]
    model_cfg = payload["model"]
    train_cfg = payload["train"]

    if not isinstance(runtime_cfg.get("experiment_name"), str) or not runtime_cfg["experiment_name"].strip():
        raise ValueError("`runtime.experiment_name` 必须是非空字符串")
    if not isinstance(runtime_cfg.get("device"), str) or not runtime_cfg["device"].strip():
        raise ValueError("`runtime.device` 必须是非空字符串")
    if not isinstance(runtime_cfg.get("use_amp"), bool):
        raise ValueError("`runtime.use_amp` 必须是 bool")
    _require_non_negative_int(int(runtime_cfg.get("seed", 0)), "runtime.seed")

    _require_non_negative_int(int(data_cfg.get("num_workers", 0)), "data.num_workers")
    _require_positive_int(int(data_cfg.get("window_length", 0)), "data.window_length")
    _require_positive_int(int(data_cfg.get("window_stride", 0)), "data.window_stride")
    _require_positive_int(int(data_cfg.get("synthetic_num_samples", 0)), "data.synthetic_num_samples")
    val_ratio = data_cfg.get("val_ratio", 0.0)
    if not isinstance(val_ratio, (int, float)) or not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError(f"`data.val_ratio` 必须在 [0, 1) 内，实际为 {val_ratio!r}")

    for key in ("dataset_path", "checkpoint_path", "infer_input_path", "eval_input_path"):
        value = path_cfg.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"`path.{key}` 必须是字符串")
    for key in ("train_output_dir", "infer_output_path", "eval_output_dir"):
        value = path_cfg.get(key, "")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`path.{key}` 必须是非空字符串")

    for name in ("ecg_encoder", "ppg_encoder"):
        encoder_cfg = model_cfg[name]
        _require_positive_int(int(encoder_cfg.get("input_channels", 0)), f"model.{name}.input_channels")
        _require_positive_int(int(encoder_cfg.get("branch_channels", 0)), f"model.{name}.branch_channels")
        kernels = encoder_cfg.get("kernel_sizes", [])
        if not isinstance(kernels, list) or not kernels:
            raise ValueError(f"`model.{name}.kernel_sizes` 必须是非空列表")
        for idx, kernel in enumerate(kernels):
            _require_positive_int(int(kernel), f"model.{name}.kernel_sizes[{idx}]")
        _require_positive_int(int(encoder_cfg.get("output_channels", 0)), f"model.{name}.output_channels")
        if not isinstance(encoder_cfg.get("use_projection"), bool):
            raise ValueError(f"`model.{name}.use_projection` 必须是 bool")
        _require_positive_int(int(encoder_cfg.get("gn_groups", 0)), f"model.{name}.gn_groups")

    conditional_cfg = model_cfg["conditional_model"]
    _require_positive_int(int(conditional_cfg.get("cond_channels", 0)), "model.conditional_model.cond_channels")
    _require_positive_int(int(conditional_cfg.get("joint_channels", 0)), "model.conditional_model.joint_channels")
    _require_positive_int(int(conditional_cfg.get("base_channels", 0)), "model.conditional_model.base_channels")
    _require_positive_int(int(conditional_cfg.get("gn_groups", 0)), "model.conditional_model.gn_groups")

    diffusion_cfg = model_cfg["diffusion"]
    _require_positive_int(int(diffusion_cfg.get("num_steps", 0)), "model.diffusion.num_steps")
    beta_start = _require_positive_float(diffusion_cfg.get("beta_start", 0.0), "model.diffusion.beta_start")
    beta_end = _require_positive_float(diffusion_cfg.get("beta_end", 0.0), "model.diffusion.beta_end")
    if beta_end <= beta_start:
        raise ValueError("`model.diffusion.beta_end` 必须大于 `model.diffusion.beta_start`")

    _require_positive_float(loss_cfg.get("diffusion_weight", 0.0), "loss.diffusion_weight")
    derivative_weight = loss_cfg.get("derivative_weight", 0.0)
    if not isinstance(derivative_weight, (int, float)) or float(derivative_weight) < 0.0:
        raise ValueError("`loss.derivative_weight` 必须是非负数")

    stage_name = train_cfg.get("stage_name")
    if not isinstance(stage_name, str) or stage_name not in {"ecg_pretrain", "ppg_pretrain", "joint"}:
        raise ValueError("`train.stage_name` 必须是 ecg_pretrain / ppg_pretrain / joint 之一")
    _require_positive_int(int(train_cfg.get("batch_size", 0)), "train.batch_size")
    _require_positive_int(int(train_cfg.get("val_batch_size", 0)), "train.val_batch_size")
    _require_positive_int(int(train_cfg.get("epochs", 0)), "train.epochs")
    _require_positive_float(train_cfg.get("lr", 0.0), "train.lr")
    weight_decay = train_cfg.get("weight_decay", 0.0)
    if not isinstance(weight_decay, (int, float)) or float(weight_decay) < 0.0:
        raise ValueError("`train.weight_decay` 必须是非负数")
    _require_positive_float(train_cfg.get("grad_clip", 0.0), "train.grad_clip")
    _require_positive_int(int(train_cfg.get("save_every_epochs", 0)), "train.save_every_epochs")
    _require_positive_int(int(train_cfg.get("max_steps_per_epoch", 0)), "train.max_steps_per_epoch")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """读取 YAML 配置，并返回经过基础校验的嵌套字典。"""
    target = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not target.exists():
        raise FileNotFoundError(f"配置文件不存在: {target}")

    with target.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是 dict: {target}")

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in payload or not isinstance(payload[key], dict):
            raise ValueError(f"配置缺少必需字段 `{key}`，且其值必须为 dict")

    model_cfg = payload["model"]
    for key in REQUIRED_MODEL_KEYS:
        if key not in model_cfg or not isinstance(model_cfg[key], dict):
            raise ValueError(f"`model.{key}` 缺失，且其值必须为 dict")

    path_dataset = payload["path"].get("dataset_path", "")
    if path_dataset and not payload["data"].get("data_path"):
        payload["data"]["data_path"] = path_dataset

    _validate_config(payload)
    return deepcopy(payload)
