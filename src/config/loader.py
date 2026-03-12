from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from .config import ExperimentConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.yaml"


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml_payload(config_path: Path | None = None) -> Dict[str, Any]:
    target = config_path or DEFAULT_CONFIG_PATH
    if not target.exists():
        raise FileNotFoundError(f"配置文件不存在: {target}")
    with target.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是映射类型: {target}")
    return payload


def _model_root(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload.get("model") or payload.get("models", {})


def _train_root(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload.get("train") or payload.get("training", {})


def list_available_models(config_path: Path | None = None) -> list[str]:
    payload = _read_yaml_payload(config_path)
    variants = _model_root(payload).get("variants", {})
    return sorted(variants.keys())


def list_available_train_stages(config_path: Path | None = None) -> list[str]:
    payload = _read_yaml_payload(config_path)
    stages = _train_root(payload).get("stages", {})
    return sorted(stages.keys())


def select_model_section(payload: Dict[str, Any], model_name: str | None = None) -> Dict[str, Any]:
    model_section = _model_root(payload)
    variants = model_section.get("variants", {})
    selected_name = model_name or model_section.get("selected")
    if not selected_name:
        raise ValueError("配置文件缺少 model.selected，且未传入 model_name")
    if selected_name not in variants:
        raise KeyError(f"未找到模型配置: {selected_name}")
    model_payload = deepcopy(variants[selected_name])
    model_payload["name"] = selected_name
    return model_payload


def select_train_section(payload: Dict[str, Any], stage: str | None = None) -> Dict[str, Any]:
    train_section = _train_root(payload)
    common_payload = deepcopy(train_section.get("common", {}))
    stages = train_section.get("stages", {})
    selected_stage = stage or train_section.get("selected_stage") or "ecg_pretrain"
    if selected_stage not in stages:
        raise KeyError(f"未找到训练阶段配置: {selected_stage}")
    train_payload = _merge_dicts(common_payload, stages[selected_stage])
    train_payload["stage_name"] = selected_stage
    return train_payload


def load_experiment_config(
    config_path: Path | None = None,
    model_name: str | None = None,
    stage: str | None = None,
) -> ExperimentConfig:
    payload = _read_yaml_payload(config_path)
    flat_payload = {
        "runtime": deepcopy(payload.get("runtime", {})),
        "data": deepcopy(payload.get("data", {})),
        "path": deepcopy(payload.get("path", {})),
        "loss": deepcopy(payload.get("loss", {})),
        "model": select_model_section(payload, model_name=model_name),
        "train": select_train_section(payload, stage=stage),
    }
    cfg = ExperimentConfig.from_flat_dict(flat_payload)
    cfg.validate()
    return cfg
