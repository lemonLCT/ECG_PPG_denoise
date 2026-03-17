from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(device: str) -> torch.device:
    """解析设备字符串；当 CUDA 不可用时回退到 CPU。"""
    dev = device.strip().lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(dev)


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，并返回 `Path`。"""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def dump_config_snapshot(config: dict[str, Any], output_dir: str | Path) -> Path:
    """将配置字典写入 YAML 快照。"""
    out_dir = ensure_dir(output_dir)
    target = out_dir / "config_snapshot.yaml"
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
) -> Path:
    """保存 checkpoint。"""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """加载 checkpoint，并按需恢复优化器与 scaler。"""
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def generate_demo_signals(batch_size: int = 1, signal_length: int = 512, seed: int = 42) -> dict[str, torch.Tensor]:
    """生成最小可运行 ECG/PPG 带噪示例信号。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, signal_length, dtype=np.float32)[None, :]
    freqs = rng.uniform(1.0, 4.0, size=(batch_size, 1)).astype(np.float32)
    phase = rng.uniform(0.0, np.pi, size=(batch_size, 1)).astype(np.float32)
    clean_ecg = np.sin(2.0 * np.pi * freqs * t + phase)
    clean_ppg = np.cos(2.0 * np.pi * (0.7 * freqs) * t + 0.3 * phase)
    noisy_ecg = clean_ecg + rng.normal(0.0, 0.2, size=clean_ecg.shape).astype(np.float32)
    noisy_ppg = clean_ppg + rng.normal(0.0, 0.2, size=clean_ppg.shape).astype(np.float32)
    return {
        "noisy_ecg": torch.from_numpy(noisy_ecg[:, None, :]).float(),
        "noisy_ppg": torch.from_numpy(noisy_ppg[:, None, :]).float(),
    }
