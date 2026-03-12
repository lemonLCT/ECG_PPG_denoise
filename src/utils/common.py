"""通用工具函数。"""

from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from config import ExperimentConfig


def seed_everything(seed: int) -> None:
    """固定随机种子，尽量保证实验可复现。"""
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
    """确保目录存在，并返回 `Path` 对象。"""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def dump_config_snapshot(config: ExperimentConfig, output_dir: str | Path) -> Path:
    """将实验配置写入 YAML 快照。"""
    out_dir = ensure_dir(output_dir)
    target = out_dir / "config_snapshot.yaml"
    target.write_text(
        yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    config: ExperimentConfig,
) -> Path:
    """保存训练 checkpoint。"""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """加载 checkpoint，并按需恢复优化器与缩放器状态。"""
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return payload


def generate_demo_signals(batch_size: int = 1, signal_length: int = 512, seed: int = 42) -> Dict[str, torch.Tensor]:
    """生成最小可运行的 ECG/PPG 带噪示例信号。"""
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
