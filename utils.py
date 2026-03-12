"""根目录通用工具。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def generate_demo_signals(batch_size: int = 1, signal_length: int = 512, seed: int = 42) -> Dict[str, torch.Tensor]:
    """生成最小可运行的合成 noisy 信号。"""
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
