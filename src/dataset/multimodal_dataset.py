from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DataConfig
from .QTdataset import build_qt_train_val_datasets


REQUIRED_KEYS = ("clean_ecg", "noisy_ecg", "clean_ppg", "noisy_ppg")


def _to_numpy_array(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    elif arr.ndim == 3 and arr.shape[1] == 1:
        pass
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.transpose(arr, (0, 2, 1))
    else:
        raise ValueError(f"{name} 形状不合法，期望 [N,T] 或 [N,1,T]，实际 {arr.shape}")
    return arr


def _window_data(data: np.ndarray, window_length: int, stride: int) -> np.ndarray:
    num_samples, channels, total_length = data.shape
    if channels != 1:
        raise ValueError(f"仅支持单通道输入，实际 channels={channels}")
    if total_length < window_length:
        pad_width = window_length - total_length
        data = np.pad(data, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
        total_length = window_length
    if total_length == window_length:
        return data

    windows = []
    for idx in range(num_samples):
        starts = list(range(0, total_length - window_length + 1, stride))
        if starts[-1] != total_length - window_length:
            starts.append(total_length - window_length)
        for start in starts:
            end = start + window_length
            windows.append(data[idx : idx + 1, :, start:end])
    return np.concatenate(windows, axis=0)


def load_multimodal_arrays(path: str | Path) -> Dict[str, np.ndarray]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".npz":
        raw = np.load(file_path, allow_pickle=True)
        payload = {key: raw[key] for key in REQUIRED_KEYS}
    elif suffix in {".pt", ".pth"}:
        raw = torch.load(file_path, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError(".pt/.pth 数据必须是 dict，并包含 clean/noisy ECG/PPG 键")
        payload = {key: raw[key] for key in REQUIRED_KEYS}
    elif suffix == ".npy":
        raw = np.load(file_path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
            obj = raw.item()
            if not isinstance(obj, dict):
                raise ValueError(".npy object 数据必须是 dict")
            payload = {key: obj[key] for key in REQUIRED_KEYS}
        else:
            raise ValueError(".npy 目前仅支持保存 dict object 的格式")
    else:
        raise ValueError(f"不支持的数据后缀: {suffix}，请使用 .npz/.pt/.npy")

    return {name: _to_numpy_array(value, name) for name, value in payload.items()}


class MultimodalSignalDataset(Dataset[Dict[str, torch.Tensor]]):
    """通用 ECG/PPG 数据集。"""

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        val_ratio: float = 0.1,
        window_length: int = 512,
        window_stride: int = 256,
        seed: int = 42,
        source_name: str = "generic",
    ) -> None:
        self.source_name = source_name
        arrays = load_multimodal_arrays(data_path)
        arrays = {
            key: _window_data(value, window_length=window_length, stride=window_stride)
            for key, value in arrays.items()
        }
        total = arrays["clean_ecg"].shape[0]
        indices = np.arange(total)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        val_count = int(total * val_ratio)
        if val_ratio > 0.0 and val_count == 0 and total > 1:
            val_count = 1
        train_idx = indices[val_count:]
        val_idx = indices[:val_count]
        chosen_idx = train_idx if split == "train" else val_idx
        if chosen_idx.size == 0:
            chosen_idx = indices

        self.clean_ecg = torch.from_numpy(arrays["clean_ecg"][chosen_idx]).float()
        self.noisy_ecg = torch.from_numpy(arrays["noisy_ecg"][chosen_idx]).float()
        self.clean_ppg = torch.from_numpy(arrays["clean_ppg"][chosen_idx]).float()
        self.noisy_ppg = torch.from_numpy(arrays["noisy_ppg"][chosen_idx]).float()

    def __len__(self) -> int:
        return int(self.clean_ecg.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "clean_ecg": self.clean_ecg[index],
            "noisy_ecg": self.noisy_ecg[index],
            "clean_ppg": self.clean_ppg[index],
            "noisy_ppg": self.noisy_ppg[index],
            "modality_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
        }


class SyntheticMultimodalDataset(Dataset[Dict[str, torch.Tensor]]):
    """无真实数据时的最小可运行随机数据集。"""

    def __init__(self, num_samples: int = 64, signal_length: int = 512, seed: int = 42) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, 1.0, signal_length, dtype=np.float32)[None, :]
        freqs = rng.uniform(1.0, 6.0, size=(num_samples, 1)).astype(np.float32)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(num_samples, 1)).astype(np.float32)

        clean_ecg = np.sin(2.0 * np.pi * freqs * t + phases)
        clean_ppg = np.cos(2.0 * np.pi * (freqs * 0.8) * t + 0.5 * phases)
        noise_ecg = rng.normal(0.0, 0.15, size=clean_ecg.shape).astype(np.float32)
        noise_ppg = rng.normal(0.0, 0.15, size=clean_ppg.shape).astype(np.float32)

        self.clean_ecg = torch.from_numpy(clean_ecg[:, None, :]).float()
        self.noisy_ecg = torch.from_numpy((clean_ecg + noise_ecg)[:, None, :]).float()
        self.clean_ppg = torch.from_numpy(clean_ppg[:, None, :]).float()
        self.noisy_ppg = torch.from_numpy((clean_ppg + noise_ppg)[:, None, :]).float()

    def __len__(self) -> int:
        return int(self.clean_ecg.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "clean_ecg": self.clean_ecg[index],
            "noisy_ecg": self.noisy_ecg[index],
            "clean_ppg": self.clean_ppg[index],
            "noisy_ppg": self.noisy_ppg[index],
            "modality_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
        }


def build_train_val_datasets(
    cfg: DataConfig,
    seed: int,
    use_qt_dataset: bool = False,
    qt_noise_version: int = 1,
) -> Tuple[Dataset, Dataset]:
    if use_qt_dataset:
        return build_qt_train_val_datasets(
            noise_version=qt_noise_version,
            val_ratio=cfg.val_ratio,
            seed=seed,
        )

    if cfg.data_path and Path(cfg.data_path).exists():
        train_ds: Dataset = MultimodalSignalDataset(
            data_path=cfg.data_path,
            split="train",
            val_ratio=cfg.val_ratio,
            window_length=cfg.window_length,
            window_stride=cfg.window_stride,
            seed=seed,
        )
        val_ds: Dataset = MultimodalSignalDataset(
            data_path=cfg.data_path,
            split="val",
            val_ratio=cfg.val_ratio,
            window_length=cfg.window_length,
            window_stride=cfg.window_stride,
            seed=seed,
        )
        return train_ds, val_ds

    synthetic = SyntheticMultimodalDataset(
        num_samples=cfg.synthetic_num_samples,
        signal_length=cfg.window_length,
        seed=seed,
    )
    val_size = max(1, int(len(synthetic) * cfg.val_ratio))
    train_size = max(1, len(synthetic) - val_size)
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(synthetic, [train_size, val_size], generator=generator)
    return train_ds, val_ds
