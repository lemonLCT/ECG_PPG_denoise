from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_channel_first(arr: np.ndarray, name: str) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 2:
        data = data[:, None, :]
    elif data.ndim == 3 and data.shape[1] == 1:
        pass
    elif data.ndim == 3 and data.shape[2] == 1:
        data = np.transpose(data, (0, 2, 1))
    else:
        raise ValueError(f"{name} 形状不合法，期望 [N,T]/[N,1,T]/[N,T,1]，实际 {data.shape}")
    return data


def unpack_qt_return(dataset_return: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将 Data_Preparation 返回值整理为 channel-first 的 ECG 训练/验证数组。"""
    if len(dataset_return) != 4:
        raise ValueError("Data_Preparation 返回值长度必须是 4: [X_train, y_train, X_test, y_test]")
    x_train, y_train, x_test, y_test = dataset_return
    return (
        _to_channel_first(x_train, "X_train"),
        _to_channel_first(y_train, "y_train"),
        _to_channel_first(x_test, "X_test"),
        _to_channel_first(y_test, "y_test"),
    )


class QTDataset(Dataset[dict[str, torch.Tensor]]):
    """
    QT ECG-only 数据集，适配当前多模态训练接口。

    `__getitem__` 返回:
    - `clean_ecg:[1,T]`
    - `noisy_ecg:[1,T]`
    - `clean_ppg:[1,T]`，固定为 0
    - `noisy_ppg:[1,T]`，固定为 0
    - `modality_mask:[2] = [1,0]`
    """

    def __init__(self, noisy_ecg: np.ndarray, clean_ecg: np.ndarray) -> None:
        if noisy_ecg.shape != clean_ecg.shape:
            raise ValueError(f"noisy_ecg 和 clean_ecg 形状不一致: {noisy_ecg.shape} vs {clean_ecg.shape}")
        self.noisy_ecg = torch.from_numpy(_to_channel_first(noisy_ecg, "noisy_ecg")).float()
        self.clean_ecg = torch.from_numpy(_to_channel_first(clean_ecg, "clean_ecg")).float()
        self.noisy_ppg = torch.zeros_like(self.noisy_ecg)
        self.clean_ppg = torch.zeros_like(self.clean_ecg)
        self.modality_mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.noisy_ecg.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "clean_ecg": self.clean_ecg[index],
            "noisy_ecg": self.noisy_ecg[index],
            "clean_ppg": self.clean_ppg[index],
            "noisy_ppg": self.noisy_ppg[index],
            "modality_mask": self.modality_mask.clone(),
        }


def _resolve_qt_data_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "db" / "QTDB"
    if (data_root / "qt-database-1.0.0").exists() and (
        data_root / "mit-bih-noise-stress-test-database-1.0.0"
    ).exists():
        return data_root
    raise FileNotFoundError(
        "未找到 QT/NSTDB 数据目录。当前 fix 链路要求数据位于 "
        f"{data_root}，且其中包含 qt-database-1.0.0 和 "
        "mit-bih-noise-stress-test-database-1.0.0。"
    )


def load_qt_arrays(noise_version: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    调用 QT Data_Preparation，并返回:
    - `train_noisy_ecg:[N,1,T]`
    - `train_clean_ecg:[N,1,T]`
    - `val_noisy_ecg:[M,1,T]`
    - `val_clean_ecg:[M,1,T]`
    """
    selected_data_root = _resolve_qt_data_root()
    from data.Data_Preparation.data_preparation import Data_Preparation

    dataset_return = Data_Preparation(noise_version=noise_version, data_root=selected_data_root)
    return unpack_qt_return(dataset_return)


def build_qt_train_val_datasets(noise_version: int = 1) -> tuple[QTDataset, QTDataset]:
    """构造 QT 专用训练/验证数据集。"""
    train_noisy_ecg, train_clean_ecg, val_noisy_ecg, val_clean_ecg = load_qt_arrays(noise_version=noise_version)
    train_ds = QTDataset(noisy_ecg=train_noisy_ecg, clean_ecg=train_clean_ecg)
    val_ds = QTDataset(noisy_ecg=val_noisy_ecg, clean_ecg=val_clean_ecg)
    return train_ds, val_ds
