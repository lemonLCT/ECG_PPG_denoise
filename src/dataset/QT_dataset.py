from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Sequence, Tuple

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


def unpack_qt_return(dataset_return: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(dataset_return) != 4:
        raise ValueError("Data_Preparation 返回值长度必须是 4: [X_train, y_train, X_test, y_test]")
    x_train, y_train, x_test, y_test = dataset_return
    return (
        _to_channel_first(x_train, "X_train"),
        _to_channel_first(y_train, "y_train"),
        _to_channel_first(x_test, "X_test"),
        _to_channel_first(y_test, "y_test"),
    )


class QTCompatibleDataset(Dataset):
    """将 QT ECG-only 数据适配为当前统一训练接口。"""

    def __init__(self, noisy_ecg: np.ndarray, clean_ecg: np.ndarray) -> None:
        if noisy_ecg.shape != clean_ecg.shape:
            raise ValueError(f"noisy_ecg 和 clean_ecg 形状不一致: {noisy_ecg.shape} vs {clean_ecg.shape}")
        self.noisy_ecg = torch.from_numpy(noisy_ecg).float()
        self.clean_ecg = torch.from_numpy(clean_ecg).float()
        self.noisy_ppg = torch.zeros_like(self.noisy_ecg)
        self.clean_ppg = torch.zeros_like(self.clean_ecg)

    def __len__(self) -> int:
        return int(self.noisy_ecg.shape[0])

    def __getitem__(self, index: int):
        return {
            "clean_ecg": self.clean_ecg[index],
            "noisy_ecg": self.noisy_ecg[index],
            "clean_ppg": self.clean_ppg[index],
            "noisy_ppg": self.noisy_ppg[index],
            "modality_mask": torch.tensor([1.0, 0.0], dtype=torch.float32),
        }


def build_qt_train_val_datasets(noise_version: int = 1) -> Tuple[Dataset, Dataset]:
    project_root = Path(__file__).resolve().parents[2]
    src_root = Path(__file__).resolve().parents[1]
    candidate_data_roots = [project_root / "dataset", src_root / "dataset"]

    selected_data_root: Path | None = None
    for root in candidate_data_roots:
        if (root / "qt-database-1.0.0").exists() and (root / "mit-bih-noise-stress-test-database-1.0.0").exists():
            selected_data_root = root
            break
    if selected_data_root is None:
        raise FileNotFoundError(
            "未找到 QT/NSTDB 数据目录。请确保以下任一位置包含 qt-database-1.0.0 和 mit-bih-noise-stress-test-database-1.0.0: "
            f"{candidate_data_roots[0]} 或 {candidate_data_roots[1]}"
        )

    from Data_Preparation.data_preparation import Data_Preparation

    @contextmanager
    def _chdir(path: Path):
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    runtime_cwd = selected_data_root.parent
    with _chdir(runtime_cwd):
        dataset_return = Data_Preparation(noise_version=noise_version)

    x_train, y_train, x_test, y_test = unpack_qt_return(dataset_return)
    train_ds = QTCompatibleDataset(noisy_ecg=x_train, clean_ecg=y_train)
    val_ds = QTCompatibleDataset(noisy_ecg=x_test, clean_ecg=y_test)
    return train_ds, val_ds
