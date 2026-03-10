"""基于现有 Data_Preparation 的 PyTorch Dataset 适配层。"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    DataLoader = Any  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Data_Preparation.data_preparation import Data_Preparation

DEFAULT_WORK_DIRS = ("data", "outputs", "outputs/checkpoints", "outputs/logs")


def _ensure_torch_available() -> None:
    if torch is None:
        raise ModuleNotFoundError("未检测到 torch，请先安装 PyTorch。")


def prepare_workdir(project_root: str | Path | None = None) -> Path:
    """创建并返回训练所需工作目录。"""

    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    for relative_dir in DEFAULT_WORK_DIRS:
        (root / relative_dir).mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def _working_directory(target_dir: Path):
    previous_dir = Path.cwd()
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


class ECGTorchDataset(Dataset):
    """ECG 去噪 map-style Dataset。"""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        split: str,
        dtype: "torch.dtype | None" = None,
    ) -> None:
        _ensure_torch_available()

        x_np = np.asarray(inputs)
        y_np = np.asarray(targets)
        if x_np.shape != y_np.shape:
            raise ValueError(
                f"{split} 集输入和标签形状不一致: {x_np.shape} vs {y_np.shape}"
            )
        if x_np.ndim != 3:
            raise ValueError(f"{split} 集应为 [N, T, C]，当前维度: {x_np.ndim}")
        if x_np.shape[0] == 0:
            raise ValueError(f"{split} 集为空，无法构建 Dataset。")

        tensor_dtype = dtype or torch.float32
        self.inputs = torch.as_tensor(x_np, dtype=tensor_dtype)
        self.targets = torch.as_tensor(y_np, dtype=tensor_dtype)
        self.split = split

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"{self.split} 集索引越界: {index}")
        return {
            "noisy": self.inputs[index],
            "clean": self.targets[index],
            "index": index,
            "split": self.split,
        }

    @property
    def shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(self.inputs.shape), tuple(self.targets.shape)


def get_train_test_dataset(
    noise_version: int = 1,
    dtype: "torch.dtype | None" = None,
    project_root: str | Path | None = None,
) -> tuple[ECGTorchDataset, ECGTorchDataset]:
    """调用预处理并直接返回 train/test Dataset。"""

    _ensure_torch_available()
    root = prepare_workdir(project_root=project_root)
    with _working_directory(root):
        x_train, y_train, x_test, y_test = Data_Preparation(noise_version=noise_version)

    train_dataset = ECGTorchDataset(x_train, y_train, split="train", dtype=dtype)
    test_dataset = ECGTorchDataset(x_test, y_test, split="test", dtype=dtype)
    return train_dataset, test_dataset


def build_dataloaders(
    batch_size: int = 64,
    test_batch_size: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    noise_version: int = 1,
    dtype: "torch.dtype | None" = None,
    project_root: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """返回 train/test DataLoader。"""

    _ensure_torch_available()
    train_dataset, test_dataset = get_train_test_dataset(
        noise_version=noise_version,
        dtype=dtype,
        project_root=project_root,
    )

    final_test_batch_size = test_batch_size or batch_size
    final_pin_memory = bool(pin_memory and torch.cuda.is_available())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=final_pin_memory,
        drop_last=drop_last,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=final_test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=final_pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader


def _print_dataset_preview(name: str, dataset: ECGTorchDataset) -> None:
    x_shape, y_shape = dataset.shape
    print(f"{name} 集输入形状: {x_shape}")
    print(f"{name} 集标签形状: {y_shape}")
    print(f"{name} 集样本数量: {len(dataset)}")

    sample = dataset[0]
    print(f"{name} 集首个样本键: {list(sample.keys())}")
    print(f"{name} 集首个 noisy 形状: {tuple(sample['noisy'].shape)}")
    print(f"{name} 集首个 clean 形状: {tuple(sample['clean'].shape)}")
    print(f"{name} 集首个 noisy 内容:\n{sample['noisy'].squeeze()}")
    print(f"{name} 集首个 clean 内容:\n{sample['clean'].squeeze()}")


if __name__ == "__main__":
    train_ds, test_ds = get_train_test_dataset(noise_version=1)
    _print_dataset_preview("train", train_ds)
    _print_dataset_preview("test", test_ds)
