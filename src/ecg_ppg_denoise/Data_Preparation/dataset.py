"""兼容入口：转发到 src/ecg_ppg_denoise/data/dataset.py。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_ppg_denoise.data.dataset import (  # noqa: E402
    ECGTorchDataset,
    build_dataloaders,
    get_train_test_dataset,
    prepare_workdir,
)

__all__ = [
    "ECGTorchDataset",
    "build_dataloaders",
    "get_train_test_dataset",
    "prepare_workdir",
]


if __name__ == "__main__":
    train_ds, test_ds = get_train_test_dataset(noise_version=1)
    print(f"train dataset size: {len(train_ds)}")
    print(f"test dataset size: {len(test_ds)}")
    print(f"train sample keys: {list(train_ds[0].keys())}")
