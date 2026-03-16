from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
sys.modules.pop("utils", None)

from dataset import QTDataset, unpack_qt_return

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import compute_ecg_metrics


def test_unpack_qt_return_and_dataset_shapes() -> None:
    x_train = np.ones((2, 16), dtype=np.float32)
    y_train = np.zeros((2, 16), dtype=np.float32)
    x_test = np.ones((1, 16), dtype=np.float32)
    y_test = np.zeros((1, 16), dtype=np.float32)

    train_x, train_y, test_x, test_y = unpack_qt_return((x_train, y_train, x_test, y_test))
    assert train_x.shape == (2, 1, 16)
    assert train_y.shape == (2, 1, 16)
    assert test_x.shape == (1, 1, 16)
    assert test_y.shape == (1, 1, 16)

    dataset = QTDataset(noisy_ecg=x_train, clean_ecg=y_train)
    sample = dataset[1]
    assert set(sample.keys()) == {"clean_ecg", "noisy_ecg", "clean_ppg", "noisy_ppg", "modality_mask"}
    assert sample["clean_ecg"].shape == (1, 16)
    assert sample["noisy_ecg"].shape == (1, 16)
    assert sample["clean_ppg"].shape == (1, 16)
    assert sample["noisy_ppg"].shape == (1, 16)
    assert torch.equal(sample["modality_mask"], torch.tensor([1.0, 0.0], dtype=torch.float32))


def test_compute_ecg_metrics_has_expected_keys() -> None:
    n, t = 8, 256
    clean = np.random.randn(n, 1, t).astype(np.float32)
    noisy = clean + 0.20 * np.random.randn(n, 1, t).astype(np.float32)
    denoised = clean + 0.05 * np.random.randn(n, 1, t).astype(np.float32)

    metrics = compute_ecg_metrics(clean_ecg=clean, noisy_ecg=noisy, denoised_ecg=denoised)
    expected = {"SSD", "MAD", "PRD", "COS_SIM", "SNR_in", "SNR_out", "SNR_improvement"}
    assert set(metrics.keys()) == expected
    for value in metrics.values():
        assert np.isfinite(value) or np.isnan(value)
