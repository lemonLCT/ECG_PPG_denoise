"""QT 适配与评估指标 smoke 测试。"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

from ecg_ppg_denoise.data.QT_dataset import QTCompatibleDataset, unpack_qt_return

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import compute_ecg_metrics


def test_qt_unpack_and_dataset_shapes() -> None:
    n, t = 4, 512
    x_train = np.random.randn(n, t, 1).astype(np.float32)
    y_train = np.random.randn(n, t, 1).astype(np.float32)
    x_test = np.random.randn(n, t, 1).astype(np.float32)
    y_test = np.random.randn(n, t, 1).astype(np.float32)

    x_train_cf, y_train_cf, x_test_cf, y_test_cf = unpack_qt_return([x_train, y_train, x_test, y_test])
    assert x_train_cf.shape == (n, 1, t)
    assert y_train_cf.shape == (n, 1, t)
    assert x_test_cf.shape == (n, 1, t)
    assert y_test_cf.shape == (n, 1, t)

    ds = QTCompatibleDataset(noisy_ecg=x_train_cf, clean_ecg=y_train_cf)
    sample = ds[0]
    assert sample["clean_ecg"].shape == (1, t)
    assert sample["noisy_ecg"].shape == (1, t)
    assert sample["clean_ppg"].shape == (1, t)
    assert sample["noisy_ppg"].shape == (1, t)
    assert float(sample["clean_ppg"].abs().sum().item()) == 0.0
    assert sample["modality_mask"].tolist() == [1.0, 0.0]


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
