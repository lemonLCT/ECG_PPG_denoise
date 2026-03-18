from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import sys
from types import SimpleNamespace

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
sys.modules.pop("utils", None)

from dataset import QTDataset, split_qt_train_val_arrays, unpack_qt_return

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate
from evaluate import build_noise_segment_summary, compute_ecg_metric_arrays, compute_ecg_metrics, summarize_metric_arrays


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


def test_metric_summary_contains_mean_and_std() -> None:
    n, t = 6, 128
    clean = np.random.randn(n, 1, t).astype(np.float32)
    noisy = clean + 0.15 * np.random.randn(n, 1, t).astype(np.float32)
    denoised = clean + 0.03 * np.random.randn(n, 1, t).astype(np.float32)

    metric_arrays = compute_ecg_metric_arrays(clean_ecg=clean, noisy_ecg=noisy, denoised_ecg=denoised)
    summary = summarize_metric_arrays(metric_arrays)

    assert set(summary.keys()) == {"SSD", "MAD", "PRD", "COS_SIM", "SNR_in", "SNR_out", "SNR_improvement"}
    for stats in summary.values():
        assert set(stats.keys()) == {"mean", "std"}
        assert np.isfinite(stats["mean"]) or np.isnan(stats["mean"])
        assert np.isfinite(stats["std"]) or np.isnan(stats["std"])


def test_build_noise_segment_summary_groups_by_noise_level() -> None:
    metric_arrays = {
        "SSD": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "MAD": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
    }
    noise_levels = np.array([0.3, 0.7, 1.2, 1.8], dtype=np.float64)

    summary = build_noise_segment_summary(metric_arrays, noise_levels, segments=(0.2, 0.6, 1.0, 2.0))

    assert set(summary.keys()) == {"0.2<noise<=0.6", "0.6<noise<=1.0", "1.0<noise<=2.0"}
    assert summary["0.2<noise<=0.6"]["SSD"]["mean"] == 1.0
    assert summary["0.6<noise<=1.0"]["MAD"]["mean"] == 0.2
    assert summary["1.0<noise<=2.0"]["SSD"]["mean"] == 3.5


def test_split_qt_train_val_arrays_uses_train_only_and_keeps_7_3_ratio() -> None:
    total = 10
    train_noisy = np.arange(total * 8, dtype=np.float32).reshape(total, 8)
    train_clean = train_noisy + 1000.0

    split_train_noisy, split_train_clean, val_noisy, val_clean = split_qt_train_val_arrays(
        train_noisy_ecg=train_noisy,
        train_clean_ecg=train_clean,
        val_ratio=0.3,
        seed=123,
    )

    assert split_train_noisy.shape[0] == 7
    assert split_train_clean.shape[0] == 7
    assert val_noisy.shape[0] == 3
    assert val_clean.shape[0] == 3

    train_rows = {tuple(row.tolist()) for row in split_train_noisy.reshape(split_train_noisy.shape[0], -1)}
    val_rows = {tuple(row.tolist()) for row in val_noisy.reshape(val_noisy.shape[0], -1)}
    full_rows = {tuple(row.tolist()) for row in train_noisy.reshape(train_noisy.shape[0], -1)}
    assert train_rows.isdisjoint(val_rows)
    assert train_rows | val_rows == full_rows


def test_load_eval_arrays_can_switch_to_bidmc(monkeypatch) -> None:
    calls: list[str] = []

    class FakeDataset:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            base = np.full((1, 8), float(index), dtype=np.float32)
            return {
                "clean_ecg": base,
                "noisy_ecg": base + 0.1,
                "clean_ppg": base + 0.2,
                "noisy_ppg": base + 0.3,
                "modality_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
            }

    def fake_build_bidmc_test_dataset(*, config):
        calls.append("bidmc")
        assert "bidmc" in config
        return FakeDataset()

    monkeypatch.setattr(evaluate, "build_bidmc_test_dataset", fake_build_bidmc_test_dataset)
    args = SimpleNamespace(use_bidmc_dataset=True, use_qt_dataset=False, qt_noise_version=1, input_path=None)

    arrays = evaluate._load_eval_arrays(args, {"bidmc": {"path": {"bidmc_root": "unused"}}})

    assert calls == ["bidmc"]
    assert arrays["clean_ecg"].shape == (2, 1, 8)
    assert arrays["noisy_ppg"].shape == (2, 1, 8)
