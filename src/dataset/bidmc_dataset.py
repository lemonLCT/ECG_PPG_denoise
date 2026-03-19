from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PPG_NOISE_MODULE_PATH = PROJECT_ROOT / "data" / "ppg_noise_generate.py"
NSTDB_ROOT = PROJECT_ROOT / "data" / "db" / "QTDB" / "mit-bih-noise-stress-test-database-1.0.0"


def _stable_seed(*parts: str, base_seed: int = 0) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) + int(base_seed)


def _normalize_record_name(record_path: Path) -> str:
    return record_path.stem


def _is_waveform_record(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.startswith("bidmc") and not stem.endswith("n")


def _normalize_channel_name(name: str) -> str:
    return "".join(str(name).strip().upper().split())


def _resolve_channel_index(signal_names: Sequence[str], preferred_name: str, fallback_names: Sequence[str] = ()) -> int | None:
    normalized_signal_names = [_normalize_channel_name(name) for name in signal_names]
    candidates = [_normalize_channel_name(preferred_name), *[_normalize_channel_name(name) for name in fallback_names]]

    for candidate in candidates:
        if candidate in normalized_signal_names:
            return normalized_signal_names.index(candidate)

    for candidate in candidates:
        for index, normalized_name in enumerate(normalized_signal_names):
            if normalized_name.endswith(candidate) or candidate in normalized_name:
                return index
    return None


def _load_ppg_noise_module():
    module_name = "project_ppg_noise_generate_runtime"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, PPG_NOISE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 PPG 噪声模块: {PPG_NOISE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _require_bidmc_dict(config: dict[str, Any]) -> dict[str, Any]:
    if "bidmc" not in config or not isinstance(config["bidmc"], dict):
        raise ValueError("配置缺少 `bidmc` 顶层字段")
    return config["bidmc"]


def _resolve_bidmc_root(config: dict[str, Any]) -> Path:
    bidmc_cfg = _require_bidmc_dict(config)
    root = Path(bidmc_cfg["path"]["bidmc_root"]).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"BIDMC 数据目录不存在: {root}")
    return root


def _resolve_nstdb_root(config: dict[str, Any]) -> Path:
    bidmc_cfg = _require_bidmc_dict(config)
    configured_root = str(bidmc_cfg.get("path", {}).get("nstdb_root", "")).strip()
    root = Path(configured_root).expanduser().resolve() if configured_root else NSTDB_ROOT
    if not root.exists():
        raise FileNotFoundError(
            "未找到 NSTDB 根目录: "
            f"{root}。BIDMC 数据集构造默认需要 MIT-BIH Noise Stress Test Database 作为 ECG 噪声源，"
            "请下载后放到该目录，或在 `bidmc.path.nstdb_root` 中显式配置路径。"
        )
    return root


def _list_bidmc_record_ids(root: Path) -> list[str]:
    record_ids = sorted(_normalize_record_name(path) for path in root.glob("bidmc*.hea") if _is_waveform_record(path))
    if not record_ids:
        raise ValueError(f"未在 {root} 下找到可用 BIDMC 波形记录")
    return record_ids


def _split_record_ids(record_ids: Sequence[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    if abs((float(train_ratio) + float(val_ratio) + float(test_ratio)) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1")

    total = len(record_ids)
    if total < 3:
        raise ValueError(f"BIDMC 记录数过少，无法切分 train/val/test: {total}")

    indices = np.arange(total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_count = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio)))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]
    return (
        [record_ids[idx] for idx in train_idx],
        [record_ids[idx] for idx in val_idx],
        [record_ids[idx] for idx in test_idx],
    )


def _compute_window_starts(total_length: int, window_length: int, stride: int) -> list[int]:
    if total_length <= window_length:
        return [0]
    starts = list(range(0, total_length - window_length + 1, stride))
    final_start = total_length - window_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _ensure_length(signal_values: np.ndarray, target_length: int) -> np.ndarray:
    if signal_values.shape[0] >= target_length:
        return signal_values[:target_length]
    pad_width = target_length - signal_values.shape[0]
    return np.pad(signal_values, (0, pad_width), mode="constant")


def _take_cyclic_segment(noise_pool: np.ndarray, start: int, length: int) -> np.ndarray:
    if noise_pool.ndim != 1:
        raise ValueError("噪声池必须是一维数组")
    if noise_pool.size == 0:
        raise ValueError("噪声池不能为空")
    start_idx = int(start) % int(noise_pool.size)
    if start_idx + length <= noise_pool.size:
        return noise_pool[start_idx : start_idx + length].astype(np.float32, copy=False)
    first = noise_pool[start_idx:]
    remaining = length - first.size
    repeats = int(np.ceil(remaining / noise_pool.size))
    second = np.tile(noise_pool, repeats + 1)[:remaining]
    return np.concatenate([first, second], axis=0).astype(np.float32, copy=False)


def _normalize_pair(clean_window: np.ndarray, noisy_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pair_min = float(min(np.min(clean_window), np.min(noisy_window)))
    pair_max = float(max(np.max(clean_window), np.max(noisy_window)))
    if not np.isfinite(pair_min) or not np.isfinite(pair_max):
        raise ValueError("窗口包含非有限数值")
    if abs(pair_max - pair_min) < 1e-8:
        return np.zeros_like(clean_window, dtype=np.float32), np.zeros_like(noisy_window, dtype=np.float32)
    scale = pair_max - pair_min
    clean_norm = 2.0 * ((clean_window - pair_min) / scale) - 1.0
    noisy_norm = 2.0 * ((noisy_window - pair_min) / scale) - 1.0
    return clean_norm.astype(np.float32), noisy_norm.astype(np.float32)


def _scale_noise_window(clean_window: np.ndarray, noise_window: np.ndarray, ratio: float) -> np.ndarray:
    clean_range = float(np.ptp(clean_window))
    noise_range = float(np.ptp(noise_window))
    if clean_range <= 1e-8 or noise_range <= 1e-8:
        return np.zeros_like(noise_window, dtype=np.float32)
    target_noise_range = float(ratio) * clean_range
    alpha = target_noise_range / noise_range
    return (noise_window * alpha).astype(np.float32)


def _load_bw_noise_pools(noise_version: int, nstdb_root: Path) -> tuple[np.ndarray, np.ndarray]:
    if not (nstdb_root / "bw.hea").exists():
        raise FileNotFoundError(
            "未找到 NSTDB 的 `bw` 记录文件，期望目录为: "
            f"{nstdb_root}。请确认目录下至少包含 `bw.hea` 和 `bw.dat`。"
        )
    bw_signals, _ = wfdb.rdsamp(str(nstdb_root / "bw"))
    bw_signals = np.asarray(bw_signals, dtype=np.float32)
    halfway = bw_signals.shape[0] // 2
    bw_noise_channel1_a = bw_signals[:halfway, 0]
    bw_noise_channel1_b = bw_signals[halfway:-1, 0]
    bw_noise_channel2_a = bw_signals[:halfway, 1]
    bw_noise_channel2_b = bw_signals[halfway:-1, 1]
    if noise_version == 1:
        return bw_noise_channel1_a, bw_noise_channel2_b
    if noise_version == 2:
        return bw_noise_channel2_a, bw_noise_channel1_b
    raise ValueError("ecg_noise_version 只能是 1 或 2")


def _build_ppg_generation_params(bidmc_data_cfg: dict[str, Any]):
    module = _load_ppg_noise_module()
    prob = tuple(module.build_prob_from_artifact_types(bidmc_data_cfg["ppg_noise_artifact_types"]))
    artifact_param_path_value = str(bidmc_data_cfg.get("ppg_noise_artifact_param_path", "")).strip()
    artifact_param_path = Path(artifact_param_path_value).expanduser().resolve() if artifact_param_path_value else None
    return module.GenerationParams(
        prob=prob,
        artifact_types=tuple(int(v) for v in bidmc_data_cfg["ppg_noise_artifact_types"]),
        dur_mu=tuple(float(v) for v in bidmc_data_cfg["ppg_noise_dur_mu"]),
        rms_shape=tuple(float(v) for v in bidmc_data_cfg["ppg_noise_rms_shape"]),
        rms_scale=tuple(float(v) for v in bidmc_data_cfg["ppg_noise_rms_scale"]),
        slope_mean=tuple(float(v) for v in bidmc_data_cfg["ppg_noise_slope_mean"]),
        slope_std=tuple(float(v) for v in bidmc_data_cfg["ppg_noise_slope_std"]),
        slope_override=None,
        artifact_param_path=artifact_param_path,
    )


def _generate_ppg_noise_full(record_id: str, signal_length: int, bidmc_data_cfg: dict[str, Any]) -> np.ndarray:
    module = _load_ppg_noise_module()
    params = _build_ppg_generation_params(bidmc_data_cfg)
    config = module.GenerationConfig(
        output_dir=PROJECT_ROOT / "artifacts" / "ppg_noise",
        output_stem=f"{record_id}_ppg_noise",
        duration_samples=int(signal_length),
        sampling_rate_hz=float(bidmc_data_cfg["sampling_rate_hz"]),
        seed=_stable_seed(record_id, "ppg", base_seed=int(bidmc_data_cfg["bidmc_ppg_noise_seed"])),
        preset_name=None,
        save_states=False,
        params=params,
    )
    generated = module.synthesize_ppg_noise(config)
    return np.asarray(generated.signal_values, dtype=np.float32)


class BIDMCDataset(Dataset[dict[str, torch.Tensor | str]]):
    """BIDMC 双模态对齐训练集。"""

    def __init__(
        self,
        clean_ecg: np.ndarray,
        noisy_ecg: np.ndarray,
        clean_ppg: np.ndarray,
        noisy_ppg: np.ndarray,
        record_ids: Sequence[str],
        window_starts: Sequence[int],
        sampling_rates_hz: Sequence[float],
    ) -> None:
        if not (
            clean_ecg.shape == noisy_ecg.shape == clean_ppg.shape == noisy_ppg.shape
            and clean_ecg.ndim == 3
            and clean_ecg.shape[1] == 1
        ):
            raise ValueError("BIDMCDataset 输入数组必须具有相同的 [N,1,T] 形状")
        sample_count = clean_ecg.shape[0]
        if not (len(record_ids) == len(window_starts) == len(sampling_rates_hz) == sample_count):
            raise ValueError("元数据长度必须与样本数一致")

        self.clean_ecg = torch.from_numpy(clean_ecg).float()
        self.noisy_ecg = torch.from_numpy(noisy_ecg).float()
        self.clean_ppg = torch.from_numpy(clean_ppg).float()
        self.noisy_ppg = torch.from_numpy(noisy_ppg).float()
        self.record_ids = list(record_ids)
        self.window_starts = [int(v) for v in window_starts]
        self.sampling_rates_hz = [float(v) for v in sampling_rates_hz]
        self.modality_mask = torch.tensor([1.0, 1.0], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.clean_ecg.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "clean_ecg": self.clean_ecg[index],
            "noisy_ecg": self.noisy_ecg[index],
            "clean_ppg": self.clean_ppg[index],
            "noisy_ppg": self.noisy_ppg[index],
            "modality_mask": self.modality_mask.clone(),
            "record_id": self.record_ids[index],
            "window_start": torch.tensor(self.window_starts[index], dtype=torch.int64),
            "sampling_rate_hz": torch.tensor(self.sampling_rates_hz[index], dtype=torch.float32),
        }


def _build_dataset_for_records(record_ids: Sequence[str], root: Path, bidmc_data_cfg: dict[str, Any], split_name: str) -> BIDMCDataset:
    window_length = int(bidmc_data_cfg["window_length"])
    window_stride = int(bidmc_data_cfg["window_stride"])
    expected_fs = float(bidmc_data_cfg["sampling_rate_hz"])
    ecg_channel_name = str(bidmc_data_cfg["ecg_channel_name"])
    ppg_channel_name = str(bidmc_data_cfg["ppg_channel_name"])
    ecg_fallback_names = tuple(str(name) for name in bidmc_data_cfg.get("ecg_channel_fallback_names", []))
    ppg_fallback_names = tuple(str(name) for name in bidmc_data_cfg.get("ppg_channel_fallback_names", []))
    nstdb_root = Path(str(bidmc_data_cfg["nstdb_root"])).expanduser().resolve()
    train_noise_pool, test_noise_pool = _load_bw_noise_pools(int(bidmc_data_cfg["ecg_noise_version"]), nstdb_root)
    ecg_noise_pool = train_noise_pool if split_name in {"train", "val"} else test_noise_pool

    ecg_ratio_low, ecg_ratio_high = (float(v) for v in bidmc_data_cfg["ecg_noise_ratio_range"])
    ppg_ratio_low, ppg_ratio_high = (float(v) for v in bidmc_data_cfg["ppg_noise_ratio_range"])

    clean_ecg_windows: list[np.ndarray] = []
    noisy_ecg_windows: list[np.ndarray] = []
    clean_ppg_windows: list[np.ndarray] = []
    noisy_ppg_windows: list[np.ndarray] = []
    record_id_windows: list[str] = []
    window_starts: list[int] = []
    sampling_rates_hz: list[float] = []

    for record_id in record_ids:
        signals, fields = wfdb.rdsamp(str(root / record_id))
        signal_names = list(fields["sig_name"])
        fs = float(fields["fs"])
        if abs(fs - expected_fs) > 1e-6:
            raise ValueError(f"{record_id} 采样率为 {fs} Hz，与配置要求的 {expected_fs} Hz 不一致")
        ecg_index = _resolve_channel_index(signal_names, ecg_channel_name, ecg_fallback_names)
        ppg_index = _resolve_channel_index(signal_names, ppg_channel_name, ppg_fallback_names)
        if ecg_index is None or ppg_index is None:
            logger.warning(
                "跳过记录 %s：配置 ECG=%s PPG=%s，实际通道=%s",
                record_id,
                [ecg_channel_name, *ecg_fallback_names],
                [ppg_channel_name, *ppg_fallback_names],
                signal_names,
            )
            continue

        clean_ecg_full = np.asarray(signals[:, ecg_index], dtype=np.float32)
        clean_ppg_full = np.asarray(signals[:, ppg_index], dtype=np.float32)
        total_length = int(min(clean_ecg_full.shape[0], clean_ppg_full.shape[0]))
        clean_ecg_full = _ensure_length(clean_ecg_full[:total_length], total_length)
        clean_ppg_full = _ensure_length(clean_ppg_full[:total_length], total_length)

        ecg_noise_seed = _stable_seed(record_id, split_name, "ecg_offset", base_seed=int(bidmc_data_cfg["split_seed"]))
        ecg_offset_rng = np.random.default_rng(ecg_noise_seed)
        ecg_noise_full = _take_cyclic_segment(ecg_noise_pool, int(ecg_offset_rng.integers(0, ecg_noise_pool.shape[0])), total_length)
        ppg_noise_full = _generate_ppg_noise_full(record_id=record_id, signal_length=total_length, bidmc_data_cfg=bidmc_data_cfg)

        ecg_ratio_rng = np.random.default_rng(_stable_seed(record_id, split_name, "ecg_ratio", base_seed=int(bidmc_data_cfg["split_seed"])))
        ppg_ratio_rng = np.random.default_rng(_stable_seed(record_id, split_name, "ppg_ratio", base_seed=int(bidmc_data_cfg["bidmc_ppg_noise_seed"])))

        for start in _compute_window_starts(total_length, window_length, window_stride):
            end = start + window_length
            clean_ecg_window = _ensure_length(clean_ecg_full[start:end], window_length)
            clean_ppg_window = _ensure_length(clean_ppg_full[start:end], window_length)
            ecg_noise_window = _ensure_length(ecg_noise_full[start:end], window_length)
            ppg_noise_window = _ensure_length(ppg_noise_full[start:end], window_length)

            ecg_ratio = float(ecg_ratio_rng.uniform(ecg_ratio_low, ecg_ratio_high))
            ppg_ratio = float(ppg_ratio_rng.uniform(ppg_ratio_low, ppg_ratio_high))
            noisy_ecg_window = clean_ecg_window + _scale_noise_window(clean_ecg_window, ecg_noise_window, ecg_ratio)
            noisy_ppg_window = clean_ppg_window + _scale_noise_window(clean_ppg_window, ppg_noise_window, ppg_ratio)

            clean_ecg_norm, noisy_ecg_norm = _normalize_pair(clean_ecg_window, noisy_ecg_window)
            clean_ppg_norm, noisy_ppg_norm = _normalize_pair(clean_ppg_window, noisy_ppg_window)

            clean_ecg_windows.append(clean_ecg_norm[None, :])
            noisy_ecg_windows.append(noisy_ecg_norm[None, :])
            clean_ppg_windows.append(clean_ppg_norm[None, :])
            noisy_ppg_windows.append(noisy_ppg_norm[None, :])
            record_id_windows.append(record_id)
            window_starts.append(start)
            sampling_rates_hz.append(fs)

    if not clean_ecg_windows:
        raise ValueError(f"{split_name} 切分为空，请检查 BIDMC 路径、记录筛选或通道名配置")

    return BIDMCDataset(
        clean_ecg=np.stack(clean_ecg_windows, axis=0).astype(np.float32),
        noisy_ecg=np.stack(noisy_ecg_windows, axis=0).astype(np.float32),
        clean_ppg=np.stack(clean_ppg_windows, axis=0).astype(np.float32),
        noisy_ppg=np.stack(noisy_ppg_windows, axis=0).astype(np.float32),
        record_ids=record_id_windows,
        window_starts=window_starts,
        sampling_rates_hz=sampling_rates_hz,
    )


def build_bidmc_train_val_test_datasets(config: dict[str, Any], split_seed: int | None = None) -> tuple[BIDMCDataset, BIDMCDataset, BIDMCDataset]:
    bidmc_cfg = _require_bidmc_dict(config)
    bidmc_data_cfg = dict(bidmc_cfg["data"])
    root = _resolve_bidmc_root(config)
    bidmc_data_cfg["nstdb_root"] = str(_resolve_nstdb_root(config))
    record_ids = _list_bidmc_record_ids(root)
    effective_seed = int(bidmc_data_cfg["split_seed"] if split_seed is None else split_seed)
    train_ids, val_ids, test_ids = _split_record_ids(
        record_ids=record_ids,
        train_ratio=float(bidmc_data_cfg["train_ratio"]),
        val_ratio=float(bidmc_data_cfg["val_ratio"]),
        test_ratio=float(bidmc_data_cfg["test_ratio"]),
        seed=effective_seed,
    )
    return (
        _build_dataset_for_records(train_ids, root, bidmc_data_cfg, split_name="train"),
        _build_dataset_for_records(val_ids, root, bidmc_data_cfg, split_name="val"),
        _build_dataset_for_records(test_ids, root, bidmc_data_cfg, split_name="test"),
    )


def build_bidmc_train_val_datasets(config: dict[str, Any], split_seed: int | None = None) -> tuple[BIDMCDataset, BIDMCDataset]:
    train_ds, val_ds, _ = build_bidmc_train_val_test_datasets(config=config, split_seed=split_seed)
    return train_ds, val_ds


def build_bidmc_test_dataset(config: dict[str, Any], split_seed: int | None = None) -> BIDMCDataset:
    _, _, test_ds = build_bidmc_train_val_test_datasets(config=config, split_seed=split_seed)
    return test_ds
