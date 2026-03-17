from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import soxr
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 soxr 依赖，请先安装 requirements.in 中声明的 soxr。"
    ) from exc


DEFAULT_RESAMPLE_QUALITY = "VHQ"


def ppg_resample(source_hz: float, target_hz: float, signal: Iterable[float]) -> np.ndarray:
    """对 PPG 一维时序做高质量重采样。"""
    return _resample_signal(signal=signal, source_hz=source_hz, target_hz=target_hz)


def ecg_resample(source_hz: float, target_hz: float, signal: Iterable[float]) -> np.ndarray:
    """对 ECG 一维时序做高质量重采样。"""
    return _resample_signal(signal=signal, source_hz=source_hz, target_hz=target_hz)


def _resample_signal(signal: Iterable[float], source_hz: float, target_hz: float) -> np.ndarray:
    signal_array = np.asarray(signal)
    if signal_array.ndim != 1:
        raise ValueError(f"signal 必须是一维时序，当前 shape={signal_array.shape}")
    if signal_array.size == 0:
        raise ValueError("signal 不能为空")
    if source_hz <= 0:
        raise ValueError(f"source_hz 必须大于 0，当前为 {source_hz}")
    if target_hz <= 0:
        raise ValueError(f"target_hz 必须大于 0，当前为 {target_hz}")

    output_dtype = signal_array.dtype if np.issubdtype(signal_array.dtype, np.floating) else np.float64
    working_signal = signal_array.astype(np.float32, copy=False)

    if np.isclose(source_hz, target_hz):
        return working_signal.astype(output_dtype, copy=True)

    resampled_signal = soxr.resample(
        working_signal,
        in_rate=float(source_hz),
        out_rate=float(target_hz),
        quality=DEFAULT_RESAMPLE_QUALITY,
    )
    return np.asarray(resampled_signal, dtype=output_dtype)
