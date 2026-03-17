from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import signal

DEFAULT_OUTPUT_DIR = Path("artifacts") / "ppg_noise"
DEFAULT_PRESET_NAME = "demo"
DEFAULT_PRESET_PARAMS = {
    "prob": [0.25, 0.25, 0.25, 0.25],
    "dur_mu": [12.0, 4.0, 4.0, 4.0, 4.0],
    "rms_shape": [2.0, 2.0, 2.0, 2.0],
    "rms_scale": [0.35, 0.45, 0.55, 0.75],
    "slope": [-6.0, -8.0, -10.0, -12.0],
}
FIR_ORDER = 250
FILTER_WARMUP_SAMPLES = 200
SVG_WIDTH = 1280
SVG_HEIGHT = 420
SVG_MARGIN_X = 72
SVG_MARGIN_Y = 48
MAX_PLOT_POINTS = 2048


@dataclass(frozen=True)
class GenerationParams:
    prob: tuple[float, float, float, float]
    dur_mu: tuple[float, float, float, float, float]
    rms_shape: tuple[float, float, float, float]
    rms_scale: tuple[float, float, float, float]
    slope: tuple[float, float, float, float]


@dataclass(frozen=True)
class GenerationConfig:
    output_dir: Path
    output_stem: str
    duration_samples: int
    sampling_rate_hz: float
    seed: int | None
    preset_name: str | None
    save_states: bool
    params: GenerationParams


@dataclass(frozen=True)
class GenerationArtifacts:
    csv_path: Path
    svg_path: Path
    metadata_path: Path
    states_path: Path | None
    sample_count: int


@dataclass(frozen=True)
class GeneratedNoise:
    signal_values: np.ndarray
    state_values: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用纯 Python 复现 gen_PPG_artifacts.m 的逻辑，生成 PPG 噪声并导出为 CSV、SVG 和 JSON。"
    )
    parser.add_argument(
        "--duration-samples",
        type=int,
        required=True,
        help="要生成的噪声长度，单位为采样点。",
    )
    parser.add_argument(
        "--sampling-rate-hz",
        type=float,
        required=True,
        help="采样率，单位 Hz。",
    )
    parser.add_argument(
        "--preset",
        choices=(DEFAULT_PRESET_NAME,),
        default=None,
        help="使用预置参数；不指定时需显式提供全部 5 组参数。",
    )
    parser.add_argument(
        "--prob",
        nargs=4,
        type=float,
        default=None,
        metavar=("P1", "P2", "P3", "P4"),
        help="4 种伪影类型的转移概率，4 个值之和必须为 1。",
    )
    parser.add_argument(
        "--dur-mu",
        nargs=5,
        type=float,
        default=None,
        metavar=("D0", "D1", "D2", "D3", "D4"),
        help="无伪影和 4 种伪影的平均持续时间，单位秒。",
    )
    parser.add_argument(
        "--rms-shape",
        nargs=4,
        type=float,
        default=None,
        metavar=("S1", "S2", "S3", "S4"),
        help="Gamma 分布 shape 参数。",
    )
    parser.add_argument(
        "--rms-scale",
        nargs=4,
        type=float,
        default=None,
        metavar=("C1", "C2", "C3", "C4"),
        help="Gamma 分布 scale 参数。",
    )
    parser.add_argument(
        "--slope",
        nargs=4,
        type=float,
        default=None,
        metavar=("K1", "K2", "K3", "K4"),
        help="4 种伪影的 PSD slope 参数。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；指定后可复现同一组噪声。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录，默认 artifacts/ppg_noise。",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="ppg_noise",
        help="输出文件名前缀，默认 ppg_noise。",
    )
    parser.add_argument(
        "--save-states",
        action="store_true",
        help="额外导出状态序列，0 表示无伪影，1-4 表示 4 类伪影。",
    )
    return parser


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    params = resolve_generation_params(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_stem = args.output_stem.strip()

    if args.duration_samples <= 0:
        raise ValueError(f"duration_samples 必须大于 0，当前为 {args.duration_samples}")
    if args.sampling_rate_hz <= 0:
        raise ValueError(f"sampling_rate_hz 必须大于 0，当前为 {args.sampling_rate_hz}")
    if not output_stem:
        raise ValueError("output_stem 不能为空")

    return GenerationConfig(
        output_dir=output_dir,
        output_stem=output_stem,
        duration_samples=args.duration_samples,
        sampling_rate_hz=args.sampling_rate_hz,
        seed=args.seed,
        preset_name=args.preset,
        save_states=bool(args.save_states),
        params=params,
    )


def resolve_generation_params(args: argparse.Namespace) -> GenerationParams:
    preset = DEFAULT_PRESET_PARAMS if args.preset == DEFAULT_PRESET_NAME else {}
    prob = _resolve_vector(args.prob, preset.get("prob"), 4, "prob")
    dur_mu = _resolve_vector(args.dur_mu, preset.get("dur_mu"), 5, "dur_mu")
    rms_shape = _resolve_vector(args.rms_shape, preset.get("rms_shape"), 4, "rms_shape")
    rms_scale = _resolve_vector(args.rms_scale, preset.get("rms_scale"), 4, "rms_scale")
    slope = _resolve_vector(args.slope, preset.get("slope"), 4, "slope")

    prob_total = math.fsum(prob)
    if not math.isclose(prob_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"prob 的 4 个值之和必须为 1，当前为 {prob_total:.12g}")
    if any(value < 0 for value in prob):
        raise ValueError("prob 不能包含负值")
    _ensure_positive(dur_mu, "dur_mu")
    _ensure_positive(rms_shape, "rms_shape")
    _ensure_positive(rms_scale, "rms_scale")
    _ensure_finite(slope, "slope")

    return GenerationParams(
        prob=tuple(prob),
        dur_mu=tuple(dur_mu),
        rms_shape=tuple(rms_shape),
        rms_scale=tuple(rms_scale),
        slope=tuple(slope),
    )


def _resolve_vector(
    user_values: Sequence[float] | None,
    preset_values: Sequence[float] | None,
    expected_len: int,
    name: str,
) -> list[float]:
    values = list(user_values) if user_values is not None else list(preset_values or [])
    if len(values) != expected_len:
        raise ValueError(f"{name} 需要 {expected_len} 个值，当前收到 {len(values)} 个")
    return [float(value) for value in values]


def _ensure_positive(values: Iterable[float], name: str) -> None:
    if any((not math.isfinite(value)) or value <= 0 for value in values):
        raise ValueError(f"{name} 的所有值都必须大于 0")


def _ensure_finite(values: Iterable[float], name: str) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} 不能包含 NaN 或无穷大")


def generate_ppg_noise(config: GenerationConfig) -> GenerationArtifacts:
    generated = synthesize_ppg_noise(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_dir / f"{config.output_stem}.csv"
    svg_path = config.output_dir / f"{config.output_stem}.svg"
    metadata_path = config.output_dir / f"{config.output_stem}.json"
    states_path = config.output_dir / f"{config.output_stem}_states.csv" if config.save_states else None

    write_signal_csv(csv_path, generated.signal_values)
    write_signal_svg(svg_path, generated.signal_values, config.sampling_rate_hz)
    write_metadata(metadata_path, config, generated)
    if states_path is not None:
        write_states_csv(states_path, generated.state_values)

    return GenerationArtifacts(
        csv_path=csv_path,
        svg_path=svg_path,
        metadata_path=metadata_path,
        states_path=states_path,
        sample_count=int(generated.signal_values.size),
    )


def synthesize_ppg_noise(config: GenerationConfig) -> GeneratedNoise:
    rng = np.random.default_rng(config.seed)
    filters = build_artifact_filters(config.sampling_rate_hz, config.params.slope)
    state_transition = build_state_transition_matrix(config.params.prob)
    duration_means = np.asarray(config.params.dur_mu, dtype=np.float64) * float(config.sampling_rate_hz)
    min_interval_samples = max(1, int(math.ceil(config.sampling_rate_hz)))

    signal_chunks: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []
    generated_length = 0
    current_state = 0

    while generated_length < config.duration_samples:
        interval_samples = round_nonnegative(rng.exponential(duration_means[current_state]))
        interval_samples = max(interval_samples, min_interval_samples)

        remaining_after_interval = config.duration_samples - (generated_length + interval_samples)
        if remaining_after_interval < min_interval_samples:
            interval_samples = config.duration_samples - generated_length

        if interval_samples <= 0:
            break

        if current_state == 0:
            chunk = np.zeros(interval_samples, dtype=np.float64)
        else:
            chunk = generate_artifact_chunk(
                rng=rng,
                filter_coefficients=filters[current_state - 1],
                rms_shape=config.params.rms_shape[current_state - 1],
                rms_scale=config.params.rms_scale[current_state - 1],
                interval_samples=interval_samples,
            )

        signal_chunks.append(chunk)
        state_chunks.append(np.full(interval_samples, current_state, dtype=np.int16))
        generated_length += interval_samples

        transition_cdf = np.cumsum(state_transition[current_state])
        current_state = int(np.searchsorted(transition_cdf, rng.random(), side="right"))
        current_state = min(current_state, state_transition.shape[0] - 1)

    signal_values = np.concatenate(signal_chunks)[: config.duration_samples]
    state_values = np.concatenate(state_chunks)[: config.duration_samples]
    return GeneratedNoise(signal_values=signal_values, state_values=state_values)


def round_nonnegative(value: float) -> int:
    return int(math.floor(max(value, 0.0) + 0.5))


def build_state_transition_matrix(prob: Sequence[float]) -> np.ndarray:
    transition = np.zeros((5, 5), dtype=np.float64)
    transition[0, 0] = 0.0
    transition[0, 1:] = np.asarray(prob, dtype=np.float64)
    transition[1:, 0] = 1.0
    return transition


def build_artifact_filters(sampling_rate_hz: float, slopes: Sequence[float]) -> np.ndarray:
    normalized_frequency = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    physical_frequency = normalized_frequency * (sampling_rate_hz / 2.0)
    filters: list[np.ndarray] = []

    for slope in slopes:
        desired = np.zeros_like(normalized_frequency)
        desired[2:] = np.sqrt(10.0 ** ((np.log10(physical_frequency[2:]) * slope) / 10.0))
        bands = np.column_stack([normalized_frequency[:-1], normalized_frequency[1:]])
        gains = np.column_stack([desired[:-1], desired[1:]])
        filters.append(signal.firls(FIR_ORDER + 1, bands, gains, fs=2.0))

    return np.asarray(filters, dtype=np.float64)


def generate_artifact_chunk(
    rng: np.random.Generator,
    filter_coefficients: np.ndarray,
    rms_shape: float,
    rms_scale: float,
    interval_samples: int,
) -> np.ndarray:
    white_noise = rng.standard_normal(interval_samples + FILTER_WARMUP_SAMPLES)
    artifact = signal.lfilter(filter_coefficients, [1.0], white_noise)[FILTER_WARMUP_SAMPLES:]
    artifact_std = float(np.std(artifact))
    if artifact_std <= 0.0:
        artifact = np.zeros_like(artifact)
    else:
        artifact = (artifact - np.mean(artifact)) / artifact_std
    amplitude = float(rng.gamma(shape=rms_shape, scale=rms_scale))
    return amplitude * artifact


def write_signal_csv(path: Path, signal_values: np.ndarray) -> None:
    lines = ["sample_index,ppg_noise"]
    lines.extend(f"{index},{format_float(float(value))}" for index, value in enumerate(signal_values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_states_csv(path: Path, state_values: np.ndarray) -> None:
    lines = ["sample_index,state"]
    lines.extend(f"{index},{int(value)}" for index, value in enumerate(state_values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_signal_svg(path: Path, signal_values: np.ndarray, sampling_rate_hz: float) -> None:
    plot_signal = downsample_for_plot(signal_values, MAX_PLOT_POINTS)
    min_value = float(np.min(plot_signal))
    max_value = float(np.max(plot_signal))
    if math.isclose(min_value, max_value):
        padding = 1.0 if math.isclose(min_value, 0.0) else abs(min_value) * 0.1
        min_value -= padding
        max_value += padding

    plot_width = SVG_WIDTH - SVG_MARGIN_X * 2
    plot_height = SVG_HEIGHT - SVG_MARGIN_Y * 2
    points: list[str] = []

    for index, value in enumerate(plot_signal):
        x_ratio = 0.0 if len(plot_signal) == 1 else index / (len(plot_signal) - 1)
        y_ratio = (value - min_value) / (max_value - min_value)
        x = SVG_MARGIN_X + x_ratio * plot_width
        y = SVG_MARGIN_Y + (1.0 - y_ratio) * plot_height
        points.append(f"{x:.2f},{y:.2f}")

    duration_seconds = signal_values.size / sampling_rate_hz
    svg_text = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
  <rect width="100%" height="100%" fill="#fcfcfd" />
  <rect x="{SVG_MARGIN_X}" y="{SVG_MARGIN_Y}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#d9dde7" stroke-width="1.5" />
  <line x1="{SVG_MARGIN_X}" y1="{SVG_HEIGHT / 2:.2f}" x2="{SVG_WIDTH - SVG_MARGIN_X}" y2="{SVG_HEIGHT / 2:.2f}" stroke="#e6e9f0" stroke-width="1" />
  <polyline fill="none" stroke="#136f63" stroke-width="1.5" points="{' '.join(points)}" />
  <text x="{SVG_MARGIN_X}" y="28" font-size="22" font-family="Segoe UI, Arial, sans-serif" fill="#102a43">PPG 噪声波形</text>
  <text x="{SVG_MARGIN_X}" y="{SVG_HEIGHT - 14}" font-size="14" font-family="Segoe UI, Arial, sans-serif" fill="#486581">采样点数: {signal_values.size} | 采样率: {format_float(sampling_rate_hz)} Hz | 时长: {duration_seconds:.3f} s</text>
  <text x="{SVG_WIDTH - SVG_MARGIN_X}" y="28" text-anchor="end" font-size="14" font-family="Segoe UI, Arial, sans-serif" fill="#486581">min={format_float(float(np.min(signal_values)))}, max={format_float(float(np.max(signal_values)))}</text>
</svg>
"""
    path.write_text(svg_text, encoding="utf-8")


def downsample_for_plot(signal_values: np.ndarray, max_points: int) -> list[float]:
    if signal_values.size <= max_points:
        return [float(value) for value in signal_values]

    sample_positions = np.linspace(0, signal_values.size - 1, max_points, dtype=np.int64)
    return [float(signal_values[index]) for index in sample_positions]


def write_metadata(path: Path, config: GenerationConfig, generated: GeneratedNoise) -> None:
    state_ids, state_counts = np.unique(generated.state_values, return_counts=True)
    payload = {
        "output_dir": str(config.output_dir),
        "output_stem": config.output_stem,
        "duration_samples": config.duration_samples,
        "sampling_rate_hz": config.sampling_rate_hz,
        "seed": config.seed,
        "preset_name": config.preset_name,
        "save_states": config.save_states,
        "params": asdict(config.params),
        "sample_count": int(generated.signal_values.size),
        "signal_summary": {
            "mean": float(np.mean(generated.signal_values)),
            "std": float(np.std(generated.signal_values)),
            "rms": float(np.sqrt(np.mean(np.square(generated.signal_values)))),
            "min": float(np.min(generated.signal_values)),
            "max": float(np.max(generated.signal_values)),
        },
        "state_counts": {str(int(state_id)): int(count) for state_id, count in zip(state_ids, state_counts)},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_float(value: float) -> str:
    return f"{value:.15g}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")

    try:
        args = build_parser().parse_args()
        config = build_generation_config(args)
        artifacts = generate_ppg_noise(config)
        print(f"CSV 已生成: {artifacts.csv_path}")
        print(f"SVG 已生成: {artifacts.svg_path}")
        print(f"元数据已生成: {artifacts.metadata_path}")
        if artifacts.states_path is not None:
            print(f"状态序列已生成: {artifacts.states_path}")
        print(f"采样点数: {artifacts.sample_count}")
        return 0
    except Exception as exc:
        print(f"PPG 噪声生成失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
