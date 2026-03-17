from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config
from dataset import build_qt_test_dataset, load_multimodal_arrays
from models import DDPM
from utils.common import ensure_dir, resolve_device
from utils.logging import build_logger
from utils.metrics import COS_SIM, MAD, PRD, SNR, SNR_improvement, SSD


DEFAULT_NOISE_SEGMENTS = (0.2, 0.6, 1.0, 1.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="参照 DeScoD-ECG 风格评估 ECG 去噪指标")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型 checkpoint 路径")
    parser.add_argument("--input-path", type=str, default=None, help="输入数据(.npz/.pt/.npy)")
    parser.add_argument("--use-qt-dataset", action="store_true", help="使用 QT Data_Preparation 评估集")
    parser.add_argument("--qt-noise-version", type=int, default=1, choices=[1, 2], help="QT 噪声版本")
    parser.add_argument("--mode", type=str, default="ecg", choices=["ecg", "ppg", "joint"], help="推理模式")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--shots", type=int, default=1, help="同一样本重复采样次数，>1 时取平均")
    parser.add_argument("--max-samples", type=int, default=0, help=">0 时仅评估前 N 条")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _stack_from_qt_test(noise_version: int) -> dict[str, np.ndarray]:
    test_ds = build_qt_test_dataset(noise_version=noise_version)
    return {
        "clean_ecg": test_ds.clean_ecg.detach().cpu().numpy(),
        "noisy_ecg": test_ds.noisy_ecg.detach().cpu().numpy(),
        "clean_ppg": test_ds.clean_ppg.detach().cpu().numpy(),
        "noisy_ppg": test_ds.noisy_ppg.detach().cpu().numpy(),
    }


def _load_eval_arrays(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.use_qt_dataset:
        return _stack_from_qt_test(noise_version=args.qt_noise_version)
    if args.input_path is None:
        raise ValueError("未启用 --use-qt-dataset 时，必须提供 --input-path")
    return load_multimodal_arrays(args.input_path)


def _select_inputs(
    mode: str,
    noisy_ecg: torch.Tensor,
    noisy_ppg: torch.Tensor,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    if mode == "ecg":
        return noisy_ecg, None, torch.tensor([1.0, 0.0], device=noisy_ecg.device)
    if mode == "ppg":
        return None, noisy_ppg, torch.tensor([0.0, 1.0], device=noisy_ppg.device)
    return noisy_ecg, noisy_ppg, torch.tensor([1.0, 1.0], device=noisy_ecg.device)


def _safe_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _safe_std(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def _metric_vector(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def compute_ecg_metric_arrays(clean_ecg: np.ndarray, noisy_ecg: np.ndarray, denoised_ecg: np.ndarray) -> dict[str, np.ndarray]:
    y_clean = clean_ecg.reshape(clean_ecg.shape[0], -1)
    y_noisy = noisy_ecg.reshape(noisy_ecg.shape[0], -1)
    y_denoised = denoised_ecg.reshape(denoised_ecg.shape[0], -1)
    return {
        "SSD": _metric_vector(SSD(y_clean, y_denoised)),
        "MAD": _metric_vector(MAD(y_clean, y_denoised)),
        "PRD": _metric_vector(PRD(y_clean, y_denoised)),
        "COS_SIM": _metric_vector(COS_SIM(y_clean[..., None], y_denoised[..., None])),
        "SNR_in": _metric_vector(SNR(y_clean, y_noisy)),
        "SNR_out": _metric_vector(SNR(y_clean, y_denoised)),
        "SNR_improvement": _metric_vector(SNR_improvement(y_noisy, y_denoised, y_clean)),
    }


def summarize_metric_arrays(metric_arrays: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        metric_name: {
            "mean": _safe_mean(metric_values),
            "std": _safe_std(metric_values),
        }
        for metric_name, metric_values in metric_arrays.items()
    }


def compute_ecg_metrics(clean_ecg: np.ndarray, noisy_ecg: np.ndarray, denoised_ecg: np.ndarray) -> dict[str, float]:
    summary = summarize_metric_arrays(
        compute_ecg_metric_arrays(clean_ecg=clean_ecg, noisy_ecg=noisy_ecg, denoised_ecg=denoised_ecg)
    )
    return {metric_name: stats["mean"] for metric_name, stats in summary.items()}


def _load_qt_noise_levels() -> Optional[np.ndarray]:
    rnd_path = PROJECT_ROOT / "data" / "db" / "QTDB" / "rnd_test.npy"
    if not rnd_path.exists():
        return None
    noise_levels = np.load(rnd_path, allow_pickle=False)
    return np.asarray(noise_levels, dtype=np.float64).reshape(-1)


def build_noise_segment_summary(
    metric_arrays: dict[str, np.ndarray],
    noise_levels: np.ndarray,
    segments: tuple[float, ...] = DEFAULT_NOISE_SEGMENTS,
) -> dict[str, dict[str, dict[str, float]]]:
    if len(segments) < 2:
        raise ValueError("噪声区间至少需要两个边界值")
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for left, right in zip(segments[:-1], segments[1:]):
        label = f"{left}<noise<={right}"
        idx = np.argwhere((noise_levels >= left) & (noise_levels <= right)).reshape(-1)
        if idx.size == 0:
            summary[label] = {}
            continue
        sliced = {metric_name: metric_values[idx] for metric_name, metric_values in metric_arrays.items()}
        summary[label] = summarize_metric_arrays(sliced)
    return summary


def _load_checkpoint_state(ckpt_path: Path, device: torch.device) -> dict:
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"不支持的 checkpoint 格式: {type(payload)!r}")


def _run_sampling(
    model: DDPM,
    y_ecg: Optional[torch.Tensor],
    y_ppg: Optional[torch.Tensor],
    modality_mask: torch.Tensor,
    num_steps: Optional[int],
    use_ddim: bool,
    shots: int,
) -> torch.Tensor:
    if shots <= 0:
        raise ValueError("shots 必须大于 0")
    shot_outputs = []
    for _ in range(shots):
        result = model.generate(
            noisy_ecg=y_ecg,
            noisy_ppg=y_ppg,
            modality_mask=modality_mask,
            num_steps=num_steps,
            use_ddim=use_ddim,
        )
        shot_outputs.append(result["denoised_ecg"].detach())
    return torch.stack(shot_outputs, dim=0).mean(dim=0)


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device

    logger = build_logger("evaluate")
    device = resolve_device(cfg["runtime"]["device"])

    checkpoint_path = args.checkpoint or cfg["path"]["checkpoint_path"]
    if not checkpoint_path:
        raise ValueError("必须通过 --checkpoint 或 config.path.checkpoint_path 提供模型权重路径")
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    if args.mode == "ppg":
        raise ValueError("当前 evaluate.py 参照 DeScoD-ECG，仅支持含 ECG 输出的 ecg/joint 模式")

    model = DDPM(base_model=None, config=cfg, device=device).to(device)
    model.load_state_dict(_load_checkpoint_state(ckpt_path, device))
    model.eval()
    logger.info("加载 checkpoint: %s", ckpt_path)

    if args.input_path is None and cfg["path"]["eval_input_path"]:
        args.input_path = cfg["path"]["eval_input_path"]
    arrays = _load_eval_arrays(args)
    if args.max_samples > 0:
        arrays = {key: value[: args.max_samples] for key, value in arrays.items()}

    clean_ecg = arrays["clean_ecg"]
    noisy_ecg = arrays["noisy_ecg"]
    noisy_ppg = arrays["noisy_ppg"]

    denoised_chunks = []
    total_samples = int(clean_ecg.shape[0])
    total_batches = (total_samples + args.batch_size - 1) // args.batch_size if total_samples > 0 else 0

    with torch.inference_mode():
        with tqdm(
            range(0, total_samples, args.batch_size),
            total=total_batches,
            desc=f"Evaluate-{args.shots}shot",
            unit="batch",
            dynamic_ncols=True,
        ) as progress_bar:
            for start in progress_bar:
                end = min(total_samples, start + args.batch_size)
                ecg_batch = torch.from_numpy(noisy_ecg[start:end]).float().to(device)
                ppg_batch = torch.from_numpy(noisy_ppg[start:end]).float().to(device)
                y_ecg, y_ppg, modality_mask = _select_inputs(args.mode, ecg_batch, ppg_batch)
                denoised_batch = _run_sampling(
                    model=model,
                    y_ecg=y_ecg,
                    y_ppg=y_ppg,
                    modality_mask=modality_mask,
                    num_steps=args.num_steps,
                    use_ddim=args.use_ddim,
                    shots=args.shots,
                )
                denoised_chunks.append(denoised_batch.cpu().numpy())
                progress_bar.set_postfix(processed=end, total=total_samples)

    denoised_ecg = np.concatenate(denoised_chunks, axis=0) if denoised_chunks else np.empty_like(clean_ecg)
    metric_arrays = compute_ecg_metric_arrays(clean_ecg=clean_ecg, noisy_ecg=noisy_ecg, denoised_ecg=denoised_ecg)
    metric_summary = summarize_metric_arrays(metric_arrays)
    logger.info("ECG 评估结果（mean±std）: %s", metric_summary)

    payload: dict[str, object] = {
        "mode": args.mode,
        "shots": args.shots,
        "num_steps": args.num_steps,
        "use_ddim": args.use_ddim,
        "metrics": metric_summary,
    }

    if args.use_qt_dataset:
        noise_levels = _load_qt_noise_levels()
        if noise_levels is not None and noise_levels.shape[0] == clean_ecg.shape[0]:
            payload["noise_segments"] = build_noise_segment_summary(metric_arrays, noise_levels)
        else:
            logger.info("未找到与测试集长度一致的 QT 噪声强度数组，跳过噪声分段统计")

    out_dir = ensure_dir(args.output_dir or cfg["path"]["eval_output_dir"])
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(
        out_dir / "eval_outputs.npz",
        clean_ecg=clean_ecg,
        noisy_ecg=noisy_ecg,
        denoised_ecg=denoised_ecg,
        SSD=metric_arrays["SSD"],
        MAD=metric_arrays["MAD"],
        PRD=metric_arrays["PRD"],
        COS_SIM=metric_arrays["COS_SIM"],
        SNR_in=metric_arrays["SNR_in"],
        SNR_out=metric_arrays["SNR_out"],
        SNR_improvement=metric_arrays["SNR_improvement"],
    )
    logger.info("评估结果已保存: %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
