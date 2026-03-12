from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_ppg_denoise.config import load_experiment_config
from ecg_ppg_denoise.data import build_qt_train_val_datasets, load_multimodal_arrays
from ecg_ppg_denoise.models import ModalityFlexibleConditionalDiffusion
from ecg_ppg_denoise.utils import build_logger, ensure_dir, resolve_device
from ecg_ppg_denoise.utils.metrics import COS_SIM, MAD, PRD, SNR, SNR_improvement, SSD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 ECG 去噪指标")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    parser.add_argument("--model-name", type=str, default=None, help="YAML 中的模型配置名")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--input-path", type=str, default=None, help="输入数据(.npz/.pt/.npy)")
    parser.add_argument("--use-qt-dataset", action="store_true", help="使用 QT Data_Preparation 评估集")
    parser.add_argument("--qt-noise-version", type=int, default=1, choices=[1, 2], help="QT 噪声版本")
    parser.add_argument("--mode", type=str, default="ecg", choices=["ecg", "ppg", "joint"], help="推理模式")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help=">0 时仅评估前 N 条")
    parser.add_argument("--output-dir", type=str, default="artifacts/eval")
    return parser.parse_args()


def _stack_from_qt_val(noise_version: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, val_ds = build_qt_train_val_datasets(noise_version=noise_version)
    clean_ecg = val_ds.clean_ecg.detach().cpu().numpy()
    noisy_ecg = val_ds.noisy_ecg.detach().cpu().numpy()
    noisy_ppg = val_ds.noisy_ppg.detach().cpu().numpy()
    return clean_ecg, noisy_ecg, noisy_ppg


def _load_eval_arrays(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.use_qt_dataset:
        return _stack_from_qt_val(noise_version=args.qt_noise_version)
    if args.input_path is None:
        raise ValueError("未启用 --use-qt-dataset 时，必须提供 --input-path")
    arrays = load_multimodal_arrays(args.input_path)
    return arrays["clean_ecg"], arrays["noisy_ecg"], arrays["noisy_ppg"]


def _select_inputs(
    mode: str,
    noisy_ecg: torch.Tensor,
    noisy_ppg: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if mode == "ecg":
        return noisy_ecg, None
    if mode == "ppg":
        return None, noisy_ppg
    return noisy_ecg, noisy_ppg


def _safe_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def compute_ecg_metrics(clean_ecg: np.ndarray, noisy_ecg: np.ndarray, denoised_ecg: np.ndarray) -> Dict[str, float]:
    y_clean = clean_ecg.reshape(clean_ecg.shape[0], -1)
    y_noisy = noisy_ecg.reshape(noisy_ecg.shape[0], -1)
    y_denoised = denoised_ecg.reshape(denoised_ecg.shape[0], -1)

    return {
        "SSD": _safe_mean(SSD(y_clean, y_denoised)),
        "MAD": _safe_mean(MAD(y_clean, y_denoised)),
        "PRD": _safe_mean(PRD(y_clean, y_denoised)),
        "COS_SIM": _safe_mean(COS_SIM(y_clean[..., None], y_denoised[..., None])),
        "SNR_in": _safe_mean(SNR(y_clean, y_noisy)),
        "SNR_out": _safe_mean(SNR(y_clean, y_denoised)),
        "SNR_improvement": _safe_mean(SNR_improvement(y_noisy, y_denoised, y_clean)),
    }


def main() -> int:
    args = parse_args()
    cfg = load_experiment_config(
        config_path=Path(args.config) if args.config else None,
        model_name=args.model_name,
    )
    if args.device is not None:
        cfg.runtime.device = args.device
    cfg.validate()

    logger = build_logger("evaluate")
    device = resolve_device(cfg.runtime.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    model = ModalityFlexibleConditionalDiffusion(cfg.model, cfg.loss).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    logger.info("加载 checkpoint: %s", ckpt_path)

    clean_ecg, noisy_ecg, noisy_ppg = _load_eval_arrays(args)
    if args.max_samples > 0:
        clean_ecg = clean_ecg[: args.max_samples]
        noisy_ecg = noisy_ecg[: args.max_samples]
        noisy_ppg = noisy_ppg[: args.max_samples]

    denoised_chunks = []
    for start in range(0, clean_ecg.shape[0], args.batch_size):
        end = min(clean_ecg.shape[0], start + args.batch_size)
        ecg_batch = torch.from_numpy(noisy_ecg[start:end]).float().to(device)
        ppg_batch = torch.from_numpy(noisy_ppg[start:end]).float().to(device)
        y_ecg, y_ppg = _select_inputs(args.mode, ecg_batch, ppg_batch)

        with torch.no_grad():
            result = model.denoise_signal(
                y_ecg=y_ecg,
                y_ppg=y_ppg,
                num_steps=args.num_steps,
                use_ddim=args.use_ddim,
            )
        denoised_chunks.append(result["denoised_ecg"].detach().cpu().numpy())

    denoised_ecg = np.concatenate(denoised_chunks, axis=0)
    metrics = compute_ecg_metrics(clean_ecg=clean_ecg, noisy_ecg=noisy_ecg, denoised_ecg=denoised_ecg)
    logger.info("ECG 评估指标: %s", metrics)

    out_dir = ensure_dir(args.output_dir)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(
        out_dir / "eval_outputs.npz",
        clean_ecg=clean_ecg,
        noisy_ecg=noisy_ecg,
        denoised_ecg=denoised_ecg,
    )
    logger.info("评估结果已保存: %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
