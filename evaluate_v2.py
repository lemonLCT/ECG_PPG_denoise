from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import build_bidmc_test_dataset, build_qt_test_dataset
from models.DDPM import DDPM
from models.HNF import ConditionalModel
from utils.common import dump_config_snapshot, ensure_dir, resolve_device, seed_everything
from utils.logging import build_logger
from utils.metrics import COS_SIM, MAD, PRD, SNR, SNR_improvement, SSD

DEFAULT_CONFIG_PATH = SRC_DIR / "config" / "bidmc_v2.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BIDMC/QT ECG 单模态 DDPM v2 评估入口")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="v2 YAML 配置文件路径")
    parser.add_argument("--dataset", type=str, default="bidmc", choices=["bidmc", "qt"], help="选择评估数据集")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型 checkpoint 路径")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dataset-path", type=str, default=None, help="数据集根目录；BIDMC 对应 bidmc_root，QT 对应 qt_root")
    parser.add_argument("--qt-noise-version", type=int, default=1, choices=[1, 2], help="QT 噪声版本")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    with Path(args.config).expanduser().open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    if args.seed is not None:
        cfg["runtime"]["seed"] = args.seed
        cfg["bidmc"]["data"]["split_seed"] = args.seed
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.output_dir is not None:
        cfg["path"]["eval_output_dir"] = args.output_dir
    if args.batch_size is not None:
        cfg["evaluate"]["batch_size"] = args.batch_size
    if args.num_steps is not None:
        cfg["evaluate"]["num_steps"] = args.num_steps
        cfg["diffusion"]["num_steps"] = args.num_steps
    if args.max_samples is not None:
        cfg["evaluate"]["max_samples"] = args.max_samples
    if args.dataset == "bidmc" and args.dataset_path is not None:
        cfg["bidmc"]["path"]["bidmc_root"] = args.dataset_path
    if args.dataset == "qt" and args.dataset_path is not None:
        cfg.setdefault("path", {})
        cfg["path"]["qt_root"] = args.dataset_path
    if args.checkpoint is not None:
        cfg["path"]["checkpoint_path"] = args.checkpoint

    checkpoint_path = str(cfg["path"].get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise ValueError("必须通过 --checkpoint 或 path.checkpoint_path 提供 checkpoint")

    device = resolve_device(str(cfg["runtime"]["device"]))
    seed_everything(int(cfg["runtime"]["seed"]))

    output_dir = Path(cfg["path"]["eval_output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = ensure_dir(output_dir.resolve())
    cfg["path"]["eval_output_dir"] = str(output_dir)
    dump_config_snapshot(cfg, output_dir)
    logger = build_logger("evaluate_v2", log_path=output_dir / "evaluate_v2.log")

    if args.dataset == "qt":
        dataset = build_qt_test_dataset(
            noise_version=args.qt_noise_version,
            data_root=cfg.get("path", {}).get("qt_root") or None,
        )
    else:
        dataset = build_bidmc_test_dataset(
            config=cfg,
            split_seed=int(cfg["bidmc"]["data"]["split_seed"]),
        )

    if int(cfg["evaluate"]["max_samples"]) > 0:
        max_samples = int(cfg["evaluate"]["max_samples"])
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["evaluate"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = DDPM(
        base_model=ConditionalModel(feats=int(cfg["hnf"]["feats"])),
        config=cfg,
        device=device,
    ).to(device)
    checkpoint_file = Path(checkpoint_path).expanduser()
    if not checkpoint_file.is_absolute():
        checkpoint_file = PROJECT_ROOT / checkpoint_file
    payload = torch.load(checkpoint_file.resolve(), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    losses: list[float] = []
    clean_batches: list[np.ndarray] = []
    noisy_batches: list[np.ndarray] = []
    denoised_batches: list[np.ndarray] = []

    with torch.inference_mode():
        eval_bar = tqdm(dataloader, desc=f"Evaluate-v2-{args.dataset}", unit="batch", dynamic_ncols=True, leave=False)
        for batch in eval_bar:
            clean_ecg = batch["clean_ecg"].to(device, non_blocking=True)
            noisy_ecg = batch["noisy_ecg"].to(device, non_blocking=True)
            loss = model(clean_ecg, noisy_ecg)
            denoised = model.denoising(noisy_ecg)
            losses.append(float(loss.detach().cpu().item()))
            clean_batches.append(clean_ecg.cpu().numpy())
            noisy_batches.append(noisy_ecg.cpu().numpy())
            denoised_batches.append(denoised.cpu().numpy())
            eval_bar.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

    clean_ecg = np.concatenate(clean_batches, axis=0) if clean_batches else np.empty((0, 1, 0), dtype=np.float32)
    noisy_ecg = np.concatenate(noisy_batches, axis=0) if noisy_batches else np.empty((0, 1, 0), dtype=np.float32)
    denoised_ecg = np.concatenate(denoised_batches, axis=0) if denoised_batches else np.empty((0, 1, 0), dtype=np.float32)

    y_clean = clean_ecg.reshape(clean_ecg.shape[0], -1)
    y_noisy = noisy_ecg.reshape(noisy_ecg.shape[0], -1)
    y_denoised = denoised_ecg.reshape(denoised_ecg.shape[0], -1)
    metric_arrays = {
        "SSD": np.asarray(SSD(y_clean, y_denoised), dtype=np.float64).reshape(-1),
        "MAD": np.asarray(MAD(y_clean, y_denoised), dtype=np.float64).reshape(-1),
        "PRD": np.asarray(PRD(y_clean, y_denoised), dtype=np.float64).reshape(-1),
        "COS_SIM": np.asarray(COS_SIM(y_clean[..., None], y_denoised[..., None]), dtype=np.float64).reshape(-1),
        "SNR_in": np.asarray(SNR(y_clean, y_noisy), dtype=np.float64).reshape(-1),
        "SNR_out": np.asarray(SNR(y_clean, y_denoised), dtype=np.float64).reshape(-1),
        "SNR_improvement": np.asarray(SNR_improvement(y_noisy, y_denoised, y_clean), dtype=np.float64).reshape(-1),
    }
    metric_summary = {}
    for name, values in metric_arrays.items():
        valid = values[np.isfinite(values)]
        metric_summary[name] = {
            "mean": float(np.mean(valid)) if valid.size else float("nan"),
            "std": float(np.std(valid)) if valid.size else float("nan"),
        }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_file.resolve()),
                "dataset": args.dataset,
                "num_steps": int(cfg["diffusion"]["num_steps"]),
                "test_loss": (sum(losses) / len(losses)) if losses else 0.0,
                "metrics": metric_summary,
                "checkpoint_epoch": payload.get("epoch"),
                "checkpoint_global_step": payload.get("global_step"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    np.savez(
        output_dir / "eval_outputs.npz",
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
