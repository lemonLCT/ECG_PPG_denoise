from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_ppg_denoise.config import ExperimentConfig, load_experiment_config
from ecg_ppg_denoise.data import build_train_val_datasets
from ecg_ppg_denoise.models import ModalityFlexibleConditionalDiffusion
from ecg_ppg_denoise.trainers import TrainEngine
from ecg_ppg_denoise.utils import (
    build_logger,
    dump_config_snapshot,
    ensure_dir,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seed_everything,
)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 ModalityFlexibleConditionalDiffusion")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--model-name", type=str, default=None, help="YAML 中的模型配置名")
    parser.add_argument("--stage", type=str, required=True, choices=["ecg_pretrain", "ppg_pretrain", "joint"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="继续训练的 checkpoint 路径")
    parser.add_argument("--use-qt-dataset", action="store_true", help="启用 QT Data_Preparation 适配数据集")
    parser.add_argument("--qt-noise-version", type=int, default=1, choices=[1, 2], help="QT 噪声版本，1 或 2")
    return parser.parse_args()


def apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.seed is not None:
        cfg.runtime.seed = args.seed
    if args.device is not None:
        cfg.runtime.device = args.device
    if args.output_dir is not None:
        cfg.runtime.output_dir = args.output_dir
    if args.data_path is not None:
        cfg.data.data_path = args.data_path
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
        cfg.data.val_batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.max_steps_per_epoch is not None:
        cfg.train.max_steps_per_epoch = args.max_steps_per_epoch
    cfg.validate()
    return cfg

def build_grad_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config) if args.config else None
    cfg = load_experiment_config(config_path=config_path, model_name=args.model_name, stage=args.stage)
    cfg = apply_overrides(cfg, args)

    logger = build_logger("train")
    device = resolve_device(cfg.runtime.device)
    seed_everything(cfg.runtime.seed)
    output_dir = ensure_dir(cfg.runtime.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    dump_config_snapshot(cfg, output_dir)

    logger.info("Git SHA: %s", _git_sha())
    logger.info("设备: %s | 阶段: %s", device, args.stage)
    logger.info("输出目录: %s", output_dir)

    train_ds, val_ds = build_train_val_datasets(
        cfg.data,
        seed=cfg.runtime.seed,
        use_qt_dataset=args.use_qt_dataset,
        qt_noise_version=args.qt_noise_version,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = ModalityFlexibleConditionalDiffusion(cfg.model, cfg.loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = build_grad_scaler(enabled=cfg.runtime.use_amp and device.type == "cuda")

    start_epoch = 0
    global_step = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, scaler=scaler, map_location=device)
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        logger.info("已恢复训练: epoch=%d, global_step=%d", start_epoch, global_step)

    engine = TrainEngine(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        stage=args.stage,
        modality_dropout=cfg.train.modality_dropout,
        use_amp=cfg.runtime.use_amp,
        grad_clip=cfg.train.grad_clip,
    )

    best_val = float("inf")
    for epoch in range(start_epoch, cfg.train.epochs):
        logger.info("==== Epoch %d/%d ====", epoch + 1, cfg.train.epochs)
        train_metrics = engine.train_one_epoch(
            dataloader=train_loader,
            max_steps=cfg.train.max_steps_per_epoch,
            log_interval=cfg.train.log_interval,
            logger=logger,
        )
        val_metrics = engine.validate_one_epoch(
            dataloader=val_loader,
            max_steps=max(1, cfg.train.max_steps_per_epoch // 2),
            logger=logger,
        )
        global_step += cfg.train.max_steps_per_epoch

        latest_path = ckpt_dir / "latest.pt"
        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            config=cfg,
        )

        if (epoch + 1) % cfg.train.save_every_epochs == 0:
            epoch_path = ckpt_dir / f"epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                path=epoch_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                config=cfg,
            )

        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            best_path = ckpt_dir / "best.pt"
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                config=cfg,
            )
            logger.info("刷新 best checkpoint: %s (val_total=%.6f)", best_path, best_val)

        logger.info("train_metrics=%s", train_metrics)
        logger.info("val_metrics=%s", val_metrics)

    logger.info("训练完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
