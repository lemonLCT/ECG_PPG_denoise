from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config
from dataset import build_bidmc_train_val_datasets, build_qt_train_val_datasets, build_train_val_datasets
from models import DDPM
from trainers import TrainEngine
from utils.common import (
    dump_config_snapshot,
    ensure_dir,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from utils.logging import build_logger


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _resolve_project_path(path_value: str | Path) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练多模态扩散模型")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--stage", type=str, default=None, choices=["ecg_pretrain", "ppg_pretrain", "joint"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="继续训练的 checkpoint 路径")
    parser.add_argument("--use-bidmc-dataset", action="store_true", help="启用 BIDMC 双模态数据集")
    parser.add_argument("--use-qt-dataset", action="store_true", help="启用 QT Data_Preparation 适配数据集")
    parser.add_argument("--qt-noise-version", type=int, default=1, choices=[1, 2], help="QT 噪声版本")
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.stage is not None:
        cfg["train"]["stage_name"] = args.stage
    if args.seed is not None:
        cfg["runtime"]["seed"] = args.seed
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.output_dir is not None:
        cfg["path"]["train_output_dir"] = args.output_dir
    if args.dataset_path is not None:
        cfg["path"]["dataset_path"] = args.dataset_path
        cfg["data"]["data_path"] = args.dataset_path
        if "bidmc" in cfg and "path" in cfg["bidmc"]:
            cfg["bidmc"]["path"]["bidmc_root"] = args.dataset_path
        cfg["path"]["qt_root"] = args.dataset_path
    elif cfg["path"].get("dataset_path"):
        cfg["data"]["data_path"] = cfg["path"]["dataset_path"]
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
        cfg["train"]["val_batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.max_steps_per_epoch is not None:
        cfg["train"]["max_steps_per_epoch"] = args.max_steps_per_epoch
    return cfg


def build_data_namespace(cfg: dict) -> SimpleNamespace:
    return SimpleNamespace(
        data_path=cfg["data"].get("data_path", ""),
        num_workers=int(cfg["data"]["num_workers"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        window_length=int(cfg["data"]["window_length"]),
        window_stride=int(cfg["data"]["window_stride"]),
        synthetic_num_samples=int(cfg["data"]["synthetic_num_samples"]),
    )


def build_datasets(cfg: dict, args: argparse.Namespace):
    if args.use_bidmc_dataset and args.use_qt_dataset:
        raise ValueError("BIDMC 与 QT 数据集开关不能同时启用")

    if args.use_bidmc_dataset:
        split_seed = int(cfg["bidmc"]["data"]["split_seed"])
        return build_bidmc_train_val_datasets(config=cfg, split_seed=split_seed)

    if args.use_qt_dataset:
        return build_qt_train_val_datasets(
            noise_version=args.qt_noise_version,
            val_ratio=float(cfg["data"]["val_ratio"]),
            seed=int(cfg["runtime"]["seed"]),
            data_root=cfg["path"].get("qt_root") or None,
        )

    data_cfg = build_data_namespace(cfg)
    return build_train_val_datasets(
        data_cfg,
        seed=int(cfg["runtime"]["seed"]),
        use_qt_dataset=False,
        qt_noise_version=args.qt_noise_version,
    )


def build_grad_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    cfg = apply_overrides(cfg, args)

    output_dir = ensure_dir(_resolve_project_path(cfg["path"]["train_output_dir"]))
    cfg["path"]["train_output_dir"] = str(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    logger = build_logger("train", log_path=output_dir / "train.log")
    device = resolve_device(cfg["runtime"]["device"])
    seed_everything(int(cfg["runtime"]["seed"]))
    dump_config_snapshot(cfg, output_dir)

    logger.info("Git SHA: %s", _git_sha())
    logger.info("设备: %s | 阶段: %s", device, cfg["train"]["stage_name"])
    logger.info("输出目录: %s", output_dir)

    train_ds, val_ds = build_datasets(cfg, args)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["val_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    model = DDPM(base_model=None, config=cfg, device=device).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg["train"]["epochs"])),
    )
    scaler = build_grad_scaler(enabled=bool(cfg["runtime"]["use_amp"]) and device.type == "cuda")

    start_epoch = 0
    global_step = 0
    if args.resume:
        payload = load_checkpoint(
            _resolve_project_path(args.resume),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        logger.info("已恢复训练 epoch=%d global_step=%d", start_epoch, global_step)

    engine = TrainEngine(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        stage=cfg["train"]["stage_name"],
        use_amp=bool(cfg["runtime"]["use_amp"]),
        grad_clip=float(cfg["train"]["grad_clip"]),
    )

    best_val = float("inf")
    epochs = int(cfg["train"]["epochs"])
    max_steps_per_epoch = int(cfg["train"]["max_steps_per_epoch"])
    for epoch in range(start_epoch, epochs):
        logger.info("==== Epoch %d/%d ====", epoch + 1, epochs)
        train_metrics = engine.train_one_epoch(
            dataloader=train_loader,
            max_steps=max_steps_per_epoch,
            logger=logger,
        )
        val_metrics = engine.validate_one_epoch(
            dataloader=val_loader,
            max_steps=max(1, max_steps_per_epoch // 2),
            logger=logger,
        )
        global_step += max_steps_per_epoch
        scheduler.step()
        logger.info("当前学习率 %.8f", float(optimizer.param_groups[0]["lr"]))
        logger.info("train_metrics=%s", train_metrics)
        logger.info("val_metrics=%s", val_metrics)

        latest_path = ckpt_dir / "latest.pt"
        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            config=cfg,
        )

        if (epoch + 1) % int(cfg["train"]["save_every_epochs"]) == 0:
            epoch_path = ckpt_dir / f"epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                path=epoch_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
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
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config=cfg,
            )
            logger.info("刷新 best checkpoint: %s (val_total=%.6f)", best_path, best_val)

    logger.info("训练完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
