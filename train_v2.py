from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import build_bidmc_train_val_datasets, build_qt_train_val_datasets
from models.DDPM import DDPM
from models.HNF import ConditionalModel
from utils.common import dump_config_snapshot, ensure_dir, resolve_device, seed_everything
from utils.logging import build_logger

DEFAULT_CONFIG_PATH = SRC_DIR / "config" / "bidmc_v2.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BIDMC/QT ECG 单模态 DDPM v2 训练入口")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="v2 YAML 配置文件路径")
    parser.add_argument("--dataset", type=str, default="qt", choices=["bidmc", "qt"], help="选择训练数据集")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None, help="继续训练的 checkpoint 路径")
    parser.add_argument("--dataset-path", type=str, default=None, help="数据集根目录；BIDMC 对应 bidmc_root，QT 对应 qt_root")
    parser.add_argument("--qt-noise-version", type=int, default=1, choices=[1, 2], help="QT 噪声版本")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
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
        cfg["path"]["train_output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
        cfg["train"]["val_batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.dataset == "bidmc" and args.dataset_path is not None:
        cfg["bidmc"]["path"]["bidmc_root"] = args.dataset_path
    if args.dataset == "qt" and args.dataset_path is not None:
        cfg.setdefault("path", {})
        cfg["path"]["qt_root"] = args.dataset_path
    if args.max_steps_per_epoch is not None:
        cfg["train"]["max_steps_per_epoch"] = args.max_steps_per_epoch

    device = resolve_device(str(cfg["runtime"]["device"]))
    use_amp = bool(cfg["runtime"].get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    seed_everything(int(cfg["runtime"]["seed"]))

    output_dir = Path(cfg["path"]["train_output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = ensure_dir(output_dir.resolve())
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    cfg["path"]["train_output_dir"] = str(output_dir)
    dump_config_snapshot(cfg, output_dir)

    logger = build_logger("train_v2", log_path=output_dir / "train_v2.log")
    logger.info("设备: %s", device)
    logger.info("数据集: %s", args.dataset)
    logger.info("输出目录: %s", output_dir)

    if args.dataset == "qt":
        train_ds, val_ds = build_qt_train_val_datasets(
            noise_version=args.qt_noise_version,
            val_ratio=float(cfg.get("data", {}).get("val_ratio", 0.3)),
            seed=int(cfg["runtime"]["seed"]),
            data_root=cfg.get("path", {}).get("qt_root") or None,
        )
    else:
        train_ds, val_ds = build_bidmc_train_val_datasets(
            config=cfg,
            split_seed=int(cfg["bidmc"]["data"]["split_seed"]),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["val_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = DDPM(
        base_model=ConditionalModel(feats=int(cfg["hnf"]["feats"])),
        config=cfg,
        device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg["train"]["epochs"])),
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        payload = torch.load(resume_path.resolve(), map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if payload.get("scaler_state_dict"):
            scaler.load_state_dict(payload["scaler_state_dict"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        logger.info("恢复训练: epoch=%d global_step=%d", start_epoch, global_step)

    epochs = int(cfg["train"]["epochs"])
    max_steps_per_epoch = int(cfg["train"]["max_steps_per_epoch"])
    grad_clip = float(cfg["train"]["grad_clip"])
    best_val_loss = float("inf")

    epoch_bar = tqdm(
        range(start_epoch, epochs),
        total=max(0, epochs - start_epoch),
        desc=f"Epochs-{args.dataset}",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        model.train()
        train_losses: list[float] = []
        train_bar = tqdm(
            train_loader,
            total=min(len(train_loader), max_steps_per_epoch),
            desc=f"Train-{args.dataset}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )
        for step, batch in enumerate(train_bar):
            if step >= max_steps_per_epoch:
                break
            clean_ecg = batch["clean_ecg"].to(device, non_blocking=True)
            noisy_ecg = batch["noisy_ecg"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
            with autocast_ctx:
                loss = model(clean_ecg, noisy_ecg)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            train_bar.set_postfix(loss=f"{sum(train_losses) / len(train_losses):.4f}")

        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            val_bar = tqdm(
                val_loader,
                total=min(len(val_loader), max_steps_per_epoch),
                desc=f"Val-{args.dataset}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
            for step, batch in enumerate(val_bar):
                if step >= max_steps_per_epoch:
                    break
                clean_ecg = batch["clean_ecg"].to(device, non_blocking=True)
                noisy_ecg = batch["noisy_ecg"].to(device, non_blocking=True)
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
                with autocast_ctx:
                    loss = model(clean_ecg, noisy_ecg)
                val_losses.append(float(loss.detach().cpu().item()))
                val_bar.set_postfix(loss=f"{sum(val_losses) / len(val_losses):.4f}")

        train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        global_step += min(len(train_loader), max_steps_per_epoch)
        scheduler.step()

        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "config": cfg,
        }
        torch.save(payload, ckpt_dir / "latest.pt")

        if (epoch + 1) % int(cfg["train"]["save_every_epochs"]) == 0:
            torch.save(payload, ckpt_dir / f"epoch_{epoch + 1:04d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(payload, ckpt_dir / "best.pt")

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
        logger.info(
            "epoch=%d dataset=%s train_loss=%.6f val_loss=%.6f lr=%.8f",
            epoch + 1,
            args.dataset,
            train_loss,
            val_loss,
            float(optimizer.param_groups[0]["lr"]),
        )

    epoch_bar.close()
    logger.info("训练完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
