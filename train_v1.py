from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config
from dataset import build_bidmc_train_val_datasets, build_qt_train_val_datasets, build_train_val_datasets
from models.model_v1 import DDPMv1, train as train_step
from utils.common import dump_config_snapshot, ensure_dir, load_checkpoint, resolve_device, save_checkpoint, seed_everything
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
    parser = argparse.ArgumentParser(description="训练 DDPMv1 双模态 HNF 扩散模型")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
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
    parser.add_argument("--use-qt-dataset", action="store_true", help="启用 QT 数据集")
    parser.add_argument("--qt-noise-version", type=int, default=1, choices=[1, 2], help="QT 噪声版本")
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
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


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _aggregate_metrics(sums: dict[str, float], steps: int) -> dict[str, float]:
    if steps <= 0:
        return {}
    return {key: value / steps for key, value in sums.items()}


def validate_one_epoch(
    model: DDPMv1,
    dataloader: DataLoader,
    device: torch.device,
    max_steps: int,
) -> dict[str, float]:
    model.eval()
    metric_sums = {
        "total_loss": 0.0,
        "diffusion_loss": 0.0,
        "ecg_diffusion_loss": 0.0,
        "ppg_diffusion_loss": 0.0,
    }
    step_count = 0
    with torch.no_grad():
        total_steps = max(1, min(len(dataloader), max_steps))
        with tqdm(
            dataloader,
            total=total_steps,
            desc="Val",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ) as progress_bar:
            for step, batch in enumerate(progress_bar):
                if step >= max_steps:
                    break
                batch = _move_batch(batch, device)
                outputs = model(
                    noisy_ecg=batch["noisy_ecg"],
                    noisy_ppg=batch["noisy_ppg"],
                    clean_ecg=batch["clean_ecg"],
                    clean_ppg=batch["clean_ppg"],
                )
                for key in metric_sums:
                    metric_sums[key] += float(outputs[key].detach().item())
                step_count += 1
                progress_bar.set_postfix(
                    total=float(outputs["total_loss"].detach().item()),
                    ecg=float(outputs["ecg_diffusion_loss"].detach().item()),
                    ppg=float(outputs["ppg_diffusion_loss"].detach().item()),
                )
    return _aggregate_metrics(metric_sums, step_count)


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    cfg = apply_overrides(cfg, args)

    output_dir = ensure_dir(_resolve_project_path(cfg["path"]["train_output_dir"]))
    cfg["path"]["train_output_dir"] = str(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    logger = build_logger("train_v1", log_path=output_dir / "train_v1.log")
    device = resolve_device(cfg["runtime"]["device"])
    seed_everything(int(cfg["runtime"]["seed"]))
    dump_config_snapshot(cfg, output_dir)

    logger.info("Git SHA: %s", _git_sha())
    logger.info("设备: %s", device)
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

    model = DDPMv1(base_model=None, config=cfg, device=device).to(device)
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
        payload = load_checkpoint(
            _resolve_project_path(args.resume),
            model=model,
            optimizer=optimizer,
            scaler=None,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        logger.info("已恢复训练 epoch=%d global_step=%d", start_epoch, global_step)

    best_val = float("inf")
    epochs = int(cfg["train"]["epochs"])
    max_steps_per_epoch = int(cfg["train"]["max_steps_per_epoch"])
    for epoch in range(start_epoch, epochs):
        model.train()
        metric_sums = {
            "total_loss": 0.0,
            "diffusion_loss": 0.0,
            "ecg_diffusion_loss": 0.0,
            "ppg_diffusion_loss": 0.0,
        }
        step_count = 0
        logger.info("==== Epoch %d/%d ====", epoch + 1, epochs)

        total_steps = max(1, min(len(train_loader), max_steps_per_epoch))
        with tqdm(
            train_loader,
            total=total_steps,
            desc=f"Train {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ) as progress_bar:
            for step, batch in enumerate(progress_bar):
                if step >= max_steps_per_epoch:
                    break
                batch = _move_batch(batch, device)
                outputs = train_step(
                    model=model,
                    optimizer=optimizer,
                    noisy_ecg=batch["noisy_ecg"],
                    noisy_ppg=batch["noisy_ppg"],
                    clean_ecg=batch["clean_ecg"],
                    clean_ppg=batch["clean_ppg"],
                )
                for key in metric_sums:
                    metric_sums[key] += float(outputs[key].detach().item())
                step_count += 1
                progress_bar.set_postfix(
                    total=float(outputs["total_loss"].detach().item()),
                    ecg=float(outputs["ecg_diffusion_loss"].detach().item()),
                    ppg=float(outputs["ppg_diffusion_loss"].detach().item()),
                )

        train_metrics = _aggregate_metrics(metric_sums, step_count)
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            max_steps=max(1, max_steps_per_epoch // 2),
        )
        global_step += step_count
        scheduler.step()

        logger.info("当前学习率 %.8f", float(optimizer.param_groups[0]["lr"]))
        logger.info("train_metrics=%s", train_metrics)
        logger.info("val_metrics=%s", val_metrics)

        latest_path = ckpt_dir / "latest_v1.pt"
        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scaler=None,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            config=cfg,
        )

        if (epoch + 1) % int(cfg["train"]["save_every_epochs"]) == 0:
            epoch_path = ckpt_dir / f"epoch_v1_{epoch + 1:04d}.pt"
            save_checkpoint(
                path=epoch_path,
                model=model,
                optimizer=optimizer,
                scaler=None,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config=cfg,
            )

        if val_metrics and val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            best_path = ckpt_dir / "best_v1.pt"
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scaler=None,
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
