from __future__ import annotations

import argparse
import json
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
from dataset import build_bidmc_test_dataset, build_qt_test_dataset, build_train_val_datasets
from models.model_v1 import DDPMv1, evaluate as evaluate_step
from utils.common import ensure_dir, load_checkpoint, resolve_device, seed_everything
from utils.logging import build_logger


def _resolve_project_path(path_value: str | Path) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 DDPMv1 双模态 HNF 扩散模型")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型 checkpoint 路径")
    parser.add_argument("--input-path", type=str, default=None, help="输入数据(.npz/.pt/.npy)或数据集根目录")
    parser.add_argument("--use-bidmc-dataset", action="store_true", help="使用 BIDMC 测试集评估")
    parser.add_argument("--use-qt-dataset", action="store_true", help="使用 QT 测试集评估")
    parser.add_argument("--qt-noise-version", type=int, default=1, choices=[1, 2], help="QT 噪声版本")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0, help=">0 时仅评估前 N 个样本")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def build_data_namespace(cfg: dict) -> SimpleNamespace:
    return SimpleNamespace(
        data_path=cfg["data"].get("data_path", ""),
        num_workers=int(cfg["data"]["num_workers"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        window_length=int(cfg["data"]["window_length"]),
        window_stride=int(cfg["data"]["window_stride"]),
        synthetic_num_samples=int(cfg["data"]["synthetic_num_samples"]),
    )


def build_eval_dataset(cfg: dict, args: argparse.Namespace):
    if args.use_bidmc_dataset and args.use_qt_dataset:
        raise ValueError("BIDMC 与 QT 数据集开关不能同时启用")

    if args.input_path is not None:
        cfg["path"]["dataset_path"] = args.input_path
        cfg["data"]["data_path"] = args.input_path
        cfg["path"]["qt_root"] = args.input_path
        if "bidmc" in cfg and "path" in cfg["bidmc"]:
            cfg["bidmc"]["path"]["bidmc_root"] = args.input_path

    if args.use_bidmc_dataset:
        return build_bidmc_test_dataset(config=cfg)

    if args.use_qt_dataset:
        return build_qt_test_dataset(
            noise_version=args.qt_noise_version,
            data_root=cfg["path"].get("qt_root") or None,
        )

    data_cfg = build_data_namespace(cfg)
    _, val_ds = build_train_val_datasets(
        data_cfg,
        seed=int(cfg["runtime"]["seed"]),
        use_qt_dataset=False,
        qt_noise_version=args.qt_noise_version,
    )
    return val_ds


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.seed is not None:
        cfg["runtime"]["seed"] = args.seed

    logger = build_logger("evaluate_v1")
    device = resolve_device(cfg["runtime"]["device"])
    seed_everything(int(cfg["runtime"]["seed"]))

    checkpoint_path = args.checkpoint or cfg["path"]["checkpoint_path"]
    if not checkpoint_path:
        raise ValueError("必须通过 --checkpoint 或 config.path.checkpoint_path 提供模型权重路径")
    checkpoint_path = _resolve_project_path(checkpoint_path)

    model = DDPMv1(base_model=None, config=cfg, device=device).to(device)
    load_checkpoint(checkpoint_path, model=model, optimizer=None, scaler=None, scheduler=None, map_location=device)
    model.eval()
    logger.info("加载 checkpoint: %s", checkpoint_path)

    dataset = build_eval_dataset(cfg, args)
    if args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))

    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    metric_sums = {
        "total_loss": 0.0,
        "diffusion_loss": 0.0,
        "ecg_diffusion_loss": 0.0,
        "ppg_diffusion_loss": 0.0,
    }
    batches = 0
    with tqdm(
        dataloader,
        total=len(dataloader),
        desc="Evaluate",
        unit="batch",
        dynamic_ncols=True,
    ) as progress_bar:
        for batch in progress_bar:
            batch = _move_batch(batch, device)
            outputs = evaluate_step(
                model=model,
                noisy_ecg=batch["noisy_ecg"],
                noisy_ppg=batch["noisy_ppg"],
                clean_ecg=batch["clean_ecg"],
                clean_ppg=batch["clean_ppg"],
            )
            for key in metric_sums:
                metric_sums[key] += float(outputs[key].detach().item())
            batches += 1
            progress_bar.set_postfix(
                total=float(outputs["total_loss"].detach().item()),
                ecg=float(outputs["ecg_diffusion_loss"].detach().item()),
                ppg=float(outputs["ppg_diffusion_loss"].detach().item()),
            )

    if batches == 0:
        raise ValueError("评估数据集为空，无法计算指标")

    summary = {key: value / batches for key, value in metric_sums.items()}
    logger.info("evaluate_metrics=%s", summary)

    output_dir = ensure_dir(_resolve_project_path(args.output_dir or cfg["path"]["eval_output_dir"]))
    metrics_path = output_dir / "metrics_v1.json"
    payload = {
        "checkpoint": str(checkpoint_path),
        "num_batches": batches,
        "metrics": summary,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("评估结果已保存: %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
