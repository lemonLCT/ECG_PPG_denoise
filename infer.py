from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_ppg_denoise.config import load_experiment_config
from ecg_ppg_denoise.data import load_multimodal_arrays
from ecg_ppg_denoise.models import ModalityFlexibleConditionalDiffusion
from ecg_ppg_denoise.utils import build_logger, ensure_dir, generate_demo_signals, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="推理 ModalityFlexibleConditionalDiffusion")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--model-name", type=str, default=None, help="YAML 中的模型配置名")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型 checkpoint，可不传")
    parser.add_argument("--input-path", type=str, default=None, help="输入数据 .npz/.pt/.npy")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "ecg", "ppg", "joint"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--use-ddim", action="store_true")
    return parser.parse_args()


def _prepare_input_from_file(path: str | Path) -> Tuple[torch.Tensor, torch.Tensor]:
    arrays = load_multimodal_arrays(path)
    noisy_ecg = torch.from_numpy(arrays["noisy_ecg"][0:1]).float()
    noisy_ppg = torch.from_numpy(arrays["noisy_ppg"][0:1]).float()
    return noisy_ecg, noisy_ppg


def _select_mode_inputs(
    mode: str,
    noisy_ecg: Optional[torch.Tensor],
    noisy_ppg: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if mode == "joint":
        return noisy_ecg, noisy_ppg
    if mode == "ecg":
        return noisy_ecg, None
    if mode == "ppg":
        return None, noisy_ppg
    if noisy_ecg is not None and noisy_ppg is not None:
        return noisy_ecg, noisy_ppg
    return noisy_ecg, noisy_ppg


def main() -> int:
    args = parse_args()
    cfg = load_experiment_config(
        config_path=Path(args.config) if args.config else None,
        model_name=args.model_name,
    )
    if args.device is not None:
        cfg.runtime.device = args.device
    cfg.validate()

    logger = build_logger("infer")
    device = resolve_device(cfg.runtime.device)
    model = ModalityFlexibleConditionalDiffusion(cfg.model).to(device)
    model.eval()

    checkpoint_path = args.checkpoint or cfg.path.checkpoint_path
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model"])
        logger.info("已加载 checkpoint: %s", ckpt_path)
    else:
        logger.info("未提供 checkpoint，使用随机初始化参数执行推理")

    input_path = args.input_path or cfg.path.infer_input_path
    if input_path:
        noisy_ecg, noisy_ppg = _prepare_input_from_file(input_path)
    else:
        demo = generate_demo_signals(batch_size=1, signal_length=cfg.data.window_length, seed=cfg.runtime.seed)
        noisy_ecg = demo["noisy_ecg"]
        noisy_ppg = demo["noisy_ppg"]

    y_ecg, y_ppg = _select_mode_inputs(args.mode, noisy_ecg, noisy_ppg)
    if y_ecg is not None:
        y_ecg = y_ecg.to(device)
    if y_ppg is not None:
        y_ppg = y_ppg.to(device)

    with torch.no_grad():
        result = model.denoise_signal(
            y_ecg=y_ecg,
            y_ppg=y_ppg,
            num_steps=args.num_steps,
            use_ddim=args.use_ddim,
        )

    output_path = Path(args.output_path or cfg.path.infer_output_path)
    ensure_dir(output_path.parent)
    mask_row = result["modality_mask"][0].detach().cpu()
    payload = {"modality_mask": result["modality_mask"].detach().cpu().numpy()}
    if bool(mask_row[0].item()):
        payload["denoised_ecg"] = result["denoised_ecg"].detach().cpu().numpy()
        if y_ecg is not None:
            payload["noisy_ecg"] = y_ecg.detach().cpu().numpy()
    else:
        payload["generated_ecg"] = result["denoised_ecg"].detach().cpu().numpy()
    if bool(mask_row[1].item()):
        payload["denoised_ppg"] = result["denoised_ppg"].detach().cpu().numpy()
        if y_ppg is not None:
            payload["noisy_ppg"] = y_ppg.detach().cpu().numpy()
    else:
        payload["generated_ppg"] = result["denoised_ppg"].detach().cpu().numpy()
    np.savez(output_path, **payload)
    logger.info("推理完成，结果已保存到 %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
