from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import DDPM


class _NullAutocast:
    """未启用 AMP 时的空上下文。"""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _build_autocast(enabled: bool):
    if not enabled:
        return _NullAutocast()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def _resolve_total_steps(dataloader: DataLoader, max_steps: int) -> int:
    try:
        return max(1, min(len(dataloader), max_steps))
    except TypeError:
        return max(1, max_steps)


def sample_modality_mask(batch_size: int, stage: str, device: torch.device, dropout_prob: float = 0.0) -> Tensor:
    """按训练阶段采样模态存在性掩码，输出 `mask:[B,2]`。"""
    if stage == "ecg_pretrain":
        return torch.tensor([1.0, 0.0], device=device).repeat(batch_size, 1)
    if stage == "ppg_pretrain":
        return torch.tensor([0.0, 1.0], device=device).repeat(batch_size, 1)
    if stage != "joint":
        raise ValueError(f"未知训练阶段: {stage}")

    states = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device)
    indices = torch.randint(0, states.shape[0], size=(batch_size,), device=device)
    mask = states[indices].clone()

    if dropout_prob > 0.0:
        drop = (torch.rand_like(mask) < dropout_prob).float()
        mask = mask * (1.0 - drop)
        both_zero = mask.sum(dim=1) == 0
        if both_zero.any():
            picks = torch.randint(0, 2, size=(int(both_zero.sum().item()),), device=device)
            mask[both_zero] = 0.0
            mask[both_zero, picks] = 1.0
    return mask


class TrainEngine:
    """单机训练循环封装。"""

    def __init__(
        self,
        model: DDPM,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler],
        device: torch.device,
        stage: str,
        modality_dropout: float,
        use_amp: bool,
        grad_clip: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.stage = stage
        self.modality_dropout = modality_dropout
        self.use_amp = use_amp and device.type == "cuda"
        self.grad_clip = grad_clip

    def _move_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {k: v.to(self.device, non_blocking=True) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def _progress_bar(self, total_steps: int, desc: str):
        return tqdm(total=total_steps, desc=desc, unit="batch", dynamic_ncols=True, leave=False)

    @staticmethod
    def _set_metrics_postfix(progress_bar: tqdm, sums: Dict[str, float], step_count: int) -> None:
        progress_bar.set_postfix(
            total=f"{sums['total_loss'] / step_count:.4f}",
            diff=f"{sums['diffusion_loss'] / step_count:.4f}",
            deriv=f"{sums['derivative_loss'] / step_count:.4f}",
        )

    def train_one_epoch(self, dataloader: DataLoader, max_steps: int, logger: logging.Logger) -> Dict[str, float]:
        self.model.train()
        sums = {"total_loss": 0.0, "diffusion_loss": 0.0, "derivative_loss": 0.0}
        step_count = 0
        total_steps = _resolve_total_steps(dataloader, max_steps)

        with self._progress_bar(total_steps=total_steps, desc="Train") as progress_bar:
            for step, batch in enumerate(dataloader):
                if step >= max_steps:
                    break
                step_count += 1
                batch = self._move_batch(batch)
                batch_size = batch["clean_ecg"].shape[0]
                modality_mask = sample_modality_mask(
                    batch_size=batch_size,
                    stage=self.stage,
                    device=self.device,
                    dropout_prob=self.modality_dropout if self.stage == "joint" else 0.0,
                )

                self.optimizer.zero_grad(set_to_none=True)
                with _build_autocast(enabled=self.use_amp):
                    outputs = self.model(
                        noisy_ecg=batch["noisy_ecg"],
                        noisy_ppg=batch["noisy_ppg"],
                        clean_ecg=batch["clean_ecg"],
                        clean_ppg=batch["clean_ppg"],
                        modality_mask=modality_mask,
                        train_gen_flag=0,
                    )
                    loss = outputs["total_loss"]

                if self.scaler is not None and self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.optimizer.step()

                for key in sums:
                    sums[key] += float(outputs[key].detach().cpu().item())

                progress_bar.update(1)
                self._set_metrics_postfix(progress_bar, sums, step_count)

        if step_count == 0:
            step_count = 1
        metrics = {k: v / step_count for k, v in sums.items()}
        logger.info("训练 epoch 完成: %s", metrics)
        return metrics

    @torch.no_grad()
    def validate_one_epoch(self, dataloader: DataLoader, max_steps: int, logger: logging.Logger) -> Dict[str, float]:
        self.model.eval()
        sums = {"total_loss": 0.0, "diffusion_loss": 0.0, "derivative_loss": 0.0}
        step_count = 0
        total_steps = _resolve_total_steps(dataloader, max_steps)

        with self._progress_bar(total_steps=total_steps, desc="Val") as progress_bar:
            for step, batch in enumerate(dataloader):
                if step >= max_steps:
                    break
                step_count += 1
                batch = self._move_batch(batch)
                batch_size = batch["clean_ecg"].shape[0]
                modality_mask = sample_modality_mask(
                    batch_size=batch_size,
                    stage=self.stage,
                    device=self.device,
                    dropout_prob=0.0,
                )

                outputs = self.model(
                    noisy_ecg=batch["noisy_ecg"],
                    noisy_ppg=batch["noisy_ppg"],
                    clean_ecg=batch["clean_ecg"],
                    clean_ppg=batch["clean_ppg"],
                    modality_mask=modality_mask,
                    train_gen_flag=0,
                )
                for key in sums:
                    sums[key] += float(outputs[key].detach().cpu().item())

                progress_bar.update(1)
                self._set_metrics_postfix(progress_bar, sums, step_count)

        if step_count == 0:
            step_count = 1
        metrics = {k: v / step_count for k, v in sums.items()}
        logger.info("验证 epoch 完成: %s", metrics)
        return metrics


class ExperimentRunner:
    """最小 smoke 运行器。"""

    def __init__(self, config: dict, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run_smoke(self) -> None:
        device = torch.device("cpu")
        model = DDPM(base_model=None, config=self.config, device=device).to(device)
        batch_size = 2
        length = int(self.config["data"]["window_length"])
        clean_ecg = torch.randn(batch_size, 1, length, device=device)
        clean_ppg = torch.randn(batch_size, 1, length, device=device)
        noisy_ecg = clean_ecg + 0.1 * torch.randn_like(clean_ecg)
        noisy_ppg = clean_ppg + 0.1 * torch.randn_like(clean_ppg)
        modality_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], device=device)

        outputs = model(
            noisy_ecg=noisy_ecg,
            noisy_ppg=noisy_ppg,
            clean_ecg=clean_ecg,
            clean_ppg=clean_ppg,
            modality_mask=modality_mask,
            train_gen_flag=0,
        )
        self.logger.info("smoke 通过，总损失 %.6f", outputs["total_loss"].item())
