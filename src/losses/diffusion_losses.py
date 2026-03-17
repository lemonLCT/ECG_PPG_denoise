from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from losses.masked_losses import masked_derivative_l1, masked_mse

if TYPE_CHECKING:
    from models.main_model import DDPM


class DiffusionLoss(nn.Module):
    """
    扩散训练损失。
    输入:
    - `clean_ecg/clean_ppg:[B,1,T]`
    - `noisy_ecg/noisy_ppg:[B,1,T]`
    - `modality_mask:[B,2]`
    输出:
    - `total_loss`
    - `diffusion_loss`
    - `derivative_loss`
    - 以及训练中间量
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.diffusion_weight = float(cfg.get("diffusion_weight", 1.0))
        self.derivative_weight = float(cfg.get("derivative_weight", 0.0))

    @staticmethod
    def _ensure_3d(x: Tensor, name: str) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"{name} 期望 [B,1,T]，实际为 {tuple(x.shape)}")
        return x

    def forward(
        self,
        model: "DDPM",
        clean_ecg: Tensor,
        clean_ppg: Tensor,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        modality_mask: Tensor,
    ) -> dict[str, Tensor]:
        clean_ecg = self._ensure_3d(clean_ecg, "clean_ecg")
        clean_ppg = self._ensure_3d(clean_ppg, "clean_ppg")
        noisy_ecg = self._ensure_3d(noisy_ecg, "noisy_ecg")
        noisy_ppg = self._ensure_3d(noisy_ppg, "noisy_ppg")
        modality_mask = model.normalize_modality_mask(
            modality_mask,
            batch_size=clean_ecg.shape[0],
            device=clean_ecg.device,
            dtype=clean_ecg.dtype,
        )

        clean_ecg, clean_ppg = model._mask_signal_pair(clean_ecg, clean_ppg, modality_mask)
        noisy_ecg, noisy_ppg = model._fill_missing_noisy_pair(noisy_ecg, noisy_ppg, modality_mask)

        batch_size = clean_ecg.shape[0]
        device = clean_ecg.device
        t = model.diffusion.sample_timesteps(batch_size=batch_size, device=device)

        noise_ecg = torch.randn_like(clean_ecg)
        noise_ppg = torch.randn_like(clean_ppg)
        x_t_ecg = model.diffusion.q_sample(clean_ecg, t, noise_ecg)
        x_t_ppg = model.diffusion.q_sample(clean_ppg, t, noise_ppg)

        outputs = model.predict_noise_from_xt(
            x_t_ecg=x_t_ecg,
            x_t_ppg=x_t_ppg,
            cond_ecg=noisy_ecg,
            cond_ppg=noisy_ppg,
            t=t,
            modality_mask=modality_mask,
        )
        x0_hat = model.predict_x0_pair(
            x_t_ecg=x_t_ecg,
            x_t_ppg=x_t_ppg,
            t=t,
            pred_noise_ecg=outputs["pred_noise_ecg"],
            pred_noise_ppg=outputs["pred_noise_ppg"],
        )

        ecg_mask = modality_mask[:, 0]
        ppg_mask = modality_mask[:, 1]
        diffusion_loss = 0.5 * (
            masked_mse(outputs["pred_noise_ecg"], noise_ecg, ecg_mask)
            + masked_mse(outputs["pred_noise_ppg"], noise_ppg, ppg_mask)
        )
        derivative_loss = 0.5 * (
            masked_derivative_l1(x0_hat["x0_hat_ecg"], clean_ecg, ecg_mask)
            + masked_derivative_l1(x0_hat["x0_hat_ppg"], clean_ppg, ppg_mask)
        )
        total_loss = self.diffusion_weight * diffusion_loss + self.derivative_weight * derivative_loss

        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "derivative_loss": derivative_loss,
            "t": t,
            "noise_ecg": noise_ecg,
            "noise_ppg": noise_ppg,
            "x_t_ecg": x_t_ecg,
            "x_t_ppg": x_t_ppg,
            **x0_hat,
            **outputs,
        }
