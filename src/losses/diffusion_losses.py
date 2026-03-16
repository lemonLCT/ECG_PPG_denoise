from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch
from torch import Tensor, nn

from config import LossConfig
from losses.masked_losses import masked_derivative_l1, masked_mse

if TYPE_CHECKING:
    from models.unified_diffusion_model import ModalityFlexibleConditionalDiffusion


class UnifiedDiffusionLoss(nn.Module):
    """统一扩散训练损失，负责时间步采样、前向加噪和多项损失聚合。"""

    def __init__(self, loss_cfg: LossConfig) -> None:
        super().__init__()
        self.loss_cfg = loss_cfg

    @staticmethod
    def _ensure_3d(x: Tensor, name: str) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"{name} 期望 [B,1,T]，实际为 {tuple(x.shape)}")
        return x

    def forward(
        self,
        model: "ModalityFlexibleConditionalDiffusion",
        clean_ecg: Tensor,
        clean_ppg: Tensor,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        modality_mask: Tensor,
    ) -> Dict[str, Tensor]:
        clean_ecg = self._ensure_3d(clean_ecg, "clean_ecg")
        clean_ppg = self._ensure_3d(clean_ppg, "clean_ppg")
        noisy_ecg = self._ensure_3d(noisy_ecg, "noisy_ecg")
        noisy_ppg = self._ensure_3d(noisy_ppg, "noisy_ppg")

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
            t=t,
            modality_mask=modality_mask,
            cond_ecg=noisy_ecg,
            cond_ppg=noisy_ppg,
        )

        ecg_mask = modality_mask[:, 0]
        ppg_mask = modality_mask[:, 1]
        diff_loss_ecg = masked_mse(outputs["pred_noise_ecg"], noise_ecg, ecg_mask)
        diff_loss_ppg = masked_mse(outputs["pred_noise_ppg"], noise_ppg, ppg_mask)
        diffusion_loss = 0.5 * (diff_loss_ecg + diff_loss_ppg)

        der_loss_ecg = masked_derivative_l1(outputs["x0_hat_ecg"], clean_ecg, ecg_mask)
        der_loss_ppg = masked_derivative_l1(outputs["x0_hat_ppg"], clean_ppg, ppg_mask)
        derivative_loss = 0.5 * (der_loss_ecg + der_loss_ppg)

        total_loss = (
            self.loss_cfg.diffusion_weight * diffusion_loss
            + self.loss_cfg.derivative_weight * derivative_loss
        )

        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "derivative_loss": derivative_loss,
            "t": t,
            "noise_ecg": noise_ecg,
            "noise_ppg": noise_ppg,
            "x_t_ecg": x_t_ecg,
            "x_t_ppg": x_t_ppg,
            **outputs,
        }
