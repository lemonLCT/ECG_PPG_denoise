from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from config import LossConfig, ModelConfig
from losses import masked_derivative_l1, masked_l1, masked_mse
from models.blocks import UnifiedNoisePredictor1D
from models.diffusion_schedule import DiffusionSchedule1D
from models.encoders import QualityAssessor1D, SignalConditionEncoder1D
from models.fusion import ModalityAwareFusion


class ModalityFlexibleConditionalDiffusion(nn.Module):
    """统一的 ECG/PPG 条件扩散模型。"""

    def __init__(self, model_cfg: ModelConfig, loss_cfg: Optional[LossConfig] = None) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg or LossConfig()

        self.ecg_encoder = SignalConditionEncoder1D(
            out_channels=model_cfg.cond_channels,
            gn_groups=model_cfg.gn_groups,
        )
        self.ppg_encoder = SignalConditionEncoder1D(
            out_channels=model_cfg.cond_channels,
            gn_groups=model_cfg.gn_groups,
        )
        self.ecg_quality = QualityAssessor1D(channels=model_cfg.cond_channels, gn_groups=model_cfg.gn_groups)
        self.ppg_quality = QualityAssessor1D(channels=model_cfg.cond_channels, gn_groups=model_cfg.gn_groups)
        self.fusion = ModalityAwareFusion(channels=model_cfg.cond_channels, joint_channels=model_cfg.joint_channels)

        self.noise_predictor = UnifiedNoisePredictor1D(
            cond_channels=model_cfg.cond_channels,
            joint_channels=model_cfg.joint_channels,
            base_channels=model_cfg.base_channels,
            gn_groups=model_cfg.gn_groups,
        )

        self.ecg_output_head = nn.Conv1d(1, 1, kernel_size=1)
        self.ppg_output_head = nn.Conv1d(1, 1, kernel_size=1)

        self.diffusion = DiffusionSchedule1D(
            num_steps=model_cfg.diffusion_steps,
            beta_start=model_cfg.beta_start,
            beta_end=model_cfg.beta_end,
        )

    @staticmethod
    def _ensure_3d(x: Tensor, name: str) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"{name} 期望 [B,1,T]，实际为 {tuple(x.shape)}")
        return x

    @staticmethod
    def build_modality_mask(has_ecg: bool, has_ppg: bool, batch_size: int, device: torch.device) -> Tensor:
        mask = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)
        mask[:, 0] = 1.0 if has_ecg else 0.0
        mask[:, 1] = 1.0 if has_ppg else 0.0
        if not has_ecg and not has_ppg:
            raise ValueError("至少需要一个可用模态")
        return mask

    @staticmethod
    def _signal_mask(modality_mask: Tensor, index: int, ref: Tensor) -> Tensor:
        return modality_mask[:, index].to(dtype=ref.dtype, device=ref.device).view(-1, 1, 1)

    def _mask_signal_pair(self, ecg: Tensor, ppg: Tensor, modality_mask: Tensor) -> tuple[Tensor, Tensor]:
        ecg_mask = self._signal_mask(modality_mask, 0, ecg)
        ppg_mask = self._signal_mask(modality_mask, 1, ppg)
        return ecg * ecg_mask, ppg * ppg_mask

    def _encode_conditions(self, cond_ecg: Tensor, cond_ppg: Tensor, modality_mask: Tensor) -> Dict[str, Tensor]:
        cond_ecg, cond_ppg = self._mask_signal_pair(cond_ecg, cond_ppg, modality_mask)
        feat_ecg = self.ecg_encoder(cond_ecg)
        feat_ppg = self.ppg_encoder(cond_ppg)
        q_map_ecg = self.ecg_quality(feat_ecg)
        q_map_ppg = self.ppg_quality(feat_ppg)
        q_map_ecg = q_map_ecg * self._signal_mask(modality_mask, 0, q_map_ecg)
        q_map_ppg = q_map_ppg * self._signal_mask(modality_mask, 1, q_map_ppg)
        c_ecg, c_ppg, c_joint = self.fusion(feat_ecg, feat_ppg, q_map_ecg, q_map_ppg, modality_mask)
        return {
            "feat_ecg": feat_ecg,
            "feat_ppg": feat_ppg,
            "q_map_ecg": q_map_ecg,
            "q_map_ppg": q_map_ppg,
            "c_ecg": c_ecg,
            "c_ppg": c_ppg,
            "c_joint": c_joint,
        }

    def predict_noise_from_xt(
        self,
        x_t_ecg: Tensor,
        x_t_ppg: Tensor,
        t: Tensor,
        modality_mask: Tensor,
        cond_ecg: Optional[Tensor] = None,
        cond_ppg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        x_t_ecg = self._ensure_3d(x_t_ecg, "x_t_ecg")
        x_t_ppg = self._ensure_3d(x_t_ppg, "x_t_ppg")
        cond_ecg = x_t_ecg if cond_ecg is None else self._ensure_3d(cond_ecg, "cond_ecg")
        cond_ppg = x_t_ppg if cond_ppg is None else self._ensure_3d(cond_ppg, "cond_ppg")
        x_t_ecg, x_t_ppg = self._mask_signal_pair(x_t_ecg, x_t_ppg, modality_mask)

        enc = self._encode_conditions(cond_ecg=cond_ecg, cond_ppg=cond_ppg, modality_mask=modality_mask)
        x_t_pair = torch.cat([x_t_ecg, x_t_ppg], dim=1)
        pred_eps_pair = self.noise_predictor(
            x_t_pair=x_t_pair,
            t=t,
            c_joint=enc["c_joint"],
            c_ecg=enc["c_ecg"],
            c_ppg=enc["c_ppg"],
            modality_mask=modality_mask,
        )

        pred_noise_ecg = pred_eps_pair[:, 0:1, :]
        pred_noise_ppg = pred_eps_pair[:, 1:2, :]
        x0_hat_pair = self.diffusion.predict_x0_from_eps(x_t=x_t_pair, t=t, eps=pred_eps_pair)
        x0_hat_ecg = self.ecg_output_head(x0_hat_pair[:, 0:1, :])
        x0_hat_ppg = self.ppg_output_head(x0_hat_pair[:, 1:2, :])

        return {
            "pred_noise_ecg": pred_noise_ecg,
            "pred_noise_ppg": pred_noise_ppg,
            "x0_hat_ecg": x0_hat_ecg,
            "x0_hat_ppg": x0_hat_ppg,
            "q_map_ecg": enc["q_map_ecg"],
            "q_map_ppg": enc["q_map_ppg"],
            "c_ecg": enc["c_ecg"],
            "c_ppg": enc["c_ppg"],
            "c_joint": enc["c_joint"],
        }

    def forward(self, noisy_ecg: Tensor, noisy_ppg: Tensor, t: Tensor, modality_mask: Tensor) -> Dict[str, Tensor]:
        return self.predict_noise_from_xt(
            x_t_ecg=noisy_ecg,
            x_t_ppg=noisy_ppg,
            t=t,
            modality_mask=modality_mask,
            cond_ecg=noisy_ecg,
            cond_ppg=noisy_ppg,
        )

    def compute_losses(
        self,
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
        t = self.diffusion.sample_timesteps(batch_size=batch_size, device=device)
        noise_ecg = torch.randn_like(clean_ecg)
        noise_ppg = torch.randn_like(clean_ppg)
        x_t_ecg = self.diffusion.q_sample(clean_ecg, t, noise_ecg)
        x_t_ppg = self.diffusion.q_sample(clean_ppg, t, noise_ppg)

        outputs = self.predict_noise_from_xt(
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

        rec_loss_ecg = masked_l1(outputs["x0_hat_ecg"], clean_ecg, ecg_mask)
        rec_loss_ppg = masked_l1(outputs["x0_hat_ppg"], clean_ppg, ppg_mask)
        reconstruction_loss = 0.5 * (rec_loss_ecg + rec_loss_ppg)

        der_loss_ecg = masked_derivative_l1(outputs["x0_hat_ecg"], clean_ecg, ecg_mask)
        der_loss_ppg = masked_derivative_l1(outputs["x0_hat_ppg"], clean_ppg, ppg_mask)
        derivative_loss = 0.5 * (der_loss_ecg + der_loss_ppg)

        total_loss = (
            self.loss_cfg.diffusion_weight * diffusion_loss
            + self.loss_cfg.reconstruction_weight * reconstruction_loss
            + self.loss_cfg.derivative_weight * derivative_loss
        )

        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "reconstruction_loss": reconstruction_loss,
            "derivative_loss": derivative_loss,
            **outputs,
        }

    @torch.no_grad()
    def denoise_signal(
        self,
        y_ecg: Optional[Tensor] = None,
        y_ppg: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        use_ddim: bool = False,
    ) -> Dict[str, Tensor]:
        if y_ecg is None and y_ppg is None:
            raise ValueError("y_ecg 和 y_ppg 不能同时为空")

        if y_ecg is not None:
            y_ecg = self._ensure_3d(y_ecg, "y_ecg")
        if y_ppg is not None:
            y_ppg = self._ensure_3d(y_ppg, "y_ppg")

        ref = y_ecg if y_ecg is not None else y_ppg
        assert ref is not None
        if y_ecg is not None and y_ppg is not None and y_ecg.shape != y_ppg.shape:
            raise ValueError(f"y_ecg 和 y_ppg 形状必须一致，实际为 {tuple(y_ecg.shape)} 和 {tuple(y_ppg.shape)}")

        batch_size, _, length = ref.shape
        device = ref.device
        has_ecg = y_ecg is not None
        has_ppg = y_ppg is not None
        modality_mask = self.build_modality_mask(has_ecg=has_ecg, has_ppg=has_ppg, batch_size=batch_size, device=device)

        if y_ecg is None:
            y_ecg = torch.zeros(batch_size, 1, length, device=device, dtype=ref.dtype)
        if y_ppg is None:
            y_ppg = torch.zeros(batch_size, 1, length, device=device, dtype=ref.dtype)
        y_ecg = self._ensure_3d(y_ecg, "y_ecg")
        y_ppg = self._ensure_3d(y_ppg, "y_ppg")

        x_t_pair = torch.randn(batch_size, 2, length, device=device, dtype=ref.dtype)
        total_steps = self.diffusion.num_steps if num_steps is None else int(num_steps)
        if total_steps <= 0:
            raise ValueError("num_steps 必须 > 0")

        time_grid = torch.linspace(
            self.diffusion.num_steps - 1,
            0,
            steps=total_steps,
            device=device,
            dtype=torch.float32,
        ).long()

        for scalar_t in time_grid:
            t = torch.full((batch_size,), int(scalar_t.item()), device=device, dtype=torch.long)
            out = self.predict_noise_from_xt(
                x_t_ecg=x_t_pair[:, 0:1, :],
                x_t_ppg=x_t_pair[:, 1:2, :],
                t=t,
                modality_mask=modality_mask,
                cond_ecg=y_ecg,
                cond_ppg=y_ppg,
            )
            pred_eps = torch.cat([out["pred_noise_ecg"], out["pred_noise_ppg"]], dim=1)
            if use_ddim:
                x_t_pair = self.diffusion.ddim_sample_step(x_t_pair, t, pred_eps)
            else:
                x_t_pair = self.diffusion.p_sample(x_t_pair, t, pred_eps)

        denoised_ecg = self.ecg_output_head(x_t_pair[:, 0:1, :])
        denoised_ppg = self.ppg_output_head(x_t_pair[:, 1:2, :])
        denoised_ecg, denoised_ppg = self._mask_signal_pair(denoised_ecg, denoised_ppg, modality_mask)
        return {
            "denoised_ecg": denoised_ecg,
            "denoised_ppg": denoised_ppg,
            "modality_mask": modality_mask,
        }
