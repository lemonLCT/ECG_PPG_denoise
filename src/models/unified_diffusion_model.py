from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from config import ModelConfig
from models.blocks import UnifiedNoisePredictor1D
from models.diffusion_schedule import DiffusionSchedule1D
from models.encoders import SignalConditionEncoder1D
from models.quality_assessor import QualityAssessor1D


class ModalityFlexibleConditionalDiffusion(nn.Module):
    """统一的 ECG/PPG 条件扩散模型。"""

    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.cond_channels = self._resolve_cond_channels(model_cfg)

        self.ecg_encoder = SignalConditionEncoder1D(
            in_channels=model_cfg.ecg_encoder.input_channels,
            branch_channels=model_cfg.ecg_encoder.branch_channels,
            kernel_sizes=model_cfg.ecg_encoder.kernel_sizes,
            out_channels=model_cfg.ecg_encoder.output_channels,
            use_projection=model_cfg.ecg_encoder.use_projection,
            gn_groups=model_cfg.gn_groups,
        )
        self.ppg_encoder = SignalConditionEncoder1D(
            in_channels=model_cfg.ppg_encoder.input_channels,
            branch_channels=model_cfg.ppg_encoder.branch_channels,
            kernel_sizes=model_cfg.ppg_encoder.kernel_sizes,
            out_channels=model_cfg.ppg_encoder.output_channels,
            use_projection=model_cfg.ppg_encoder.use_projection,
            gn_groups=model_cfg.gn_groups,
        )
        self.ecg_quality = QualityAssessor1D(
            channels=self.cond_channels,
            hidden_channels=model_cfg.quality_assessor.hidden_channels,
            gn_groups=model_cfg.gn_groups,
        )
        self.ppg_quality = QualityAssessor1D(
            channels=self.cond_channels,
            hidden_channels=model_cfg.quality_assessor.hidden_channels,
            gn_groups=model_cfg.gn_groups,
        )
        # 暂不启用跨模态特征融合，局部条件直接使用各自编码特征。
        self.context_proj = nn.Sequential(
            nn.Linear(self.cond_channels * 2 + 4, model_cfg.joint_channels),
            nn.GELU(),
            nn.Linear(model_cfg.joint_channels, model_cfg.joint_channels),
        )

        self.noise_predictor = UnifiedNoisePredictor1D(
            cond_channels=self.cond_channels,
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
    def _resolve_cond_channels(model_cfg: ModelConfig) -> int:
        ecg_channels = model_cfg.ecg_encoder.resolved_output_channels
        ppg_channels = model_cfg.ppg_encoder.resolved_output_channels
        if ecg_channels != ppg_channels:
            raise ValueError(
                f"ECG/PPG encoder 输出通道必须一致，实际为 {ecg_channels} 和 {ppg_channels}"
            )
        model_cfg.ecg_encoder.output_channels = ecg_channels
        model_cfg.ppg_encoder.output_channels = ppg_channels
        if model_cfg.cond_channels != ecg_channels:
            model_cfg.cond_channels = ecg_channels
        return ecg_channels

    @staticmethod
    def _ensure_3d(x: Tensor, name: str) -> Tensor:
        """将输入规范到 `[B,1,T]`。支持 `[B,T] -> [B,1,T]`。"""
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"{name} 期望 [B,1,T]，实际为 {tuple(x.shape)}")
        return x

    @staticmethod
    def build_modality_mask(has_ecg: bool, has_ppg: bool, batch_size: int, device: torch.device) -> Tensor:
        """构造模态掩码 `modality_mask:[B,2]`，列顺序为 `[ecg, ppg]`。"""
        mask = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)
        mask[:, 0] = 1.0 if has_ecg else 0.0
        mask[:, 1] = 1.0 if has_ppg else 0.0
        if not has_ecg and not has_ppg:
            raise ValueError("至少需要一个可用模态")
        return mask

    @staticmethod
    def _signal_mask(modality_mask: Tensor, index: int, ref: Tensor) -> Tensor:
        """将 `modality_mask[:,index]` 扩展成与信号广播兼容的 `[B,1,1]`。"""
        return modality_mask[:, index].to(dtype=ref.dtype, device=ref.device).view(-1, 1, 1)

    def _mask_signal_pair(self, ecg: Tensor, ppg: Tensor, modality_mask: Tensor) -> tuple[Tensor, Tensor]:
        """输入 `ecg/ppg:[B,1,T]`，输出按模态掩码置零后的同形状张量。"""
        ecg_mask = self._signal_mask(modality_mask, 0, ecg)
        ppg_mask = self._signal_mask(modality_mask, 1, ppg)
        return ecg * ecg_mask, ppg * ppg_mask

    def _encode_conditions(self, cond_ecg: Tensor, cond_ppg: Tensor, modality_mask: Tensor) -> Dict[str, Tensor]:
        """
        输入:
        - `cond_ecg/cond_ppg:[B,1,T]`
        - `modality_mask:[B,2]`
        输出:
        - `feat_ecg/feat_ppg:[B,cond_channels,T]`
        - `q_map_ecg/q_map_ppg:[B,1,T]`
        - `c_ecg/c_ppg:[B,cond_channels,T]`
        - `c_joint:[B,joint_channels]`
        """
        cond_ecg, cond_ppg = self._mask_signal_pair(cond_ecg, cond_ppg, modality_mask)
        feat_ecg = self.ecg_encoder(cond_ecg)
        feat_ppg = self.ppg_encoder(cond_ppg)
        q_map_ecg = self.ecg_quality(feat_ecg)
        q_map_ppg = self.ppg_quality(feat_ppg)
        q_map_ecg = q_map_ecg * self._signal_mask(modality_mask, 0, q_map_ecg)
        q_map_ppg = q_map_ppg * self._signal_mask(modality_mask, 1, q_map_ppg)
        c_ecg = feat_ecg
        c_ppg = feat_ppg
        pooled_ecg = c_ecg.mean(dim=-1)
        pooled_ppg = c_ppg.mean(dim=-1)
        quality_summary = torch.cat(
            [
                q_map_ecg.mean(dim=-1) * modality_mask[:, 0:1].to(dtype=q_map_ecg.dtype, device=q_map_ecg.device),
                q_map_ppg.mean(dim=-1) * modality_mask[:, 1:2].to(dtype=q_map_ppg.dtype, device=q_map_ppg.device),
            ],
            dim=1,
        )
        c_joint = self.context_proj(torch.cat([pooled_ecg, pooled_ppg, modality_mask.float(), quality_summary], dim=1))
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
        """
        输入:
        - `x_t_ecg/x_t_ppg:[B,1,T]`
        - `t:[B]`
        - `modality_mask:[B,2]`
        - `cond_ecg/cond_ppg:[B,1,T]`，为空时默认使用对应 `x_t`
        输出:
        - `pred_noise_ecg/pred_noise_ppg:[B,1,T]`
        - `x0_hat_ecg/x0_hat_ppg:[B,1,T]`
        - `q_map_ecg/q_map_ppg:[B,1,T]`
        - `c_ecg/c_ppg:[B,cond_channels,T]`
        - `c_joint:[B,joint_channels]`
        """
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
        """前向预测入口，输入 `noisy_ecg/noisy_ppg:[B,1,T]`，返回预测噪声及条件中间量。"""
        return self.predict_noise_from_xt(
            x_t_ecg=noisy_ecg,
            x_t_ppg=noisy_ppg,
            t=t,
            modality_mask=modality_mask,
            cond_ecg=noisy_ecg,
            cond_ppg=noisy_ppg,
        )

    @torch.no_grad()
    def denoise_signal(
        self,
        y_ecg: Optional[Tensor] = None,
        y_ppg: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        use_ddim: bool = False,
    ) -> Dict[str, Tensor]:
        """
        输入:
        - `y_ecg/y_ppg:[B,1,T]` 或 `[B,T]`
        输出:
        - `denoised_ecg/denoised_ppg:[B,1,T]`
        - `modality_mask:[B,2]`
        """
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
