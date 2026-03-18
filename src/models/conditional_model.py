from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.single_encoder import _make_group_norm


class SinusoidalTimeEmbedding(nn.Module):
    """标准正弦时间步嵌入。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """输入 `t:[B]`，输出 `time_emb:[B,dim]`。"""
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-scale))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class FiLMResBlock1D(nn.Module):
    """
    带 FiLM 条件注入的 1D 残差块。

    输入:
    - `x:[B,C_in,T]`
    - `cond:[B,D]`

    输出:
    - `y:[B,C_out,T]`
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, gn_groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = _make_group_norm(out_channels, gn_groups)
        self.norm2 = _make_group_norm(out_channels, gn_groups)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.act = nn.GELU()
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1.0 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(self.norm2(h))
        return h + self.skip(x)


class Downsample1D(nn.Module):
    """1D 下采样块。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """输入 `x:[B,C,T]`，输出 `y:[B,C,T/2]`。"""
        return self.conv(x)


class Upsample1D(nn.Module):
    """1D 上采样块。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """输入 `x:[B,C_in,T]`，输出 `y:[B,C_out,2T]`。"""
        return self.conv(x)


class SharedConditionalUNet1D(nn.Module):
    """
    共享噪声预测 U-Net 主干。

    输入:
    - `x_t_pair:[B,2,T]`
    - `t:[B]`
    - `c_joint:[B,joint_channels]`
    - `c_ecg/c_ppg:[B,modality_channels,T_cond]`

    输出:
    - `pred_eps_pair:[B,2,T]`
    """

    def __init__(self, modality_channels: int, joint_channels: int, base_channels: int, gn_groups: int) -> None:
        super().__init__()
        cond_dim = base_channels * 4
        self.time_embed = SinusoidalTimeEmbedding(base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(joint_channels, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.local_proj = nn.Conv1d(modality_channels * 2, base_channels, kernel_size=1)
        self.stem = nn.Conv1d(2, base_channels, kernel_size=3, padding=1)
        self.down1 = FiLMResBlock1D(base_channels, base_channels, cond_dim, gn_groups)
        self.downsample1 = Downsample1D(base_channels)
        self.down2 = FiLMResBlock1D(base_channels, base_channels * 2, cond_dim, gn_groups)
        self.downsample2 = Downsample1D(base_channels * 2)
        self.mid = FiLMResBlock1D(base_channels * 2, base_channels * 2, cond_dim, gn_groups)
        self.up2 = Upsample1D(base_channels * 2, base_channels * 2)
        self.up_block2 = FiLMResBlock1D(base_channels * 4, base_channels, cond_dim, gn_groups)
        self.up1 = Upsample1D(base_channels, base_channels)
        self.up_block1 = FiLMResBlock1D(base_channels * 2, base_channels, cond_dim, gn_groups)
        self.out = nn.Sequential(
            _make_group_norm(base_channels, gn_groups),
            nn.GELU(),
            nn.Conv1d(base_channels, 2, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t_pair: Tensor,
        t: Tensor,
        c_joint: Tensor,
        c_ecg: Tensor,
        c_ppg: Tensor,
    ) -> Tensor:
        _, _, length = x_t_pair.shape
        c_ecg_up = F.interpolate(c_ecg, size=length, mode="linear", align_corners=False)
        c_ppg_up = F.interpolate(c_ppg, size=length, mode="linear", align_corners=False)
        local_cond = torch.cat([c_ecg_up, c_ppg_up], dim=1)

        cond_vec = self.time_mlp(self.time_embed(t)) + self.joint_mlp(c_joint)
        h0 = self.stem(x_t_pair) + self.local_proj(local_cond)
        h1 = self.down1(h0, cond_vec)
        d1 = self.downsample1(h1)
        h2 = self.down2(d1, cond_vec)
        d2 = self.downsample2(h2)
        mid = self.mid(d2, cond_vec)

        u2 = self.up2(mid)
        if u2.shape[-1] != h2.shape[-1]:
            u2 = F.interpolate(u2, size=h2.shape[-1], mode="linear", align_corners=False)
        u2 = self.up_block2(torch.cat([u2, h2], dim=1), cond_vec)

        u1 = self.up1(u2)
        if u1.shape[-1] != h1.shape[-1]:
            u1 = F.interpolate(u1, size=h1.shape[-1], mode="linear", align_corners=False)
        u1 = self.up_block1(torch.cat([u1, h1], dim=1), cond_vec)
        return self.out(u1)


class ConditionalNoiseModel1D(nn.Module):
    """
    只负责噪声预测的条件网络。

    输入:
    - `x_t_ecg/x_t_ppg:[B,1,T]`
    - `noisy_feat_ecg/noisy_feat_ppg:[B,C,T]`
    - `t:[B]`

    输出:
    - `pred_noise_pair:[B,2,T]`
    - `pred_noise_ecg/pred_noise_ppg:[B,1,T]`
    - `c_ecg/c_ppg:[B,C,T]`
    - `c_joint:[B,joint_channels]`
    """

    def __init__(self, config: dict, feature_channels: int) -> None:
        super().__init__()
        model_cfg = config["model"]
        conditional_cfg = model_cfg["conditional_model"]

        configured_channels = int(conditional_cfg.get("cond_channels", feature_channels))
        if configured_channels != feature_channels:
            raise ValueError(
                "conditional_model.cond_channels 必须与 single_encoder 输出通道一致，"
                f"实际为 {configured_channels} 和 {feature_channels}"
            )

        self.single_feature_channels = feature_channels
        self.modality_channels = feature_channels
        self.joint_channels = int(conditional_cfg["joint_channels"])
        self.base_channels = int(conditional_cfg["base_channels"])
        self.gn_groups = int(conditional_cfg["gn_groups"])

        self.context_proj = nn.Sequential(
            nn.Linear(self.modality_channels * 2, self.joint_channels),
            nn.GELU(),
            nn.Linear(self.joint_channels, self.joint_channels),
        )
        self.noise_backbone = SharedConditionalUNet1D(
            modality_channels=self.modality_channels,
            joint_channels=self.joint_channels,
            base_channels=self.base_channels,
            gn_groups=self.gn_groups,
        )

    @staticmethod
    def _ensure_3d(x: Tensor, name: str) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"{name} 期望 [B,C,T]，实际为 {tuple(x.shape)}")
        return x

    def _build_conditions(
        self,
        noisy_feat_ecg: Tensor,
        noisy_feat_ppg: Tensor,
    ) -> Dict[str, Tensor]:
        """
        输入:
        - `noisy_feat_ecg/noisy_feat_ppg:[B,C,T]`

        输出:
        - `c_ecg/c_ppg:[B,C,T]`
        - `c_joint:[B,joint_channels]`
        """
        c_ecg = noisy_feat_ecg
        c_ppg = noisy_feat_ppg
        pooled_ecg = c_ecg.mean(dim=-1)
        pooled_ppg = c_ppg.mean(dim=-1)
        joint_input = torch.cat([pooled_ecg, pooled_ppg], dim=1)
        c_joint = self.context_proj(joint_input)
        return {
            "c_ecg": c_ecg,
            "c_ppg": c_ppg,
            "c_joint": c_joint,
        }

    def forward(
        self,
        x_t_ecg: Tensor,
        x_t_ppg: Tensor,
        noisy_feat_ecg: Tensor,
        noisy_feat_ppg: Tensor,
        t: Tensor,
    ) -> Dict[str, Tensor]:
        """
        输入:
        - `x_t_ecg/x_t_ppg:[B,1,T]`
        - `noisy_feat_ecg/noisy_feat_ppg:[B,C,T]`
        - `t:[B]`

        输出:
        - `pred_noise_pair:[B,2,T]`
        - `pred_noise_ecg/pred_noise_ppg:[B,1,T]`
        - `c_ecg/c_ppg:[B,C,T]`
        - `c_joint:[B,joint_channels]`
        """
        x_t_ecg = self._ensure_3d(x_t_ecg, "x_t_ecg")
        x_t_ppg = self._ensure_3d(x_t_ppg, "x_t_ppg")
        noisy_feat_ecg = self._ensure_3d(noisy_feat_ecg, "noisy_feat_ecg")
        noisy_feat_ppg = self._ensure_3d(noisy_feat_ppg, "noisy_feat_ppg")
        if t.ndim != 1 or t.shape[0] != x_t_ecg.shape[0]:
            raise ValueError(f"t 期望 [B]，实际为 {tuple(t.shape)}")

        cond = self._build_conditions(noisy_feat_ecg, noisy_feat_ppg)
        x_t_pair = torch.cat([x_t_ecg, x_t_ppg], dim=1)
        pred_noise_pair = self.noise_backbone(
            x_t_pair=x_t_pair,
            t=t,
            c_joint=cond["c_joint"],
            c_ecg=cond["c_ecg"],
            c_ppg=cond["c_ppg"],
        )
        return {
            "pred_noise_pair": pred_noise_pair,
            "pred_noise_ecg": pred_noise_pair[:, 0:1, :],
            "pred_noise_ppg": pred_noise_pair[:, 1:2, :],
            "noisy_feat_ecg": noisy_feat_ecg,
            "noisy_feat_ppg": noisy_feat_ppg,
            **cond,
        }
