from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _make_group_norm(num_channels: int, max_groups: int) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


class ConvNormGELU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = _make_group_norm(out_channels, gn_groups)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """输入 `x:[B,C_in,T]`，输出 `[B,C_out,T]`。"""
        return self.act(self.norm(self.conv(x)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """输入时间步 `t:[B]`，输出时间嵌入 `[B,dim]`。"""
        half = self.dim // 2
        device = t.device
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-scale))
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class FiLMResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, gn_groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = _make_group_norm(out_channels, gn_groups)
        self.norm2 = _make_group_norm(out_channels, gn_groups)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.act = nn.GELU()
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """输入 `x:[B,C_in,T]`、`cond:[B,D]`，输出 `[B,C_out,T]`。"""
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
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """输入 `x:[B,C,T]`，输出 `[B,C,T/2]`。"""
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """输入 `x:[B,C_in,T]`，输出 `[B,C_out,2T]`。"""
        return self.conv(x)


class UnifiedNoisePredictor1D(nn.Module):
    """共享噪声预测主干，输入 [B,2,T]，输出 [B,2,T]。"""

    def __init__(
        self,
        cond_channels: int = 128,
        joint_channels: int = 256,
        base_channels: int = 64,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        cond_dim = base_channels * 4
        self.time_embed = SinusoidalTimeEmbedding(base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(joint_channels + 2, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.local_proj = nn.Conv1d(cond_channels * 2, base_channels, kernel_size=1)
        self.stem = nn.Conv1d(2, base_channels, kernel_size=3, padding=1)

        self.down1 = FiLMResBlock1D(base_channels, base_channels, cond_dim=cond_dim, gn_groups=gn_groups)
        self.downsample1 = Downsample1D(base_channels)
        self.down2 = FiLMResBlock1D(base_channels, base_channels * 2, cond_dim=cond_dim, gn_groups=gn_groups)
        self.downsample2 = Downsample1D(base_channels * 2)

        self.mid = FiLMResBlock1D(base_channels * 2, base_channels * 2, cond_dim=cond_dim, gn_groups=gn_groups)

        self.up2 = Upsample1D(base_channels * 2, base_channels * 2)
        self.up_block2 = FiLMResBlock1D(base_channels * 4, base_channels, cond_dim=cond_dim, gn_groups=gn_groups)
        self.up1 = Upsample1D(base_channels, base_channels)
        self.up_block1 = FiLMResBlock1D(base_channels * 2, base_channels, cond_dim=cond_dim, gn_groups=gn_groups)

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
        modality_mask: Tensor,
    ) -> Tensor:
        """
        输入:
        - `x_t_pair:[B,2,T]`
        - `t:[B]`
        - `c_joint:[B,joint_channels]`
        - `c_ecg/c_ppg:[B,cond_channels,T_cond]`
        - `modality_mask:[B,2]`
        输出:
        - 预测噪声 `pred_eps_pair:[B,2,T]`
        """
        batch_size, _, length = x_t_pair.shape
        c_ecg_up = F.interpolate(c_ecg, size=length, mode="linear", align_corners=False)
        c_ppg_up = F.interpolate(c_ppg, size=length, mode="linear", align_corners=False)

        ecg_mask = modality_mask[:, 0].view(batch_size, 1, 1)
        ppg_mask = modality_mask[:, 1].view(batch_size, 1, 1)
        local_cond = torch.cat([c_ecg_up * ecg_mask, c_ppg_up * ppg_mask], dim=1)

        cond_vec = self.time_mlp(self.time_embed(t)) + self.joint_mlp(torch.cat([c_joint, modality_mask.float()], dim=1))
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
