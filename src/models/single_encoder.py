from __future__ import annotations

import torch
from torch import Tensor, nn


def _make_group_norm(num_channels: int, max_groups: int) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvNormGELU(nn.Module):
    """卷积 + GroupNorm + GELU 组合块。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, gn_groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = _make_group_norm(out_channels, gn_groups)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """输入 `x:[B,C_in,T]`，输出 `y:[B,C_out,T]`。"""
        return self.act(self.norm(self.conv(x)))


class SingleEncoder1D(nn.Module):
    """
    ECG/PPG 单模态编码器。
    输入:
    - `signal:[B,1,T]`
    - `mask:[B]` 或标量，`0` 表示该模态缺失
    输出:
    - `encoded:[B,C,T]`
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.input_channels = int(config.get("input_channels", 1))
        self.branch_channels = int(config.get("branch_channels", 16))
        self.kernel_sizes = tuple(int(kernel) for kernel in config.get("kernel_sizes", [1, 3, 5, 7, 9, 11]))
        self.output_channels = int(config.get("output_channels", self.branch_channels * len(self.kernel_sizes)))
        self.use_projection = bool(config.get("use_projection", False))
        self.gn_groups = int(config.get("gn_groups", 8))
        self.concat_channels = self.branch_channels * len(self.kernel_sizes)

        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    self.input_channels,
                    self.branch_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        for layer in self.branches:
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        if self.use_projection and self.output_channels != self.concat_channels:
            self.projection: nn.Module = ConvNormGELU(
                in_channels=self.concat_channels,
                out_channels=self.output_channels,
                kernel_size=1,
                gn_groups=self.gn_groups,
            )
            self.out_channels = self.output_channels
        else:
            self.projection = nn.Identity()
            self.out_channels = self.concat_channels

    @staticmethod
    def _prepare_mask(mask: Tensor | float | int, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        mask_tensor = torch.as_tensor(mask, device=device, dtype=dtype)
        if mask_tensor.ndim == 0:
            mask_tensor = mask_tensor.repeat(batch_size)
        elif mask_tensor.ndim == 2 and mask_tensor.shape[-1] == 1:
            mask_tensor = mask_tensor.view(batch_size)
        elif mask_tensor.ndim != 1:
            raise ValueError(f"encoder mask 期望标量或 [B]，实际为 {tuple(mask_tensor.shape)}")
        if mask_tensor.shape[0] != batch_size:
            raise ValueError(f"encoder mask batch 大小不匹配，期望 {batch_size}，实际为 {mask_tensor.shape[0]}")
        return mask_tensor.view(batch_size, 1, 1)

    def forward(self, signal: Tensor, mask: Tensor | float | int) -> Tensor:
        """
        输入:
        - `signal:[B,1,T]`
        - `mask:[B]` 或标量
        输出:
        - `encoded:[B,C,T]`
        """
        if signal.ndim != 3:
            raise ValueError(f"signal 期望 [B,1,T]，实际为 {tuple(signal.shape)}")
        mask_tensor = self._prepare_mask(mask, batch_size=signal.shape[0], device=signal.device, dtype=signal.dtype)
        branch_outputs = [branch(signal) for branch in self.branches]
        features = torch.cat(branch_outputs, dim=1)
        encoded = self.projection(features)
        return encoded * mask_tensor
