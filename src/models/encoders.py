from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from models.blocks import ConvNormGELU


class SignalConditionEncoder1D(nn.Module):
    """仿 UniCardio singleEncoder 的多卷积核并行编码器。"""

    def __init__(
        self,
        in_channels: int = 1,
        branch_channels: int = 16,
        kernel_sizes: Sequence[int] = (1, 3, 5, 7, 9, 11),
        out_channels: int = 96,
        use_projection: bool = False,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes 不能为空")

        self.kernel_sizes = tuple(int(kernel) for kernel in kernel_sizes)
        self.branch_channels = int(branch_channels)
        self.concat_channels = self.branch_channels * len(self.kernel_sizes)
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(in_channels, self.branch_channels, kernel_size=kernel, padding=kernel // 2)
                for kernel in self.kernel_sizes
            ]
        )
        for layer in self.conv_layers:
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        if use_projection and out_channels != self.concat_channels:
            self.output_projection: nn.Module = ConvNormGELU(
                self.concat_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                gn_groups=gn_groups,
            )
            self.out_channels = out_channels
        else:
            self.output_projection = nn.Identity()
            self.out_channels = self.concat_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        输入 `x:[B,1,T]`。
        每个分支输出 `[B,branch_channels,T]`，拼接后为 `[B,concat_channels,T]`，
        若启用 projection，则最终输出 `[B,out_channels,T]`，否则输出 `[B,concat_channels,T]`。
        """
        branch_outputs = [conv(x) for conv in self.conv_layers]
        features = torch.cat(branch_outputs, dim=1)
        return self.output_projection(features)
