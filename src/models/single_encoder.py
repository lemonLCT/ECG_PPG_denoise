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

    def forward(self, signal: Tensor) -> Tensor:
        """
        输入:
        - `signal:[B,1,T]`

        输出:
        - `encoded:[B,C,T]`
        """
        if signal.ndim != 3:
            raise ValueError(f"signal 期望 [B,1,T]，实际为 {tuple(signal.shape)}")

        branch_outputs = [branch(signal) for branch in self.branches]

        features = torch.cat(branch_outputs, dim=1)
        # 维度投射
        features = self.projection(features)
        return features


class SignalEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=80, kernel_sizes=[1, 3, 7, 11]):
        super().__init__()
        # Store the number of output channels per convolution
        self.N = out_channels

        # Create multiple convolutional layers with different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, 20, kernel_size=ks, padding=ks // 2)
            for ks in kernel_sizes
        ])

        # Initialize weights using kaiming normal initialization
        for layer in self.conv_layers:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        # Apply all convolution layers and collect their outputs
        outputs = [conv(x) for conv in self.conv_layers]

        # Concatenate all outputs along the channel dimension
        concatenated_output = torch.cat(outputs, dim=1)  # Shape: (B, 6*N, L)

        # Apply the final 1x1 convolution
        # transformed_output = self.final_transform1(concatenated_output)  # Shape: (B, N, L)
        # transformed_output = F.silu(transformed_output)
        # transformed_output = self.final_transform2(transformed_output)  # Shape: (B, N, L)

        return concatenated_output