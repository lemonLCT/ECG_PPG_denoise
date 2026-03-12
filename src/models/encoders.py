from __future__ import annotations

from torch import Tensor, nn

from models.blocks import ConvNormGELU


class SignalConditionEncoder1D(nn.Module):
    """单模态条件编码器，输入 [B,1,T]，输出 [B,C,T/8]。"""

    def __init__(self, out_channels: int = 128, gn_groups: int = 8) -> None:
        super().__init__()
        hidden = max(out_channels // 2, 32)
        self.net = nn.Sequential(
            ConvNormGELU(1, hidden, kernel_size=7, stride=2, gn_groups=gn_groups),
            ConvNormGELU(hidden, hidden, kernel_size=5, stride=2, gn_groups=gn_groups),
            ConvNormGELU(hidden, out_channels, kernel_size=5, stride=2, gn_groups=gn_groups),
            ConvNormGELU(out_channels, out_channels, kernel_size=3, stride=1, gn_groups=gn_groups),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class QualityAssessor1D(nn.Module):
    """质量评估器，仅输出局部质量图 q_map。"""

    def __init__(self, channels: int = 128, gn_groups: int = 8) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.net = nn.Sequential(
            ConvNormGELU(channels, hidden, kernel_size=3, stride=1, gn_groups=gn_groups),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.net(x))
