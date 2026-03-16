from __future__ import annotations

from torch import Tensor, nn

from models.blocks import ConvNormGELU


class QualityAssessor1D(nn.Module):
    """质量评估器，仅输出局部质量图 q_map。"""

    def __init__(self, channels: int = 128, hidden_channels: int = 64, gn_groups: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormGELU(channels, hidden_channels, kernel_size=3, stride=1, gn_groups=gn_groups),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """输入特征 `x:[B,C,T]`，输出局部质量图 `q_map:[B,1,T]`。"""
        return self.act(self.net(x))
