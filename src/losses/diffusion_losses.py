from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


class DiffusionLoss(_Loss):
    """
    仅负责计算预测噪声和真实高斯噪声之间的损失。

    输入:
    - `pred_noise:[B,1,T]`
    - `target_noise:[B,1,T]`

    输出:
    - 标量损失 `loss:[]`
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        reduction = str(cfg.get("reduction", "mean"))
        super().__init__(reduction=reduction)

    def forward(self, pred_noise: Tensor, target_noise: Tensor) -> Tensor:
        if pred_noise.shape != target_noise.shape:
            raise ValueError(
                "pred_noise 和 target_noise 形状必须一致，"
                f"实际为 {tuple(pred_noise.shape)} 和 {tuple(target_noise.shape)}"
            )
        return F.mse_loss(pred_noise, target_noise, reduction=self.reduction)
