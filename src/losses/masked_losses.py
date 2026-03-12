"""支持 modality_mask 的损失函数。"""

from __future__ import annotations

from torch import Tensor
import torch


def _normalize_mask(mask: Tensor, ref: Tensor) -> Tensor:
    if mask.ndim == 1:
        mask = mask[:, None, None]
    elif mask.ndim == 2:
        mask = mask[:, :, None]
    while mask.ndim < ref.ndim:
        mask = mask.unsqueeze(-1)
    return mask.to(dtype=ref.dtype, device=ref.device)


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """对 mask=1 的位置计算 MSE。"""
    w = _normalize_mask(mask, pred)
    diff = (pred - target) ** 2
    full_w = w.expand_as(diff)
    num = (diff * full_w).sum()
    den = full_w.sum().clamp(min=1.0)
    return num / den


def masked_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """对 mask=1 的位置计算 L1。"""
    w = _normalize_mask(mask, pred)
    diff = torch.abs(pred - target)
    full_w = w.expand_as(diff)
    num = (diff * full_w).sum()
    den = full_w.sum().clamp(min=1.0)
    return num / den


def masked_derivative_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """一阶导数 L1。"""
    pred_d = pred[..., 1:] - pred[..., :-1]
    target_d = target[..., 1:] - target[..., :-1]
    return masked_l1(pred_d, target_d, mask)
