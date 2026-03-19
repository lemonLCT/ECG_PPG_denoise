from __future__ import annotations

import torch

from models.HNF import ConditionalModel, HNFBlock


def test_hnf_block_supports_channel_projection() -> None:
    block = HNFBlock(input_size=8, hidden_size=16, dilation=2)
    x = torch.randn(2, 8, 64)

    y = block(x)

    assert y.shape == (2, 16, 64)
    assert torch.isfinite(y).all()


def test_hnf_conditional_model_preserves_signal_shape() -> None:
    model = ConditionalModel(feats=32)
    x = torch.randn(2, 1, 128)
    cond = torch.randn(2, 1, 128)
    noise_scale = torch.rand(2, 1)

    y = model(x, cond, noise_scale)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
