from __future__ import annotations

import pytest
import torch

from ecg_ppg_denoise.config import ExperimentConfig
from ecg_ppg_denoise.models import ModalityFlexibleConditionalDiffusion


def _build_small_components() -> ModalityFlexibleConditionalDiffusion:
    cfg = ExperimentConfig()
    cfg.model.signal_length = 128
    cfg.model.diffusion_steps = 20
    cfg.model.base_channels = 32
    cfg.model.cond_channels = 64
    cfg.model.joint_channels = 128
    cfg.data.window_length = 128
    cfg.validate()
    return ModalityFlexibleConditionalDiffusion(cfg.model, cfg.loss)


def test_forward_supports_three_modality_modes() -> None:
    model = _build_small_components()
    batch_size, length = 2, 128
    noisy_ecg = torch.randn(batch_size, 1, length)
    noisy_ppg = torch.randn(batch_size, 1, length)
    t = torch.randint(low=0, high=model.diffusion.num_steps, size=(batch_size,))

    for mask in (
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
    ):
        out = model(noisy_ecg=noisy_ecg, noisy_ppg=noisy_ppg, t=t, modality_mask=mask)
        assert out["pred_noise_ecg"].shape == (batch_size, 1, length)
        assert out["pred_noise_ppg"].shape == (batch_size, 1, length)
        assert out["x0_hat_ecg"].shape == (batch_size, 1, length)
        assert out["x0_hat_ppg"].shape == (batch_size, 1, length)
        assert out["q_map_ecg"].shape == (batch_size, 1, length)
        assert out["q_map_ppg"].shape == (batch_size, 1, length)
        assert "q_score_ecg" not in out
        assert "q_score_ppg" not in out


def test_one_training_step_runs() -> None:
    model = _build_small_components()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size, length = 2, 128
    clean_ecg = torch.randn(batch_size, 1, length)
    clean_ppg = torch.randn(batch_size, 1, length)
    noisy_ecg = clean_ecg + 0.1 * torch.randn_like(clean_ecg)
    noisy_ppg = clean_ppg + 0.1 * torch.randn_like(clean_ppg)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    out = model.compute_losses(
        clean_ecg=clean_ecg,
        clean_ppg=clean_ppg,
        noisy_ecg=noisy_ecg,
        noisy_ppg=noisy_ppg,
        modality_mask=mask,
    )
    loss = out["total_loss"]
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def test_missing_modality_is_masked_before_backbone() -> None:
    model = _build_small_components()
    captured: dict[str, torch.Tensor] = {}

    def _capture_stem_input(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        del module, output
        captured["x_t_pair"] = inputs[0].detach().clone()

    hook = model.noise_predictor.stem.register_forward_hook(_capture_stem_input)
    try:
        batch_size, length = 2, 128
        x_t_ecg = torch.randn(batch_size, 1, length)
        x_t_ppg = torch.randn(batch_size, 1, length)
        cond_ecg = torch.randn(batch_size, 1, length)
        cond_ppg = torch.randn(batch_size, 1, length)
        t = torch.randint(low=0, high=model.diffusion.num_steps, size=(batch_size,))
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        model.predict_noise_from_xt(
            x_t_ecg=x_t_ecg,
            x_t_ppg=x_t_ppg,
            t=t,
            modality_mask=mask,
            cond_ecg=cond_ecg,
            cond_ppg=cond_ppg,
        )
    finally:
        hook.remove()

    assert "x_t_pair" in captured
    assert torch.allclose(captured["x_t_pair"][:, 1:2, :], torch.zeros_like(captured["x_t_pair"][:, 1:2, :]))


def test_default_loss_reduces_to_diffusion_objective() -> None:
    model = _build_small_components()
    batch_size, length = 2, 128
    clean_ecg = torch.randn(batch_size, 1, length)
    clean_ppg = torch.randn(batch_size, 1, length)
    noisy_ecg = clean_ecg + 0.1 * torch.randn_like(clean_ecg)
    noisy_ppg = clean_ppg + 0.1 * torch.randn_like(clean_ppg)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    out = model.compute_losses(
        clean_ecg=clean_ecg,
        clean_ppg=clean_ppg,
        noisy_ecg=noisy_ecg,
        noisy_ppg=noisy_ppg,
        modality_mask=mask,
    )

    assert torch.allclose(out["total_loss"], out["diffusion_loss"])


def test_denoise_signal_accepts_batched_2d_input() -> None:
    model = _build_small_components()
    batch_size, length = 2, 128
    noisy_ecg = torch.randn(batch_size, length)

    out = model.denoise_signal(y_ecg=noisy_ecg, num_steps=2)

    assert out["denoised_ecg"].shape == (batch_size, 1, length)
    assert torch.allclose(out["denoised_ppg"], torch.zeros_like(out["denoised_ppg"]))


def test_denoise_signal_requires_matching_modal_shapes() -> None:
    model = _build_small_components()
    noisy_ecg = torch.randn(2, 1, 128)
    noisy_ppg = torch.randn(2, 1, 64)

    with pytest.raises(ValueError):
        model.denoise_signal(y_ecg=noisy_ecg, y_ppg=noisy_ppg, num_steps=2)
