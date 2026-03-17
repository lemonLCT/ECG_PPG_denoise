from __future__ import annotations

from copy import deepcopy

import torch

from config import load_config
from models import DDPM


def _build_small_config() -> dict:
    cfg = deepcopy(load_config())
    cfg["data"]["window_length"] = 64
    cfg["model"]["main_model"]["signal_length"] = 64
    cfg["model"]["diffusion"]["num_steps"] = 8
    cfg["model"]["conditional_model"]["base_channels"] = 32
    cfg["model"]["conditional_model"]["joint_channels"] = 64
    cfg["train"]["batch_size"] = 2
    cfg["train"]["val_batch_size"] = 2
    return cfg


def test_predict_noise_outputs_expected_shapes() -> None:
    cfg = _build_small_config()
    model = DDPM(base_model=None, config=cfg, device="cpu")
    batch_size, length = 2, 64
    x_t_ecg = torch.randn(batch_size, 1, length)
    x_t_ppg = torch.randn(batch_size, 1, length)
    cond_ecg = torch.randn(batch_size, 1, length)
    cond_ppg = torch.randn(batch_size, 1, length)
    t = torch.randint(low=0, high=model.diffusion.num_steps, size=(batch_size,))
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    out = model.predict_noise_from_xt(
        x_t_ecg=x_t_ecg,
        x_t_ppg=x_t_ppg,
        cond_ecg=cond_ecg,
        cond_ppg=cond_ppg,
        t=t,
        modality_mask=mask,
    )

    assert out["pred_noise_pair"].shape == (batch_size, 2, length)
    assert out["pred_noise_ecg"].shape == (batch_size, 1, length)
    assert out["pred_noise_ppg"].shape == (batch_size, 1, length)
    assert out["feat_ecg"].shape[0] == batch_size
    assert out["feat_ecg"].shape[-1] == length
    assert out["feat_ppg"].shape[0] == batch_size
    assert out["feat_ppg"].shape[-1] == length
    assert out["c_ecg"].shape[0] == batch_size
    assert out["c_ecg"].shape[-1] == length
    assert out["c_ppg"].shape[0] == batch_size
    assert out["c_ppg"].shape[-1] == length
    assert out["c_joint"].shape[0] == batch_size


def test_one_training_step_runs() -> None:
    cfg = _build_small_config()
    model = DDPM(base_model=None, config=cfg, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size, length = 2, 64
    clean_ecg = torch.randn(batch_size, 1, length)
    clean_ppg = torch.randn(batch_size, 1, length)
    noisy_ecg = clean_ecg + 0.1 * torch.randn_like(clean_ecg)
    noisy_ppg = clean_ppg + 0.1 * torch.randn_like(clean_ppg)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    out = model(
        noisy_ecg=noisy_ecg,
        noisy_ppg=noisy_ppg,
        clean_ecg=clean_ecg,
        clean_ppg=clean_ppg,
        modality_mask=mask,
        train_gen_flag=0,
    )

    loss = out["total_loss"]
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def test_missing_modality_uses_gaussian_placeholder_for_noisy_path() -> None:
    cfg = _build_small_config()
    model = DDPM(base_model=None, config=cfg, device="cpu")
    batch_size, length = 2, 64
    clean_ecg = torch.randn(batch_size, 1, length)
    clean_ppg = torch.randn(batch_size, 1, length)
    noisy_ecg = clean_ecg + 0.1 * torch.randn_like(clean_ecg)
    noisy_ppg = clean_ppg + 0.1 * torch.randn_like(clean_ppg)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    out = model(
        noisy_ecg=noisy_ecg,
        noisy_ppg=noisy_ppg,
        clean_ecg=clean_ecg,
        clean_ppg=clean_ppg,
        modality_mask=mask,
        train_gen_flag=0,
    )

    missing_ppg_xt = out["x_t_ppg"][1]
    assert not torch.allclose(missing_ppg_xt, torch.zeros_like(missing_ppg_xt))


def test_generate_masks_missing_modality_output() -> None:
    cfg = _build_small_config()
    model = DDPM(base_model=None, config=cfg, device="cpu")
    batch_size, length = 2, 64
    noisy_ecg = torch.randn(batch_size, 1, length)

    out = model(
        noisy_ecg=noisy_ecg,
        noisy_ppg=None,
        modality_mask=torch.tensor([1.0, 0.0]),
        train_gen_flag=1,
        num_steps=2,
    )

    assert out["denoised_ecg"].shape == (batch_size, 1, length)
    assert out["denoised_ppg"].shape == (batch_size, 1, length)
    assert torch.allclose(out["denoised_ppg"], torch.zeros_like(out["denoised_ppg"]))
