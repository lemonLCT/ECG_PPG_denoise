from __future__ import annotations

import torch
from torch import Tensor, nn


class DiffusionSchedule1D(nn.Module):
    """标准 DDPM 1D 调度器。"""

    def __init__(self, num_steps: int = 50, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        if num_steps <= 1:
            raise ValueError("num_steps 必须 > 1")

        betas = torch.linspace(beta_start, beta_end, steps=num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.num_steps = int(num_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(low=0, high=self.num_steps, size=(batch_size,), device=device, dtype=torch.long)

    @staticmethod
    def _extract(buffer: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
        out = buffer.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def predict_x0_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar.clamp(min=1e-8)

    def p_sample(self, x_t: Tensor, t: Tensor, pred_eps: Tensor) -> Tensor:
        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

        model_mean = sqrt_recip_alpha_t * (x_t - beta_t * pred_eps / sqrt_one_minus_alpha_bar_t.clamp(min=1e-8))
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(t.shape[0], *([1] * (x_t.ndim - 1)))
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    def ddim_sample_step(self, x_t: Tensor, t: Tensor, pred_eps: Tensor) -> Tensor:
        return self.p_sample(x_t=x_t, t=t, pred_eps=pred_eps)
