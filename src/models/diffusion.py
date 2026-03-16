from __future__ import annotations

import torch
from torch import Tensor, nn


class Diffusion1D(nn.Module):
    """
    1D DDPM 调度器。
    负责:
    - 采样时间步
    - 前向扩散 `q_sample`
    - 由噪声预测反推 `x0_hat`
    - 单步反向采样 `p_sample`
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_steps = int(config.get("num_steps", 200))
        beta_start = float(config.get("beta_start", 1e-4))
        beta_end = float(config.get("beta_end", 2e-2))
        if self.num_steps <= 1:
            raise ValueError("num_steps 必须大于 1")

        betas = torch.linspace(beta_start, beta_end, steps=self.num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

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
        """输出随机时间步 `t:[B]`。"""
        return torch.randint(0, self.num_steps, size=(batch_size,), device=device, dtype=torch.long)

    @staticmethod
    def _extract(buffer: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
        """从调度表中提取批量系数，输出形状 `[B,1,...,1]`。"""
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
