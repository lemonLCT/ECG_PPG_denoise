from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class Diffusion1D(nn.Module):
    """
    1D 扩散调度器。

    负责：
    - 训练阶段的前向加噪 `q_sample`
    - 由预测噪声恢复 `x0_hat`
    - DDPM 单步逆扩散
    - DDIM 单步逆扩散
    - 完整逆扩散采样循环
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_steps = int(config.get("num_steps", 200))
        self.ddim_eta = float(config.get("ddim_eta", 0.0))
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
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer("posterior_log_variance_clipped", torch.log(self.posterior_variance))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

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

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, pred_eps: Tensor) -> Tensor:
        sqrt_recip_alpha_bar = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alpha_bar = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * pred_eps

    def predict_x0_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        return self.predict_start_from_noise(x_t=x_t, t=t, pred_eps=eps)

    def q_posterior(self, x_start: Tensor, x_t: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        posterior_mean = self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance

    def p_mean_variance(
        self,
        x_t: Tensor,
        t: Tensor,
        pred_eps: Tensor,
        clip_denoised: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        x0_hat = self.predict_start_from_noise(x_t=x_t, t=t, pred_eps=pred_eps)
        if clip_denoised:
            x0_hat = x0_hat.clamp(-1.0, 1.0)
        model_mean, posterior_log_variance = self.q_posterior(x_start=x0_hat, x_t=x_t, t=t)
        return model_mean, posterior_log_variance, x0_hat

    def p_sample(
        self,
        x_t: Tensor,
        t: Tensor,
        pred_eps: Tensor,
        clip_denoised: bool = False,
    ) -> Tensor:
        model_mean, posterior_log_variance, _ = self.p_mean_variance(
            x_t=x_t,
            t=t,
            pred_eps=pred_eps,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(t.shape[0], *([1] * (x_t.ndim - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

    def ddim_sample_step(
        self,
        x_t: Tensor,
        t: Tensor,
        next_t: Tensor | None,
        pred_eps: Tensor,
        eta: float = 0.0,
        clip_denoised: bool = False,
    ) -> Tensor:
        x0_hat = self.predict_start_from_noise(x_t=x_t, t=t, pred_eps=pred_eps)
        if clip_denoised:
            x0_hat = x0_hat.clamp(-1.0, 1.0)

        if next_t is None:
            return x0_hat

        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_next = self._extract(self.alphas_cumprod, next_t, x_t.shape)
        sigma = eta * torch.sqrt((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t)) * torch.sqrt(
            1.0 - alpha_bar_t / alpha_bar_next
        )
        noise = torch.randn_like(x_t) if eta > 0.0 else torch.zeros_like(x_t)
        direction = torch.sqrt((1.0 - alpha_bar_next - sigma.square()).clamp(min=0.0)) * pred_eps
        return torch.sqrt(alpha_bar_next) * x0_hat + direction + sigma * noise

    def build_sampling_timesteps(self, num_sampling_steps: int | None = None) -> Tensor:
        if num_sampling_steps is None or int(num_sampling_steps) >= self.num_steps:
            return torch.arange(self.num_steps - 1, -1, -1, dtype=torch.long)

        total = int(num_sampling_steps)
        if total <= 0:
            raise ValueError("num_sampling_steps 必须大于 0")
        if total == 1:
            return torch.tensor([self.num_steps - 1], dtype=torch.long)

        step_positions = torch.arange(total, dtype=torch.float64)
        scaled = torch.round(step_positions * (self.num_steps - 1) / (total - 1)).to(torch.long)
        scaled = torch.unique_consecutive(scaled)
        if scaled.shape[0] != total:
            raise ValueError("采样时间步构造失败，得到重复时间步，请检查 num_sampling_steps 设置")
        return torch.flip(scaled, dims=[0])

    def sample_loop(
        self,
        shape: tuple[int, ...] | torch.Size,
        pred_eps_fn: Callable[[Tensor, Tensor], Tensor],
        num_sampling_steps: int | None = None,
        use_ddim: bool = False,
        clip_denoised: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if len(shape) != 3:
            raise ValueError(f"shape 必须为 [B,C,T]，实际为 {tuple(shape)}")

        sample_device = self.betas.device if device is None else device
        sample_dtype = self.betas.dtype if dtype is None else dtype
        x_t = torch.randn(*shape, device=sample_device, dtype=sample_dtype)
        sampling_timesteps = self.build_sampling_timesteps(num_sampling_steps)

        for index, timestep in enumerate(sampling_timesteps):
            t = torch.full((shape[0],), int(timestep.item()), device=sample_device, dtype=torch.long)
            pred_eps = pred_eps_fn(x_t, t)
            if pred_eps.shape != x_t.shape:
                raise ValueError(
                    "pred_eps_fn 返回形状必须与 x_t 一致，"
                    f"实际为 {tuple(pred_eps.shape)} vs {tuple(x_t.shape)}"
                )

            if use_ddim:
                next_t = None
                if index + 1 < sampling_timesteps.shape[0]:
                    next_value = int(sampling_timesteps[index + 1].item())
                    next_t = torch.full((shape[0],), next_value, device=sample_device, dtype=torch.long)
                x_t = self.ddim_sample_step(
                    x_t=x_t,
                    t=t,
                    next_t=next_t,
                    pred_eps=pred_eps,
                    eta=self.ddim_eta,
                    clip_denoised=clip_denoised,
                )
            else:
                x_t = self.p_sample(
                    x_t=x_t,
                    t=t,
                    pred_eps=pred_eps,
                    clip_denoised=clip_denoised,
                )

        return x_t
