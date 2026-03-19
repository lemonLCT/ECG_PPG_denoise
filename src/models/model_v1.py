from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from losses.diffusion_losses import DiffusionLoss
from models.HNF import multiConditionModel
from models.diffusion import Diffusion1D


class DDPMv1(nn.Module):
    """双模态 HNF 条件扩散模型。"""

    def __init__(self, base_model: nn.Module | type[nn.Module] | None, config: dict, device: str | torch.device) -> None:
        super().__init__()
        self.config = config
        self.device_obj = torch.device(device)
        self.model_cfg = config["model"]
        self.hnf_cfg = self.model_cfg.get("hnf", {})

        hnf_feats = int(self.hnf_cfg.get("feats", 80))
        if base_model is None:
            self.base_model = multiConditionModel(feats=hnf_feats)
        elif isinstance(base_model, nn.Module):
            self.base_model = base_model
        elif isinstance(base_model, type):
            self.base_model = base_model(feats=hnf_feats)
        else:
            raise TypeError("base_model 必须是 nn.Module、nn.Module 子类或 None")

        self.diffusion = Diffusion1D(self.model_cfg["diffusion"])
        self.loss_fn = DiffusionLoss(config.get("loss", {}))
        self.diffusion_weight = float(config.get("loss", {}).get("diffusion_weight", 1.0))
        self.to(self.device_obj)

    @staticmethod
    def _ensure_3d(x: Tensor, name: str) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"{name} 期望 [B,1,T]，实际为 {tuple(x.shape)}")
        return x

    def _build_diffusion_targets(self, clean_ecg: Tensor, clean_ppg: Tensor) -> Dict[str, Tensor]:
        batch_size = clean_ecg.shape[0]
        device = clean_ecg.device
        t = self.diffusion.sample_timesteps(batch_size=batch_size, device=device)
        noise_ecg = torch.randn_like(clean_ecg)
        noise_ppg = torch.randn_like(clean_ppg)
        x_t_ecg = self.diffusion.q_sample(clean_ecg, t, noise_ecg)
        x_t_ppg = self.diffusion.q_sample(clean_ppg, t, noise_ppg)
        noise_scale = self.diffusion.sqrt_alphas_cumprod.gather(0, t).view(batch_size, 1)
        return {
            "t": t,
            "noise_scale": noise_scale,
            "noise_ecg": noise_ecg,
            "noise_ppg": noise_ppg,
            "x_t_ecg": x_t_ecg,
            "x_t_ppg": x_t_ppg,
        }

    def predict_noise_from_xt(
        self,
        x_t_ecg: Tensor,
        x_t_ppg: Tensor,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        noise_scale: Tensor,
    ) -> Dict[str, Tensor]:
        x_t_ecg = self._ensure_3d(x_t_ecg, "x_t_ecg")
        x_t_ppg = self._ensure_3d(x_t_ppg, "x_t_ppg")
        noisy_ecg = self._ensure_3d(noisy_ecg, "noisy_ecg")
        noisy_ppg = self._ensure_3d(noisy_ppg, "noisy_ppg")

        if x_t_ecg.shape != x_t_ppg.shape:
            raise ValueError(f"x_t_ecg 与 x_t_ppg 形状必须一致，实际为 {tuple(x_t_ecg.shape)} 与 {tuple(x_t_ppg.shape)}")
        if noisy_ecg.shape != noisy_ppg.shape:
            raise ValueError(f"noisy_ecg 与 noisy_ppg 形状必须一致，实际为 {tuple(noisy_ecg.shape)} 与 {tuple(noisy_ppg.shape)}")
        if x_t_ecg.shape != noisy_ecg.shape:
            raise ValueError(
                f"扩散输入与参考输入形状必须一致，实际为 {tuple(x_t_ecg.shape)} 与 {tuple(noisy_ecg.shape)}"
            )

        pred_noise_ecg, pred_noise_ppg = self.base_model(
            x_t_ecg=x_t_ecg,
            cond_ecg=noisy_ecg,
            x_t_ppg=x_t_ppg,
            cond_ppg=noisy_ppg,
            noise_scale=noise_scale,
        )
        return {
            "pred_noise_ecg": pred_noise_ecg,
            "pred_noise_ppg": pred_noise_ppg,
        }

    def forward(
        self,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        clean_ecg: Tensor,
        clean_ppg: Tensor,
    ) -> Dict[str, Tensor]:
        clean_ecg = self._ensure_3d(clean_ecg, "clean_ecg")
        clean_ppg = self._ensure_3d(clean_ppg, "clean_ppg")
        noisy_ecg = self._ensure_3d(noisy_ecg, "noisy_ecg")
        noisy_ppg = self._ensure_3d(noisy_ppg, "noisy_ppg")

        if clean_ecg.shape != clean_ppg.shape:
            raise ValueError(f"clean_ecg 与 clean_ppg 形状必须一致，实际为 {tuple(clean_ecg.shape)} 与 {tuple(clean_ppg.shape)}")
        if noisy_ecg.shape != noisy_ppg.shape:
            raise ValueError(f"noisy_ecg 与 noisy_ppg 形状必须一致，实际为 {tuple(noisy_ecg.shape)} 与 {tuple(noisy_ppg.shape)}")
        if clean_ecg.shape != noisy_ecg.shape:
            raise ValueError(
                f"干净信号与参考含噪信号形状必须一致，实际为 {tuple(clean_ecg.shape)} 与 {tuple(noisy_ecg.shape)}"
            )

        diffusion_targets = self._build_diffusion_targets(clean_ecg, clean_ppg)
        outputs = self.predict_noise_from_xt(
            x_t_ecg=diffusion_targets["x_t_ecg"],
            x_t_ppg=diffusion_targets["x_t_ppg"],
            noisy_ecg=noisy_ecg,
            noisy_ppg=noisy_ppg,
            noise_scale=diffusion_targets["noise_scale"],
        )

        ecg_loss = self.loss_fn(outputs["pred_noise_ecg"], diffusion_targets["noise_ecg"])
        ppg_loss = self.loss_fn(outputs["pred_noise_ppg"], diffusion_targets["noise_ppg"])
        diffusion_loss = ecg_loss + ppg_loss
        total_loss = self.diffusion_weight * diffusion_loss

        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "ecg_diffusion_loss": ecg_loss,
            "ppg_diffusion_loss": ppg_loss,
            **diffusion_targets,
            **outputs,
        }




def train(
    model: DDPMv1,
    optimizer: torch.optim.Optimizer,
    noisy_ecg: Tensor,
    noisy_ppg: Tensor,
    clean_ecg: Tensor,
    clean_ppg: Tensor,
) -> Dict[str, Tensor]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(
        noisy_ecg=noisy_ecg,
        noisy_ppg=noisy_ppg,
        clean_ecg=clean_ecg,
        clean_ppg=clean_ppg,
    )
    outputs["total_loss"].backward()
    optimizer.step()
    return outputs


def evaluate(
    model: DDPMv1,
    noisy_ecg: Tensor,
    noisy_ppg: Tensor,
    clean_ecg: Tensor,
    clean_ppg: Tensor,
) -> Dict[str, Tensor]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(
            noisy_ecg=noisy_ecg,
            noisy_ppg=noisy_ppg,
            clean_ecg=clean_ecg,
            clean_ppg=clean_ppg,
        )
    if was_training:
        model.train()
    return outputs


__all__ = ["DDPMv1", "train", "evaluate"]
