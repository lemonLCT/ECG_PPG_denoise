from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from losses.diffusion_losses import DiffusionLoss
from models.conditional_model import ConditionalNoiseModel1D
from models.diffusion import Diffusion1D


class DDPM(nn.Module):
    """
    对外主模型接口。
    训练:
    - `forward(train_gen_flag=0)` 返回 loss 字典
    推理:
    - `forward(train_gen_flag=1)` 返回生成/去噪结果
    """

    def __init__(self, base_model: nn.Module | type[nn.Module] | None, config: dict, device: str | torch.device) -> None:
        super().__init__()
        self.config = config
        self.device_obj = torch.device(device)
        self.model_cfg = config["model"]

        if base_model is None:
            self.base_model = ConditionalNoiseModel1D(config)
        elif isinstance(base_model, nn.Module):
            self.base_model = base_model
        elif isinstance(base_model, type):
            self.base_model = base_model(config)
        else:
            raise TypeError("base_model 必须是 nn.Module、nn.Module 子类或 None")

        self.diffusion = Diffusion1D(self.model_cfg["diffusion"])
        self.loss_fn = DiffusionLoss(config.get("loss", {}))
        self.ecg_output_head = nn.Conv1d(1, 1, kernel_size=1)
        self.ppg_output_head = nn.Conv1d(1, 1, kernel_size=1)
        self.to(self.device_obj)

    @staticmethod
    def _ensure_3d(x: Tensor, name: str) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"{name} 期望 [B,1,T]，实际为 {tuple(x.shape)}")
        return x

    @staticmethod
    def normalize_modality_mask(mask: Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if mask is None:
            return torch.ones(batch_size, 2, device=device, dtype=dtype)
        mask_tensor = torch.as_tensor(mask, device=device, dtype=dtype)
        if mask_tensor.ndim == 1:
            if mask_tensor.shape[0] != 2:
                raise ValueError(f"mask 期望 [2] 或 [B,2]，实际为 {tuple(mask_tensor.shape)}")
            mask_tensor = mask_tensor.unsqueeze(0).repeat(batch_size, 1)
        if mask_tensor.ndim != 2 or mask_tensor.shape[1] != 2:
            raise ValueError(f"mask 期望 [B,2]，实际为 {tuple(mask_tensor.shape)}")
        if mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.repeat(batch_size, 1)
        if mask_tensor.shape[0] != batch_size:
            raise ValueError(f"mask batch 大小不匹配，期望 {batch_size}，实际为 {mask_tensor.shape[0]}")
        if torch.any(mask_tensor.sum(dim=1) == 0):
            raise ValueError("每个样本至少需要一个可用模态")
        return mask_tensor

    @staticmethod
    def _mask_signal_pair(ecg: Tensor, ppg: Tensor, modality_mask: Tensor) -> tuple[Tensor, Tensor]:
        ecg_mask = modality_mask[:, 0].view(-1, 1, 1)
        ppg_mask = modality_mask[:, 1].view(-1, 1, 1)
        return ecg * ecg_mask, ppg * ppg_mask

    @staticmethod
    def _fill_missing_noisy_pair(ecg: Tensor, ppg: Tensor, modality_mask: Tensor) -> tuple[Tensor, Tensor]:
        """将缺失模态的 noisy/noised 信号替换为同形状高斯噪声。"""
        ecg_mask = modality_mask[:, 0].view(-1, 1, 1)
        ppg_mask = modality_mask[:, 1].view(-1, 1, 1)
        noisy_ecg = ecg * ecg_mask + torch.randn_like(ecg) * (1.0 - ecg_mask)
        noisy_ppg = ppg * ppg_mask + torch.randn_like(ppg) * (1.0 - ppg_mask)
        return noisy_ecg, noisy_ppg

    def predict_noise_from_xt(
        self,
        x_t_ecg: Tensor,
        x_t_ppg: Tensor,
        cond_ecg: Tensor,
        cond_ppg: Tensor,
        t: Tensor,
        modality_mask: Tensor,
    ) -> Dict[str, Tensor]:
        x_t_ecg = self._ensure_3d(x_t_ecg, "x_t_ecg")
        x_t_ppg = self._ensure_3d(x_t_ppg, "x_t_ppg")
        cond_ecg = self._ensure_3d(cond_ecg, "cond_ecg")
        cond_ppg = self._ensure_3d(cond_ppg, "cond_ppg")
        if x_t_ecg.shape != x_t_ppg.shape or cond_ecg.shape != cond_ppg.shape:
            raise ValueError("ECG/PPG 输入形状必须一致")

        modality_mask = self.normalize_modality_mask(modality_mask, x_t_ecg.shape[0], x_t_ecg.device, x_t_ecg.dtype)
        return self.base_model(
            x_t_ecg=x_t_ecg,
            x_t_ppg=x_t_ppg,
            cond_ecg=cond_ecg,
            cond_ppg=cond_ppg,
            t=t,
            modality_mask=modality_mask,
        )

    def predict_x0_pair(
        self,
        x_t_ecg: Tensor,
        x_t_ppg: Tensor,
        t: Tensor,
        pred_noise_ecg: Tensor,
        pred_noise_ppg: Tensor,
    ) -> Dict[str, Tensor]:
        x_t_pair = torch.cat([x_t_ecg, x_t_ppg], dim=1)
        pred_eps_pair = torch.cat([pred_noise_ecg, pred_noise_ppg], dim=1)
        x0_hat_pair = self.diffusion.predict_x0_from_eps(x_t_pair, t, pred_eps_pair)
        return {
            "x0_hat_ecg": self.ecg_output_head(x0_hat_pair[:, 0:1, :]),
            "x0_hat_ppg": self.ppg_output_head(x0_hat_pair[:, 1:2, :]),
        }

    def one_modal_train(
        self,
        clean_ecg: Tensor,
        clean_ppg: Tensor,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        modality_mask: Tensor,
    ) -> Dict[str, Tensor]:
        return self.loss_fn(self, clean_ecg, clean_ppg, noisy_ecg, noisy_ppg, modality_mask)

    def two_modal_train(
        self,
        clean_ecg: Tensor,
        clean_ppg: Tensor,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        modality_mask: Tensor,
    ) -> Dict[str, Tensor]:
        return self.loss_fn(self, clean_ecg, clean_ppg, noisy_ecg, noisy_ppg, modality_mask)

    def compute_training_outputs(
        self,
        clean_ecg: Tensor,
        clean_ppg: Tensor,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        modality_mask: Tensor | None,
    ) -> Dict[str, Tensor]:
        clean_ecg = self._ensure_3d(clean_ecg, "clean_ecg")
        clean_ppg = self._ensure_3d(clean_ppg, "clean_ppg")
        noisy_ecg = self._ensure_3d(noisy_ecg, "noisy_ecg")
        noisy_ppg = self._ensure_3d(noisy_ppg, "noisy_ppg")
        if clean_ecg.shape != clean_ppg.shape or noisy_ecg.shape != noisy_ppg.shape:
            raise ValueError("训练时 ECG/PPG 形状必须一致")

        modality_mask = self.normalize_modality_mask(modality_mask, clean_ecg.shape[0], clean_ecg.device, clean_ecg.dtype)
        if torch.all(modality_mask.sum(dim=1) == 1):
            return self.one_modal_train(clean_ecg, clean_ppg, noisy_ecg, noisy_ppg, modality_mask)
        return self.two_modal_train(clean_ecg, clean_ppg, noisy_ecg, noisy_ppg, modality_mask)

    def _run_reverse_process(
        self,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        modality_mask: Tensor,
        num_steps: Optional[int],
        use_ddim: bool,
    ) -> Dict[str, Tensor]:
        noisy_ecg, noisy_ppg = self._fill_missing_noisy_pair(noisy_ecg, noisy_ppg, modality_mask)
        batch_size, _, length = noisy_ecg.shape
        device = noisy_ecg.device
        x_t_pair = torch.randn(batch_size, 2, length, device=device, dtype=noisy_ecg.dtype)
        total_steps = self.diffusion.num_steps if num_steps is None else int(num_steps)
        if total_steps <= 0:
            raise ValueError("num_steps 必须大于 0")

        time_grid = torch.linspace(
            self.diffusion.num_steps - 1,
            0,
            steps=total_steps,
            device=device,
            dtype=torch.float32,
        ).long()

        for scalar_t in time_grid:
            t = torch.full((batch_size,), int(scalar_t.item()), device=device, dtype=torch.long)
            out = self.predict_noise_from_xt(
                x_t_ecg=x_t_pair[:, 0:1, :],
                x_t_ppg=x_t_pair[:, 1:2, :],
                cond_ecg=noisy_ecg,
                cond_ppg=noisy_ppg,
                t=t,
                modality_mask=modality_mask,
            )
            pred_eps = torch.cat([out["pred_noise_ecg"], out["pred_noise_ppg"]], dim=1)
            if use_ddim:
                x_t_pair = self.diffusion.ddim_sample_step(x_t_pair, t, pred_eps)
            else:
                x_t_pair = self.diffusion.p_sample(x_t_pair, t, pred_eps)

        denoised_ecg = self.ecg_output_head(x_t_pair[:, 0:1, :])
        denoised_ppg = self.ppg_output_head(x_t_pair[:, 1:2, :])
        denoised_ecg, denoised_ppg = self._mask_signal_pair(denoised_ecg, denoised_ppg, modality_mask)
        return {
            "denoised_ecg": denoised_ecg,
            "denoised_ppg": denoised_ppg,
            "modality_mask": modality_mask,
        }

    def one_modal_generate(
        self,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        modality_mask: Tensor,
        num_steps: Optional[int],
        use_ddim: bool,
    ) -> Dict[str, Tensor]:
        return self._run_reverse_process(noisy_ecg, noisy_ppg, modality_mask, num_steps, use_ddim)

    def two_modal_generate(
        self,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        modality_mask: Tensor,
        num_steps: Optional[int],
        use_ddim: bool,
    ) -> Dict[str, Tensor]:
        return self._run_reverse_process(noisy_ecg, noisy_ppg, modality_mask, num_steps, use_ddim)

    def generate(
        self,
        noisy_ecg: Optional[Tensor] = None,
        noisy_ppg: Optional[Tensor] = None,
        modality_mask: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        use_ddim: bool = False,
    ) -> Dict[str, Tensor]:
        if noisy_ecg is None and noisy_ppg is None:
            raise ValueError("noisy_ecg 和 noisy_ppg 不能同时为空")

        has_ecg = noisy_ecg is not None
        has_ppg = noisy_ppg is not None
        ref = noisy_ecg if noisy_ecg is not None else noisy_ppg
        assert ref is not None
        ref = self._ensure_3d(ref, "reference_signal")
        batch_size, _, length = ref.shape
        device = ref.device

        if noisy_ecg is not None:
            noisy_ecg = self._ensure_3d(noisy_ecg, "noisy_ecg")
        else:
            noisy_ecg = torch.zeros(batch_size, 1, length, device=device, dtype=ref.dtype)
        if noisy_ppg is not None:
            noisy_ppg = self._ensure_3d(noisy_ppg, "noisy_ppg")
        else:
            noisy_ppg = torch.zeros(batch_size, 1, length, device=device, dtype=ref.dtype)
        if noisy_ecg.shape != noisy_ppg.shape:
            raise ValueError(f"noisy_ecg 和 noisy_ppg 形状必须一致，实际为 {tuple(noisy_ecg.shape)} 和 {tuple(noisy_ppg.shape)}")

        if modality_mask is None:
            modality_mask = torch.tensor([float(has_ecg), float(has_ppg)], device=device, dtype=ref.dtype)
        modality_mask = self.normalize_modality_mask(modality_mask, batch_size, device, ref.dtype)

        if torch.all(modality_mask.sum(dim=1) == 1):
            return self.one_modal_generate(noisy_ecg, noisy_ppg, modality_mask, num_steps, use_ddim)
        return self.two_modal_generate(noisy_ecg, noisy_ppg, modality_mask, num_steps, use_ddim)

    def forward(
        self,
        noisy_ecg: Optional[Tensor] = None,
        noisy_ppg: Optional[Tensor] = None,
        clean_ecg: Optional[Tensor] = None,
        clean_ppg: Optional[Tensor] = None,
        modality_mask: Optional[Tensor] = None,
        train_gen_flag: int = 0,
        num_steps: Optional[int] = None,
        use_ddim: bool = False,
    ) -> Dict[str, Tensor]:
        if train_gen_flag == 0:
            if clean_ecg is None or clean_ppg is None or noisy_ecg is None or noisy_ppg is None:
                raise ValueError("训练模式下必须提供 clean/noisy 的 ECG 和 PPG")
            return self.compute_training_outputs(
                clean_ecg=clean_ecg,
                clean_ppg=clean_ppg,
                noisy_ecg=noisy_ecg,
                noisy_ppg=noisy_ppg,
                modality_mask=modality_mask,
            )
        if train_gen_flag == 1:
            return self.generate(
                noisy_ecg=noisy_ecg,
                noisy_ppg=noisy_ppg,
                modality_mask=modality_mask,
                num_steps=num_steps,
                use_ddim=use_ddim,
            )
        raise ValueError(f"train_gen_flag 仅支持 0 或 1，实际为 {train_gen_flag}")
