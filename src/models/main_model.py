from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from losses.diffusion_losses import DiffusionLoss
from models.conditional_model import ConditionalNoiseModel1D
from models.diffusion import Diffusion1D
from models.single_encoder import SingleEncoder1D


class DDPM(nn.Module):
    """
    对外主模型接口。

    训练:
    - `forward(...)` 根据模态组合分发到单模态或双模态训练路径

    推理:
    - `generate(...)` 返回生成/去噪结果
    """

    def __init__(self, base_model: nn.Module | type[nn.Module] | None, config: dict, device: str | torch.device) -> None:
        super().__init__()
        self.config = config
        self.device_obj = torch.device(device)
        self.model_cfg = config["model"]

        self.clean_ecg_encoder = SingleEncoder1D(self.model_cfg["ecg_encoder"])
        self.clean_ppg_encoder = SingleEncoder1D(self.model_cfg["ppg_encoder"])
        self.noisy_ecg_encoder = SingleEncoder1D(self.model_cfg["ecg_encoder"])
        self.noisy_ppg_encoder = SingleEncoder1D(self.model_cfg["ppg_encoder"])

        self.ecg_feature_channels = self.clean_ecg_encoder.out_channels
        self.ppg_feature_channels = self.clean_ppg_encoder.out_channels
        if self.ecg_feature_channels != self.ppg_feature_channels:
            raise ValueError(
                "ECG 和 PPG encoder 的输出通道必须一致，"
                f"实际为 {self.ecg_feature_channels} 和 {self.ppg_feature_channels}"
            )

        if base_model is None:
            self.base_model = ConditionalNoiseModel1D(config, feature_channels=self.ecg_feature_channels)
        elif isinstance(base_model, nn.Module):
            self.base_model = base_model
        elif isinstance(base_model, type):
            self.base_model = base_model(config, feature_channels=self.ecg_feature_channels)
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

    @staticmethod
    def _ensure_feature_3d(x: Tensor, name: str) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"{name} 期望 [B,C,T]，实际为 {tuple(x.shape)}")
        return x

    @staticmethod
    def _validate_modality_mask(mask: Tensor | None, device: torch.device, dtype: torch.dtype) -> Tensor:
        if mask is None:
            return torch.tensor([1.0, 1.0], device=device, dtype=dtype)
        mask_tensor = torch.as_tensor(mask, device=device, dtype=dtype)
        if mask_tensor.ndim != 1 or mask_tensor.shape[0] != 2:
            raise ValueError(f"modality_mask 期望 [2]，实际为 {tuple(mask_tensor.shape)}")
        if not torch.all((mask_tensor == 0) | (mask_tensor == 1)):
            raise ValueError(f"modality_mask 只允许由 0/1 组成，实际为 {mask_tensor.tolist()}")
        if int(mask_tensor.sum().item()) not in (1, 2):
            raise ValueError(f"modality_mask 只支持 [1,0]、[0,1]、[1,1]，实际为 {mask_tensor.tolist()}")
        return mask_tensor

    @staticmethod
    def _mask_signal_pair(ecg: Tensor, ppg: Tensor, modality_mask: Tensor) -> tuple[Tensor, Tensor]:
        ecg_mask = modality_mask[0].view(1, 1, 1)
        ppg_mask = modality_mask[1].view(1, 1, 1)
        return ecg * ecg_mask, ppg * ppg_mask

    @staticmethod
    def _fill_missing_noisy_pair(ecg: Tensor, ppg: Tensor, modality_mask: Tensor) -> tuple[Tensor, Tensor]:
        """将缺失模态的 noisy/noised 信号替换为同形状高斯噪声。"""
        ecg_mask = modality_mask[0].view(1, 1, 1)
        ppg_mask = modality_mask[1].view(1, 1, 1)
        noisy_ecg = ecg * ecg_mask + torch.randn_like(ecg) * (1.0 - ecg_mask)
        noisy_ppg = ppg * ppg_mask + torch.randn_like(ppg) * (1.0 - ppg_mask)
        return noisy_ecg, noisy_ppg

    def _encode_training_features(
        self,
        clean_ecg: Tensor,
        clean_ppg: Tensor,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        modality_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """
        输入:
        - `clean_ecg/clean_ppg/noisy_ecg/noisy_ppg:[B,1,T]`
        - `modality_mask:[2]`

        输出:
        - `clean_feat_ecg/clean_feat_ppg/noisy_feat_ecg/noisy_feat_ppg:[B,C,T]`
        """
        return {
            "clean_feat_ecg": self.clean_ecg_encoder(clean_ecg, modality_mask[0]),
            "clean_feat_ppg": self.clean_ppg_encoder(clean_ppg, modality_mask[1]),
            "noisy_feat_ecg": self.noisy_ecg_encoder(noisy_ecg, modality_mask[0]),
            "noisy_feat_ppg": self.noisy_ppg_encoder(noisy_ppg, modality_mask[1]),
        }

    def _encode_generation_features(self, noisy_ecg: Tensor, noisy_ppg: Tensor, modality_mask: Tensor) -> Dict[str, Tensor]:
        """
        输入:
        - `noisy_ecg/noisy_ppg:[B,1,T]`
        - `modality_mask:[2]`

        输出:
        - `clean_feat_*:[B,C,T]`，推理阶段置零
        - `noisy_feat_*:[B,C,T]`
        """
        noisy_feat_ecg = self.noisy_ecg_encoder(noisy_ecg, modality_mask[0])
        noisy_feat_ppg = self.noisy_ppg_encoder(noisy_ppg, modality_mask[1])
        return {
            "clean_feat_ecg": torch.zeros_like(noisy_feat_ecg),
            "clean_feat_ppg": torch.zeros_like(noisy_feat_ppg),
            "noisy_feat_ecg": noisy_feat_ecg,
            "noisy_feat_ppg": noisy_feat_ppg,
        }

    def _build_diffusion_targets(self, clean_ecg: Tensor, clean_ppg: Tensor) -> Dict[str, Tensor]:
        """
        输入:
        - `clean_ecg/clean_ppg:[B,1,T]`

        输出:
        - `t:[B]`
        - `noise_ecg/noise_ppg:[B,1,T]`
        - `x_t_ecg/x_t_ppg:[B,1,T]`
        """
        batch_size = clean_ecg.shape[0]
        device = clean_ecg.device
        t = self.diffusion.sample_timesteps(batch_size=batch_size, device=device)
        noise_ecg = torch.randn_like(clean_ecg)
        noise_ppg = torch.randn_like(clean_ppg)
        x_t_ecg = self.diffusion.q_sample(clean_ecg, t, noise_ecg)
        x_t_ppg = self.diffusion.q_sample(clean_ppg, t, noise_ppg)
        return {
            "t": t,
            "noise_ecg": noise_ecg,
            "noise_ppg": noise_ppg,
            "x_t_ecg": x_t_ecg,
            "x_t_ppg": x_t_ppg,
        }

    def predict_noise_from_xt(
        self,
        x_t_ecg: Tensor,
        x_t_ppg: Tensor,
        clean_feat_ecg: Tensor,
        clean_feat_ppg: Tensor,
        noisy_feat_ecg: Tensor,
        noisy_feat_ppg: Tensor,
        t: Tensor,
        modality_mask: Tensor,
    ) -> Dict[str, Tensor]:
        x_t_ecg = self._ensure_3d(x_t_ecg, "x_t_ecg")
        x_t_ppg = self._ensure_3d(x_t_ppg, "x_t_ppg")
        clean_feat_ecg = self._ensure_feature_3d(clean_feat_ecg, "clean_feat_ecg")
        clean_feat_ppg = self._ensure_feature_3d(clean_feat_ppg, "clean_feat_ppg")
        noisy_feat_ecg = self._ensure_feature_3d(noisy_feat_ecg, "noisy_feat_ecg")
        noisy_feat_ppg = self._ensure_feature_3d(noisy_feat_ppg, "noisy_feat_ppg")
        if x_t_ecg.shape != x_t_ppg.shape:
            raise ValueError("x_t_ecg 和 x_t_ppg 形状必须一致")
        modality_mask = self._validate_modality_mask(modality_mask, x_t_ecg.device, x_t_ecg.dtype)
        return self.base_model(
            x_t_ecg=x_t_ecg,
            x_t_ppg=x_t_ppg,
            clean_feat_ecg=clean_feat_ecg,
            clean_feat_ppg=clean_feat_ppg,
            noisy_feat_ecg=noisy_feat_ecg,
            noisy_feat_ppg=noisy_feat_ppg,
            t=t,
            modality_mask=modality_mask,
        )

    def one_modal_train(
        self,
        clean_ecg: Tensor,
        clean_ppg: Tensor,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        modality_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """
        单模态训练路径。

        输入:
        - `clean_ecg/clean_ppg/noisy_ecg/noisy_ppg:[B,1,T]`
        - `modality_mask:[2]`，只能是 `[1,0]` 或 `[0,1]`

        输出:
        - `total_loss:[]`
        - `diffusion_loss:[]`
        - 以及训练中间量
        """
        modality_mask = self._validate_modality_mask(modality_mask, clean_ecg.device, clean_ecg.dtype)
        if int(modality_mask.sum().item()) != 1:
            raise ValueError(f"one_modal_train 只接受单模态 mask，实际为 {modality_mask.tolist()}")

        clean_ecg, clean_ppg = self._mask_signal_pair(clean_ecg, clean_ppg, modality_mask)
        noisy_ecg, noisy_ppg = self._fill_missing_noisy_pair(noisy_ecg, noisy_ppg, modality_mask)
        diffusion_targets = self._build_diffusion_targets(clean_ecg, clean_ppg)
        encoded = self._encode_training_features(clean_ecg, clean_ppg, noisy_ecg, noisy_ppg, modality_mask)
        outputs = self.predict_noise_from_xt(
            x_t_ecg=diffusion_targets["x_t_ecg"],
            x_t_ppg=diffusion_targets["x_t_ppg"],
            clean_feat_ecg=encoded["clean_feat_ecg"],
            clean_feat_ppg=encoded["clean_feat_ppg"],
            noisy_feat_ecg=encoded["noisy_feat_ecg"],
            noisy_feat_ppg=encoded["noisy_feat_ppg"],
            t=diffusion_targets["t"],
            modality_mask=modality_mask,
        )

        ecg_loss = torch.zeros((), device=clean_ecg.device, dtype=clean_ecg.dtype)
        ppg_loss = torch.zeros((), device=clean_ecg.device, dtype=clean_ecg.dtype)
        if bool(modality_mask[0].item()):
            ecg_loss = self.loss_fn(outputs["pred_noise_ecg"], diffusion_targets["noise_ecg"])
            diffusion_loss = ecg_loss
        else:
            ppg_loss = self.loss_fn(outputs["pred_noise_ppg"], diffusion_targets["noise_ppg"])
            diffusion_loss = ppg_loss

        total_loss = self.diffusion_weight * diffusion_loss
        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "ecg_diffusion_loss": ecg_loss,
            "ppg_diffusion_loss": ppg_loss,
            "modality_mask": modality_mask,
            **diffusion_targets,
            **encoded,
            **outputs,
        }

    def two_modal_train(
        self,
        clean_ecg: Tensor,
        clean_ppg: Tensor,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
    ) -> Dict[str, Tensor]:
        """
        双模态训练路径。

        输入:
        - `clean_ecg/clean_ppg/noisy_ecg/noisy_ppg:[B,1,T]`

        输出:
        - `total_loss:[]`
        - `diffusion_loss:[]`
        - 以及训练中间量
        """
        modality_mask = torch.tensor([1.0, 1.0], device=clean_ecg.device, dtype=clean_ecg.dtype)
        diffusion_targets = self._build_diffusion_targets(clean_ecg, clean_ppg)
        encoded = self._encode_training_features(clean_ecg, clean_ppg, noisy_ecg, noisy_ppg, modality_mask)
        outputs = self.predict_noise_from_xt(
            x_t_ecg=diffusion_targets["x_t_ecg"],
            x_t_ppg=diffusion_targets["x_t_ppg"],
            clean_feat_ecg=encoded["clean_feat_ecg"],
            clean_feat_ppg=encoded["clean_feat_ppg"],
            noisy_feat_ecg=encoded["noisy_feat_ecg"],
            noisy_feat_ppg=encoded["noisy_feat_ppg"],
            t=diffusion_targets["t"],
            modality_mask=modality_mask,
        )

        ecg_loss = self.loss_fn(outputs["pred_noise_ecg"], diffusion_targets["noise_ecg"])
        ppg_loss = self.loss_fn(outputs["pred_noise_ppg"], diffusion_targets["noise_ppg"])
        diffusion_loss = 0.5 * (ecg_loss + ppg_loss)
        total_loss = self.diffusion_weight * diffusion_loss
        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "ecg_diffusion_loss": ecg_loss,
            "ppg_diffusion_loss": ppg_loss,
            "modality_mask": modality_mask,
            **diffusion_targets,
            **encoded,
            **outputs,
        }

    def _run_reverse_process(
        self,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        modality_mask: Tensor,
        num_steps: Optional[int],
        use_ddim: bool,
    ) -> Dict[str, Tensor]:
        modality_mask = self._validate_modality_mask(modality_mask, noisy_ecg.device, noisy_ecg.dtype)
        noisy_ecg, noisy_ppg = self._fill_missing_noisy_pair(noisy_ecg, noisy_ppg, modality_mask)
        encoded = self._encode_generation_features(noisy_ecg, noisy_ppg, modality_mask)
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
                clean_feat_ecg=encoded["clean_feat_ecg"],
                clean_feat_ppg=encoded["clean_feat_ppg"],
                noisy_feat_ecg=encoded["noisy_feat_ecg"],
                noisy_feat_ppg=encoded["noisy_feat_ppg"],
                t=t,
                modality_mask=modality_mask,
            )
            pred_eps = torch.cat([out["pred_noise_ecg"], out["pred_noise_ppg"]], dim=1)
            if use_ddim:
                x_t_pair = self.diffusion.ddim_sample_step(x_t_pair, t, pred_eps)
            else:
                x_t_pair = self.diffusion.p_sample(x_t_pair, t, pred_eps)

        denoised_ecg = x_t_pair[:, 0:1, :]
        denoised_ppg = x_t_pair[:, 1:2, :]
        denoised_ecg, denoised_ppg = self._mask_signal_pair(denoised_ecg, denoised_ppg, modality_mask)
        return {
            "denoised_ecg": denoised_ecg,
            "denoised_ppg": denoised_ppg,
            "modality_mask": modality_mask,
            **encoded,
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
            raise ValueError(
                f"noisy_ecg 和 noisy_ppg 形状必须一致，实际为 {tuple(noisy_ecg.shape)} 和 {tuple(noisy_ppg.shape)}"
            )

        if modality_mask is None:
            modality_mask = torch.tensor([float(has_ecg), float(has_ppg)], device=device, dtype=ref.dtype)
        modality_mask = self._validate_modality_mask(modality_mask, device, ref.dtype)

        if int(modality_mask.sum().item()) == 1:
            return self.one_modal_generate(noisy_ecg, noisy_ppg, modality_mask, num_steps, use_ddim)
        return self.two_modal_generate(noisy_ecg, noisy_ppg, modality_mask, num_steps, use_ddim)

    def forward(
        self,
        noisy_ecg: Tensor,
        noisy_ppg: Tensor,
        clean_ecg: Tensor,
        clean_ppg: Tensor,
        modality_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        clean_ecg = self._ensure_3d(clean_ecg, "clean_ecg")
        clean_ppg = self._ensure_3d(clean_ppg, "clean_ppg")
        noisy_ecg = self._ensure_3d(noisy_ecg, "noisy_ecg")
        noisy_ppg = self._ensure_3d(noisy_ppg, "noisy_ppg")
        if clean_ecg.shape != clean_ppg.shape or noisy_ecg.shape != noisy_ppg.shape:
            raise ValueError("训练时 ECG/PPG 形状必须一致")

        modality_mask = self._validate_modality_mask(modality_mask, clean_ecg.device, clean_ecg.dtype)
        if int(modality_mask.sum().item()) == 1:
            return self.one_modal_train(clean_ecg, clean_ppg, noisy_ecg, noisy_ppg, modality_mask)
        return self.two_modal_train(clean_ecg, clean_ppg, noisy_ecg, noisy_ppg)
