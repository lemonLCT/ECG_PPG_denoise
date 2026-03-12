from __future__ import annotations

import torch
from torch import Tensor, nn


class ModalityAwareFusion(nn.Module):
    """模态存在性 + 质量双重门控融合。"""

    def __init__(self, channels: int = 128, joint_channels: int = 256) -> None:
        super().__init__()
        self.ecg_missing_token = nn.Parameter(torch.zeros(1, channels, 1))
        self.ppg_missing_token = nn.Parameter(torch.zeros(1, channels, 1))

        self.ecg_to_ppg = nn.Conv1d(channels, channels, kernel_size=1)
        self.ppg_to_ecg = nn.Conv1d(channels, channels, kernel_size=1)
        self.ecg_fuse = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.ppg_fuse = nn.Conv1d(channels * 2, channels, kernel_size=1)

        self.joint_proj = nn.Sequential(
            nn.Linear(channels * 2 + 4, joint_channels),
            nn.GELU(),
            nn.Linear(joint_channels, joint_channels),
        )

    @staticmethod
    def _apply_missing_token(feat: Tensor, token: Tensor, mask: Tensor) -> Tensor:
        return feat * mask + token.expand(feat.shape[0], -1, feat.shape[-1]) * (1.0 - mask)

    def forward(
        self,
        feat_ecg: Tensor,
        feat_ppg: Tensor,
        q_map_ecg: Tensor,
        q_map_ppg: Tensor,
        modality_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ecg_mask = modality_mask[:, 0].view(-1, 1, 1)
        ppg_mask = modality_mask[:, 1].view(-1, 1, 1)

        base_ecg = self._apply_missing_token(feat_ecg, self.ecg_missing_token, ecg_mask)
        base_ppg = self._apply_missing_token(feat_ppg, self.ppg_missing_token, ppg_mask)

        aux_for_ecg = self.ppg_to_ecg(base_ppg) * q_map_ppg * ppg_mask
        aux_for_ppg = self.ecg_to_ppg(base_ecg) * q_map_ecg * ecg_mask

        c_ecg = self.ecg_fuse(torch.cat([base_ecg, aux_for_ecg], dim=1))
        c_ppg = self.ppg_fuse(torch.cat([base_ppg, aux_for_ppg], dim=1))

        pooled_ecg = c_ecg.mean(dim=-1)
        pooled_ppg = c_ppg.mean(dim=-1)
        quality_summary = torch.cat(
            [
                q_map_ecg.mean(dim=-1) * modality_mask[:, 0:1].to(dtype=q_map_ecg.dtype, device=q_map_ecg.device),
                q_map_ppg.mean(dim=-1) * modality_mask[:, 1:2].to(dtype=q_map_ppg.dtype, device=q_map_ppg.device),
            ],
            dim=1,
        )
        joint_input = torch.cat([pooled_ecg, pooled_ppg, modality_mask.float(), quality_summary], dim=1)
        c_joint = self.joint_proj(joint_input)
        return c_ecg, c_ppg, c_joint
