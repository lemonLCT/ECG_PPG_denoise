from __future__ import annotations

from math import log as ln

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchsummary import summary

class Conv1d(nn.Conv1d):
    """带 Kaiming 初始化的 1D 卷积。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1)
        return x


class HNFBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()

        self.filters = nn.ModuleList([
            Conv1d(input_size, hidden_size // 4, 3, dilation=dilation, padding=1 * dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size // 4, 5, dilation=dilation, padding=2 * dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size // 4, 9, dilation=dilation, padding=4 * dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size // 4, 15, dilation=dilation, padding=7 * dilation, padding_mode='reflect'),
        ])

        self.conv_1 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode='reflect')

        self.norm = nn.InstanceNorm1d(hidden_size // 2)

        self.conv_2 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode='reflect')

    def forward(self, x):
        residual = x

        filts = []
        for layer in self.filters:
            filts.append(layer(x))

        filts = torch.cat(filts, dim=1)

        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)

        filts = F.leaky_relu(torch.cat([self.norm(nfilts), filts], dim=1), 0.2)

        filts = F.leaky_relu(self.conv_2(filts), 0.2)

        return filts + residual


class Bridge(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoding = FeatureWiseAffine(input_size, hidden_size, use_affine_level=1)
        self.input_conv = Conv1d(input_size, input_size, 3, padding=1, padding_mode='reflect')
        self.output_conv = Conv1d(input_size, hidden_size, 3, padding=1, padding_mode='reflect')

    def forward(self, x, noise_embed):
        x = self.input_conv(x)
        x = self.encoding(x, noise_embed)
        return self.output_conv(x)


class ConditionalModel(nn.Module):
    def __init__(self, feats=64):
        super(ConditionalModel, self).__init__()
        self.stream_x = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 1),
        ])

        self.stream_cond = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 1),
        ])

        self.embed = PositionalEncoding(feats)

        self.bridge = nn.ModuleList([
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
        ])

        self.conv_out = Conv1d(feats, 1, 9, padding=4, padding_mode='reflect')

    def forward(self, x, cond, noise_scale):
        noise_embed = self.embed(noise_scale)
        xs = []
        for layer, br in zip(self.stream_x, self.bridge):
            x = layer(x)
            xs.append(br(x, noise_embed))

        for x, layer in zip(xs, self.stream_cond):
            cond = layer(cond) + x

        return self.conv_out(cond)


class multiConditionModel(nn.Module):
    def __init__(self, feats=80):
        super(multiConditionModel, self).__init__()

        self.ecg_embed = PositionalEncoding(feats)
        self.ecg_x_in = nn.Sequential(
            Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        self.ecg_x_blocks = nn.ModuleList([
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 1),
        ])
        self.ecg_bridge = nn.ModuleList([
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
        ])
        self.ecg_cond_in = nn.Sequential(
            Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        self.ecg_cond_blocks = nn.ModuleList([
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 1),
        ])
        self.ecg_out = Conv1d(feats, 1, 9, padding=4, padding_mode='reflect')

        self.ppg_embed = PositionalEncoding(feats)
        self.ppg_x_in = nn.Sequential(
            Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        self.ppg_x_blocks = nn.ModuleList([
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 1),
        ])
        self.ppg_bridge = nn.ModuleList([
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
        ])
        self.ppg_cond_in = nn.Sequential(
            Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        self.ppg_cond_blocks = nn.ModuleList([
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 1),
        ])
        self.ppg_out = Conv1d(feats, 1, 9, padding=4, padding_mode='reflect')

    def forward(self, x_t_ecg, cond_ecg, x_t_ppg, cond_ppg, noise_scale):
        if x_t_ecg.ndim != 3 or cond_ecg.ndim != 3 or x_t_ppg.ndim != 3 or cond_ppg.ndim != 3:
            raise ValueError("x_t_ecg、cond_ecg、x_t_ppg、cond_ppg 都必须是 [B, 1, T] 张量。")
        if x_t_ecg.shape != cond_ecg.shape:
            raise ValueError(f"ECG 的 x_t 和 cond 形状必须一致，当前为 {tuple(x_t_ecg.shape)} 与 {tuple(cond_ecg.shape)}")
        if x_t_ppg.shape != cond_ppg.shape:
            raise ValueError(f"PPG 的 x_t 和 cond 形状必须一致，当前为 {tuple(x_t_ppg.shape)} 与 {tuple(cond_ppg.shape)}")

        ecg_noise_embed = self.ecg_embed(noise_scale)
        ecg_x = self.ecg_x_in(x_t_ecg)
        ecg_skips = []
        for layer, bridge in zip(self.ecg_x_blocks, self.ecg_bridge):
            ecg_x = layer(ecg_x)
            ecg_skips.append(bridge(ecg_x, ecg_noise_embed))

        ecg_cond = self.ecg_cond_in(cond_ecg)
        for skip, layer in zip(ecg_skips, self.ecg_cond_blocks):
            ecg_cond = layer(ecg_cond) + skip
        pred_noise_ecg = self.ecg_out(ecg_cond)

        ppg_noise_embed = self.ppg_embed(noise_scale)
        ppg_x = self.ppg_x_in(x_t_ppg)
        ppg_skips = []
        for layer, bridge in zip(self.ppg_x_blocks, self.ppg_bridge):
            ppg_x = layer(ppg_x)
            ppg_skips.append(bridge(ppg_x, ppg_noise_embed))

        ppg_cond = self.ppg_cond_in(cond_ppg)
        for skip, layer in zip(ppg_skips, self.ppg_cond_blocks):
            ppg_cond = layer(ppg_cond) + skip
        pred_noise_ppg = self.ppg_out(ppg_cond)

        return pred_noise_ecg, pred_noise_ppg


if __name__ == "__main__":
    net = ConditionalModel(80)
    # leaf = frontend.Leaf(sample_rate=400, n_filters=128, window_len=65, window_stride=40).cuda()
    x = torch.randn(10, 1, 512)
    y = torch.randn(10, 1)
    z = net(x, x, y)

    # print(z.shape)

    summary(net)
