"""Compact three-frame center-heatmap and visibility network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(ConvBlock(channels, channels), ConvBlock(channels, channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.layers(inputs))


class TemporalHeatmapNet(nn.Module):
    """Stack RGB frames on channels and emit a stride-four center heatmap."""

    output_stride = 4

    def __init__(self, context_frames: int = 3, base_channels: int = 24):
        super().__init__()
        self.context_frames = context_frames
        self.encoder = nn.Sequential(
            ConvBlock(context_frames * 3, base_channels, stride=2),
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4),
            ResidualBlock(base_channels * 4),
        )
        self.heatmap_head = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 2),
            nn.Conv2d(base_channels * 2, 1, 1),
        )
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(inputs)
        return self.heatmap_head(features), self.visibility_head(features).squeeze(1)


def temporal_loss(
    heatmap_logits: torch.Tensor,
    visibility_logits: torch.Tensor,
    target_heatmap: torch.Tensor,
    target_visibility: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    heatmap_positive_weight: float = 40.0,
    visibility_loss_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    pixel_weight = 1.0 + heatmap_positive_weight * target_heatmap
    heatmap_per_pixel = F.binary_cross_entropy_with_logits(
        heatmap_logits, target_heatmap, reduction="none"
    )
    heatmap_per_sample = (heatmap_per_pixel * pixel_weight).mean(dim=(1, 2, 3))
    visibility_per_sample = F.binary_cross_entropy_with_logits(
        visibility_logits, target_visibility, reduction="none"
    )
    denominator = sample_weight.sum().clamp_min(1.0)
    heatmap_loss = (heatmap_per_sample * sample_weight).sum() / denominator
    visibility_loss = (visibility_per_sample * sample_weight).sum() / denominator
    total = heatmap_loss + visibility_loss_weight * visibility_loss
    return total, {
        "heatmap_loss": float(heatmap_loss.detach()),
        "visibility_loss": float(visibility_loss.detach()),
        "total_loss": float(total.detach()),
    }

