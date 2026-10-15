"""Dual-Stream Spatial-Frequency and Temporal Biometric Transformer Architectures.

Implements SOTA deepfake detection models:
1. DualStreamDeepfakeDetector: Spatial RGB backbone (EfficientNet / ConvNeXt) coupled with
   a high-pass Laplacian frequency stream to capture GAN/Diffusion upsampling grid lines.
2. SpatioTemporalBiometricTransformer: Ingests sequences of frame embeddings along with
   involuntary physiological time-series (EAR, rPPG, FACS) using Multi-Head Self-Attention.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

logger = logging.getLogger(__name__)


class HighPassResidualBlock(nn.Module):
    """Isolates high-frequency spectral artifacts from GAN/Diffusion generators."""

    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        # Fixed 3x3 Laplacian edge/high-pass kernel
        laplacian_3x3 = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("laplacian", laplacian_3x3.repeat(in_channels, 1, 1, 1))

        # Lightweight frequency convolutional feature extractor
        self.conv_stream = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract high-pass frequency residual vector.

        Args:
            x: Input frame tensor (B, 3, H, W).

        Returns:
            Frequency feature tensor (B, out_channels).
        """
        # Grouped convolution applies Laplacian independently to each RGB channel
        residual = F.conv2d(x, self.laplacian, padding=1, groups=x.size(1))
        return self.conv_stream(residual)


class DualStreamDeepfakeDetector(nn.Module):
    """Dual-Stream Spatial-Frequency Neural Network for Single Frame & Video Inspection."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
        freq_dim: int = 64,
        drop_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.freq_dim = freq_dim

        # 1. Spatial Stream Backbone
        if timm is not None:
            self.spatial_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            spatial_dim = getattr(self.spatial_backbone, "num_features", 1280)
        else:
            raise ImportError("timm library is required for DualStreamDeepfakeDetector.")

        # 2. Frequency Residual Stream
        self.freq_stream = HighPassResidualBlock(in_channels=3, out_channels=freq_dim)

        # 3. Multimodal Dense Fusion Head
        total_dim = spatial_dim + freq_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(64, num_classes),
        )

        logger.info(
            "DualStreamDeepfakeDetector initialized: Spatial=%s (%d-D) + Frequency (%d-D) = %d-D fusion.",
            backbone_name,
            spatial_dim,
            freq_dim,
            total_dim,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract fused spatial-frequency embedding vector.

        Args:
            x: Input batch (B, 3, 224, 224).

        Returns:
            Concatenated embedding vector (B, total_dim).
        """
        f_spatial = self.spatial_backbone(x)
        f_freq = self.freq_stream(x)
        return torch.cat([f_spatial, f_freq], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute bonafide vs deepfake classification logits.

        Args:
            x: Input image tensor (B, 3, 224, 224).

        Returns:
            Unnormalized logits (B, num_classes).
        """
        fused = self.extract_features(x)
        return self.fusion_head(fused)


class SpatioTemporalBiometricTransformer(nn.Module):
    """Multi-Head Self-Attention model fusing temporal frame sequences with biological dynamics."""

    def __init__(
        self,
        frame_feature_dim: int = 256,
        bio_feature_dim: int = 5,  # EAR, Mouth Lift, Forehead rPPG, Left Cheek rPPG, Right Cheek rPPG
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Projections
        self.frame_proj = nn.Linear(frame_feature_dim, hidden_dim // 2)
        self.bio_proj = nn.Linear(bio_feature_dim, hidden_dim // 2)

        # Positional Encoding for temporal steps
        self.pos_encoder = nn.Parameter(torch.randn(1, 64, hidden_dim))

        # Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=drop_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, frame_features: torch.Tensor, bio_features: torch.Tensor) -> torch.Tensor:
        """Classify temporal sequence.

        Args:
            frame_features: (B, T, frame_feature_dim) spatial embeddings.
            bio_features: (B, T, bio_feature_dim) biological time-series signals.

        Returns:
            Logits (B, num_classes).
        """
        b, t, _ = frame_features.shape

        # Embed both streams
        f_embed = self.frame_proj(frame_features)
        b_embed = self.bio_proj(bio_features)
        x = torch.cat([f_embed, b_embed], dim=-1)  # (B, T, hidden_dim)

        # Add positional embedding (slice to actual sequence length T)
        x = x + self.pos_encoder[:, :t, :]

        # Temporal attention
        trans_out = self.transformer(x)  # (B, T, hidden_dim)

        # Mean pooling across time steps
        pooled = torch.mean(trans_out, dim=1)

        return self.head(pooled)
