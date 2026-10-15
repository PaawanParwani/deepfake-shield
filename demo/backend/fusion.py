"""Learned Cross-Modal Attention Fusion for Multimodal Deepfake Verification.

Implements bidirectional cross-attention and adaptive reliability gating between
audio representations (e.g., WavLM / AASIST) and visual representations (e.g., EfficientNet-V2 / ConvNeXt):
1. Bidirectional Cross-Modal Multi-Head Attention (A -> V and V -> A)
2. Dynamic Reliability Gating (weights audio vs visual evidence under noise/occlusion)
3. Multimodal Classifier Head with Uncertainty & Contribution Estimation
4. Calibrated Multimodal Decision Engine
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossAttentionBlock(nn.Module):
    """Bidirectional Cross-Attention between Audio and Visual embeddings."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        """Initialize cross-attention block.

        Args:
            embed_dim: Common feature embedding dimension.
            num_heads: Number of attention heads.
            dropout: Attention and feedforward dropout rate.
        """
        super().__init__()
        self.mha_a2v = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_v2a = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_a1 = nn.LayerNorm(embed_dim)
        self.norm_v1 = nn.LayerNorm(embed_dim)

        self.ffn_a = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_v = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.norm_a2 = nn.LayerNorm(embed_dim)
        self.norm_v2 = nn.LayerNorm(embed_dim)

    def forward(
        self, audio_feat: torch.Tensor, video_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with bidirectional cross-attention.

        Args:
            audio_feat: Audio features (B, Ta, D).
            video_feat: Video features (B, Tv, D).

        Returns:
            Tuple of contextualized (audio_out, video_out) with shapes (B, Ta, D) and (B, Tv, D).
        """
        # Audio attends to Video
        attn_a2v, _ = self.mha_a2v(query=audio_feat, key=video_feat, value=video_feat)
        a_mid = self.norm_a1(audio_feat + attn_a2v)
        audio_out = self.norm_a2(a_mid + self.ffn_a(a_mid))

        # Video attends to Audio
        attn_v2a, _ = self.mha_v2a(query=video_feat, key=audio_feat, value=audio_feat)
        v_mid = self.norm_v1(video_feat + attn_v2a)
        video_out = self.norm_v2(v_mid + self.ffn_v(v_mid))

        return audio_out, video_out


class CrossAttentionMultimodalFusion(nn.Module):
    """End-to-end multimodal deepfake detector fusing speech and face video features."""

    def __init__(
        self,
        audio_dim: int = 768,
        video_dim: int = 1280,
        fusion_dim: int = 256,
        num_heads: int = 4,
        num_classes: int = 2,
    ) -> None:
        """Initialize multimodal fusion architecture.

        Args:
            audio_dim: Input dimension of audio feature embeddings.
            video_dim: Input dimension of visual feature embeddings.
            fusion_dim: Dimension of unified projection space.
            num_heads: Number of attention heads.
            num_classes: Classification categories (2 = authentic vs synthetic).
        """
        super().__init__()
        self.fusion_dim = fusion_dim

        # Modality projectors
        self.proj_audio = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.proj_video = nn.Sequential(
            nn.Linear(video_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Cross-Attention interaction layers
        self.cross_attn = CrossAttentionBlock(embed_dim=fusion_dim, num_heads=num_heads)

        # Modality Reliability Gating Network (estimates reliability of each stream)
        self.reliability_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=-1),
        )

        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        audio_emb: torch.Tensor,
        video_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute multimodal predictions and attention weights.

        Args:
            audio_emb: Audio representations (B, D_audio) or (B, T_audio, D_audio).
            video_emb: Video representations (B, D_video) or (B, T_video, D_video).

        Returns:
            Dictionary containing logits, probabilities, and modality weights.
        """
        if audio_emb.dim() == 2:
            audio_emb = audio_emb.unsqueeze(1)
        if video_emb.dim() == 2:
            video_emb = video_emb.unsqueeze(1)

        # Project to common dimension
        h_a = self.proj_audio(audio_emb)  # (B, T_a, fusion_dim)
        h_v = self.proj_video(video_emb)  # (B, T_v, fusion_dim)

        # Bidirectional cross-attention
        ctx_a, ctx_v = self.cross_attn(h_a, h_v)

        # Global temporal pooling
        pool_a = ctx_a.mean(dim=1)  # (B, fusion_dim)
        pool_v = ctx_v.mean(dim=1)  # (B, fusion_dim)

        # Compute dynamic gating weights (alpha_audio, alpha_video)
        gate_input = torch.cat([pool_a, pool_v], dim=-1)
        weights = self.reliability_gate(gate_input)  # (B, 2)
        w_a = weights[:, 0:1]
        w_v = weights[:, 1:2]

        # Gated fusion
        fused = w_a * pool_a + w_v * pool_v  # (B, fusion_dim)

        # Classification logits
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probabilities": probs,
            "p_fake": probs[:, 1],
            "audio_weight": w_a.squeeze(-1),
            "video_weight": w_v.squeeze(-1),
            "fused_embedding": fused,
        }


def fuse_multimodal_predictions(
    p_fake_audio: Optional[float],
    p_fake_video: Optional[float],
    p_bio_liveness: Optional[float] = None,
    audio_snr: Optional[float] = None,
    face_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Production decision engine combining single-stream predictions with biological pulse evidence.

    Handles unimodal fallback gracefully when only audio or only video is available.

    Args:
        p_fake_audio: Audio synthetic probability [0.0, 1.0].
        p_fake_video: Video synthetic probability [0.0, 1.0].
        p_bio_liveness: Biological liveness probability (rPPG + blinks) [0.0, 1.0].
        audio_snr: Audio signal-to-noise ratio in dB.
        face_confidence: MTCNN face detection confidence.

    Returns:
        Structured multimodal decision dictionary.
    """
    has_audio = p_fake_audio is not None and not np.isnan(p_fake_audio)
    has_video = p_fake_video is not None and not np.isnan(p_fake_video)

    if not has_audio and not has_video:
        return {
            "verdict": "unknown",
            "p_fake": 0.0,
            "confidence": 0.0,
            "reason": "Neither audio nor video stream provided.",
            "modalities": [],
        }

    # Dynamic weighting based on stream signal quality
    w_audio = 0.5
    w_video = 0.5

    if has_audio and not has_video:
        w_audio = 1.0
        w_video = 0.0
    elif has_video and not has_audio:
        w_audio = 0.0
        w_video = 1.0
    else:
        # Both present: calibrate weights based on signal quality
        if face_confidence is not None and face_confidence > 0.95:
            w_video += 0.1
            w_audio -= 0.1
        if audio_snr is not None and audio_snr < 10.0:  # Noisy audio
            w_audio = max(0.2, w_audio - 0.2)
            w_video = 1.0 - w_audio

    # Weighted baseline fake probability
    p_fake_combined = 0.0
    if has_audio:
        p_fake_combined += w_audio * float(p_fake_audio)
    if has_video:
        p_fake_combined += w_video * float(p_fake_video)

    # Biological liveness modulation (if biological pulse & blinking are confirmed bonafide)
    if p_bio_liveness is not None and p_bio_liveness > 0.85 and has_video:
        # Strong biological signal pulls down false-positive deepfake flags
        p_fake_combined = p_fake_combined * (1.0 - 0.40 * (p_bio_liveness - 0.85) / 0.15)

    p_fake_combined = float(np.clip(p_fake_combined, 0.0, 1.0))

    # Determine verdict and confidence
    if p_fake_combined >= 0.65:
        verdict = "suspicious"
        confidence = p_fake_combined
    elif p_fake_combined <= 0.35:
        verdict = "authentic"
        confidence = 1.0 - p_fake_combined
    else:
        verdict = "inconclusive"
        confidence = max(p_fake_combined, 1.0 - p_fake_combined)

    return {
        "verdict": verdict,
        "p_fake": round(p_fake_combined, 4),
        "confidence": round(float(confidence), 4),
        "weights": {
            "audio": round(float(w_audio), 3),
            "video": round(float(w_video), 3),
        },
        "modalities": [m for m, active in [("audio", has_audio), ("video", has_video)] if active],
        "biological_liveness": round(float(p_bio_liveness), 4) if p_bio_liveness is not None else None,
    }
