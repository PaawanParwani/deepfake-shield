"""Wav2Vec2-based Audio Deepfake Classifier.

This module defines the deep learning model architecture using Hugging Face's
pretrained Wav2Vec2Model backbone combined with mean temporal pooling and a
linear classification head for bonafide vs. spoof speech discrimination.
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Wav2Vec2AudioClassifier(nn.Module):
    """Deepfake audio classifier utilizing Wav2Vec2 representation backbone.

    Extracts contextual representation sequence from raw audio waveforms,
    applies mean temporal pooling, and classifies via a linear projection head.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
        backbone_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Wav2Vec2 classifier.

        Args:
            model_name: HuggingFace model identifier or local directory.
            num_classes: Number of target classes (2: bonafide vs spoof).
            freeze_backbone: Whether to freeze backbone weights during initialization.
            dropout: Dropout probability applied before linear classification head.
            backbone_config: Serialized Wav2Vec2 config for offline checkpoint loading.
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if backbone_config is not None:
            self.backbone = Wav2Vec2Model(Wav2Vec2Config.from_dict(backbone_config))
            logger.info("Initialized Wav2Vec2 backbone from the supplied checkpoint config.")
        else:
            try:
                self.backbone = Wav2Vec2Model.from_pretrained(model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load pretrained Wav2Vec2 weights for '{model_name}'. "
                    "Training from a random backbone is disabled."
                ) from exc
            logger.info("Loaded pretrained backbone weights from %s", model_name)

        self.hidden_size: int = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the Wav2Vec2 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Wav2Vec2 backbone frozen.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the Wav2Vec2 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Wav2Vec2 backbone unfrozen.")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification.

        Args:
            waveform: Float tensor of shape (batch_size, sequence_length) at 16kHz.

        Returns:
            Class logits of shape (batch_size, num_classes).
        """
        # Ensure 2D tensor: (batch_size, sequence_length)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Extract features through Wav2Vec2 backbone
        outputs = self.backbone(waveform)
        # last_hidden_state: (batch_size, time_steps, hidden_size)
        hidden_states = outputs.last_hidden_state

        # Mean temporal pooling across time dimension
        pooled = torch.mean(hidden_states, dim=1)

        # Compute output logits
        logits = self.classifier(pooled)
        return logits

    def get_optimizer(
        self,
        lr_backbone: float = 1e-5,
        lr_head: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> torch.optim.AdamW:
        """Construct AdamW optimizer with distinct learning rates for backbone and head.

        Args:
            lr_backbone: Learning rate for backbone parameters.
            lr_head: Learning rate for classification head parameters.
            weight_decay: Weight decay factor for L2 regularization.

        Returns:
            Configured torch.optim.AdamW optimizer.
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.classifier.parameters() if p.requires_grad]

        param_groups: List[Dict[str, Any]] = []
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay}
            )
        if head_params:
            param_groups.append(
                {"params": head_params, "lr": lr_head, "weight_decay": weight_decay}
            )

        if not param_groups:
            raise ValueError("No trainable parameters found in model.")

        optimizer = torch.optim.AdamW(param_groups)
        return optimizer


class RNNAudioClassifier(nn.Module):
    """Recurrent Neural Network (CRNN: Strided Conv1d + Bidirectional GRU) for Audio Deepfake Detection."""

    def __init__(self, hidden_dim: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=16, stride=8),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=4, stride=2),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.conv(x).transpose(1, 2)
        out, _ = self.rnn(feat)
        pooled = torch.mean(out, dim=1)
        return self.fc(pooled)


def build_model(
    model_name: str = "facebook/wav2vec2-base",
    num_classes: int = 2,
    freeze_backbone: bool = True,
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
    backbone_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Factory function to build and initialize the audio classifier.

    Args:
        model_name: Model identifier or 'rnn' / 'crnn'.
        num_classes: Number of target classes.
        freeze_backbone: Whether to freeze backbone weights.
        dropout: Head dropout probability.
        device: Target computation device.

    Returns:
        Instantiated audio classifier (Wav2Vec2AudioClassifier or RNNAudioClassifier).
    """
    if "rnn" in model_name.lower() or "crnn" in model_name.lower():
        model: nn.Module = RNNAudioClassifier(hidden_dim=64, num_classes=num_classes)
    else:
        model = Wav2Vec2AudioClassifier(
            model_name=model_name,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            backbone_config=backbone_config,
        )
    if device is not None:
        model.to(device)
    return model


def main() -> None:
    """CLI smoke test running a dummy forward pass through the model."""
    parser = argparse.ArgumentParser(description="Wav2Vec2 Audio Classifier Smoke Test")
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--duration", type=float, default=4.0, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    set_seed(42)

    # Determine execution device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Instantiating model on device: {device}")
    model = build_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_backbone,
        device=device,
    )
    model.eval()

    # Generate dummy input waveform: (batch_size, sequence_length)
    seq_len = int(args.duration * args.sample_rate)
    dummy_input = torch.randn(args.batch_size, seq_len, device=device)
    logger.info(f"Input waveform shape: {dummy_input.shape}")

    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=-1)

    logger.info(f"Forward pass completed. Logits shape: {logits.shape}")
    logger.info(f"Output probabilities:\n{probs.cpu().numpy()}")
    assert logits.shape == (args.batch_size, args.num_classes), (
        f"Expected output shape ({args.batch_size}, {args.num_classes}), got {logits.shape}"
    )

    optimizer = model.get_optimizer(lr_backbone=1e-5, lr_head=1e-3)
    logger.info(f"Optimizer initialized with {len(optimizer.param_groups)} parameter groups.")
    logger.info("Model smoke test passed successfully!")


if __name__ == "__main__":
    main()
