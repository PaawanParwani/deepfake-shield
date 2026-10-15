"""EfficientNet-B0 binary classifier for FaceForensics++ deepfake detection.

Wraps timm's EfficientNet-B0 with capabilities for backbone freezing,
custom head adaptation, and AdamW optimizer configuration.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml

try:
    import timm
except ImportError:
    timm = None

try:
    import torchvision.models as tv_models
except ImportError:
    tv_models = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed integer.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeepfakeClassifier(nn.Module):
    """EfficientNet-B0 binary classifier for real vs. fake face image classification."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
        drop_rate: float = 0.2,
    ) -> None:
        """Initialize the model architecture.

        Args:
            model_name: Model architecture identifier (default: 'efficientnet_b0').
            pretrained: Whether to load ImageNet pretrained weights.
            num_classes: Number of output target classes (default: 2).
            drop_rate: Dropout probability for classifier head.
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.drop_rate = drop_rate

        if timm is not None:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
            )
            logger.info("Instantiated timm model: %s (pretrained=%s, num_classes=%d)", model_name, pretrained, num_classes)
        elif tv_models is not None and hasattr(tv_models, model_name):
            logger.warning("timm not found; falling back to torchvision %s.", model_name)
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = getattr(tv_models, model_name)(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=drop_rate, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ImportError("Neither 'timm' nor 'torchvision' could provide the requested model.")

    def get_classifier_parameters(self) -> Tuple[nn.Parameter, ...]:
        """Retrieve parameters belonging specifically to the classification head.

        Returns:
            Tuple of classifier head parameters.
        """
        if hasattr(self.backbone, "get_classifier"):
            return tuple(self.backbone.get_classifier().parameters())
        if hasattr(self.backbone, "classifier"):
            return tuple(self.backbone.classifier.parameters())
        if hasattr(self.backbone, "fc"):
            return tuple(self.backbone.fc.parameters())
        return tuple()

    def freeze_backbone(self) -> None:
        """Freeze all feature extraction parameters, leaving only the classifier head trainable."""
        classifier_params = set(id(p) for p in self.get_classifier_parameters())
        frozen_count = 0
        for p in self.backbone.parameters():
            if id(p) not in classifier_params:
                p.requires_grad = False
                frozen_count += 1
        logger.info("Froze %d backbone parameter tensors; classifier head remains trainable.", frozen_count)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the model for full end-to-end fine-tuning."""
        unfrozen_count = 0
        for p in self.backbone.parameters():
            p.requires_grad = True
            unfrozen_count += 1
        logger.info("Unfroze all %d parameter tensors for end-to-end training.", unfrozen_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.backbone(x)

    def get_optimizer(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> torch.optim.Optimizer:
        """Create AdamW optimizer for all trainable parameters.

        Args:
            lr: Learning rate.
            weight_decay: Weight decay factor.

        Returns:
            Configured AdamW optimizer.
        """
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )


def build_model(
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> DeepfakeClassifier:
    """Build and initialize DeepfakeClassifier from configuration dictionary.

    Args:
        config: Loaded config dictionary.
        device: Target torch device.

    Returns:
        DeepfakeClassifier moved to specified device.
    """
    model_name = config.get("model_name", "efficientnet_b0")
    pretrained = bool(config.get("pretrained", True))
    num_classes = int(config.get("num_classes", 2))

    model = DeepfakeClassifier(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if device is not None:
        model = model.to(device)

    return model


def main() -> None:
    """CLI interface: create model, run dummy forward pass, and verify output dimensions."""
    parser = argparse.ArgumentParser(description="Instantiate and test EfficientNet deepfake model.")
    parser.add_argument(
        "--config",
        type=str,
        default="video/configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model architecture name override (e.g. 'efficientnet_b0').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Dummy batch size for shape testing.",
    )
    args = parser.parse_args()

    set_seed(42)

    config_path = Path(args.config)
    cfg: Dict[str, Any] = {}
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    if args.model_name:
        cfg["model_name"] = args.model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Instantiating model on device: {device}")

    model = build_model(cfg, device=device)
    optimizer = model.get_optimizer(
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    print(f"Created optimizer: {type(optimizer).__name__} with {len(optimizer.param_groups[0]['params'])} param groups")

    # Test dummy forward pass
    dummy_input = torch.randn(args.batch_size, 3, int(cfg.get("image_size", 224)), int(cfg.get("image_size", 224)), device=device)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (args.batch_size, int(cfg.get("num_classes", 2))), "Output shape mismatch!"
    print("Model forward pass verified successfully.")


if __name__ == "__main__":
    main()
