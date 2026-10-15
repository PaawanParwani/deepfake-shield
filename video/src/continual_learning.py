"""Continual Learning and Elastic Weight Consolidation (EWC) for Autonomous Self-Training.

Enables the deepfake detection model to continually train itself on new human faces,
novel lighting, and emerging synthetic generators WITHOUT catastrophic forgetting.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class ElasticWeightConsolidation:
    """Calculates Fisher Information Matrix and computes EWC regularization loss."""

    def __init__(self, model: nn.Module, lambda_reg: float = 400.0) -> None:
        """Initialize EWC regularizer.

        Args:
            model: Neural network model instance.
            lambda_reg: Importance weight for previous task memory preservation.
        """
        self.model = model
        self.lambda_reg = lambda_reg
        self.params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}

        # Save initial reference parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone().detach()

    def compute_fisher(self, dataloader: Any, device: torch.device, num_samples: int = 100) -> None:
        """Compute the empirical diagonal Fisher Information Matrix over reference samples.

        Args:
            dataloader: DataLoader with baseline dataset samples (images, labels).
            device: Active PyTorch compute device.
            num_samples: Number of samples used to approximate Fisher values.
        """
        self.model.eval()
        self.fisher = {name: torch.zeros_like(param) for name, param in self.params.items()}

        samples_processed = 0
        criterion = nn.CrossEntropyLoss()

        for batch in dataloader:
            if samples_processed >= num_samples:
                break

            images = batch[0].to(device)
            labels = batch[1].to(device)
            self.model.zero_grad()

            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Accumulate squared gradients (diagonal Fisher)
                    self.fisher[name] += (param.grad.data ** 2) * images.size(0)

            samples_processed += images.size(0)

        # Normalize across processed sample count
        count = max(1, samples_processed)
        for name in self.fisher:
            self.fisher[name] /= count

        logger.info("Computed Fisher Information Matrix across %d reference samples.", samples_processed)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute quadratic EWC loss penalizing deviation from critical baseline weights.

        Args:
            model: Currently training model instance.

        Returns:
            Scalar tensor representing EWC consolidation loss.
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        if not self.fisher:
            return loss

        for name, param in model.named_parameters():
            if name in self.fisher and name in self.params:
                f = self.fisher[name].to(param.device)
                p_old = self.params[name].to(param.device)
                # Weighted quadratic deviation
                loss += (f * (param - p_old) ** 2).sum()

        return (self.lambda_reg / 2.0) * loss


class AutonomousSelfTrainer:
    """Manages the autonomous self-training lifecycle from the Experience Replay Buffer."""

    def __init__(
        self,
        model: nn.Module,
        experience_buffer: Any,
        device: torch.device,
        learning_rate: float = 1e-5,
        lambda_ewc: float = 400.0,
    ) -> None:
        """Initialize self-trainer.

        Args:
            model: Detection model instance.
            experience_buffer: SelfTrainingExperienceBuffer instance.
            device: Active PyTorch compute device.
            learning_rate: Gentle fine-tuning learning rate.
            lambda_ewc: EWC memory retention weight.
        """
        self.model = model
        self.experience_buffer = experience_buffer
        self.device = device
        self.ewc = ElasticWeightConsolidation(model, lambda_reg=lambda_ewc)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        self.updates_completed = 0

    def step(self, batch_size: int = 16) -> Dict[str, Any]:
        """Perform one self-supervised update step from the experience replay buffer.

        Args:
            batch_size: Number of pseudo-labeled samples to draw.

        Returns:
            Dictionary with training loss and metrics.
        """
        batch = self.experience_buffer.get_batch(batch_size=batch_size)
        if batch is None:
            return {
                "success": False,
                "message": f"Insufficient buffer samples (need {batch_size}, currently {self.experience_buffer.size()}).",
            }

        images_np, labels_np = batch

        # Preprocess and normalize to PyTorch tensor format
        images_tensor = torch.from_numpy(images_np).float()
        if images_tensor.dim() == 4 and images_tensor.shape[-1] == 3:
            # Convert (B, H, W, C) to (B, C, H, W) and ImageNet normalize
            images_tensor = images_tensor.permute(0, 3, 1, 2) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images_tensor = (images_tensor - mean) / std

        images = images_tensor.to(self.device)
        labels = torch.from_numpy(labels_np).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(images)
        task_loss = self.criterion(outputs, labels)
        ewc_loss = self.ewc.penalty(self.model)

        total_loss = task_loss + ewc_loss
        total_loss.backward()
        self.optimizer.step()

        self.updates_completed += 1
        acc = float((outputs.argmax(dim=1) == labels).float().mean().item()) * 100.0

        return {
            "success": True,
            "total_loss": round(float(total_loss.item()), 4),
            "task_loss": round(float(task_loss.item()), 4),
            "ewc_loss": round(float(ewc_loss.item()), 4),
            "batch_accuracy": round(acc, 1),
            "buffer_size": self.experience_buffer.size(),
            "updates_completed": self.updates_completed,
        }
