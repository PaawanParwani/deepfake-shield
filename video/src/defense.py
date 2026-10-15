"""Adversarial Defenses and Robust Video Preprocessing for Deepfake Detection.

Implements input-level stochastic transformations and adversarial training routines:
1. Randomized Resizing and Padding Defense (Mitigating gradient alignment, Xie et al.)
2. Randomized JPEG Compression Defense (Quality factor Q in [70, 90] to destroy high-frequency noise)
3. Total Variation / Spatial Gaussian Filtering Defense
4. Video Adversarial Trainer (PGD-AT on frame/crop tensors)
"""

from __future__ import annotations

import io
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RandomizedResizingPaddingDefense:
    """Randomized spatial resizing and padding defense.
    
    Breaks exact pixel-level gradient alignment commonly exploited by white-box
    and transfer attacks (e.g., FGSM, PGD, C&W).
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        scale_range: Tuple[float, float] = (0.90, 1.05),
    ) -> None:
        """Initialize defense.

        Args:
            target_size: Expected output (H, W).
            scale_range: Min and max scaling factor for random perturbation.
        """
        self.target_size = target_size
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random resizing and padding.

        Args:
            x: Tensor of shape (B, C, H, W) or (C, H, W).

        Returns:
            Defended tensor of shape (B, C, target_size[0], target_size[1]).
        """
        squeeze_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True

        B, C, H, W = x.shape
        scale = float(np.random.uniform(self.scale_range[0], self.scale_range[1]))
        new_h = int(H * scale)
        new_w = int(W * scale)

        # Bilinear resize to randomized dimensions
        resized = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # Pad or crop back to target_size
        out_h, out_w = self.target_size
        padded = torch.zeros((B, C, out_h, out_w), dtype=x.dtype, device=x.device)

        src_h = min(new_h, out_h)
        src_w = min(new_w, out_w)
        start_y = max(0, (out_h - src_h) // 2)
        start_x = max(0, (out_w - src_w) // 2)

        padded[:, :, start_y : start_y + src_h, start_x : start_x + src_w] = resized[:, :, :src_h, :src_w]

        if squeeze_batch:
            return padded.squeeze(0)
        return padded


class RandomizedJPEGCompressionDefense:
    """Non-differentiable JPEG re-compression defense.
    
    Destroys high-frequency adversarial perturbations using randomized 8x8 DCT quantization.
    """

    def __init__(self, quality_range: Tuple[int, int] = (70, 90)) -> None:
        """Initialize JPEG defense.

        Args:
            quality_range: Tuple of (min_quality, max_quality) between 1 and 100.
        """
        self.quality_range = quality_range

    def _compress_numpy(self, img_np: np.ndarray, quality: int) -> np.ndarray:
        """Compress single image uint8 (H, W, C) via PIL JPEG."""
        pil_img = Image.fromarray(img_np)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        reloaded = Image.open(buffer)
        return np.array(reloaded)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply randomized JPEG recompression to tensor.

        Args:
            x: Tensor in range [0, 1] or [-1, 1], shape (B, C, H, W) or (C, H, W).

        Returns:
            Compressed tensor matching original device and shape.
        """
        device = x.device
        dtype = x.dtype
        squeeze_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True

        B, C, H, W = x.shape
        # Check normalization range
        is_neg_one_to_one = bool(x.min() < 0.0)
        if is_neg_one_to_one:
            x_norm = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        else:
            x_norm = x.clamp(0.0, 1.0)

        # Move to CPU numpy for JPEG encoding
        x_cpu = (x_norm.detach().permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        compressed_batch = []

        for b in range(B):
            quality = int(np.random.randint(self.quality_range[0], self.quality_range[1] + 1))
            comp = self._compress_numpy(x_cpu[b], quality=quality)
            compressed_batch.append(comp)

        out_np = np.stack(compressed_batch, axis=0).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_np).permute(0, 3, 1, 2).to(device=device, dtype=dtype)

        if is_neg_one_to_one:
            out_tensor = out_tensor * 2.0 - 1.0

        if squeeze_batch:
            return out_tensor.squeeze(0)
        return out_tensor


class SpatialSmoothingDefense:
    """Gaussian spatial smoothing filter to suppress high-frequency noise."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0) -> None:
        """Initialize 2D Gaussian filter kernel.

        Args:
            kernel_size: Odd kernel size.
            sigma: Standard deviation of Gaussian.
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Build 2D Gaussian kernel
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g2d = torch.outer(g1d, g1d)
        kernel = g2d / g2d.sum()
        self.kernel = kernel.view(1, 1, kernel_size, kernel_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing per channel.

        Args:
            x: Tensor (B, C, H, W).

        Returns:
            Smoothed tensor (B, C, H, W).
        """
        squeeze_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True

        B, C, H, W = x.shape
        k = self.kernel.repeat(C, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        pad = self.kernel_size // 2
        smoothed = F.conv2d(x, k, padding=pad, groups=C)

        if squeeze_batch:
            return smoothed.squeeze(0)
        return smoothed


class VideoAdversarialTrainer:
    """Projected Gradient Descent Adversarial Training (PGD-AT) for Video Deepfake Models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        epsilon: float = 8.0 / 255.0,
        alpha: float = 2.0 / 255.0,
        steps: int = 5,
        mix_ratio: float = 0.5,
    ) -> None:
        """Initialize PGD-AT video trainer.

        Args:
            model: Deepfake classification model.
            optimizer: PyTorch optimizer.
            criterion: Classification loss (defaults to CrossEntropyLoss).
            epsilon: Maximum Linf perturbation budget.
            alpha: Step size per iteration.
            steps: Number of PGD inner steps.
            mix_ratio: Proportion of adversarial examples in each batch (0.0 to 1.0).
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.mix_ratio = mix_ratio

    def generate_adversarial_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Generate PGD perturbed batch under Linf norm.

        Args:
            images: Batch of clean frames/crops (B, C, H, W).
            labels: Ground truth labels (B,).

        Returns:
            Adversarial images bounded by [-epsilon, epsilon].
        """
        self.model.eval()
        adv_images = images.clone().detach()

        # Random start within Linf ball
        noise = torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
        adv_images = torch.clamp(adv_images + noise, 0.0, 1.0).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            loss = self.criterion(outputs, labels)
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * torch.sign(grad.detach())
            eta = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            adv_images = torch.clamp(images + eta, 0.0, 1.0).detach()

        return adv_images

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float]:
        """Perform one robust adversarial training step.

        Args:
            images: Clean input batch (B, C, H, W).
            labels: Batch labels (B,).

        Returns:
            Tuple of (loss_value, clean_loss_value).
        """
        self.model.train()
        B = images.size(0)
        n_adv = int(B * self.mix_ratio)

        if n_adv > 0:
            adv_x = self.generate_adversarial_batch(images[:n_adv], labels[:n_adv])
            combined_x = torch.cat([adv_x, images[n_adv:]], dim=0)
            combined_y = labels
        else:
            combined_x = images
            combined_y = labels

        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(combined_x)
        if hasattr(outputs, "logits"):
            outputs = outputs.logits

        loss = self.criterion(outputs, combined_y)
        loss.backward()
        self.optimizer.step()

        # Compute clean loss metric
        with torch.no_grad():
            clean_out = self.model(images)
            if hasattr(clean_out, "logits"):
                clean_out = clean_out.logits
            clean_loss = float(self.criterion(clean_out, labels).item())

        return float(loss.item()), clean_loss
