"""Visual Explainability Module for Deepfake Detection (Grad-CAM & Frequency Heatmaps).

Provides forensic visual explanations by highlighting spatial artifact regions:
1. Grad-CAM (Gradient-Weighted Class Activation Mapping) on convolutional feature maps.
2. Heatmap blending over face images with customizable colormaps (Jet, Viridis, Inferno).
3. Synthetic artifact localization (bounding boxes around highest activation clusters).
4. Dual-stream spatial-frequency spectral explanation.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAMExplainer:
    """Computes Grad-CAM spatial heatmaps for deepfake classifiers."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        """Initialize Grad-CAM explainer.

        Args:
            model: PyTorch deepfake classification model.
            target_layer: Convolutional layer to extract activations from.
                If None, automatically discovers the last conv layer in the backbone.
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_target_layer(model)
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _find_target_layer(self, model: nn.Module) -> nn.Module:
        """Heuristically identify the final convolutional layer of the backbone."""
        # Check timm EfficientNet / ConvNeXt layer names
        if hasattr(model, "backbone"):
            bb = model.backbone
            if hasattr(bb, "conv_head"):
                return bb.conv_head
            if hasattr(bb, "bn2"):
                return bb.bn2
            if hasattr(bb, "blocks") and len(bb.blocks) > 0:
                last_block = bb.blocks[-1]
                if hasattr(last_block, "conv"):
                    return last_block.conv
                return last_block
            if hasattr(bb, "features"):
                return bb.features[-1]

        # General traversal backwards for last Conv2d layer
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                last_conv = module
        if last_conv is not None:
            return last_conv

        # Fallback to model itself
        return model

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            self.activations = out.detach()

        def backward_hook(module: nn.Module, grad_in: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """Compute 2D Grad-CAM heatmap normalized to [0, 1].

        Args:
            input_tensor: Normalized image tensor (1, C, H, W).
            target_class: Index of target class (1 = deepfake / spoof).

        Returns:
            2D numpy array (H, W) in range [0.0, 1.0].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        if hasattr(output, "logits"):
            output = output.logits

        score = output[0, target_class]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            # Fallback uniform heatmap if hooks failed to fire
            H, W = input_tensor.shape[2:]
            return np.zeros((H, W), dtype=np.float32)

        # Global average pooling of gradients: alpha_k = 1/Z * sum(grad)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])  # (C,)

        # Weight activations by importance: sum_k alpha_k * A^k
        activations = self.activations[0]  # (C, h, w)
        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()  # (h, w)
        heatmap = np.maximum(heatmap, 0)  # ReLU

        # Normalize to [0, 1]
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        else:
            heatmap = np.zeros_like(heatmap)

        # Resize to input tensor dimensions
        H, W = input_tensor.shape[2:]
        heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
        return heatmap_resized

    def overlay_heatmap(
        self,
        rgb_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay heatmap on RGB image.

        Args:
            rgb_image: Input image (H, W, 3) in uint8 [0, 255].
            heatmap: 2D array (H, W) in [0.0, 1.0].
            alpha: Heatmap blend weight (0.0 to 1.0).
            colormap: OpenCV colormap constant.

        Returns:
            Blended RGB image (H, W, 3) in uint8 [0, 255].
        """
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(rgb_image, 1.0 - alpha, heatmap_color_rgb, alpha, 0)
        return blended

    def find_artifact_regions(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """Identify high-activation bounding boxes corresponding to localized manipulation artifacts.

        Args:
            heatmap: 2D array (H, W) in [0.0, 1.0].
            threshold: Activation cutoff for artifact extraction.

        Returns:
            List of detected artifact bounding boxes with activation intensities.
        """
        binary = (heatmap >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        artifacts = []
        H, W = heatmap.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= (H * W * 0.01):  # Ignore tiny noise (< 1% of face area)
                x, y, w, h = cv2.boundingRect(cnt)
                roi_intensity = float(np.mean(heatmap[y : y + h, x : x + w]))
                artifacts.append({
                    "box": [int(x), int(y), int(x + w), int(y + h)],
                    "intensity": round(roi_intensity, 3),
                    "area_percentage": round((area / (H * W)) * 100.0, 1),
                })

        # Sort by peak intensity descending
        artifacts.sort(key=lambda a: a["intensity"], reverse=True)
        return artifacts


def explain_face_frame(
    model: nn.Module,
    face_rgb: np.ndarray,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Forensic explanation pipeline generating Grad-CAM visualization for a given face crop.

    Args:
        model: Trained deepfake classifier.
        face_rgb: RGB face crop (H, W, 3) in range [0, 255].
        device: 'cpu' or 'cuda'.

    Returns:
        Dictionary with base64 visualization, artifact boxes, and forensic diagnosis.
    """
    H, W = face_rgb.shape[:2]
    # Resize to 224x224 for model input
    resized = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    norm_tensor = ((tensor - mean) / std).unsqueeze(0).to(device)

    explainer = GradCAMExplainer(model)
    heatmap = explainer.generate_heatmap(norm_tensor, target_class=1)

    # Scale heatmap back to original face crop size
    heatmap_orig = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    overlay_rgb = explainer.overlay_heatmap(face_rgb, heatmap_orig, alpha=0.55)
    artifact_boxes = explainer.find_artifact_regions(heatmap_orig, threshold=0.60)

    # Encode blended image as JPEG base64 for web display
    pil_img = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=88)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Determine most prominent forensic anomaly location
    dominant_location = "uniform_distribution"
    if artifact_boxes:
        top_box = artifact_boxes[0]["box"]
        by_mid = (top_box[1] + top_box[3]) / 2.0 / H
        if by_mid < 0.40:
            dominant_location = "periocular_region_and_eyes"
        elif by_mid > 0.65:
            dominant_location = "perioral_region_and_mouth"
        else:
            dominant_location = "nasal_and_central_facial_seam"

    return {
        "heatmap_base64": f"data:image/jpeg;base64,{b64_str}",
        "artifact_regions": artifact_boxes,
        "dominant_artifact_region": dominant_location,
        "peak_activation": round(float(np.max(heatmap)), 4),
    }
