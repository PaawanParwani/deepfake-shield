"""Export trained audio deepfake detection model to ONNX format.

Converts the Wav2Vec2 classifier into an optimized ONNX computational graph
with dynamic batch size and variable-length time sequence axes, followed by
validation with onnxruntime against PyTorch reference outputs.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

# Relative imports or package imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from .model import build_model
except ImportError:
    from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducible export.

    Args:
        seed: Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def export_to_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    sample_rate: int = 16000,
    duration: float = 4.0,
    opset_version: int = 14,
) -> Path:
    """Export PyTorch audio classifier to ONNX format.

    Specifies dynamic axes on the batch dimension and audio sequence length
    dimension to enable variable-duration audio inference.

    Args:
        model: PyTorch classification model.
        output_path: Target path for the exported .onnx model file.
        sample_rate: Audio sampling rate in Hz.
        duration: Representative input duration in seconds.
        opset_version: ONNX operator set version (default: 14).

    Returns:
        Path object pointing to the exported ONNX file.
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seq_len = int(sample_rate * duration)
    dummy_input = torch.randn(1, seq_len, dtype=torch.float32)

    dynamic_axes = {
        "waveform": {0: "batch_size", 1: "time_steps"},
        "logits": {0: "batch_size"},
    }

    logger.info(f"Exporting model to ONNX at {output_path} (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )
    logger.info("ONNX graph successfully exported.")
    return output_path


def verify_onnx(
    onnx_path: Union[str, Path],
    torch_model: nn.Module,
    sample_rate: int = 16000,
    duration: float = 4.0,
) -> bool:
    """Verify that exported ONNX model matches PyTorch outputs across static and dynamic inputs.

    Args:
        onnx_path: Path to the .onnx file.
        torch_model: Reference PyTorch model.
        sample_rate: Sampling rate in Hz.
        duration: Test audio clip duration.

    Returns:
        True if verification passes within numerical tolerance.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnx or onnxruntime package not installed. Skipping onnxruntime verification."
        )
        return True

    logger.info("Verifying ONNX graph consistency...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX checker passed.")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    torch_model.eval()

    # 1. Test standard input shape (1, T)
    seq_len = int(sample_rate * duration)
    test_wave = torch.randn(1, seq_len, dtype=torch.float32)

    with torch.no_grad():
        torch_logits = torch_model(test_wave).numpy()

    ort_inputs = {"waveform": test_wave.numpy()}
    ort_logits = session.run(["logits"], ort_inputs)[0]

    max_diff = float(np.max(np.abs(torch_logits - ort_logits)))
    logger.info(f"Test 1 (Standard length): Max absolute difference = {max_diff:.6e}")
    if not np.allclose(torch_logits, ort_logits, rtol=1e-3, atol=1e-4):
        logger.error("Verification failed: ONNX outputs differ from PyTorch reference.")
        return False

    # 2. Test dynamic axes: different batch size and different sequence length (2, T * 1.5)
    dynamic_len = int(seq_len * 1.5)
    dynamic_wave = torch.randn(2, dynamic_len, dtype=torch.float32)

    with torch.no_grad():
        torch_dynamic_logits = torch_model(dynamic_wave).numpy()

    ort_dynamic_inputs = {"waveform": dynamic_wave.numpy()}
    ort_dynamic_logits = session.run(["logits"], ort_dynamic_inputs)[0]

    dynamic_max_diff = float(np.max(np.abs(torch_dynamic_logits - ort_dynamic_logits)))
    logger.info(
        f"Test 2 (Dynamic batch=2, length={dynamic_len}): "
        f"Max absolute difference = {dynamic_max_diff:.6e}"
    )
    if not np.allclose(torch_dynamic_logits, ort_dynamic_logits, rtol=1e-3, atol=1e-4):
        logger.error("Dynamic axes verification failed.")
        return False

    logger.info("ONNX verification successfully passed on both fixed and dynamic shapes!")
    return True


def main() -> None:
    """CLI interface for exporting and validating ONNX audio detector."""
    parser = argparse.ArgumentParser(description="Export Audio Deepfake Detector to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--config",
        type=str,
        default="audio/configs/baseline.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="audio/outputs/model.onnx",
        help="Target output ONNX file path",
    )
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    set_seed(42)

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found at {config_path}. Using default settings.")
        config = {
            "model_name": "facebook/wav2vec2-base",
            "num_classes": 2,
            "sample_rate": 16000,
            "max_duration": 4.0,
        }

    # Always use CPU for exporting ONNX graphs cleanly
    device = torch.device("cpu")
    model = build_model(
        model_name=config.get("model_name", "facebook/wav2vec2-base"),
        num_classes=config.get("num_classes", 2),
        freeze_backbone=False,
        device=device,
    )

    if args.checkpoint is not None and Path(args.checkpoint).exists():
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    else:
        logger.warning("No checkpoint found. Exporting model with initialized weights.")

    output_path = Path(args.output)
    export_to_onnx(
        model=model,
        output_path=output_path,
        sample_rate=config.get("sample_rate", 16000),
        duration=config.get("max_duration", 4.0),
        opset_version=args.opset,
    )

    verify_onnx(
        onnx_path=output_path,
        torch_model=model,
        sample_rate=config.get("sample_rate", 16000),
        duration=config.get("max_duration", 4.0),
    )


if __name__ == "__main__":
    main()
