"""Export trained EfficientNet deepfake detection model to ONNX runtime format.

Exports the PyTorch model with dynamic batch dimensions and validates the exported
ONNX artifact against the native PyTorch outputs using ONNX Runtime.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from video.src.dataset import set_seed
from video.src.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    image_size: int = 224,
    opset_version: int = 14,
) -> Path:
    """Export PyTorch classifier to ONNX format with dynamic batch sizing.

    Args:
        model: PyTorch model in evaluation mode.
        output_path: Path where ONNX model should be written.
        image_size: Square image resolution.
        opset_version: ONNX operator set version.

    Returns:
        Path to the saved ONNX model.
    """
    model.eval()
    model.cpu()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    logger.info("Exporting model to ONNX at %s (opset=%d)...", output_path, opset_version)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    if onnx is not None:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX structural validation succeeded.")

    return output_path


def verify_with_onnxruntime(
    torch_model: nn.Module,
    onnx_path: Path,
    image_size: int = 224,
    test_batch_sizes: Tuple[int, ...] = (1, 4),
) -> bool:
    """Verify numeric parity between PyTorch and ONNX Runtime across different batch sizes.

    Args:
        torch_model: PyTorch model.
        onnx_path: Path to exported ONNX model.
        image_size: Model input spatial dimension.
        test_batch_sizes: Tuple of batch sizes to test dynamic batch dimension.

    Returns:
        True if all verification assertions pass, False otherwise.
    """
    if ort is None:
        logger.warning("onnxruntime is not installed. Skipping runtime verification.")
        return False

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    torch_model.eval()
    torch_model.cpu()

    all_passed = True
    for bs in test_batch_sizes:
        dummy = torch.randn(bs, 3, image_size, image_size, dtype=torch.float32)

        with torch.no_grad():
            torch_out = torch_model(dummy).numpy()

        ort_inputs = {input_name: dummy.numpy()}
        ort_out = session.run(None, ort_inputs)[0]

        max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
        logger.info(
            "Batch size %d: PyTorch vs ONNX Runtime max absolute difference = %.2e",
            bs,
            max_abs_diff,
        )

        try:
            np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-4)
            logger.info("Verification passed for batch size %d.", bs)
        except AssertionError as err:
            logger.error("Verification failed for batch size %d: %s", bs, err)
            all_passed = False

    return all_passed


def export_and_verify(
    checkpoint_path: Optional[str | Path] = None,
    config_path: str | Path = "video/configs/baseline.yaml",
    output_path: Optional[str | Path] = None,
) -> Path:
    """Load configuration and checkpoint, export to ONNX, and run verification.

    Args:
        checkpoint_path: Path to checkpoint .pt file.
        config_path: Path to configuration YAML.
        output_path: Path for output ONNX file.

    Returns:
        Path to exported ONNX file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config.get("seed", 42)))
    image_size = int(config.get("image_size", 224))
    output_dir = Path(config.get("output_dir", "video/outputs"))

    model = build_model(config, device=torch.device("cpu"))

    if checkpoint_path is None:
        default_ckpt = output_dir / "best_model.pt"
        if default_ckpt.is_file():
            checkpoint_path = default_ckpt

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        logger.info("Loading checkpoint weights from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        logger.warning("No checkpoint found; exporting initialized model.")

    if output_path is None:
        target_path = output_dir / "model.onnx"
    else:
        target_path = Path(output_path)

    saved_path = export_to_onnx(model, target_path, image_size=image_size)

    # Also copy to outputs/model.onnx for standard tracking
    common_output = Path("outputs/model.onnx")
    if saved_path.resolve() != common_output.resolve():
        try:
            common_output.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(saved_path, common_output)
            logger.info("Copied ONNX model to %s", common_output)
        except Exception as e:
            logger.debug("Could not copy to outputs/model.onnx: %s", e)

    verify_with_onnxruntime(model, saved_path, image_size=image_size)
    return saved_path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Export deepfake detection model to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="video/configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Target output path for .onnx file.",
    )
    args = parser.parse_args()

    export_and_verify(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
