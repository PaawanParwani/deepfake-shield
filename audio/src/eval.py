"""Evaluation script for Audio Deepfake Detector.

Loads a trained Wav2Vec2 checkpoint, evaluates performance on the dev or eval split
of ASVspoof 2019 LA, calculates Equal Error Rate (EER) via ROC interpolation,
AUC, and classification accuracy, generates ROC and DET plots, and outputs
metrics to outputs/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import scipy.stats as stats
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import yaml

# Relative imports or package imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from .dataset import ASVSpoof2019LADataset, create_dataloader, create_synthetic_dataset
    from .model import build_model
except ImportError:
    from dataset import ASVSpoof2019LADataset, create_dataloader, create_synthetic_dataset
    from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducible evaluation.

    Args:
        seed: Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Equal Error Rate (EER) using scipy interpolation on the ROC curve.

    Finds the intersection point where False Alarm Rate (FAR) equals
    False Rejection Rate (FRR) using brentq root-finding on linear interpolation.

    Args:
        labels: 1D array of ground truth binary labels (0 = bonafide, 1 = spoof).
        scores: 1D array of predicted confidence scores for the spoof class.

    Returns:
        Tuple of (eer, threshold, fpr, tpr, thresholds).
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning("Only one class present in evaluation. Returning default EER=0.5.")
        return 0.5, 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = fpr - fnr

    # Discrete nearest threshold point
    idx = int(np.nanargmin(np.abs(diff)))
    eer_approx = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh_approx = float(thresholds[idx])

    try:
        min_diff = np.min(diff)
        max_diff = np.max(diff)
        if min_diff <= 0.0 <= max_diff:
            # Sort monotonically by diff
            order = np.argsort(diff)
            s_diff = diff[order]
            s_fpr = fpr[order]
            s_thresh = thresholds[order]

            # Remove duplicate x coordinates
            u_diff, u_indices = np.unique(s_diff, return_index=True)
            if len(u_diff) > 1:
                eer_fn = interp1d(u_diff, s_fpr[u_indices], kind="linear")
                thresh_fn = interp1d(u_diff, s_thresh[u_indices], kind="linear")

                eer_root = float(eer_fn(0.0))
                thresh_root = float(thresh_fn(0.0))
                return eer_root, thresh_root, fpr, tpr, thresholds
    except Exception as e:
        logger.debug(f"Root-finding interpolation fell back to discrete nearest point: {e}")

    return eer_approx, thresh_approx, fpr, tpr, thresholds


def plot_roc_det(
    fpr: np.ndarray,
    tpr: np.ndarray,
    eer: float,
    auc: float,
    output_path: Path,
) -> None:
    """Generate and save combined ROC and DET curves for model evaluation.

    Args:
        fpr: False positive rate array from roc_curve.
        tpr: True positive rate array from roc_curve.
        eer: Computed Equal Error Rate.
        auc: Computed Area Under Curve.
        output_path: Destination path for the saved image.
    """
    fnr = 1.0 - tpr
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    ax1.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="#999999", lw=1, linestyle="--", label="Random Chance")
    ax1.scatter([eer], [1.0 - eer], color="#d9534f", s=50, zorder=5, label=f"EER = {eer:.4f}")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Alarm Rate (FAR / FPR)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (1 - FRR)", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # DET Curve (FAR vs FRR in percent, log scale)
    eps = 1e-4
    fpr_clipped = np.clip(fpr * 100.0, eps, 100.0)
    fnr_clipped = np.clip(fnr * 100.0, eps, 100.0)

    ax2.plot(fpr_clipped, fnr_clipped, color="#b23b3b", lw=2, label=f"DET Curve (EER = {eer * 100:.2f}%)")
    ax2.plot([eps, 100], [eps, 100], color="#999999", lw=1, linestyle="--", label="EER Diagonal")
    ax2.scatter([eer * 100.0], [eer * 100.0], color="#2b5c8f", s=50, zorder=5, label=f"EER Point")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim([0.1, 100.0])
    ax2.set_ylim([0.1, 100.0])
    ax2.set_xlabel("False Alarm Rate (%)", fontsize=11)
    ax2.set_ylabel("False Rejection Rate (%)", fontsize=11)
    ax2.set_title("Detection Error Tradeoff (DET)", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300)
    plt.close(fig)
    logger.info(f"Saved ROC/DET plot to: {output_path}")


def evaluate_model(
    checkpoint_path: Optional[Union[str, Path]],
    config: Dict[str, Any],
    split: str = "dev",
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Execute evaluation on a specified dataset split.

    Args:
        checkpoint_path: Path to checkpoint .pt file, or None for newly initialized model.
        config: Configuration dictionary.
        split: Split name ('dev' or 'eval').
        device_name: Device selection ('auto', 'cuda', 'cpu').

    Returns:
        Evaluation metrics dictionary.
    """
    set_seed(config.get("seed", 42))

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Evaluating on device: {device}")

    # Build model
    model = build_model(
        model_name=config.get("model_name", "facebook/wav2vec2-base"),
        num_classes=config.get("num_classes", 2),
        freeze_backbone=False,
        device=device,
    )

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        logger.warning("No checkpoint file provided or found. Evaluating with baseline initialized weights.")

    model.eval()

    data_dir = Path(config.get("data_dir", "audio/data/LA"))
    protocol_dir = Path(config.get("protocol_dir", "audio/data/LA/ASVspoof2019_LA_cm_protocols"))
    smoke_test = config.get("smoke_test", False)
    smoke_test_size = config.get("smoke_test_size", 500)

    from audio.src.dataset import detect_data_format
    temp_dir_obj = None
    data_format = detect_data_format(data_dir)

    if data_format == "parquet":
        logger.info("Using parquet format (HuggingFace download). Protocol files not needed.")
    elif not protocol_dir.exists():
        logger.warning("Protocol directory not found. Generating synthetic data for evaluation smoke test.")
        temp_dir_obj = tempfile.TemporaryDirectory()
        data_dir, protocol_dir = create_synthetic_dataset(
            Path(temp_dir_obj.name), num_samples=min(smoke_test_size, 30)
        )
        smoke_test = True

    dataset = ASVSpoof2019LADataset(
        data_dir=data_dir,
        protocol_dir=protocol_dir,
        split=split,
        sample_rate=config.get("sample_rate", 16000),
        max_duration=config.get("max_duration", 4.0),
        is_train=False,
        smoke_test=smoke_test,
        smoke_test_size=smoke_test_size,
        return_file_id=True,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=0,  # Single-process for deterministic eval
    )

    all_labels: List[int] = []
    all_scores: List[float] = []
    all_preds: List[int] = []
    all_ids: List[str] = []

    with torch.no_grad():
        for waveforms, labels, audio_ids in dataloader:
            waveforms = waveforms.to(device)
            logits = model(waveforms)
            probs = torch.softmax(logits, dim=-1)

            # Spoof probability (class 1)
            spoof_probs = probs[:, 1].detach().cpu().numpy()
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()

            all_labels.extend(labels.numpy().tolist())
            all_scores.extend(spoof_probs.tolist())
            all_preds.extend(preds.tolist())
            all_ids.extend(audio_ids)

    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_scores)
    preds_arr = np.array(all_preds)

    total_samples = len(labels_arr)
    num_bonafide = int(np.sum(labels_arr == 0))
    num_spoof = int(np.sum(labels_arr == 1))

    acc = float(accuracy_score(labels_arr, preds_arr)) if total_samples > 0 else 0.0
    try:
        auc = float(roc_auc_score(labels_arr, scores_arr)) if len(np.unique(labels_arr)) > 1 else 0.5
    except Exception:
        auc = 0.5

    eer, thresh, fpr, tpr, thresholds = compute_eer(labels_arr, scores_arr)

    # Class-wise accuracies
    acc_bonafide = (
        float(np.mean(preds_arr[labels_arr == 0] == 0)) if num_bonafide > 0 else 0.0
    )
    acc_spoof = float(np.mean(preds_arr[labels_arr == 1] == 1)) if num_spoof > 0 else 0.0

    output_dir = Path(config.get("output_dir", "audio/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot ROC and DET curve
    plot_file = output_dir / f"roc_det_{split}.png"
    plot_roc_det(fpr, tpr, eer, auc, plot_file)

    metrics = {
        "split": split,
        "total_samples": total_samples,
        "num_bonafide": num_bonafide,
        "num_spoof": num_spoof,
        "accuracy": acc,
        "eer": eer,
        "eer_threshold": thresh,
        "auc": auc,
        "bonafide_accuracy": acc_bonafide,
        "spoof_accuracy": acc_spoof,
        "roc_plot": str(plot_file),
    }

    metrics_path = output_dir / f"metrics_{split}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to: {metrics_path}")

    # Also update outputs/metrics.json
    main_metrics_path = output_dir / "metrics.json"
    with open(main_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Print results table
    print_results_table(metrics)

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    return metrics


def print_results_table(metrics: Dict[str, Any]) -> None:
    """Format and print an evaluation summary table to standard output.

    Args:
        metrics: Dictionary containing calculated metrics.
    """
    sep = "=" * 68
    line = "-" * 68
    print(sep)
    print(f"{'ASVSPOOF 2019 LA AUDIO EVALUATION RESULTS':^68}")
    print(sep)
    print(f" Split:             {metrics.get('split', 'N/A').upper()}")
    print(f" Total Evaluated:   {metrics.get('total_samples', 0):,}")
    print(f"   - Bonafide:      {metrics.get('num_bonafide', 0):,}")
    print(f"   - Spoof:         {metrics.get('num_spoof', 0):,}")
    print(line)
    print(f" Accuracy:          {metrics.get('accuracy', 0.0) * 100.0:6.2f} %")
    print(f" Bonafide Acc:      {metrics.get('bonafide_accuracy', 0.0) * 100.0:6.2f} %")
    print(f" Spoof Acc:         {metrics.get('spoof_accuracy', 0.0) * 100.0:6.2f} %")
    print(f" AUC-ROC:           {metrics.get('auc', 0.0) * 100.0:6.2f} %")
    print(f" Equal Error Rate:  {metrics.get('eer', 0.0) * 100.0:6.2f} %")
    print(f" EER Threshold:     {metrics.get('eer_threshold', 0.0):.6f}")
    print(sep)


def main() -> None:
    """CLI interface for evaluating audio deepfake detection."""
    parser = argparse.ArgumentParser(description="Evaluate Audio Deepfake Detection Model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--config",
        type=str,
        default="audio/configs/baseline.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "eval"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Computation device",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    evaluate_model(
        checkpoint_path=args.checkpoint,
        config=config,
        split=args.split,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
