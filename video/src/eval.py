"""Evaluation pipeline for FaceForensics++ deepfake video detection.

Loads a trained model checkpoint, evaluates on validation or test splits, computes
frame-level and video-level AUC and accuracy, performs a per-manipulation-method
breakdown, plots ROC curves, and persists metrics to JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from video.src.dataset import create_dataloader, set_seed
from video.src.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Compute accuracy and ROC AUC for binary predictions.

    Args:
        labels: 1D array of ground truth binary labels.
        scores: 1D array of predicted probabilities for positive class.

    Returns:
        Tuple of (accuracy, roc_auc).
    """
    preds = (scores >= 0.5).astype(int)
    acc = float(accuracy_score(labels, preds)) if len(labels) > 0 else 0.0

    if len(np.unique(labels)) < 2:
        auc = 0.5
    else:
        try:
            auc = float(roc_auc_score(labels, scores))
        except ValueError:
            auc = 0.5

    return acc, auc


def run_inference_on_split(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute model inference over an entire split and collect per-frame records.

    Args:
        model: Evaluated neural network model.
        dataloader: Split dataloader.
        device: Torch device.

    Returns:
        Tuple of (frame_records, video_records_dict).
    """
    model.eval()
    frame_records: List[Dict[str, Any]] = []
    video_records: Dict[str, Dict[str, Any]] = {}

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            labels = batch[1].cpu().numpy()
            video_ids = batch[2]
            methods = batch[3]

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for prob, lbl, vid, mth in zip(probs, labels, video_ids, methods):
                frame_records.append(
                    {
                        "prob": float(prob),
                        "label": int(lbl),
                        "video_id": str(vid),
                        "method": str(mth),
                    }
                )

                if vid not in video_records:
                    video_records[vid] = {
                        "probs": [],
                        "label": int(lbl),
                        "method": str(mth),
                    }
                video_records[vid]["probs"].append(float(prob))

    return frame_records, video_records


def compute_evaluation_breakdown(
    frame_records: List[Dict[str, Any]],
    video_records: Dict[str, Dict[str, Any]],
    manipulation_methods: List[str],
) -> Dict[str, Any]:
    """Compute overall and per-method metrics for both frame- and video-levels.

    Args:
        frame_records: List of individual frame prediction dictionaries.
        video_records: Dict mapping video_id to list of frame probabilities and metadata.
        manipulation_methods: List of manipulation method names.

    Returns:
        Structured evaluation metrics dictionary.
    """
    frame_labels = np.array([r["label"] for r in frame_records])
    frame_scores = np.array([r["prob"] for r in frame_records])
    frame_acc, frame_auc = compute_metrics(frame_labels, frame_scores)

    video_ids = list(video_records.keys())
    video_labels = np.array([video_records[v]["label"] for v in video_ids])
    video_scores = np.array([float(np.mean(video_records[v]["probs"])) for v in video_ids])
    video_acc, video_auc = compute_metrics(video_labels, video_scores)

    results: Dict[str, Any] = {
        "overall": {
            "total_frames": len(frame_records),
            "total_videos": len(video_records),
            "frame_auc": frame_auc,
            "frame_acc": frame_acc,
            "video_auc": video_auc,
            "video_acc": video_acc,
        },
        "per_method": {},
    }

    # Per manipulation method breakdown (evaluating each fake method vs all real samples)
    real_video_scores = [
        video_scores[i] for i in range(len(video_ids)) if video_labels[i] == 0
    ]
    real_video_labels = [0] * len(real_video_scores)

    for method in manipulation_methods:
        method_indices = [
            i
            for i, v in enumerate(video_ids)
            if video_labels[i] == 1 and video_records[v]["method"].lower() == method.lower()
        ]

        if not method_indices:
            # Check if video_id starts with method name
            method_indices = [
                i
                for i, v in enumerate(video_ids)
                if video_labels[i] == 1 and v.lower().startswith(method.lower())
            ]

        method_scores = [video_scores[i] for i in method_indices]
        method_labels = [1] * len(method_scores)

        combined_scores = np.array(real_video_scores + method_scores)
        combined_labels = np.array(real_video_labels + method_labels)

        if len(method_scores) > 0 and len(real_video_scores) > 0:
            m_acc, m_auc = compute_metrics(combined_labels, combined_scores)
        else:
            m_acc, m_auc = 0.0, 0.5

        results["per_method"][method] = {
            "num_videos": len(method_scores),
            "video_auc": m_auc,
            "video_acc": m_acc,
        }

    return results


def plot_roc_curves(
    frame_records: List[Dict[str, Any]],
    video_records: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate and save ROC curve plot.

    Args:
        frame_records: All frame evaluations.
        video_records: All video evaluations.
        output_path: Destination PNG image path.
    """
    plt.figure(figsize=(8, 6))

    # Frame level
    f_lbls = np.array([r["label"] for r in frame_records])
    f_scrs = np.array([r["prob"] for r in frame_records])
    if len(np.unique(f_lbls)) >= 2:
        fpr_f, tpr_f, _ = roc_curve(f_lbls, f_scrs)
        auc_f = roc_auc_score(f_lbls, f_scrs)
        plt.plot(fpr_f, tpr_f, label=f"Frame-level ROC (AUC = {auc_f:.3f})", color="tab:blue", lw=2)

    # Video level
    v_ids = list(video_records.keys())
    v_lbls = np.array([video_records[v]["label"] for v in v_ids])
    v_scrs = np.array([float(np.mean(video_records[v]["probs"])) for v in v_ids])
    if len(np.unique(v_lbls)) >= 2:
        fpr_v, tpr_v, _ = roc_curve(v_lbls, v_scrs)
        auc_v = roc_auc_score(v_lbls, v_scrs)
        plt.plot(fpr_v, tpr_v, label=f"Video-level ROC (AUC = {auc_v:.3f})", color="tab:green", lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("FaceForensics++ Deepfake Detection ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()
    logger.info("Saved ROC plot to %s", output_path)


def print_results_table(metrics: Dict[str, Any], split: str) -> None:
    """Print a clean formatted results summary table to console.

    Args:
        metrics: Structured metrics dictionary.
        split: Evaluated split name.
    """
    overall = metrics["overall"]
    print("\n" + "=" * 65)
    print(f"       FACEFORENSICS++ EVALUATION SUMMARY (Split: {split.upper()})")
    print("=" * 65)
    print(f"{'Metric':<30} | {'Value':<12}")
    print("-" * 65)
    print(f"{'Total Evaluated Videos':<30} | {overall['total_videos']:<12}")
    print(f"{'Total Evaluated Frames':<30} | {overall['total_frames']:<12}")
    print(f"{'Frame-Level Accuracy':<30} | {overall['frame_acc'] * 100:.2f}%")
    print(f"{'Frame-Level ROC AUC':<30} | {overall['frame_auc']:.4f}")
    print(f"{'Video-Level Accuracy':<30} | {overall['video_acc'] * 100:.2f}%")
    print(f"{'Video-Level ROC AUC':<30} | {overall['video_auc']:.4f}")
    print("-" * 65)
    print(f"{'Manipulation Method':<20} | {'Videos':<8} | {'Video AUC':<10} | {'Video Acc':<10}")
    print("-" * 65)
    for method, m_data in metrics.get("per_method", {}).items():
        print(
            f"{method:<20} | {m_data['num_videos']:<8} | {m_data['video_auc']:<10.4f} | {m_data['video_acc'] * 100:.2f}%"
        )
    print("=" * 65 + "\n")


def evaluate_split(
    checkpoint_path: Optional[str | Path] = None,
    config_path: str | Path = "video/configs/baseline.yaml",
    split: str = "test",
) -> Dict[str, Any]:
    """Execute evaluation on a chosen dataset split.

    Args:
        checkpoint_path: Path to model checkpoint file (.pt).
        config_path: Path to configuration YAML.
        split: Split to evaluate on ('test' or 'val').

    Returns:
        Evaluation metrics dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config.get("seed", 42)))

    device_setting = config.get("device", "auto")
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)
    logger.info("Evaluating on device: %s", device)

    # Resolve checkpoint
    output_dir = Path(config.get("output_dir", "video/outputs"))
    if checkpoint_path is None:
        default_ckpt = output_dir / "best_model.pt"
        if default_ckpt.is_file():
            checkpoint_path = default_ckpt
        else:
            raise FileNotFoundError(f"Trained video checkpoint not found: {default_ckpt}")

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Trained video checkpoint not found: {checkpoint_file}")
    logger.info("Loading model weights from: %s", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise ValueError("Video checkpoint does not contain a state dictionary.")

    model_config = {**config, "pretrained": False}
    model = build_model(model_config, device=device)
    model.load_state_dict(state_dict, strict=True)

    dataset, dataloader = create_dataloader(config, split=split, shuffle=False)
    if not len(dataset):
        raise ValueError(f"Evaluation split '{split}' contains no video samples.")

    frame_records, video_records = run_inference_on_split(model, dataloader, device)
    manipulation_methods = config.get(
        "manipulation_methods", ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    )
    metrics = compute_evaluation_breakdown(frame_records, video_records, manipulation_methods)

    # Print results table
    print_results_table(metrics, split=split)

    # Save metrics JSON to both output_dir and outputs/
    output_dir.mkdir(parents=True, exist_ok=True)
    json_targets = [output_dir / "metrics.json", Path("outputs/metrics.json")]
    for jt in json_targets:
        jt.parent.mkdir(parents=True, exist_ok=True)
        with open(jt, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # Save ROC plot
    plot_targets = [output_dir / "roc_curves.png", Path("outputs/roc_curves.png")]
    for pt in plot_targets:
        plot_roc_curves(frame_records, video_records, pt)

    return metrics


def main() -> None:
    """Parse CLI arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model on FF++.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="video/configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Split to evaluate on.",
    )
    args = parser.parse_args()

    evaluate_split(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
    )


if __name__ == "__main__":
    main()
