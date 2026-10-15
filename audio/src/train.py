"""Training pipeline for Audio Deepfake Detection with Wav2Vec2.

Loads configuration from YAML, initializes datasets and dataloaders, constructs
the Wav2Vec2 classifier with class-weighted cross-entropy loss and cosine
learning rate schedule, trains with torch.cuda.amp mixed precision, evaluates
Equal Error Rate (EER) each epoch, logs to Weights & Biases, and saves checkpoints
and final metrics to outputs/metrics.json.
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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
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
    """Set random seed across all libraries for deterministic execution.

    Args:
        seed: Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and optimal classification threshold.

    Computes the False Positive Rate (FPR) and False Negative Rate (FNR)
    across varying decision thresholds, finding the point where FPR == FNR.

    Args:
        labels: 1D ground truth binary labels (0 = bonafide, 1 = spoof).
        scores: 1D prediction scores or probabilities for spoof class (1).

    Returns:
        Tuple of (eer, threshold).
    """
    # Guard against single-class arrays in tiny subsets
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning("Only one class present in evaluation labels. Returning default EER=0.5.")
        return 0.5, 0.5

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = fpr - fnr

    # Find the nearest index
    idx = int(np.nanargmin(np.abs(diff)))
    eer_approx = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh_approx = float(thresholds[idx])

    # Interpolate to find exact root if diff crosses zero
    try:
        min_diff = np.min(diff)
        max_diff = np.max(diff)
        if min_diff <= 0.0 <= max_diff:
            # Sort by diff for monotonic interpolation
            sort_order = np.argsort(diff)
            sorted_diff = diff[sort_order]
            sorted_fpr = fpr[sort_order]
            sorted_thresh = thresholds[sort_order]

            # Remove duplicates for interp1d
            unique_diff, unique_idx = np.unique(sorted_diff, return_index=True)
            if len(unique_diff) > 1:
                eer_interp = interp1d(unique_diff, sorted_fpr[unique_idx], kind="linear")
                thresh_interp = interp1d(unique_diff, sorted_thresh[unique_idx], kind="linear")
                eer_root = float(eer_interp(0.0))
                thresh_root = float(thresh_interp(0.0))
                return eer_root, thresh_root
    except Exception as e:
        logger.debug(f"Interpolation for EER fell back to discrete nearest point: {e}")

    return eer_approx, thresh_approx


def evaluate_split(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate classifier performance on a given dataloader.

    Args:
        model: PyTorch classifier model.
        dataloader: Evaluation DataLoader.
        criterion: Loss function.
        device: Execution device.

    Returns:
        Dictionary of computed metrics: loss, accuracy, eer, auc, threshold.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels: list[int] = []
    all_scores: list[float] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            logits = model(waveforms)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=-1)
            # Probability of spoof (class 1)
            spoof_probs = probs[:, 1].detach().cpu().numpy()
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()

            total_loss += loss.item() * waveforms.size(0)
            total_samples += waveforms.size(0)

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_scores.extend(spoof_probs.tolist())
            all_preds.extend(preds.tolist())

    avg_loss = total_loss / max(total_samples, 1)
    labels_np = np.array(all_labels)
    scores_np = np.array(all_scores)
    preds_np = np.array(all_preds)

    acc = float(accuracy_score(labels_np, preds_np)) if total_samples > 0 else 0.0
    try:
        auc = float(roc_auc_score(labels_np, scores_np)) if len(np.unique(labels_np)) > 1 else 0.5
    except Exception:
        auc = 0.5

    eer, thresh = compute_eer(labels_np, scores_np)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "eer": eer,
        "auc": auc,
        "threshold": thresh,
    }


def train_pipeline(
    config: Dict[str, Any],
    no_wandb: bool = False,
    override_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute end-to-end model training and validation.

    Args:
        config: Configuration dictionary loaded from YAML.
        no_wandb: If True, disable Weights & Biases logging.
        override_epochs: Optional epoch count override.

    Returns:
        Dictionary containing best evaluation metrics.
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    output_dir = Path(config.get("output_dir", "audio/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine hardware device
    device_cfg = config.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")

    # W&B experiment tracking setup
    use_wandb = not no_wandb
    if use_wandb:
        try:
            import wandb

            wandb.init(
                project=config.get("wandb_project", "human-verification-audio"),
                config=config,
            )
            logger.info("Weights & Biases initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize W&B ({e}). Running without W&B.")
            use_wandb = False

    data_dir = Path(config.get("data_dir", "audio/data/LA"))
    protocol_dir = Path(config.get("protocol_dir", "audio/data/LA/ASVspoof2019_LA_cm_protocols"))
    smoke_test = config.get("smoke_test", False)
    smoke_test_size = config.get("smoke_test_size", 500)

    # Auto-detect data format and handle missing data gracefully
    from audio.src.dataset import detect_data_format
    temp_dir_obj = None
    data_format = detect_data_format(data_dir)

    if data_format == "parquet":
        logger.info("Using parquet format (HuggingFace download). Protocol files not needed.")
    elif not protocol_dir.exists():
        logger.warning(
            f"Protocol directory {protocol_dir} not found. "
            "Generating temporary synthetic dataset for end-to-end execution."
        )
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_root = Path(temp_dir_obj.name)
        data_dir, protocol_dir = create_synthetic_dataset(
            temp_root, num_samples=min(smoke_test_size, 40)
        )
        smoke_test = True

    sample_rate = config.get("sample_rate", 16000)
    max_duration = config.get("max_duration", 4.0)
    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 2)
    epochs = override_epochs if override_epochs is not None else config.get("epochs", 5)

    # Build datasets
    train_dataset = ASVSpoof2019LADataset(
        data_dir=data_dir,
        protocol_dir=protocol_dir,
        split="train",
        sample_rate=sample_rate,
        max_duration=max_duration,
        is_train=True,
        smoke_test=smoke_test,
        smoke_test_size=smoke_test_size,
    )

    dev_dataset = ASVSpoof2019LADataset(
        data_dir=data_dir,
        protocol_dir=protocol_dir,
        split="dev",
        sample_rate=sample_rate,
        max_duration=max_duration,
        is_train=False,
        smoke_test=smoke_test,
        smoke_test_size=smoke_test_size,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    dev_loader = create_dataloader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Compute class weights to counterbalance ASVspoof class imbalance
    bonafide_count = train_dataset._count_label(0)
    spoof_count = train_dataset._count_label(1)
    total_count = len(train_dataset)

    if bonafide_count > 0 and spoof_count > 0:
        w0 = total_count / (2.0 * bonafide_count)
        w1 = total_count / (2.0 * spoof_count)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)

    logger.info(
        f"Class balance: bonafide={bonafide_count}, spoof={spoof_count}. "
        f"Weights: {class_weights.tolist()}"
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Build model
    model = build_model(
        model_name=config.get("model_name", "facebook/wav2vec2-base"),
        num_classes=config.get("num_classes", 2),
        freeze_backbone=config.get("freeze_backbone", True),
        device=device,
    )

    optimizer = model.get_optimizer(
        lr_backbone=float(config.get("lr_backbone", 1e-5)),
        lr_head=float(config.get("lr_head", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-7
    )

    # Mixed precision scaler (enabled on CUDA)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_eer = float("inf")
    best_metrics: Dict[str, Any] = {}
    best_checkpoint_path = output_dir / "best_model.pt"

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_samples = 0

        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(waveforms)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * waveforms.size(0)
            train_samples += waveforms.size(0)

        scheduler.step()
        train_loss = running_loss / max(train_samples, 1)

        # Validation on dev split
        dev_metrics = evaluate_split(model, dev_loader, criterion, device)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_metrics['loss']:.4f} | "
            f"Dev Acc: {dev_metrics['accuracy']:.4f} | "
            f"Dev EER: {dev_metrics['eer']:.4f} | "
            f"Dev AUC: {dev_metrics['auc']:.4f}"
        )

        log_payload = {
            "epoch": epoch,
            "train/loss": train_loss,
            "dev/loss": dev_metrics["loss"],
            "dev/accuracy": dev_metrics["accuracy"],
            "dev/eer": dev_metrics["eer"],
            "dev/auc": dev_metrics["auc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if use_wandb:
            import wandb

            wandb.log(log_payload)

        # Track and save best model by EER
        if dev_metrics["eer"] <= best_eer:
            best_eer = dev_metrics["eer"]
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                **dev_metrics,
                "model_name": config.get("model_name"),
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eer": best_eer,
                    "metrics": dev_metrics,
                    "config": config,
                },
                best_checkpoint_path,
            )
            logger.info(f"Saved new best checkpoint to {best_checkpoint_path} (EER: {best_eer:.4f})")

    # Save final metrics JSON
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    logger.info(f"Saved best metrics to {metrics_file}")

    if use_wandb:
        import wandb

        wandb.finish()

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    return best_metrics


def main() -> None:
    """CLI interface for training the audio deepfake detector."""
    parser = argparse.ArgumentParser(description="Train Audio Deepfake Detector (Wav2Vec2)")
    parser.add_argument(
        "--config",
        type=str,
        default="audio/configs/baseline.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    best_metrics = train_pipeline(config, no_wandb=args.no_wandb, override_epochs=args.epochs)
    logger.info(f"Training run completed. Best metrics: {best_metrics}")


if __name__ == "__main__":
    main()
