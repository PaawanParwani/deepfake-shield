"""Training pipeline for EfficientNet deepfake video detection.

Trains the binary classifier using cosine annealing LR schedule, mixed precision,
Weights & Biases telemetry, and dual-level (frame-level and video-level) AUC evaluation.
Saves the best model checkpoint and final evaluation metrics to JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import yaml

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import wandb
except ImportError:
    wandb = None

try:
    from sklearn.metrics import accuracy_score, roc_auc_score
except ImportError:
    roc_auc_score = None
    accuracy_score = None

from video.src.dataset import create_dataloader, set_seed
from video.src.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Safely compute ROC AUC score with graceful fallback if single class.

    Args:
        labels: Ground truth binary labels (0 or 1).
        scores: Predicted probabilities for the positive class (class 1).

    Returns:
        ROC AUC score as a float between 0.0 and 1.0.
    """
    if len(np.unique(labels)) < 2:
        return 0.5

    if roc_auc_score is not None:
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return 0.5

    # Manual trapezoidal approximation fallback
    descending_idx = np.argsort(-scores)
    sorted_labels = labels[descending_idx]
    n_pos = np.sum(sorted_labels == 1)
    n_neg = np.sum(sorted_labels == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    cum_pos = np.cumsum(sorted_labels == 1)
    cum_neg = np.cumsum(sorted_labels == 0)
    tpr = np.concatenate([[0], cum_pos / n_pos])
    fpr = np.concatenate([[0], cum_neg / n_neg])
    return float(np.trapz(tpr, fpr))


def compute_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute standard classification accuracy.

    Args:
        labels: Ground truth labels.
        preds: Predicted binary labels.

    Returns:
        Accuracy fraction.
    """
    if accuracy_score is not None:
        return float(accuracy_score(labels, preds))
    return float(np.mean(labels == preds))


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, float]:
    """Execute one training epoch with mixed precision.

    Args:
        model: DeepfakeClassifier model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: AdamW optimizer.
        scaler: GradScaler for AMP.
        device: Active torch device.
        epoch: Current epoch index (1-based).

    Returns:
        Tuple of (mean_loss, frame_accuracy, frame_auc).
    """
    model.train()
    running_loss = 0.0
    all_labels: List[int] = []
    all_scores: List[float] = []
    all_preds: List[int] = []

    use_amp = (device.type == "cuda")

    for batch in tqdm(dataloader, desc=f"Train epoch {epoch}", unit="batch"):
        images = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()

        with autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()

        all_labels.extend(labels.cpu().numpy().tolist())
        all_scores.extend(probs.tolist())
        all_preds.extend(preds.tolist())

    n_samples = max(1, len(all_labels))
    epoch_loss = running_loss / n_samples
    epoch_acc = compute_accuracy(np.array(all_labels), np.array(all_preds))
    epoch_auc = compute_roc_auc(np.array(all_labels), np.array(all_scores))

    return epoch_loss, epoch_acc, epoch_auc


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation/test set computing frame- and video-level metrics.

    Args:
        model: DeepfakeClassifier model.
        dataloader: Evaluation dataloader.
        criterion: Loss function.
        device: Active torch device.

    Returns:
        Dictionary containing val_loss, frame_acc, frame_auc, video_acc, and video_auc.
    """
    model.eval()
    running_loss = 0.0
    all_labels: List[int] = []
    all_scores: List[float] = []
    all_preds: List[int] = []
    video_predictions: Dict[str, List[float]] = {}
    video_targets: Dict[str, int] = {}

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", unit="batch"):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            video_ids = batch[2]

            with autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            lbls = labels.cpu().numpy()

            all_labels.extend(lbls.tolist())
            all_scores.extend(probs.tolist())
            all_preds.extend(preds.tolist())

            for vid, prob, lbl in zip(video_ids, probs, lbls):
                if vid not in video_predictions:
                    video_predictions[vid] = []
                    video_targets[vid] = int(lbl)
                video_predictions[vid].append(float(prob))

    n_samples = max(1, len(all_labels))
    frame_loss = running_loss / n_samples
    frame_acc = compute_accuracy(np.array(all_labels), np.array(all_preds))
    frame_auc = compute_roc_auc(np.array(all_labels), np.array(all_scores))

    # Video-level aggregation: mean frame probability per video
    vid_labels_list = []
    vid_scores_list = []
    for vid, p_list in video_predictions.items():
        vid_labels_list.append(video_targets[vid])
        vid_scores_list.append(float(np.mean(p_list)))

    vid_labels_arr = np.array(vid_labels_list)
    vid_scores_arr = np.array(vid_scores_list)
    vid_preds_arr = (vid_scores_arr >= 0.5).astype(int)

    video_auc = compute_roc_auc(vid_labels_arr, vid_scores_arr)
    video_acc = compute_accuracy(vid_labels_arr, vid_preds_arr)

    return {
        "val_loss": frame_loss,
        "frame_acc": frame_acc,
        "frame_auc": frame_auc,
        "video_acc": video_acc,
        "video_auc": video_auc,
    }


def train_model(
    config_path: str | Path,
    no_wandb: bool = False,
) -> Dict[str, Any]:
    """Train DeepfakeClassifier according to the specified YAML configuration.

    Args:
        config_path: Path to configuration file.
        no_wandb: If True, disable Weights & Biases logging.

    Returns:
        Dictionary of final metrics.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device_setting = config.get("device", "auto")
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)
    logger.info("Training device: %s", device)

    # Datasets and loaders
    train_dataset, train_loader = create_dataloader(config, split="train", shuffle=True)
    val_dataset, val_loader = create_dataloader(config, split="val", shuffle=False)

    logger.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        labels = {sample[1] for sample in dataset.samples}
        if labels != {0, 1}:
            raise ValueError(
                f"The {split_name} split must contain real and fake videos; "
                f"found labels {sorted(labels)}."
            )

    expected_methods = set(config.get("manipulation_methods", []))
    train_methods = {
        sample[3] for sample in train_dataset.samples if sample[1] == 1
    }
    missing_methods = expected_methods - train_methods
    if missing_methods:
        raise ValueError(
            "Training crops are missing configured manipulation methods: "
            f"{', '.join(sorted(missing_methods))}. Re-extract the official train split."
        )

    # Model, optimizer, scheduler, criterion
    model = build_model(config, device=device)
    lr = float(config.get("lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    optimizer = model.get_optimizer(lr=lr, weight_decay=weight_decay)

    epochs = int(config.get("epochs", 5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # Outputs
    output_dir = Path(config.get("output_dir", "video/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_model.pt"

    # W&B initialization
    wandb_run = None
    if not no_wandb and wandb is not None:
        try:
            wandb_run = wandb.init(
                project=config.get("wandb_project", "human-verification-video"),
                config=config,
            )
            logger.info("Weights & Biases logging initialized.")
        except Exception as e:
            logger.warning("Could not initialize W&B (%s). Proceeding offline.", e)

    best_val_auc = -1.0
    best_metrics: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_auc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f Acc: %.4f AUC: %.4f | Val Loss: %.4f Frame AUC: %.4f Video AUC: %.4f Video Acc: %.4f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            train_auc,
            val_metrics["val_loss"],
            val_metrics["frame_auc"],
            val_metrics["video_auc"],
            val_metrics["video_acc"],
        )

        log_payload = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "train/auc": train_auc,
            "val/loss": val_metrics["val_loss"],
            "val/frame_acc": val_metrics["frame_acc"],
            "val/frame_auc": val_metrics["frame_auc"],
            "val/video_acc": val_metrics["video_acc"],
            "val/video_auc": val_metrics["video_auc"],
            "lr": current_lr,
        }

        if wandb_run is not None:
            wandb.log(log_payload)

        # Track best model checkpoint by video-level AUC (or frame AUC)
        eval_score = val_metrics["video_auc"]
        if eval_score > best_val_auc:
            best_val_auc = eval_score
            best_metrics = {
                "best_epoch": epoch,
                "best_video_auc": float(best_val_auc),
                "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                "train_loss": float(train_loss),
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "metrics": best_metrics,
                },
                best_checkpoint_path,
            )
            logger.info("Saved new best checkpoint to %s with Video AUC: %.4f", best_checkpoint_path, best_val_auc)

    # Save metrics.json in both output_dir and outputs/
    metrics_json_paths = [
        output_dir / "metrics.json",
        Path("outputs/metrics.json"),
    ]
    for m_path in metrics_json_paths:
        m_path.parent.mkdir(parents=True, exist_ok=True)
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)

    logger.info("Saved final metrics to %s", metrics_json_paths[0])

    if wandb_run is not None:
        wandb.finish()

    return best_metrics


def main() -> None:
    """Parse CLI arguments and run training."""
    parser = argparse.ArgumentParser(description="Train FaceForensics++ deepfake classifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="video/configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=False,
        help="Disable Weights & Biases logging.",
    )
    args = parser.parse_args()

    train_model(config_path=args.config, no_wandb=args.no_wandb)


if __name__ == "__main__":
    main()
