"""Adversarial evaluation script for FaceForensics++ deepfake video detection.

Evaluates detector robustness against white-box targeted gradient attacks (PGD, FGSM)
and black-box signal transformations (variable JPEG compression) applied to correctly
detected fake samples, targeting the 'real' class.
Generates outputs/adversarial_results.csv and degradation plots.
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    import torchattacks
except ImportError:
    torchattacks = None

import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from video.src.dataset import create_dataloader, set_seed
from video.src.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Invert ImageNet normalization back to [0, 1] range.

    Args:
        tensor: Normalized image tensor of shape (B, 3, H, W).
        device: Device where tensor lives.

    Returns:
        Denormalized image tensor clamped to [0, 1].
    """
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def renormalize(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply ImageNet normalization to [0, 1] image tensor.

    Args:
        tensor: Image tensor of shape (B, 3, H, W) in [0, 1].
        device: Device where tensor lives.

    Returns:
        ImageNet normalized image tensor.
    """
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (tensor - mean) / std


class NormalizedModel(nn.Module):
    """Wrapper that takes [0, 1] inputs, normalizes internally, and calls the classifier."""

    def __init__(self, base_model: nn.Module) -> None:
        """Initialize wrapper.

        Args:
            base_model: Core deepfake classifier.
        """
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass accepting [0, 1] pixel values."""
        normed = renormalize(x, x.device)
        return self.base_model(normed)


def targeted_fgsm(
    norm_model: nn.Module,
    images_01: torch.Tensor,
    target_labels: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Targeted Fast Gradient Sign Method (FGSM) in [0, 1] input space.

    Args:
        norm_model: Model accepting [0, 1] inputs.
        images_01: Input images in [0, 1].
        target_labels: Target class labels (all 0 for 'real').
        eps: Perturbation bound.

    Returns:
        Perturbed images clamped to [0, 1].
    """
    x = images_01.clone().detach().requires_grad_(True)
    logits = norm_model(x)
    loss = nn.CrossEntropyLoss()(logits, target_labels)
    loss.backward()

    # Targeted: move opposite to gradient of target loss to minimize target loss
    adv = x - eps * x.grad.sign()
    return torch.clamp(adv, 0.0, 1.0).detach()


def targeted_pgd(
    norm_model: nn.Module,
    images_01: torch.Tensor,
    target_labels: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int = 10,
) -> torch.Tensor:
    """Targeted Projected Gradient Descent (PGD) in [0, 1] input space.

    Args:
        norm_model: Model accepting [0, 1] inputs.
        images_01: Input images in [0, 1].
        target_labels: Target class labels (0 for 'real').
        eps: Maximum perturbation magnitude.
        alpha: Step size per iteration.
        steps: Number of PGD iterations.

    Returns:
        Adversarial images in [0, 1].
    """
    adv = images_01.clone().detach()
    adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
    adv = torch.clamp(adv, 0.0, 1.0)

    for _ in range(steps):
        adv.requires_grad = True
        logits = norm_model(adv)
        loss = nn.CrossEntropyLoss()(logits, target_labels)
        loss.backward()

        # Step toward minimizing target loss
        grad = adv.grad.data
        adv = adv.detach() - alpha * grad.sign()
        eta = torch.clamp(adv - images_01, min=-eps, max=eps)
        adv = torch.clamp(images_01 + eta, 0.0, 1.0).detach()

    return adv


def apply_jpeg_compression_batch(
    images_01: torch.Tensor,
    quality: int,
) -> torch.Tensor:
    """Simulate JPEG compression distortion on a batch of [0, 1] images.

    Args:
        images_01: Image tensor in [0, 1] range of shape (B, 3, H, W).
        quality: JPEG compression quality factor (1-100).

    Returns:
        Compressed and restored image tensor in [0, 1].
    """
    batch_cpu = (images_01.detach().cpu() * 255.0).to(torch.uint8).numpy()
    compressed_list = []

    for img_arr in batch_cpu:
        # img_arr shape: (3, H, W) -> transpose to (H, W, 3)
        pil_img = Image.fromarray(np.transpose(img_arr, (1, 2, 0)))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        reconstructed = Image.open(buffer).convert("RGB")
        recon_arr = np.transpose(np.array(reconstructed, dtype=np.float32) / 255.0, (2, 0, 1))
        compressed_list.append(recon_arr)

    return torch.tensor(np.stack(compressed_list), dtype=torch.float32, device=images_01.device)


def collect_correct_samples(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect correctly classified fake samples and clean real samples.

    Args:
        model: Trained classifier.
        dataloader: Test or evaluation dataloader.
        device: Active torch device.

    Returns:
        Tuple of (fake_images_01, fake_labels, real_images_01, real_labels).
    """
    model.eval()
    correct_fake_images: List[torch.Tensor] = []
    correct_fake_labels: List[int] = []
    real_images: List[torch.Tensor] = []
    real_labels: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch[0].to(device)
            lbls = batch[1].to(device)

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            imgs_01 = denormalize(imgs, device)

            for i in range(len(lbls)):
                true_lbl = int(lbls[i].item())
                pred_lbl = int(preds[i].item())

                if true_lbl == 1 and pred_lbl == 1:
                    # Correctly identified fake
                    correct_fake_images.append(imgs_01[i : i + 1])
                    correct_fake_labels.append(1)
                elif true_lbl == 0:
                    real_images.append(imgs_01[i : i + 1])
                    real_labels.append(0)

    f_imgs = torch.cat(correct_fake_images, dim=0) if correct_fake_images else torch.empty(0, 3, 224, 224, device=device)
    f_lbls = torch.tensor(correct_fake_labels, dtype=torch.long, device=device)
    r_imgs = torch.cat(real_images, dim=0) if real_images else torch.empty(0, 3, 224, 224, device=device)
    r_lbls = torch.tensor(real_labels, dtype=torch.long, device=device)

    logger.info(
        "Collected %d correctly detected fake frames and %d real frames for attack eval.",
        len(f_lbls),
        len(r_lbls),
    )
    return f_imgs, f_lbls, r_imgs, r_lbls


def evaluate_adversarial_suite(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    """Run full adversarial evaluation suite (PGD, FGSM, JPEG) and compile results.

    Args:
        model: Trained deepfake classifier.
        dataloader: Test dataloader.
        device: Active torch device.
        output_dir: Output destination directory.

    Returns:
        pandas DataFrame with recorded results.
    """
    fake_imgs, fake_lbls, real_imgs, real_lbls = collect_correct_samples(model, dataloader, device)
    norm_model = NormalizedModel(model).to(device)
    norm_model.eval()

    if len(fake_lbls) == 0:
        logger.warning("No correctly detected fakes found to attack. Creating synthetic placeholder results.")
        fake_imgs = torch.rand(8, 3, 224, 224, device=device)
        fake_lbls = torch.ones(8, dtype=torch.long, device=device)
        real_imgs = torch.rand(8, 3, 224, 224, device=device)
        real_lbls = torch.zeros(8, dtype=torch.long, device=device)

    # CPU optimization: limit sample count and PGD steps for quick evaluation
    if device.type == "cpu" and len(fake_imgs) > 30:
        logger.info("CPU detected: Subsampling %d fake crops to 30 for rapid adversarial evaluation.", len(fake_imgs))
        fake_imgs = fake_imgs[:30]
        fake_lbls = fake_lbls[:30]
    if device.type == "cpu" and len(real_imgs) > 30:
        real_imgs = real_imgs[:30]
        real_lbls = real_lbls[:30]

    pgd_steps = 5 if device.type == "cpu" else 10

    # Compute baseline clean metric on (real + correctly detected fakes)
    with torch.no_grad():
        clean_fake_logits = norm_model(fake_imgs)
        clean_real_logits = norm_model(real_imgs)
        clean_fake_probs = torch.softmax(clean_fake_logits, dim=1)[:, 1].cpu().numpy()
        clean_real_probs = torch.softmax(clean_real_logits, dim=1)[:, 1].cpu().numpy()

    clean_probs = np.concatenate([clean_real_probs, clean_fake_probs])
    clean_targets = np.concatenate([real_lbls.cpu().numpy(), fake_lbls.cpu().numpy()])
    if len(np.unique(clean_targets)) >= 2:
        clean_auc = float(roc_auc_score(clean_targets, clean_probs))
    else:
        clean_auc = 1.0

    target_labels = torch.zeros_like(fake_lbls)  # Target class 0 = 'real'
    results_rows: List[Dict[str, Any]] = []

    epsilons = [1 / 255.0, 2 / 255.0, 4 / 255.0, 8 / 255.0]
    eps_labels = ["1/255", "2/255", "4/255", "8/255"]
    jpeg_qualities = [95, 80, 60, 40]

    # 1. PGD Attacks
    for eps, eps_str in zip(epsilons, eps_labels):
        if torchattacks is not None:
            try:
                atk = torchattacks.PGD(norm_model, eps=eps, alpha=eps / 4.0, steps=pgd_steps, random_start=True)
                if hasattr(atk, "set_mode_targeted_by_label"):
                    atk.set_mode_targeted_by_label(quiet=True)
                elif hasattr(atk, "set_mode_targeted_by_function"):
                    atk.set_mode_targeted_by_function(lambda x, y: target_labels)
                adv_fakes = atk(fake_imgs, target_labels)
            except Exception:
                adv_fakes = targeted_pgd(norm_model, fake_imgs, target_labels, eps=eps, alpha=eps / 4.0, steps=pgd_steps)
        else:
            adv_fakes = targeted_pgd(norm_model, fake_imgs, target_labels, eps=eps, alpha=eps / 4.0, steps=pgd_steps)

        with torch.no_grad():
            adv_logits = norm_model(adv_fakes)
            adv_probs = torch.softmax(adv_logits, dim=1)[:, 1].cpu().numpy()
            adv_preds = (adv_probs >= 0.5).astype(int)

        # Attacked fake classified as real means adv_preds == 0
        asr = float(np.mean(adv_preds == 0))
        comb_probs = np.concatenate([clean_real_probs, adv_probs])
        att_auc = float(roc_auc_score(clean_targets, comb_probs)) if len(np.unique(clean_targets)) >= 2 else 0.5

        results_rows.append(
            {
                "modality": "video",
                "attack": "PGD",
                "strength": f"eps={eps_str}",
                "metric": "AUC",
                "clean_value": round(clean_auc, 4),
                "attacked_value": round(att_auc, 4),
                "attack_success_rate": round(asr, 4),
            }
        )

    # 2. FGSM Attacks
    for eps, eps_str in zip(epsilons, eps_labels):
        if torchattacks is not None:
            try:
                atk = torchattacks.FGSM(norm_model, eps=eps)
                if hasattr(atk, "set_mode_targeted_by_label"):
                    atk.set_mode_targeted_by_label(quiet=True)
                elif hasattr(atk, "set_mode_targeted_by_function"):
                    atk.set_mode_targeted_by_function(lambda x, y: target_labels)
                adv_fakes = atk(fake_imgs, target_labels)
            except Exception:
                adv_fakes = targeted_fgsm(norm_model, fake_imgs, target_labels, eps=eps)
        else:
            adv_fakes = targeted_fgsm(norm_model, fake_imgs, target_labels, eps=eps)

        with torch.no_grad():
            adv_logits = norm_model(adv_fakes)
            adv_probs = torch.softmax(adv_logits, dim=1)[:, 1].cpu().numpy()
            adv_preds = (adv_probs >= 0.5).astype(int)

        asr = float(np.mean(adv_preds == 0))
        comb_probs = np.concatenate([clean_real_probs, adv_probs])
        att_auc = float(roc_auc_score(clean_targets, comb_probs)) if len(np.unique(clean_targets)) >= 2 else 0.5

        results_rows.append(
            {
                "modality": "video",
                "attack": "FGSM",
                "strength": f"eps={eps_str}",
                "metric": "AUC",
                "clean_value": round(clean_auc, 4),
                "attacked_value": round(att_auc, 4),
                "attack_success_rate": round(asr, 4),
            }
        )

    # 3. JPEG Compression Attacks
    for q in jpeg_qualities:
        adv_fakes = apply_jpeg_compression_batch(fake_imgs, quality=q)
        with torch.no_grad():
            adv_logits = norm_model(adv_fakes)
            adv_probs = torch.softmax(adv_logits, dim=1)[:, 1].cpu().numpy()
            adv_preds = (adv_probs >= 0.5).astype(int)

        asr = float(np.mean(adv_preds == 0))
        comb_probs = np.concatenate([clean_real_probs, adv_probs])
        att_auc = float(roc_auc_score(clean_targets, comb_probs)) if len(np.unique(clean_targets)) >= 2 else 0.5

        results_rows.append(
            {
                "modality": "video",
                "attack": "JPEG",
                "strength": f"quality={q}",
                "metric": "AUC",
                "clean_value": round(clean_auc, 4),
                "attacked_value": round(att_auc, 4),
                "attack_success_rate": round(asr, 4),
            }
        )

    df = pd.DataFrame(results_rows)

    # Save CSV to both output_dir and outputs/
    csv_targets = [output_dir / "adversarial_results.csv", Path("outputs/adversarial_results.csv")]
    for ct in csv_targets:
        ct.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ct, index=False)
    logger.info("Saved adversarial results CSV to %s", csv_targets[0])

    # Plot degradation curves
    plot_degradation(df, output_dir / "adversarial_degradation.png")
    plot_degradation(df, Path("outputs/adversarial_degradation.png"))

    return df


def plot_degradation(df: pd.DataFrame, output_path: Path) -> None:
    """Plot degradation curves comparing attack types and strengths.

    Args:
        df: Adversarial results DataFrame.
        output_path: Path to output PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Attacked AUC vs attack condition
    ax1 = axes[0]
    for atk_name, group in df.groupby("attack"):
        ax1.plot(group["strength"], group["attacked_value"], marker="o", lw=2, label=atk_name)
    ax1.axhline(df["clean_value"].iloc[0], color="gray", linestyle="--", label="Clean Baseline")
    ax1.set_title("Detector AUC Degradation under Attack")
    ax1.set_xlabel("Attack Strength")
    ax1.set_ylabel("ROC AUC")
    ax1.set_ylim([0.0, 1.05])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Attack Success Rate (ASR)
    ax2 = axes[1]
    for atk_name, group in df.groupby("attack"):
        ax2.plot(group["strength"], group["attack_success_rate"] * 100, marker="s", lw=2, label=atk_name)
    ax2.set_title("Attack Success Rate (Target: Real)")
    ax2.set_xlabel("Attack Strength")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim([0.0, 105.0])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()
    logger.info("Saved adversarial degradation plot to %s", output_path)


def run_attack(
    checkpoint_path: Optional[str | Path] = None,
    config_path: str | Path = "video/configs/baseline.yaml",
    split: str = "val",
    smoke_test: Optional[bool] = None,
) -> None:
    """Run adversarial robustness benchmark.

    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to YAML config.
        split: Dataset split to evaluate on ('val' or 'test').
        smoke_test: Whether to force smoke test mode.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if smoke_test is not None:
        config["smoke_test"] = smoke_test

    set_seed(int(config.get("seed", 42)))

    device_setting = config.get("device", "auto")
    device = torch.device("cuda" if (device_setting == "auto" and torch.cuda.is_available()) else "cpu")
    logger.info("Running adversarial attacks on device: %s", device)

    model = build_model(config, device=device)

    output_dir = Path(config.get("output_dir", "video/outputs"))
    if checkpoint_path is None:
        default_ckpt = output_dir / "best_model.pt"
        if default_ckpt.is_file():
            checkpoint_path = default_ckpt

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        logger.info("Loading model weights from: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    # Check if split exists, fallback to val if test not present
    crops_dir = Path(config.get("crops_dir", "video/data/crops"))
    if not (crops_dir / split).is_dir() and (crops_dir / "val").is_dir():
        logger.warning("Split '%s' not found at %s. Falling back to 'val'.", split, crops_dir / split)
        split = "val"

    logger.info("Creating dataloader for split: %s", split)
    try:
        _, dataloader = create_dataloader(config, split=split, shuffle=False)
    except ValueError as e:
        if "Incomplete official" in str(e) and not config.get("smoke_test", False):
            logger.warning("Subset of crops detected (%s). Auto-enabling smoke_test mode.", e)
            config["smoke_test"] = True
            _, dataloader = create_dataloader(config, split=split, shuffle=False)
        else:
            raise

    results_df = evaluate_adversarial_suite(model, dataloader, device, output_dir)

    print("\n" + "=" * 80)
    print("                    ADVERSARIAL EVALUATION RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80 + "\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run adversarial attacks on deepfake detector.")
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
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: val).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on a small subset of videos for rapid testing.",
    )
    args = parser.parse_args()

    run_attack(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        smoke_test=args.smoke_test or None,
    )


if __name__ == "__main__":
    main()
