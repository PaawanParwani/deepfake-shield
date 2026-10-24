"""Adversarial robustness evaluation for Audio Deepfake Detector.

Implements white-box Projected Gradient Descent (PGD L-inf) adversarial attack
targeting the 'bonafide' class on correctly classified spoof samples, as well as
additive Gaussian noise perturbation across diverse Signal-to-Noise Ratios (SNR).
Evaluates Degradation in Equal Error Rate (EER) and Attack Success Rate (ASR),
generating outputs/adversarial_results.csv and visualization curves.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Relative imports or package imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from .dataset import ASVSpoof2019LADataset, create_dataloader, create_synthetic_dataset, detect_data_format
    from .model import build_model
except ImportError:
    from dataset import ASVSpoof2019LADataset, create_dataloader, create_synthetic_dataset, detect_data_format
    from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Equal Error Rate (EER) from labels and spoof prediction scores.

    Args:
        labels: 1D binary labels (0 = bonafide, 1 = spoof).
        scores: 1D prediction scores for spoof class.

    Returns:
        EER value as float in [0.0, 1.0].
    """
    if len(np.unique(labels)) < 2:
        return 0.5

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = fpr - fnr
    idx = int(np.nanargmin(np.abs(diff)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)

    try:
        if np.min(diff) <= 0.0 <= np.max(diff):
            order = np.argsort(diff)
            u_diff, u_indices = np.unique(diff[order], return_index=True)
            if len(u_diff) > 1:
                fn = interp1d(u_diff, fpr[order][u_indices], kind="linear")
                eer = float(fn(0.0))
    except Exception:
        pass

    return eer


class WaveformPGDAttack:
    """Projected Gradient Descent (PGD) L-infinity attack on raw 1D waveforms.

    Generates targeted adversarial perturbations to force spoofed audio
    clips to be misclassified as bonafide speech (target class = 0).
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.001,
        alpha: Optional[float] = None,
        steps: int = 10,
        target_class: int = 0,
    ) -> None:
        """Initialize the PGD attack on waveform audio.

        Args:
            model: PyTorch audio classification model.
            eps: Maximum L-infinity perturbation norm.
            alpha: Step size per iteration. Defaults to eps / 4.
            steps: Number of PGD attack iterations.
            target_class: Target class to fool the model into predicting (0 = bonafide).
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha if alpha is not None else max(eps / 4.0, 1e-6)
        self.steps = steps
        self.target_class = target_class

    def perturbation(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Generate targeted adversarial waveforms within L-inf ball of radius eps.

        Args:
            waveforms: Clean input waveform tensor of shape (batch_size, seq_len).

        Returns:
            Adversarial waveform tensor of identical shape.
        """
        original = waveforms.clone().detach()
        # Random initial uniform perturbation
        noise = torch.empty_like(waveforms).uniform_(-self.eps, self.eps)
        perturbed = (waveforms + noise).detach()

        target_labels = torch.full(
            (waveforms.size(0),), self.target_class, dtype=torch.long, device=waveforms.device
        )

        for _ in range(self.steps):
            perturbed.requires_grad = True
            logits = self.model(perturbed)

            # Minimize loss with respect to the target class (targeted attack)
            loss = F.cross_entropy(logits, target_labels)
            grad = torch.autograd.grad(loss, perturbed, retain_graph=False, create_graph=False)[0]

            # Gradient descent step towards target
            perturbed = perturbed.detach() - self.alpha * torch.sign(grad)

            # Project back onto L-infinity ball centered at original
            diff = torch.clamp(perturbed - original, min=-self.eps, max=self.eps)
            perturbed = (original + diff).detach()

        return perturbed


def add_gaussian_noise(waveforms: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add Gaussian noise to audio waveforms at a specified Signal-to-Noise Ratio (SNR).

    Args:
        waveforms: Audio waveforms of shape (batch_size, seq_len).
        snr_db: Desired SNR level in decibels (e.g., 40, 30, 20, 10 dB).

    Returns:
        Perturbed waveform tensor with specified SNR.
    """
    # Signal power along time dimension
    sig_power = torch.mean(waveforms ** 2, dim=-1, keepdim=True)
    # Required noise power based on SNR_dB = 10 * log10(P_signal / P_noise)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    sigma = torch.sqrt(noise_power + 1e-12)

    noise = torch.randn_like(waveforms) * sigma
    return waveforms + noise


def run_adversarial_evaluation(
    checkpoint_path: Optional[Union[str, Path]],
    config: Dict[str, Any],
    device_name: str = "auto",
) -> List[Dict[str, Any]]:
    """Execute adversarial evaluation across PGD epsilons and Gaussian noise SNRs.

    Args:
        checkpoint_path: Path to checkpoint .pt file or None.
        config: YAML configuration.
        device_name: Execution device.

    Returns:
        List of result records formatted for CSV export.
    """
    set_seed(config.get("seed", 42))

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Running adversarial attacks on device: {device}")

    # Build and load model
    model = build_model(
        model_name=config.get("model_name", "facebook/wav2vec2-base"),
        num_classes=config.get("num_classes", 2),
        freeze_backbone=False,
        device=device,
    )

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        logger.warning("No checkpoint found. Running attack evaluation on initialized weights.")

    model.eval()

    data_dir = Path(config.get("data_dir", "audio/data/LA"))
    protocol_dir = Path(config.get("protocol_dir", "audio/data/LA/ASVspoof2019_LA_cm_protocols"))
    smoke_test = config.get("smoke_test", False)
    smoke_test_size = config.get("smoke_test_size", 500)

    temp_dir_obj = None
    has_parquet = (data_dir / "data").exists() and bool(list((data_dir / "data").glob("*.parquet")))
    has_flac = any((data_dir / f"ASVspoof2019_LA_{s}" / "flac").exists() for s in ["train", "dev", "eval"])

    if has_parquet:
        logger.info("Using parquet format (HuggingFace download). Protocol files not needed.")
    elif has_flac and protocol_dir.exists():
        logger.info("Using raw FLAC format with protocol directory: %s", protocol_dir)
    else:
        logger.warning("Data not found. Using synthetic dataset for attack pipeline verification.")
        temp_dir_obj = tempfile.TemporaryDirectory()
        data_dir, protocol_dir = create_synthetic_dataset(
            Path(temp_dir_obj.name), num_samples=min(smoke_test_size, 40)
        )
        smoke_test = True

    eval_dataset = ASVSpoof2019LADataset(
        data_dir=data_dir,
        protocol_dir=protocol_dir,
        split="eval",
        sample_rate=config.get("sample_rate", 16000),
        max_duration=config.get("max_duration", 4.0),
        is_train=False,
        smoke_test=smoke_test,
        smoke_test_size=smoke_test_size,
    )

    eval_loader = create_dataloader(
        eval_dataset, batch_size=config.get("batch_size", 8), shuffle=False, num_workers=0
    )

    # 1. Clean evaluation to identify correctly-detected spoof samples
    all_waveforms: List[torch.Tensor] = []
    all_labels: List[int] = []
    clean_scores: List[float] = []
    clean_preds: List[int] = []

    for waveforms, labels in eval_loader:
        all_waveforms.append(waveforms)
        all_labels.extend(labels.tolist())

        with torch.no_grad():
            waveforms = waveforms.to(device)
            logits = model(waveforms)
            probs = torch.softmax(logits, dim=-1)
            spoof_probs = probs[:, 1].cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

            clean_scores.extend(spoof_probs)
            clean_preds.extend(preds)

    labels_arr = np.array(all_labels)
    clean_scores_arr = np.array(clean_scores)
    clean_preds_arr = np.array(clean_preds)

    clean_eer = compute_eer(labels_arr, clean_scores_arr)
    logger.info(f"Clean Evaluation complete. Clean EER: {clean_eer * 100:.2f}%")

    # Filter: take only correctly detected spoof samples (label=1, pred=1)
    spoof_indices = [
        i for i, (lbl, pred) in enumerate(zip(all_labels, clean_preds)) if lbl == 1 and pred == 1
    ]
    logger.info(
        f"Correctly detected spoof samples available for attack: {len(spoof_indices)} / "
        f"{sum(1 for l in all_labels if l == 1)}"
    )

    if not spoof_indices:
        # If no spoofs were correctly classified (e.g. untrained random model), take all spoofs
        spoof_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]
        logger.warning(f"No correctly detected spoofs found. Using all {len(spoof_indices)} spoofs.")

    # CPU optimization: limit sample count for gradient attacks to ensure fast completion
    if device.type == "cpu" and len(spoof_indices) > 25:
        logger.info("CPU detected: Subsampling %d spoof samples to 25 for rapid PGD evaluation.", len(spoof_indices))
        spoof_indices = spoof_indices[:25]
    pgd_steps = 5 if device.type == "cpu" else 10

    # Flatten waveforms to single indexed tensor list
    stacked_waveforms = torch.cat(all_waveforms, dim=0)

    # Output records
    records: List[Dict[str, Any]] = []

    # Attack parameter configurations
    pgd_epsilons = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    snr_levels_db = [40.0, 30.0, 20.0, 10.0]

    # --- PGD Attacks ---
    for eps in pgd_epsilons:
        attacker = WaveformPGDAttack(
            model=model, eps=eps, alpha=eps / 4.0, steps=pgd_steps, target_class=0
        )

        attacked_scores = clean_scores_arr.copy()
        flipped_to_bonafide = 0

        # Attack in mini-batches
        batch_size = 4
        for start_idx in range(0, len(spoof_indices), batch_size):
            batch_indices = spoof_indices[start_idx : start_idx + batch_size]
            batch_waves = stacked_waveforms[batch_indices].to(device)

            adv_waves = attacker.perturbation(batch_waves)

            with torch.no_grad():
                adv_logits = model(adv_waves)
                adv_probs = torch.softmax(adv_logits, dim=-1)
                adv_spoof_probs = adv_probs[:, 1].cpu().numpy()
                adv_preds = torch.argmax(adv_logits, dim=-1).cpu().numpy()

            for k, original_idx in enumerate(batch_indices):
                attacked_scores[original_idx] = float(adv_spoof_probs[k])
                if adv_preds[k] == 0:  # Successfully tricked into bonafide
                    flipped_to_bonafide += 1

        asr = flipped_to_bonafide / max(len(spoof_indices), 1)
        attacked_eer = compute_eer(labels_arr, attacked_scores)

        logger.info(
            f"[PGD eps={eps:6.4f}] ASR: {asr * 100:5.2f}% | "
            f"Clean EER: {clean_eer * 100:5.2f}% -> Attacked EER: {attacked_eer * 100:5.2f}%"
        )

        records.append(
            {
                "modality": "audio",
                "attack": "PGD_Linf",
                "strength": str(eps),
                "metric": "EER",
                "clean_value": f"{clean_eer:.4f}",
                "attacked_value": f"{attacked_eer:.4f}",
                "attack_success_rate": f"{asr:.4f}",
            }
        )

    # --- Gaussian Noise Attacks ---
    for snr in snr_levels_db:
        attacked_scores = clean_scores_arr.copy()
        flipped_to_bonafide = 0

        for start_idx in range(0, len(spoof_indices), batch_size):
            batch_indices = spoof_indices[start_idx : start_idx + batch_size]
            batch_waves = stacked_waveforms[batch_indices].to(device)

            noisy_waves = add_gaussian_noise(batch_waves, snr_db=snr)

            with torch.no_grad():
                noisy_logits = model(noisy_waves)
                noisy_probs = torch.softmax(noisy_logits, dim=-1)
                noisy_spoof_probs = noisy_probs[:, 1].cpu().numpy()
                noisy_preds = torch.argmax(noisy_logits, dim=-1).cpu().numpy()

            for k, original_idx in enumerate(batch_indices):
                attacked_scores[original_idx] = float(noisy_spoof_probs[k])
                if noisy_preds[k] == 0:
                    flipped_to_bonafide += 1

        asr = flipped_to_bonafide / max(len(spoof_indices), 1)
        attacked_eer = compute_eer(labels_arr, attacked_scores)

        logger.info(
            f"[Gaussian SNR={snr:2.0f}dB] ASR: {asr * 100:5.2f}% | "
            f"Clean EER: {clean_eer * 100:5.2f}% -> Attacked EER: {attacked_eer * 100:5.2f}%"
        )

        records.append(
            {
                "modality": "audio",
                "attack": "Gaussian_Noise",
                "strength": f"{snr}dB",
                "metric": "EER",
                "clean_value": f"{clean_eer:.4f}",
                "attacked_value": f"{attacked_eer:.4f}",
                "attack_success_rate": f"{asr:.4f}",
            }
        )

    # Save to outputs/adversarial_results.csv
    output_dir = Path(config.get("output_dir", "audio/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "adversarial_results.csv"

    fieldnames = [
        "modality",
        "attack",
        "strength",
        "metric",
        "clean_value",
        "attacked_value",
        "attack_success_rate",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Saved adversarial attack results to: {csv_file}")

    # Plot degradation curve
    plot_path = output_dir / "audio_adversarial_degradation.png"
    plot_degradation(records, plot_path)

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    return records


def plot_degradation(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Plot degradation curves comparing attack strength against ASR and EER.

    Args:
        records: List of attack result dictionaries.
        output_path: Path to save the generated image.
    """
    pgd_records = [r for r in records if r["attack"] == "PGD_Linf"]
    noise_records = [r for r in records if r["attack"] == "Gaussian_Noise"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. PGD: Epsilon vs Attack Success Rate
    if pgd_records:
        eps_vals = [float(r["strength"]) for r in pgd_records]
        pgd_asrs = [float(r["attack_success_rate"]) * 100.0 for r in pgd_records]
        pgd_eers = [float(r["attacked_value"]) * 100.0 for r in pgd_records]
        clean_eer = float(pgd_records[0]["clean_value"]) * 100.0

        ax1.plot(eps_vals, pgd_asrs, "o-", color="#d9534f", lw=2, markersize=6)
        ax1.set_xscale("log")
        ax1.set_xlabel("PGD Epsilon (L-infinity)", fontsize=11)
        ax1.set_ylabel("Attack Success Rate (%)", fontsize=11)
        ax1.set_title("PGD Attack Success Rate vs Epsilon", fontsize=12, fontweight="bold")
        ax1.grid(True, which="both", linestyle=":", alpha=0.6)
        ax1.set_ylim([-5, 105])

        # 2. PGD: Epsilon vs Attacked EER
        ax2.plot(eps_vals, pgd_eers, "s-", color="#2b5c8f", lw=2, markersize=6, label="Attacked EER")
        ax2.axhline(clean_eer, color="#999999", linestyle="--", label=f"Clean EER ({clean_eer:.1f}%)")
        ax2.set_xscale("log")
        ax2.set_xlabel("PGD Epsilon (L-infinity)", fontsize=11)
        ax2.set_ylabel("Equal Error Rate (%)", fontsize=11)
        ax2.set_title("Detector EER Degradation under PGD", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(True, which="both", linestyle=":", alpha=0.6)

    # 3. Gaussian Noise: SNR vs Attack Success Rate
    if noise_records:
        snr_vals = [float(r["strength"].replace("dB", "")) for r in noise_records]
        noise_asrs = [float(r["attack_success_rate"]) * 100.0 for r in noise_records]
        noise_eers = [float(r["attacked_value"]) * 100.0 for r in noise_records]
        clean_eer = float(noise_records[0]["clean_value"]) * 100.0

        ax3.plot(snr_vals, noise_asrs, "o-", color="#e67e22", lw=2, markersize=6)
        ax3.invert_xaxis()  # Lower SNR = more noise (stronger attack) to the right
        ax3.set_xlabel("Signal-to-Noise Ratio (dB) [Reversed]", fontsize=11)
        ax3.set_ylabel("Attack Success Rate (%)", fontsize=11)
        ax3.set_title("Gaussian Noise ASR vs Noise Level", fontsize=12, fontweight="bold")
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.set_ylim([-5, 105])

        # 4. Gaussian Noise: SNR vs Attacked EER
        ax4.plot(snr_vals, noise_eers, "s-", color="#16a085", lw=2, markersize=6, label="Attacked EER")
        ax4.axhline(clean_eer, color="#999999", linestyle="--", label=f"Clean EER ({clean_eer:.1f}%)")
        ax4.invert_xaxis()
        ax4.set_xlabel("Signal-to-Noise Ratio (dB) [Reversed]", fontsize=11)
        ax4.set_ylabel("Equal Error Rate (%)", fontsize=11)
        ax4.set_title("Detector EER Degradation under Gaussian Noise", fontsize=12, fontweight="bold")
        ax4.legend()
        ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300)
    plt.close(fig)
    logger.info(f"Saved adversarial degradation plots to: {output_path}")


def main() -> None:
    """CLI interface for running adversarial attacks against audio detector."""
    parser = argparse.ArgumentParser(description="Adversarial Attack Evaluation for Audio Detector")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--config",
        type=str,
        default="audio/configs/baseline.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target execution device",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on a small subset of samples for rapid verification",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.smoke_test:
        config["smoke_test"] = True

    results = run_adversarial_evaluation(
        checkpoint_path=args.checkpoint,
        config=config,
        device_name=args.device,
    )
    logger.info(f"Adversarial evaluation completed. {len(results)} attack points recorded.")


if __name__ == "__main__":
    main()
