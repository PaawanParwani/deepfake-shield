"""ASVspoof 2019 Logical Access (LA) Dataset and DataLoader.

This module provides dataset parsing, audio loading, resampling,
cropping, normalization, and DataLoader construction for the
ASVspoof 2019 Logical Access (LA) deepfake detection track.

Supports two data formats:
1. **Parquet** (Hugging Face download): audio bytes embedded in parquet files
2. **Raw FLAC** (official download): .flac files + protocol .txt files
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAP: Dict[str, int] = {
    "bonafide": 0,
    "spoof": 1,
}
INV_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

# Mapping from parquet split names to ASVspoof split names
PARQUET_SPLIT_MAP: Dict[str, str] = {
    "train": "train",
    "dev": "validation",
    "eval": "test",
    "validation": "validation",
    "test": "test",
}


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_data_format(data_dir: Path) -> str:
    """Auto-detect whether data is in parquet or raw FLAC format.

    Args:
        data_dir: Root data directory.

    Returns:
        'parquet' or 'flac' indicating the detected format.
    """
    # Check for parquet files
    parquet_dir = data_dir / "data"
    if parquet_dir.exists():
        parquet_files = list(parquet_dir.glob("*.parquet"))
        if parquet_files:
            logger.info(f"Detected parquet format ({len(parquet_files)} files in {parquet_dir})")
            return "parquet"

    # Check for FLAC directory structure
    for split_name in ["train", "dev", "eval"]:
        flac_dir = data_dir / f"ASVspoof2019_LA_{split_name}" / "flac"
        if flac_dir.exists():
            logger.info(f"Detected raw FLAC format (found {flac_dir})")
            return "flac"

    # Default to parquet if nothing found (will fail gracefully later)
    logger.warning("Could not detect data format. Defaulting to parquet.")
    return "parquet"


class ASVSpoof2019LADataset(Dataset):
    """PyTorch Dataset for the ASVspoof 2019 Logical Access (LA) dataset.

    Auto-detects data format:
    - Parquet (HuggingFace): reads audio bytes + labels from .parquet files
    - Raw FLAC (official): reads .flac files with protocol .txt files

    Audio is resampled to target rate, cropped/padded to fixed duration, and normalized.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        protocol_dir: Union[str, Path] = "",
        split: str = "train",
        sample_rate: int = 16000,
        max_duration: float = 4.0,
        is_train: bool = True,
        smoke_test: bool = False,
        smoke_test_size: int = 500,
        return_file_id: bool = False,
    ) -> None:
        """Initialize the ASVspoof 2019 LA dataset.

        Args:
            data_dir: Root directory of ASVspoof LA data (e.g. audio/data/LA).
            protocol_dir: Directory containing protocol .txt files (for FLAC format).
            split: Data split ('train', 'dev', or 'eval').
            sample_rate: Target audio sampling rate in Hz (default 16000).
            max_duration: Target audio clip duration in seconds (default 4.0).
            is_train: True for random crop, False for center crop.
            smoke_test: If True, limit the dataset to smoke_test_size samples.
            smoke_test_size: Number of samples to load in smoke test mode.
            return_file_id: If True, returns (waveform, label, audio_id).
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.protocol_dir = Path(protocol_dir) if protocol_dir else self.data_dir
        self.split = split.lower()
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * max_duration)
        self.is_train = is_train
        self.return_file_id = return_file_id
        self.smoke_test = smoke_test
        self.smoke_test_size = smoke_test_size

        # Auto-detect format
        self.data_format = detect_data_format(self.data_dir)

        if self.data_format == "parquet":
            self._load_parquet()
        else:
            self.audio_dir = self._find_audio_dir()
            self.protocol_file = self._find_protocol_file()
            self.samples: List[Tuple[str, int, str]] = []
            self._load_protocol()
            # For FLAC format, parquet_data won't be used
            self.parquet_data = None

        logger.info(
            f"Loaded {len(self)} samples for split '{self.split}' "
            f"(bonafide: {self._count_label(0)}, spoof: {self._count_label(1)})"
        )

    def _count_label(self, label: int) -> int:
        """Count samples with a given label."""
        if self.data_format == "parquet":
            return int((self.parquet_data["key"] == label).sum())
        else:
            return sum(1 for s in self.samples if s[1] == label)

    # ── Parquet loading ───────────────────────────────────────────────

    def _load_parquet(self) -> None:
        """Load data from parquet files (HuggingFace format).

        Parquet columns: speaker_id, audio_file_name, audio (dict with 'bytes' + 'path'),
        system_id, key (int: 0=bonafide, 1=spoof).
        """
        import pandas as pd

        parquet_split = PARQUET_SPLIT_MAP.get(self.split, self.split)
        parquet_dir = self.data_dir / "data"

        # Find the parquet file for this split
        parquet_files = sorted(parquet_dir.glob(f"{parquet_split}*.parquet"))
        if not parquet_files:
            # Try with original split name
            parquet_files = sorted(parquet_dir.glob(f"{self.split}*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found for split '{self.split}' "
                f"(tried '{parquet_split}' and '{self.split}') in {parquet_dir}"
            )

        logger.info(f"Loading {len(parquet_files)} parquet file(s) for split '{parquet_split}'")

        dfs = [pd.read_parquet(f) for f in parquet_files]
        self.parquet_data = pd.concat(dfs, ignore_index=True)

        if self.smoke_test:
            # Shuffle before sampling to ensure both classes are represented
            self.parquet_data = self.parquet_data.sample(
                n=min(self.smoke_test_size, len(self.parquet_data)),
                random_state=42,
            ).reset_index(drop=True)
            logger.info(f"Smoke test: limited to {len(self.parquet_data)} samples")

    def _load_audio_from_parquet(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Load a single sample from parquet data.

        Args:
            idx: Row index in the parquet dataframe.

        Returns:
            Tuple of (waveform, label, audio_file_name).
        """
        row = self.parquet_data.iloc[idx]
        audio_dict = row["audio"]
        label = int(row["key"])
        audio_id = str(row.get("audio_file_name", f"sample_{idx}"))

        # Decode audio bytes
        audio_bytes = audio_dict["bytes"]
        waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        waveform = torch.from_numpy(waveform)

        # Handle multi-channel
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=-1)  # (samples, channels) -> (samples,)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        return waveform, label, audio_id

    # ── FLAC loading (original protocol-based format) ─────────────────

    def _find_audio_dir(self) -> Path:
        """Locate the flac audio directory for the split."""
        candidates = [
            self.data_dir / f"ASVspoof2019_LA_{self.split}" / "flac",
            self.data_dir / self.split / "flac",
            self.data_dir / "flac",
            self.data_dir / f"ASVspoof2019_LA_{self.split}",
            self.data_dir,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return self.data_dir / f"ASVspoof2019_LA_{self.split}" / "flac"

    def _find_protocol_file(self) -> Path:
        """Locate the protocol file for the current split."""
        candidates = [
            self.protocol_dir / f"ASVspoof2019.LA.cm.{self.split}.trl.txt",
            self.protocol_dir / f"ASVspoof2019.LA.cm.{self.split}.trn.txt",
            self.protocol_dir / f"ASVspoof2019.LA.cm.{self.split}.txt",
            self.protocol_dir / f"{self.split}.trl.txt",
            self.protocol_dir / f"{self.split}.trn.txt",
            self.protocol_dir / f"{self.split}.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        if self.protocol_dir.exists():
            for p in self.protocol_dir.glob(f"*{self.split}*.txt"):
                return p

        ext = "trn.txt" if self.split == "train" else "trl.txt"
        return self.protocol_dir / f"ASVspoof2019.LA.cm.{self.split}.{ext}"

    def _load_protocol(self) -> None:
        """Parse protocol file lines: SPEAKER_ID AUDIO_FILE_ID - ATTACK_ID KEY."""
        if not self.protocol_file.exists():
            logger.warning(
                f"Protocol file not found at {self.protocol_file}. Initializing empty dataset."
            )
            return

        with open(self.protocol_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                audio_id = parts[1]
                attack_id = parts[3]
                key_str = parts[4].lower()

                if key_str not in LABEL_MAP:
                    continue

                label = LABEL_MAP[key_str]
                self.samples.append((audio_id, label, attack_id))

                if self.smoke_test and len(self.samples) >= self.smoke_test_size:
                    break

    def _load_audio_from_flac(self, audio_id: str) -> torch.Tensor:
        """Load audio file as mono float tensor at target sample rate."""
        candidates = [
            self.audio_dir / f"{audio_id}.flac",
            self.audio_dir / f"{audio_id}.wav",
            self.audio_dir / audio_id,
        ]
        file_path = None
        for candidate in candidates:
            if candidate.exists():
                file_path = candidate
                break

        if file_path is None:
            logger.warning(f"Audio file {audio_id} not found in {self.audio_dir}. Using silence.")
            return torch.zeros(self.target_length, dtype=torch.float32)

        try:
            waveform, sr = torchaudio.load(str(file_path))
        except Exception:
            data, sr = sf.read(str(file_path), dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()

        if waveform.dim() == 2:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        return waveform

    # ── Common processing ─────────────────────────────────────────────

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """Crop or pad waveform to fixed self.target_length samples."""
        length = waveform.shape[-1]
        if length > self.target_length:
            if self.is_train:
                max_start = length - self.target_length
                start = random.randint(0, max_start)
            else:
                start = (length - self.target_length) // 2
            waveform = waveform[start : start + self.target_length]
        elif length < self.target_length:
            pad_amount = self.target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount), value=0.0)
        return waveform

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize waveform to zero mean and unit variance."""
        mean = waveform.mean()
        std = waveform.std()
        if std > 1e-7:
            waveform = (waveform - mean) / (std + 1e-7)
        else:
            waveform = waveform - mean
        return waveform

    def __len__(self) -> int:
        """Return total number of samples."""
        if self.data_format == "parquet":
            return len(self.parquet_data)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, str]]:
        """Fetch processed audio tensor and integer label.

        Args:
            idx: Index of sample to fetch.

        Returns:
            (waveform, label) or (waveform, label, audio_id) if return_file_id is True.
        """
        if self.data_format == "parquet":
            waveform, label, audio_id = self._load_audio_from_parquet(idx)
        else:
            audio_id, label, _ = self.samples[idx]
            waveform = self._load_audio_from_flac(audio_id)

        waveform = self._crop_or_pad(waveform)
        waveform = self._normalize(waveform)

        if self.return_file_id:
            return waveform, label, audio_id
        return waveform, label


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader for ASVspoof dataset.

    Args:
        dataset: The Dataset instance.
        batch_size: Batch size for DataLoader.
        shuffle: Whether to shuffle data.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If True, copies tensors into CUDA pinned memory.
        drop_last: Set to True to drop the last incomplete batch.

    Returns:
        Configured PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )


def create_synthetic_dataset(
    temp_dir: Path, num_samples: int = 10, sample_rate: int = 16000, duration: float = 4.0
) -> Tuple[Path, Path]:
    """Create a temporary synthetic dataset for smoke testing when real data is absent.

    Args:
        temp_dir: Path to directory where dummy files should be created.
        num_samples: Number of synthetic audio files to generate.
        sample_rate: Audio sampling rate.
        duration: Audio duration in seconds.

    Returns:
        Tuple of (data_dir, protocol_dir).
    """
    data_dir = temp_dir / "LA"
    audio_dir = data_dir / "ASVspoof2019_LA_train" / "flac"
    protocol_dir = data_dir / "ASVspoof2019_LA_cm_protocols"
    audio_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    protocol_file = protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    lines = []

    num_samples_audio = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples_audio, dtype=np.float32)

    for i in range(num_samples):
        audio_id = f"LA_T_{1000000 + i}"
        speaker_id = f"LA_{i:04d}"
        key = "bonafide" if i % 2 == 0 else "spoof"
        attack_id = "-" if key == "bonafide" else f"A{i % 6 + 1:02d}"

        freq = 440.0 * (1 + 0.1 * i)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        file_path = audio_dir / f"{audio_id}.flac"
        sf.write(str(file_path), audio, sample_rate)

        lines.append(f"{speaker_id} {audio_id} - {attack_id} {key}\n")

    with open(protocol_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return data_dir, protocol_dir


def main() -> None:
    """CLI smoke test for ASVspoof dataset loading."""
    parser = argparse.ArgumentParser(description="ASVspoof 2019 LA Dataset Smoke Test")
    parser.add_argument("--data-dir", type=str, default="audio/data/LA", help="Path to data directory")
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default="audio/data/LA/ASVspoof2019_LA_cm_protocols",
        help="Path to protocol directory (for FLAC format)",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "dev", "eval"])
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for dataloader")
    parser.add_argument("--smoke-test-size", type=int, default=10, help="Number of samples to test")
    args = parser.parse_args()

    set_seed(42)
    logger.info("Running Dataset Smoke Test...")

    data_dir = Path(args.data_dir)
    data_format = detect_data_format(data_dir)

    cleanup_temp = False
    temp_dir_obj = None

    if data_format == "parquet":
        # Parquet mode — no protocol files needed
        logger.info("Using parquet format from HuggingFace download")
        try:
            dataset = ASVSpoof2019LADataset(
                data_dir=data_dir,
                split=args.split,
                sample_rate=16000,
                max_duration=4.0,
                is_train=True,
                smoke_test=True,
                smoke_test_size=args.smoke_test_size,
            )
        except Exception as e:
            logger.error(f"Failed to load parquet data: {e}")
            logger.info("Creating synthetic dataset for smoke test instead.")
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_root = Path(temp_dir_obj.name)
            syn_data_dir, syn_protocol_dir = create_synthetic_dataset(
                temp_root, num_samples=args.smoke_test_size
            )
            cleanup_temp = True
            dataset = ASVSpoof2019LADataset(
                data_dir=syn_data_dir,
                protocol_dir=syn_protocol_dir,
                split="train",
                sample_rate=16000,
                max_duration=4.0,
                is_train=True,
                smoke_test=True,
                smoke_test_size=args.smoke_test_size,
            )
    else:
        # FLAC mode — check for protocol files
        protocol_path = Path(args.protocol_dir)
        if not protocol_path.exists():
            logger.info("Real protocol files not found. Creating synthetic dataset for smoke test.")
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_root = Path(temp_dir_obj.name)
            data_dir, protocol_dir = create_synthetic_dataset(
                temp_root, num_samples=args.smoke_test_size
            )
            cleanup_temp = True
        else:
            protocol_dir = protocol_path

        dataset = ASVSpoof2019LADataset(
            data_dir=data_dir,
            protocol_dir=protocol_dir,
            split=args.split,
            sample_rate=16000,
            max_duration=4.0,
            is_train=True,
            smoke_test=True,
            smoke_test_size=args.smoke_test_size,
        )

    try:
        dataloader = create_dataloader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 3:
                waveforms, labels, _ = batch
            else:
                waveforms, labels = batch
            logger.info(
                f"Batch {batch_idx}: Waveforms shape={waveforms.shape}, "
                f"dtype={waveforms.dtype}, Labels={labels.tolist()}"
            )
            assert waveforms.ndim == 2, f"Expected 2D tensor (B, T), got {waveforms.shape}"
            assert waveforms.shape[-1] == 64000, f"Expected 64000 samples, got {waveforms.shape[-1]}"
            if batch_idx >= 2:
                break

        logger.info("Dataset smoke test completed successfully!")
    finally:
        if cleanup_temp and temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
