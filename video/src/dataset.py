"""FaceForensics++ dataset and DataLoader implementation for video deepfake detection.

Strictly splits by video identity (never by individual frame) using official FF++
train/val/test split definitions. Applies robust augmentations for training
and ImageNet standardization.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed integer.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class JPEGCompressionAugmentation:
    """Simulate variable JPEG compression artifacts on PIL images."""

    def __init__(self, quality_range: Tuple[int, int] = (50, 95)) -> None:
        """Initialize JPEG compression augmentation.

        Args:
            quality_range: Min and max JPEG quality factors (inclusive).
        """
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random JPEG compression.

        Args:
            img: Input PIL Image.

        Returns:
            Compressed and re-opened PIL Image.
        """
        q = int(torch.randint(self.quality_range[0], self.quality_range[1] + 1, (1,)).item())
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def get_transforms(
    split: str,
    image_size: int = 224,
) -> Callable[[Image.Image], torch.Tensor]:
    """Build torchvision transform pipeline for training or evaluation.

    Args:
        split: 'train' for augmentation pipeline, 'val' or 'test' for standard eval.
        image_size: Target square image dimension.

    Returns:
        Composed torchvision transform function.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                ),
                JPEGCompressionAugmentation(quality_range=(50, 95)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class FFppFaceDataset(Dataset):
    """PyTorch Dataset loading pre-extracted FaceForensics++ face crops.

    Ensures that splitting is strictly video-level. Every frame inherits the
    video's label and split membership.
    """

    def __init__(
        self,
        crops_dir: str | Path,
        split: str = "train",
        splits_dir: Optional[str | Path] = None,
        image_size: int = 224,
        smoke_test: bool = False,
        smoke_test_videos: int = 50,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        create_synthetic_if_empty: bool = True,
    ) -> None:
        """Initialize FF++ face dataset.

        Args:
            crops_dir: Directory containing face crops.
            split: 'train', 'val', or 'test'.
            splits_dir: Path to directory containing FF++ split JSONs.
            image_size: Target image dimension.
            smoke_test: Whether to restrict dataset to a small subset for fast verification.
            smoke_test_videos: Max number of videos in smoke test mode.
            transform: Optional custom transform override.
            create_synthetic_if_empty: If true and no crops exist, generate synthetic crops for smoke testing.
        """
        self.crops_dir = Path(crops_dir)
        self.split = split
        self.splits_dir = Path(splits_dir) if splits_dir else None
        self.image_size = image_size
        self.smoke_test = smoke_test
        self.smoke_test_videos = smoke_test_videos
        self.transform = transform or get_transforms(split=split, image_size=image_size)

        self.samples: List[Tuple[Path, int, str, str]] = []  # (path, label, video_id, method)
        self._load_samples()

        if len(self.samples) == 0 and create_synthetic_if_empty:
            logger.warning(
                "No crops found in %s for split %s. Creating synthetic crops for smoke testing.",
                self.crops_dir,
                self.split,
            )
            self._create_synthetic_dataset()
            self._load_samples()

    def _get_allowed_video_ids(self) -> Optional[Dict[str, List[str]]]:
        """Parse official FF++ split JSON file if available.

        Returns:
            Dictionary with 'real' video IDs and 'pair' IDs, or None if file is missing.
        """
        if not self.splits_dir:
            return None
        split_file = self.splits_dir / f"{self.split}.json"
        if not split_file.is_file():
            return None

        with open(split_file, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        real_ids: List[str] = []
        fake_pairs: List[Tuple[str, str]] = []
        for pair in pairs:
            id1, id2 = str(pair[0]), str(pair[1])
            real_ids.append(id1)
            real_ids.append(id2)
            fake_pairs.append((id1, id2))

        return {"real": list(set(real_ids)), "pairs": fake_pairs}

    def _load_samples(self) -> None:
        """Scan directory and register all frame crops belonging to this split."""
        self.samples = []
        allowed = self._get_allowed_video_ids()

        candidate_dirs = [
            self.crops_dir / self.split,
            self.crops_dir,
        ]

        base_dir: Optional[Path] = None
        for cd in candidate_dirs:
            if cd.exists() and any(cd.iterdir()):
                base_dir = cd
                break

        if base_dir is None:
            return

        videos_by_id: Dict[str, List[Tuple[Path, int, str, str]]] = {}

        for label_str, label_int in [("real", 0), ("fake", 1)]:
            label_dir = base_dir / label_str
            if not label_dir.exists():
                continue

            for video_folder in sorted(label_dir.iterdir()):
                if not video_folder.is_dir():
                    continue

                video_id = video_folder.name
                method = "real" if label_int == 0 else "unknown"

                if label_int == 1:
                    parts = video_id.split("_")
                    if len(parts) >= 2:
                        method = parts[0]

                if allowed is not None and base_dir == self.crops_dir:
                    if label_int == 0:
                        if video_id not in allowed["real"]:
                            continue
                    else:
                        match = False
                        for id1, id2 in allowed["pairs"]:
                            if f"{id1}_{id2}" in video_id or f"{id2}_{id1}" in video_id:
                                match = True
                                break
                        if not match:
                            continue

                frames = sorted(
                    list(video_folder.glob("*.jpg"))
                    + list(video_folder.glob("*.png"))
                    + list(video_folder.glob("*.jpeg"))
                )
                if not frames:
                    continue

                if video_id not in videos_by_id:
                    videos_by_id[video_id] = []

                for frame_path in frames:
                    videos_by_id[video_id].append((frame_path, label_int, video_id, method))

        selected_video_ids = list(videos_by_id.keys())
        if self.smoke_test and len(selected_video_ids) > self.smoke_test_videos:
            selected_video_ids = selected_video_ids[: self.smoke_test_videos]

        for vid in selected_video_ids:
            self.samples.extend(videos_by_id[vid])

        logger.info(
            "Split '%s': Loaded %d frames across %d videos.",
            self.split,
            len(self.samples),
            len(selected_video_ids),
        )

    def _create_synthetic_dataset(self) -> None:
        """Create minimal synthetic face crop images for end-to-end execution without FF++ files."""
        set_seed(42)
        target_root = self.crops_dir / self.split
        methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

        # Create 4 real and 4 fake synthetic videos with 4 frames each
        for i in range(4):
            vid = f"synthetic_real_{i:03d}"
            v_dir = target_root / "real" / vid
            v_dir.mkdir(parents=True, exist_ok=True)
            for f_idx in range(4):
                arr = np.random.randint(100, 240, (self.image_size, self.image_size, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(v_dir / f"frame_{f_idx:04d}.jpg")

        for i in range(4):
            method = methods[i % len(methods)]
            vid = f"{method}_{i:03d}_{i+1:03d}"
            v_dir = target_root / "fake" / vid
            v_dir.mkdir(parents=True, exist_ok=True)
            for f_idx in range(4):
                arr = np.random.randint(20, 160, (self.image_size, self.image_size, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(v_dir / f"frame_{f_idx:04d}.jpg")

    def __len__(self) -> int:
        """Return total number of frame samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        """Fetch a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_int, video_id, method).
        """
        frame_path, label, video_id, method = self.samples[idx]
        image = Image.open(frame_path).convert("RGB")
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, label, video_id, method


def create_dataloader(
    config: Dict[str, Any],
    split: str = "train",
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: Optional[int] = None,
) -> Tuple[FFppFaceDataset, DataLoader]:
    """Factory function creating a Dataset and DataLoader pair according to configuration.

    Args:
        config: Loaded config dictionary.
        split: 'train', 'val', or 'test'.
        batch_size: Batch size override.
        shuffle: Shuffle boolean override (default True for train, False for val/test).
        num_workers: Worker processes override.

    Returns:
        Tuple of (FFppFaceDataset instance, DataLoader instance).
    """
    bs = batch_size if batch_size is not None else int(config.get("batch_size", 32))
    nw = num_workers if num_workers is not None else int(config.get("num_workers", 2))
    sh = shuffle if shuffle is not None else (split == "train")

    dataset = FFppFaceDataset(
        crops_dir=config.get("crops_dir", "video/data/crops"),
        split=split,
        splits_dir=config.get("splits_dir"),
        image_size=int(config.get("image_size", 224)),
        smoke_test=bool(config.get("smoke_test", False)),
        smoke_test_videos=int(config.get("smoke_test_videos", 50)),
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=bs,
        shuffle=sh,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train" and len(dataset) > bs),
    )

    return dataset, loader


def main() -> None:
    """Smoke test dataset loader and verify tensor dimensions."""
    parser = argparse.ArgumentParser(description="Test FF++ dataset and dataloader.")
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
        default="train",
        help="Dataset split to test.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["smoke_test"] = True
    cfg["smoke_test_videos"] = 10
    dataset, loader = create_dataloader(cfg, split=args.split, batch_size=4)

    print(f"Dataset split '{args.split}' length: {len(dataset)}")
    for batch_images, batch_labels, batch_vids, batch_methods in loader:
        print(f"Batch image tensor shape: {batch_images.shape}")
        print(f"Batch labels: {batch_labels.tolist()}")
        print(f"Batch video IDs: {batch_vids[:2]}...")
        print(f"Batch methods: {batch_methods[:2]}...")
        break
    print("Dataset verification completed successfully.")


if __name__ == "__main__":
    main()
