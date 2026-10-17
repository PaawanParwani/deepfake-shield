import json
import tempfile
import unittest
from pathlib import Path

from video.src.dataset import FFppFaceDataset


class FaceForensicsDatasetSafetyTests(unittest.TestCase):
    def test_missing_crops_fail_without_creating_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            splits_dir = root / "splits"
            splits_dir.mkdir()
            (splits_dir / "train.json").write_text("[]", encoding="utf-8")
            crops_dir = root / "crops"

            with self.assertRaisesRegex(FileNotFoundError, "No extracted"):
                FFppFaceDataset(crops_dir, "train", splits_dir)

            self.assertFalse(crops_dir.exists())

    def test_smoke_dataset_keeps_only_official_video_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            splits_dir = root / "splits"
            splits_dir.mkdir()
            (splits_dir / "train.json").write_text(
                json.dumps([["001", "002"]]),
                encoding="utf-8",
            )
            split_dir = root / "crops" / "train"
            for label, video_id in (
                ("real", "001"),
                ("real", "999"),
                ("fake", "Deepfakes_001_002"),
                ("fake", "Deepfakes_888_999"),
            ):
                video_dir = split_dir / label / video_id
                video_dir.mkdir(parents=True)
                (video_dir / "frame_0000.jpg").write_bytes(b"crop")

            dataset = FFppFaceDataset(
                root / "crops",
                "train",
                splits_dir,
                smoke_test=True,
            )

            self.assertEqual(
                {sample[2] for sample in dataset.samples},
                {"001", "Deepfakes_001_002"},
            )

    def test_incomplete_official_split_is_rejected_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            splits_dir = root / "splits"
            splits_dir.mkdir()
            (splits_dir / "train.json").write_text(
                json.dumps([["001", "002"]]),
                encoding="utf-8",
            )
            split_dir = root / "crops" / "train"
            for label, video_id in (
                ("real", "001"),
                ("fake", "Deepfakes_001_002"),
            ):
                video_dir = split_dir / label / video_id
                video_dir.mkdir(parents=True)
                (video_dir / "frame_0000.jpg").write_bytes(b"crop")

            with self.assertRaisesRegex(ValueError, "Incomplete official 'train' crops"):
                FFppFaceDataset(
                    root / "crops",
                    "train",
                    splits_dir,
                    required_manipulation_methods=("Deepfakes",),
                )


if __name__ == "__main__":
    unittest.main()
