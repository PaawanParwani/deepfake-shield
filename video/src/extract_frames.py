"""Frame extraction and face cropping pipeline for FaceForensics++ videos.

Samples frames at ~1 frame per second (up to a configurable maximum), detects faces
using MTCNN, crops with a configurable margin, resizes to target resolution,
and caches crops organized by split and label.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict containing configuration parameters.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(device_setting: str = "auto") -> torch.device:
    """Resolve the execution device (CPU or CUDA).

    Args:
        device_setting: 'auto', 'cuda', or 'cpu'.

    Returns:
        torch.device instance.
    """
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def get_video_fps_and_step(cap: cv2.VideoCapture) -> Tuple[float, int]:
    """Determine video FPS and frame step for 1 frame per second sampling.

    Args:
        cap: OpenCV VideoCapture object.

    Returns:
        Tuple of (fps, frame_step).
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    step = max(1, int(round(fps)))
    return fps, step


def extract_faces_from_video(
    video_path: Path,
    output_dir: Path,
    mtcnn: Optional[Any],
    face_margin: float = 0.2,
    image_size: int = 224,
    max_frames: int = 32,
) -> int:
    """Extract face crops from a single video and save to disk.

    Args:
        video_path: Path to the video file.
        output_dir: Destination directory for cropped face images.
        mtcnn: Initialized MTCNN detector instance.
        face_margin: Relative margin to expand the detected bounding box.
        image_size: Target width and height for face crops.
        max_frames: Maximum number of frames to extract.

    Returns:
        Number of face crops successfully saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = list(output_dir.glob("frame_*.jpg"))
    if len(existing_frames) >= 1:
        logger.debug("Skipping already processed video: %s", video_path)
        return len(existing_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, step = get_video_fps_and_step(cap)

    saved_count = 0
    frame_idx = 0

    while frame_idx < total_frames and saved_count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        box: Optional[np.ndarray] = None
        if mtcnn is not None:
            try:
                boxes, _ = mtcnn.detect(frame_rgb)
                if boxes is not None and len(boxes) > 0 and boxes[0] is not None:
                    box = boxes[0]
            except Exception as e:
                logger.debug("Detection error on frame %d of %s: %s", frame_idx, video_path.name, e)

        if box is not None:
            bx1, by1, bx2, by2 = box
            bw = bx2 - bx1
            bh = by2 - by1

            crop_x1 = max(0, int(bx1 - face_margin * bw))
            crop_y1 = max(0, int(by1 - face_margin * bh))
            crop_x2 = min(w, int(bx2 + face_margin * bw))
            crop_y2 = min(h, int(by2 + face_margin * bh))

            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                out_path = output_dir / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(out_path), crop_resized)
                saved_count += 1

        frame_idx += step

    cap.release()
    return saved_count


def collect_ffpp_split_videos(
    config: Dict[str, Any],
    split: str,
) -> List[Tuple[Path, str, str]]:
    """Locate real and fake video paths for a given FF++ split.

    Args:
        config: Loaded config dictionary.
        split: One of 'train', 'val', or 'test'.

    Returns:
        List of tuples: (video_path, 'real' | 'fake', video_id).
    """
    data_dir = Path(config.get("data_dir", "video/data/FaceForensics++"))
    splits_dir = Path(config.get("splits_dir", str(data_dir / "splits")))
    compression = config.get("compression", "c23")
    manipulation_methods = config.get(
        "manipulation_methods", ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    )

    split_file = splits_dir / f"{split}.json"
    video_records: List[Tuple[Path, str, str]] = []

    # Support both official FF++ layout and Kaggle C23 layout
    def find_real_dir() -> Optional[Path]:
        for cand in [
            data_dir / "original_sequences" / "youtube" / compression / "videos",
            data_dir / "original",
            data_dir / "original_sequences",
            data_dir / "pristine",
        ]:
            if cand.exists() and cand.is_dir():
                return cand
        return None

    def find_fake_dir(method_name: str) -> Optional[Path]:
        for cand in [
            data_dir / "manipulated_sequences" / method_name / compression / "videos",
            data_dir / method_name,
            data_dir / "manipulated_sequences" / method_name,
        ]:
            if cand.exists() and cand.is_dir():
                return cand
        return None

    real_base = find_real_dir()

    if split_file.is_file():
        with open(split_file, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        seen_real_ids = set()
        for pair in pairs:
            id1, id2 = str(pair[0]), str(pair[1])
            for orig_id in [id1, id2]:
                if orig_id not in seen_real_ids:
                    seen_real_ids.add(orig_id)
                    if real_base is not None:
                        real_path = real_base / f"{orig_id}.mp4"
                        if real_path.exists():
                            video_records.append((real_path, "real", orig_id))

            for method in manipulation_methods:
                fake_base = find_fake_dir(method)
                if fake_base is not None:
                    for candidate_name in [f"{id1}_{id2}.mp4", f"{id2}_{id1}.mp4"]:
                        fake_path = fake_base / candidate_name
                        if fake_path.exists():
                            video_id = f"{method}_{candidate_name.replace('.mp4', '')}"
                            video_records.append((fake_path, "fake", video_id))
    else:
        logger.warning(
            "Split file %s not found. Scanning %s directly for videos across layouts.",
            split_file,
            data_dir,
        )
        if real_base and real_base.exists():
            for v in real_base.glob("*.mp4"):
                video_records.append((v, "real", v.stem))

        for method in manipulation_methods:
            fake_base = find_fake_dir(method)
            if fake_base and fake_base.exists():
                for v in fake_base.glob("*.mp4"):
                    video_records.append((v, "fake", f"{method}_{v.stem}"))

    return video_records


def run_extraction(
    config_path: str | Path,
    split: str,
) -> None:
    """Execute frame extraction pipeline for a given split.

    Args:
        config_path: Path to YAML config.
        split: Dataset split ('train', 'val', 'test').
    """
    config = load_config(config_path)
    device = get_device(config.get("device", "auto"))
    logger.info("Using device: %s", device)

    crops_dir = Path(config.get("crops_dir", "video/data/crops"))
    face_margin = float(config.get("face_margin", 0.2))
    image_size = int(config.get("image_size", 224))
    max_frames = int(config.get("frames_per_video", 32))

    mtcnn = None
    if MTCNN is not None:
        try:
            mtcnn = MTCNN(
                keep_all=False,
                select_largest=True,
                post_process=False,
                device=device,
            )
            logger.info("Initialized MTCNN face detector.")
        except Exception as e:
            logger.warning("Could not initialize MTCNN with device %s: %s. Trying CPU.", device, e)
            try:
                mtcnn = MTCNN(
                    keep_all=False,
                    select_largest=True,
                    post_process=False,
                    device=torch.device("cpu"),
                )
            except Exception as e_cpu:
                logger.error("Failed to initialize MTCNN on CPU: %s", e_cpu)
    else:
        logger.warning("facenet-pytorch is not installed. MTCNN face detector unavailable.")

    videos = collect_ffpp_split_videos(config, split)
    logger.info("Found %d videos for split '%s'.", len(videos), split)

    if config.get("smoke_test", False):
        limit = int(config.get("smoke_test_videos", 50))
        reals = [v for v in videos if v[1] == "real"]
        fakes = [v for v in videos if v[1] == "fake"]
        half = max(1, limit // 2)
        videos = reals[:half] + fakes[:half]
        logger.info("Smoke test enabled: sampled %d real and %d fake videos (%d total).", len(reals[:half]), len(fakes[:half]), len(videos))

    total_extracted = 0
    for video_path, label_str, video_id in tqdm(videos, desc=f"Extracting {split}"):
        target_dir = crops_dir / split / label_str / video_id
        count = extract_faces_from_video(
            video_path=video_path,
            output_dir=target_dir,
            mtcnn=mtcnn,
            face_margin=face_margin,
            image_size=image_size,
            max_frames=max_frames,
        )
        total_extracted += count

    logger.info(
        "Finished extraction for split '%s'. Processed %d videos, saved %d face crops.",
        split,
        len(videos),
        total_extracted,
    )


def main() -> None:
    """Parse CLI arguments and run extraction."""
    parser = argparse.ArgumentParser(description="Extract face crops from FaceForensics++ videos.")
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
        default="val",
        help="Dataset split to extract (train, val, or test).",
    )
    args = parser.parse_args()
    run_extraction(config_path=args.config, split=args.split)


if __name__ == "__main__":
    main()
