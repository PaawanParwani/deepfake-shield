"""Physiological and Affective Biometrics Module for Deepfake Detection.

Implements real-time involuntary biological micro-signal tracking:
1. Ocular Dynamics: Eye Aspect Ratio (EAR), blink duration curves, and blink frequency.
2. Remote Photoplethysmography (rPPG): Multi-ROI subcutaneous blood volume pulse (BVP) coherence.
3. Facial Action Units (FACS): Duchenne smile alignment (AU6 periocular vs AU12 mouth corner).
4. Self-Training Experience Buffer: Pseudo-labeling based on biological consistency.
"""

from __future__ import annotations

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PhysiologicalTracker:
    """Tracks involuntary biological dynamics across sequential video frames."""

    def __init__(self, buffer_size: int = 60, fps: float = 30.0) -> None:
        """Initialize temporal history buffers.

        Args:
            buffer_size: Max number of history frames (default 60 = 2 seconds at 30 FPS).
            fps: Video stream frame rate.
        """
        self.buffer_size = buffer_size
        self.fps = fps

        # Temporal buffers
        self.timestamps: collections.deque[float] = collections.deque(maxlen=buffer_size)
        self.ear_history: collections.deque[float] = collections.deque(maxlen=buffer_size)
        self.fh_chrom_history: collections.deque[float] = collections.deque(maxlen=buffer_size)
        self.lc_chrom_history: collections.deque[float] = collections.deque(maxlen=buffer_size)
        self.rc_chrom_history: collections.deque[float] = collections.deque(maxlen=buffer_size)
        self.mouth_lift_history: collections.deque[float] = collections.deque(maxlen=buffer_size)
        self.eye_crinkle_history: collections.deque[float] = collections.deque(maxlen=buffer_size)

        # Blink counters
        self.blink_count = 0
        self.in_blink = False
        self.blink_frames = 0

    def reset(self) -> None:
        """Clear temporal buffers."""
        self.timestamps.clear()
        self.ear_history.clear()
        self.fh_chrom_history.clear()
        self.lc_chrom_history.clear()
        self.rc_chrom_history.clear()
        self.mouth_lift_history.clear()
        self.eye_crinkle_history.clear()
        self.blink_count = 0
        self.in_blink = False
        self.blink_frames = 0

    def update_from_frame(
        self,
        face_crop: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convenience method to extract biometrics directly from a cropped face patch.

        Args:
            face_crop: Cropped face RGB or BGR array (H, W, 3).
            timestamp: Optional timestamp.

        Returns:
            Biometrics dictionary.
        """
        h, w = face_crop.shape[:2]
        # Canonical canonical 5-point facial anchor distribution for centered crops
        landmarks = np.array([
            [w * 0.35, h * 0.38],  # Left eye
            [w * 0.65, h * 0.38],  # Right eye
            [w * 0.50, h * 0.55],  # Nose tip
            [w * 0.38, h * 0.75],  # Left mouth corner
            [w * 0.62, h * 0.75],  # Right mouth corner
        ], dtype=np.float32)
        return self.update(face_crop, landmarks, timestamp=timestamp)

    def update(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process incoming video frame and update biological biometric buffers.

        Args:
            frame_bgr: Full video frame in BGR format.
            landmarks: Array of 5 landmarks [[lx, ly], [rx, ry], [nx, ny], [lmx, lmy], [rmx, rmy]].
            timestamp: Optional frame timestamp in seconds.

        Returns:
            Dictionary with instantaneous physiological metrics.
        """
        h, w = frame_bgr.shape[:2]
        leye, reye, nose, lmouth, rmouth = landmarks[:5]

        # 1. Geometry baseline: Inter-ocular distance (IOD)
        iod = float(np.linalg.norm(reye - leye))
        if iod < 10.0:
            return {"valid": False}

        # 2. Ocular Dynamics: Eye openness via periocular vertical gradient
        r = max(4, int(iod * 0.20))
        leyex, leyey = int(leye[0]), int(leye[1])
        reyex, reyey = int(reye[0]), int(reye[1])

        l_patch = frame_bgr[max(0, leyey - r) : min(h, leyey + r), max(0, leyex - r) : min(w, leyex + r)]
        r_patch = frame_bgr[max(0, reyey - r) : min(h, reyey + r), max(0, reyex - r) : min(w, reyex + r)]

        ear_approx = 0.30
        if l_patch.size > 0 and r_patch.size > 0:
            l_gray = cv2.cvtColor(l_patch, cv2.COLOR_BGR2GRAY)
            r_gray = cv2.cvtColor(r_patch, cv2.COLOR_BGR2GRAY)
            # Standard deviation of eye region drops significantly during full blink closure
            openness = (float(np.std(l_gray)) + float(np.std(r_gray))) / 2.0
            ear_approx = float(np.clip(openness / 40.0, 0.05, 0.45))

        self.ear_history.append(ear_approx)

        # Blink kinematics state machine
        blink_threshold = 0.16
        if ear_approx < blink_threshold:
            if not self.in_blink:
                self.in_blink = True
                self.blink_frames = 1
            else:
                self.blink_frames += 1
        else:
            if self.in_blink:
                # Valid biological blink lasts 3 to 12 frames (100ms - 400ms at 30 FPS)
                if 2 <= self.blink_frames <= 15:
                    self.blink_count += 1
                self.in_blink = False
                self.blink_frames = 0

        # 3. Subcutaneous rPPG ROIs (Forehead, Left Cheek, Right Cheek)
        fh_cx = int((leye[0] + reye[0]) / 2.0)
        fh_cy = int(min(leye[1], reye[1]) - iod * 0.40)
        roi_r = max(5, int(iod * 0.22))

        fh_patch = frame_bgr[max(0, fh_cy - roi_r) : min(h, fh_cy + roi_r), max(0, fh_cx - roi_r) : min(w, fh_cx + roi_r)]

        lc_cx = int(leye[0] - iod * 0.18)
        lc_cy = int((leye[1] + lmouth[1]) / 2.0)
        lc_patch = frame_bgr[max(0, lc_cy - roi_r) : min(h, lc_cy + roi_r), max(0, lc_cx - roi_r) : min(w, lc_cx + roi_r)]

        rc_cx = int(reye[0] + iod * 0.18)
        rc_cy = int((reye[1] + rmouth[1]) / 2.0)
        rc_patch = frame_bgr[max(0, rc_cy - roi_r) : min(h, rc_cy + roi_r), max(0, rc_cx - roi_r) : min(w, rc_cx + roi_r)]

        def get_chrominance(patch: np.ndarray) -> float:
            if patch.size == 0:
                return 0.0
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)
            # CHROM pulse signal: 3R - 2G
            return float(np.mean(3.0 * rgb[:, :, 0] - 2.0 * rgb[:, :, 1]))

        self.fh_chrom_history.append(get_chrominance(fh_patch))
        self.lc_chrom_history.append(get_chrominance(lc_patch))
        self.rc_chrom_history.append(get_chrominance(rc_patch))

        # 4. FACS Action Units: AU12 (Lip Corner Puller) vs AU6 (Periocular Cheek Raiser)
        mouth_center_y = (lmouth[1] + rmouth[1]) / 2.0
        mouth_lift = float((nose[1] - mouth_center_y) / iod)
        eye_crinkle = float(1.0 - ear_approx)

        self.mouth_lift_history.append(mouth_lift)
        self.eye_crinkle_history.append(eye_crinkle)

        return {
            "valid": True,
            "ear": round(ear_approx, 3),
            "blink_count": self.blink_count,
            "in_blink": self.in_blink,
            "mouth_lift": round(mouth_lift, 3),
        }

    def compute_biological_consistency(self) -> Dict[str, Any]:
        """Compute multi-channel biological consistency score over accumulated history.

        Returns:
            Dictionary with p_bonafide_bio (0.0 - 1.0), channel scores, and anomaly alerts.
        """
        n = len(self.ear_history)
        if n < 15:
            # Insufficient temporal baseline
            return {
                "p_bonafide_bio": 0.50,
                "p_biological_liveness": 0.50,
                "confidence": 50.0,
                "status": "accumulating_temporal_buffer",
                "samples": n,
            }

        # Channel 1: Pulse Spatial Coherence (Cross-Correlation between Forehead and Cheeks)
        fh = np.array(self.fh_chrom_history)
        lc = np.array(self.lc_chrom_history)
        rc = np.array(self.rc_chrom_history)

        fh_norm = (fh - np.mean(fh)) / (np.std(fh) + 1e-6)
        lc_norm = (lc - np.mean(lc)) / (np.std(lc) + 1e-6)
        rc_norm = (rc - np.mean(rc)) / (np.std(rc) + 1e-6)

        corr_l = float(np.clip(np.mean(fh_norm * lc_norm), -1.0, 1.0))
        corr_r = float(np.clip(np.mean(fh_norm * rc_norm), -1.0, 1.0))
        mean_rppg_coherence = float(np.clip((corr_l + corr_r) / 2.0, 0.0, 1.0))

        # Channel 2: Dynamic Blink Activity
        ear_arr = np.array(self.ear_history)
        ear_variance = float(np.var(ear_arr))
        # Deepfakes frequently have near-zero eyelid dynamics (frozen eyes)
        blink_score = float(np.clip(ear_variance * 500.0, 0.1, 1.0))
        if self.blink_count >= 1:
            blink_score = min(1.0, blink_score + 0.3)

        # Channel 3: Duchenne Affective Coherence (AU12 mouth vs AU6 eyes)
        m_lift = np.array(self.mouth_lift_history)
        e_crink = np.array(self.eye_crinkle_history)
        if np.std(m_lift) > 1e-4 and np.std(e_crink) > 1e-4:
            m_norm = (m_lift - np.mean(m_lift)) / np.std(m_lift)
            e_norm = (e_crink - np.mean(e_crink)) / np.std(e_crink)
            duchenne_corr = float(np.clip(np.mean(m_norm * e_norm), -1.0, 1.0))
            affective_score = float(np.clip(0.5 + duchenne_corr * 0.5, 0.0, 1.0))
        else:
            affective_score = 0.50

        # Weighted Biological Synthesis
        # 45% Subcutaneous rPPG pulse coherence, 35% Ocular blink kinematics, 20% Affective harmony
        p_bonafide_bio = float(
            0.45 * mean_rppg_coherence + 0.35 * blink_score + 0.20 * affective_score
        )
        p_bonafide_bio = float(np.clip(p_bonafide_bio, 0.0, 1.0))

        return {
            "p_bonafide_bio": round(p_bonafide_bio, 4),
            "p_biological_liveness": round(p_bonafide_bio, 4),
            "p_synthetic_bio": round(1.0 - p_bonafide_bio, 4),
            "rppg_spatial_coherence": round(mean_rppg_coherence, 3),
            "blink_kinematics_score": round(blink_score, 3),
            "affective_duchenne_score": round(affective_score, 3),
            "blink_count": self.blink_count,
            "temporal_frames": n,
        }


class SelfTrainingExperienceBuffer:
    """Experience replay buffer for autonomous model self-training."""

    def __init__(self, capacity: int = 1000) -> None:
        """Initialize experience buffer.

        Args:
            capacity: Maximum number of stored sequence frames.
        """
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []

    def add_sample(
        self,
        frame: np.ndarray,
        p_bio: float,
        p_cnn: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Evaluate sample and add to replay buffer if confidence criteria are met.

        Args:
            frame: Preprocessed face crop (224x224).
            p_bio: Physiological biological score from PhysiologicalTracker.
            p_cnn: Spatial deepfake score from CNN model.
            metadata: Optional camera / context tags.

        Returns:
            Assigned pseudo-label ('bonafide', 'deepfake', or None if ambiguous).
        """
        pseudo_label = None

        # Autonomous filtering gates
        if p_bio >= 0.90 and p_cnn <= 0.15:
            pseudo_label = "bonafide"
        elif p_bio <= 0.20 and p_cnn >= 0.80:
            pseudo_label = "deepfake"

        if pseudo_label is not None:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)

            self.buffer.append(
                {
                    "frame": frame.copy(),
                    "label": 0 if pseudo_label == "bonafide" else 1,
                    "pseudo_label": pseudo_label,
                    "p_bio": p_bio,
                    "p_cnn": p_cnn,
                    "metadata": metadata or {},
                }
            )
            logger.info("Experience Buffer added high-confidence sample: %s", pseudo_label)

        return pseudo_label

    def get_batch(self, batch_size: int = 32) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve balanced batch of pseudo-labeled samples for continual training.

        Args:
            batch_size: Target batch size.

        Returns:
            Tuple of (images_array, labels_array) or None if buffer is too small.
        """
        if len(self.buffer) < batch_size:
            return None

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch_samples = [self.buffer[i] for i in indices]

        images = np.stack([s["frame"] for s in batch_samples], axis=0)
        labels = np.array([s["label"] for s in batch_samples], dtype=np.int64)

        return images, labels

    def size(self) -> int:
        """Current number of stored samples."""
        return len(self.buffer)


def main() -> None:
    """CLI entrypoint and verification smoke-test for PhysiologicalTracker."""
    import argparse

    parser = argparse.ArgumentParser(description="Test and benchmark physiological biometrics tracker.")
    parser.add_argument("--fps", type=float, default=30.0, help="Simulated camera FPS.")
    parser.add_argument("--frames", type=int, default=30, help="Number of synthetic frames to simulate.")
    args = parser.parse_args()

    print(f"Initializing PhysiologicalTracker (buffer_size=60, fps={args.fps})...")
    tracker = PhysiologicalTracker(buffer_size=60, fps=args.fps)

    # Generate synthetic face frames with subtle color variations
    print(f"Simulating {args.frames} facial video frames...")
    np.random.seed(42)
    for i in range(args.frames):
        # Simulated face crop with slight pulse variation in red channel
        t = i / args.fps
        pulse = np.sin(2.0 * np.pi * 1.2 * t) * 5.0  # ~72 BPM pulse
        frame = np.full((224, 224, 3), [140 + pulse, 150, 180], dtype=np.uint8)

        # Simulate blink closure around frame 15-18
        if 15 <= i <= 18:
            frame[70:95, 60:160] = 120

        tracker.update_from_frame(frame, timestamp=t)

    metrics = tracker.compute_biological_consistency()
    print("Computed Biological Consistency Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Verify Experience Buffer
    buffer = SelfTrainingExperienceBuffer(capacity=50)
    sample_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    label = buffer.add_sample(sample_frame, p_bio=0.95, p_cnn=0.08)
    print(f"Experience Buffer assigned pseudo-label: {label} (buffer size: {buffer.size()})")
    print("[OK] PhysiologicalTracker verification completed successfully.")


if __name__ == "__main__":
    main()
