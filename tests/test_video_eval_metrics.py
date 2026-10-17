import unittest

from video.src.eval import compute_evaluation_breakdown


class VideoEvaluationMetricsTests(unittest.TestCase):
    def test_per_method_breakdown_uses_video_metadata(self) -> None:
        video_records = {
            "real-1": {"probs": [0.1], "label": 0, "method": "real"},
            "real-2": {"probs": [0.2], "label": 0, "method": "real"},
            "Deepfakes_1": {"probs": [0.8], "label": 1, "method": "Deepfakes"},
            "Deepfakes_2": {"probs": [0.9], "label": 1, "method": "Deepfakes"},
        }
        frame_records = [
            {
                "prob": float(record["probs"][0]),
                "label": int(record["label"]),
                "video_id": video_id,
                "method": str(record["method"]),
            }
            for video_id, record in video_records.items()
        ]

        metrics = compute_evaluation_breakdown(
            frame_records,
            video_records,
            ["Deepfakes"],
        )

        self.assertEqual(metrics["overall"]["total_videos"], 4)
        self.assertEqual(metrics["per_method"]["Deepfakes"]["num_videos"], 2)
        self.assertEqual(metrics["per_method"]["Deepfakes"]["video_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
