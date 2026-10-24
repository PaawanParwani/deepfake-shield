"""Export PyTorch Deepfake Classifier to iOS CoreML (.mlpackage) for Apple Neural Engine (ANE).

Usage:
    python mobile/ios/export_coreml.py --checkpoint video/outputs/best_model.pt --output mobile/ios/DeepfakeDetector.mlpackage
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from video.src.model import DeepfakeClassifier

class NormalizedModel(nn.Module):
    """Wrapper that normalizes [0, 1] input image tensors using ImageNet parameters."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.mean) / self.std
        return self.base_model(normalized)


def export_to_coreml(checkpoint_path: str, output_path: str) -> None:
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    model = DeepfakeClassifier(num_classes=2, pretrained=False)
    ckpt = torch.load(ckpt_file, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    model_keys = set(model.state_dict().keys())
    clean_state_dict = {}
    for k, v in state_dict.items():
        k_clean = k.replace("model.", "").replace("module.", "")
        if k_clean in model_keys:
            clean_state_dict[k_clean] = v
        elif ("backbone." + k_clean) in model_keys:
            clean_state_dict["backbone." + k_clean] = v
        elif k_clean.replace("backbone.", "") in model_keys:
            clean_state_dict[k_clean.replace("backbone.", "")] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    norm_model = NormalizedModel(model)
    norm_model.eval()

    try:
        import coremltools as ct
    except ImportError:
        print("Notice: 'coremltools' is required for CoreML export. Install with: pip install coremltools")
        return

    try:
        print("Tracing model with dummy [0, 1] input...")
        dummy_input = torch.rand(1, 3, 224, 224)
        traced_model = torch.jit.trace(norm_model, dummy_input)

        print("Converting to Apple CoreML format with ImageType input...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.ImageType(
                    name="image",
                    shape=(1, 3, 224, 224),
                    scale=1.0 / 255.0,
                    bias=[0.0, 0.0, 0.0],
                    color_layout=ct.colorlayout.RGB,
                )
            ],
            classifier_config=ct.ClassifierConfig(class_labels=["Bonafide", "Deepfake"]),
            minimum_deployment_target=ct.target.iOS16,
        )

        mlmodel.author = "Antigravity Research"
        mlmodel.license = "Apache 2.0"
        mlmodel.short_description = "Real-time dual-modality deepfake detection for iOS FaceTime / video calls"

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(out_path))
        print(f"[OK] Successfully exported iOS CoreML package to: {out_path}")
        print("Ready for drag-and-drop into Xcode!")

    except Exception as e:
        print(f"CoreML export error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export to CoreML")
    parser.add_argument("--checkpoint", default="video/outputs/best_model.pt", help="Path to PyTorch checkpoint")
    parser.add_argument("--output", default="mobile/ios/DeepfakeDetector.mlpackage", help="Output .mlpackage path")
    args = parser.parse_args()

    export_to_coreml(args.checkpoint, args.output)
