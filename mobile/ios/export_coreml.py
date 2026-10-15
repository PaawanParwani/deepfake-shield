"""Export PyTorch Deepfake Classifier to iOS CoreML (.mlpackage) for Apple Neural Engine (ANE).

Usage:
    python mobile/ios/export_coreml.py --checkpoint video/outputs/best_model.pt --output mobile/ios/DeepfakeDetector.mlpackage
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from video.src.model import DeepfakeClassifier

def export_to_coreml(checkpoint_path: str, output_path: str) -> None:
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = DeepfakeClassifier(num_classes=2, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, dummy_input)

    try:
        import coremltools as ct
        print("Converting to Apple CoreML format with ImageType input...")
        
        # ImageNet normalization parameters
        scale = 1.0 / (255.0 * 0.226)
        red_bias = -0.485 / 0.229
        green_bias = -0.456 / 0.224
        blue_bias = -0.406 / 0.225

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

        mlmodel.save(output_path)
        print(f"✅ Successfully exported iOS CoreML package to: {output_path}")
        print("Ready for drag-and-drop into Xcode!")

    except ImportError:
        print("Notice: 'coremltools' is required for CoreML export. Install with: pip install coremltools")
    except Exception as e:
        print(f"CoreML export error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export to CoreML")
    parser.add_argument("--checkpoint", default="video/outputs/best_model.pt", help="Path to PyTorch checkpoint")
    parser.add_argument("--output", default="mobile/ios/DeepfakeDetector.mlpackage", help="Output .mlpackage path")
    args = parser.parse_args()

    export_to_coreml(args.checkpoint, args.output)
