# Video Track Spec (owner: Partner B, data science)

## Goal
Binary classifier: real vs manipulated face video, evaluated by **AUC** on the FaceForensics++ test split.

## Steps
1. **Data**: download FF++ (c23). Use official train/val/test split JSONs. Real = `original_sequences`, fake = a chosen subset of manipulation methods (e.g. Deepfakes, Face2Face, FaceSwap, NeuralTextures).
2. **Frame extraction**: ~1 frame/sec, max ~32 frames per video, cache to disk.
3. **Face crops**: MTCNN, add ~20% margin, resize 224x224. Skip frames with no detected face.
4. **Model**: `timm` `efficientnet_b0` pretrained, 2-class head. Augment with flip, mild color jitter, JPEG compression.
5. **Train**: AdamW, lr ~1e-4, batch 32, mixed precision, 5-10 epochs. Log to W&B.
6. **Evaluate**: frame-level and video-level (mean prob) AUC and accuracy. Break down per manipulation method.
7. **Attack** (see `07_ADVERSARIAL_EVAL.md`).
8. **Export** to ONNX.

## Files to create
```
video/src/extract_frames.py  video/src/dataset.py   video/src/model.py
video/src/train.py           video/src/eval.py      video/src/attack.py
video/src/export_onnx.py     video/configs/baseline.yaml
```

## Critical rule
**Split by video, never by frame.** Frames from the same video must stay in the same split or results are meaningless.

## Pitfalls
- Compression level mismatch hurts generalization; state which quality you used.
- Cross-method generalization is weak; report it honestly as a limitation.
- Cache extracted crops. Re-extracting every epoch wastes Colab hours.
