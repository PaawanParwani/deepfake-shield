# Architecture

## Overview
Two independent detection pipelines feed one demo.

```
 audio clip ──► resample 16kHz ──► wav2vec2 (fine-tuned) ──► P(spoof) ─┐
                                                                        ├─► combiner ──► Verified / Suspicious
 video clip ──► sample frames ──► MTCNN face crop ──► EfficientNet/ViT ─┘
                                   (per-frame P(fake), averaged)
```

## Audio pipeline
- Input: wav/flac, resampled to 16 kHz mono, cropped/padded to ~4 s
- Model: `facebook/wav2vec2-base` (or `wav2vec2-xls-r-300m` if compute allows) + classification head
- Output: P(spoof) in [0,1]
- Export: ONNX for demo

## Video pipeline
- Input: mp4, sample ~1 frame/sec (max ~32 frames)
- Face detection/crop: MTCNN (`facenet-pytorch`), resize 224x224
- Model: `timm` EfficientNet-B0 (or ViT-small) with 2-class head
- Aggregation: mean of per-frame P(fake)
- Export: ONNX for demo

## Combiner (simple, rule-based)
- Verdict = "Suspicious" if either modality P > 0.5
- Display both scores separately
- No learned fusion in this project (future work)

## Demo
- Backend: FastAPI, endpoints `POST /predict/audio`, `POST /predict/video`, `GET /health`
- Loads ONNX models once at startup
- Frontend: single HTML page with two file inputs and result cards

## Threat model (summary, full version goes in dissertation)
| Attacker goal | Capability | Example attack |
|---|---|---|
| Make a fake pass as real | White-box (knows model) | Gradient-based perturbation (FGSM/PGD) |
| Make a fake pass as real | Black-box / signal-level | Added noise, compression, re-recording, resampling |
| Spoof the "verified" badge itself | Network/UI access | Out of scope, document as limitation |
