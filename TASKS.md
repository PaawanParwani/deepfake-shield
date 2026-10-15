# TASKS

Legend: [ ] todo, [~] in progress, [x] done. Owner: A = cybersecurity, B = data science.

## Setup (both)
- [x] Create project with `/audio`, `/video`, `/demo`, `/docs`, `/paper`, `/dissertation`
- [x] Add `.gitignore` (data, checkpoints, .env, wandb/)
- [x] Add `requirements.txt` from `docs/03_TECH_STACK.md`
- [x] Create accounts: Hugging Face, Weights & Biases, Colab Pro
- [x] Both read `docs/01_SCOPE.md` and agree it is locked

## Audio track (A)
- [x] Download ASVspoof2019 LA + protocol files
- [x] `audio/src/dataset.py` with multi-format support (Parquet + FLAC)
- [x] `audio/src/model.py` wav2vec2 + head
- [x] `audio/src/train.py` runs end to end with mixed precision
- [x] Full training run completed (1 epoch on 500 samples, CPU)
- [x] `audio/src/eval.py`: EER, AUC, accuracy on dev (EER: 27.69%, AUC: 81.21%) and eval (EER: 32.52%)
- [x] CHECKPOINT: Real EER number saved to `audio/outputs/metrics.json` and `metrics_dev.json`
- [x] `audio/src/attack.py`: PGD + noise sweep (completed)
- [x] Adversarial results CSV + degradation plot (saved to `audio/outputs/adversarial_results.csv` and `audio_adversarial_degradation.png`)
- [x] Checkpoint loader support for PyTorch state_dict in demo backend

## Video track (B)
- [x] Download FaceForensics++ c23 data (extracted at `video/data/FaceForensics++/FaceForensics++_C23`)
- [x] `video/src/extract_frames.py` with dual layout auto-detection + MTCNN crops
- [x] `video/src/dataset.py` (strict video identity split)
- [x] `video/src/model.py` EfficientNet-B0 head
- [x] `video/src/train.py` end-to-end training (`video/outputs/best_model.pt`)
- [x] `video/src/eval.py`: frame AUC (67.87%) + video AUC (83.00%), per-method breakdown
- [x] `video/src/attack.py`: PGD/FGSM + JPEG compression sweep (`video/outputs/adversarial_results.csv` and `adversarial_degradation.png`)
- [x] Model integration and checkpoint loading verified
- [x] Fixed Zero-Face safety gate & 1:1 aspect-ratio square crop normalization with replicate padding

## Demo & Real-Time Deployment (both)
- [x] `demo/backend/app.py` with `/predict/audio`, `/predict/video`, `/predict/frame`, `/health`, `/physiology/status`, and `/predict/multimodal`
- [x] `demo/frontend/index.html` glassmorphic UI, audio/video upload, live scoring
- [x] Real-Time Chrome Extension (`extension/` Manifest V3, auto-detects Google Meet/Zoom/Teams, live AR HUD, popup with webcam test, zero lag, smooth drag)
- [x] Combine verdicts (either modality > 0.5 => Suspicious) + Calibrated Bayesian Reliability Gating
- [x] Script 60-90 s demo video for YC / academic presentation (`docs/15_DEMO_VIDEO_SCRIPT.md`)

## Writing & Academic Research
- [x] Threat model section drafted
- [x] Dissertation complete: `dissertation/DISSERTATION.md` (6 comprehensive chapters, empirical tables for audio and video clean & adversarial results, real-time safety architecture)
- [x] Research paper complete: `paper/PAPER.md` (IEEE Transactions format with complete mathematical formalisms)
- [x] YC application answers per `docs/11_YC_APPLICATION.md`
- [x] Advanced roadmap & mobile blueprint: `docs/13_ADVANCED_ROADMAP_AND_MOBILE.md`
- [x] Self-Supervised Physiological Architecture plan: `docs/14_SELF_SUPERVISED_PHYSIOLOGICAL_DYNAMICS_PLAN.md`
- [x] 60-90s Demo Video & Pitch Script: `docs/15_DEMO_VIDEO_SCRIPT.md`

## NeuroPulse: Self-Learning Physiological Biometrics (Autonomous Next Phase)
- [x] Google Colab Master Training Pipeline (`notebooks/Deepfake_Detection_Colab_Master.ipynb`)
- [x] Implement MediaPipe FaceMesh EAR (Eye Aspect Ratio) biological blink kinematics tracker (`video/src/physiology.py`)
- [x] Implement Facial Action Unit (FACS) Duchenne consistency checker (`video/src/physiology.py`)
- [x] Implement rPPG subcutaneous blood volume pulse (BVP) spatial coherence extractor (`video/src/physiology.py`)
- [x] Experience Replay Buffer + Physiological Oracle Pseudo-labeler for autonomous self-training (`video/src/physiology.py`)
- [x] Elastic Weight Consolidation (EWC) engine for continual adaptation without catastrophic forgetting (`video/src/continual_learning.py`)
- [x] Dual-Stream Spatial-Frequency CNN & Spatio-Temporal Biometric Transformer (`video/src/dual_stream_model.py`)
- [x] Export to CoreML (`.mlpackage`) for native iOS SwiftUI app (`mobile/ios/DeepfakeDetectorView.swift`) and ONNX Mobile for Android (`mobile/android/DeepfakeAnalyzer.kt`)
- [x] 1-Click Public SaaS Deployment on Hugging Face Spaces free tier (`deploy/deploy_to_huggingface.py` with auto-token discovery and asset filtering)

## Defenses & Multimodal Fusion
- [x] Defense: adversarial training (audio) (`audio/src/defense.py` with Randomized Quantization, Low-pass Acoustic Denoising, PGD-AT)
- [x] Defense: adversarial training / preprocessing (video) (`video/src/defense.py` with Randomized Resizing/Padding, Randomized JPEG Compression Q in [70, 90], Gaussian Smoothing, PGD-AT)
- [x] Learned cross-modal attention fusion of audio & visual scores (`demo/backend/fusion.py` with `CrossAttentionMultimodalFusion` and calibrated dynamic gating)

## Advanced Forensic Explainability & Research Delivery
- [x] Grad-CAM Visual Explainability module (`video/src/explainability.py` with high-resolution synthetic artifact highlighting)
- [x] Real-time Forensic Heatmap API (`POST /explain/frame` in `demo/backend/app.py`)
- [x] Publication-ready IEEE Transactions LaTeX package (`paper/main.tex` and `paper/references.bib`)
- [x] Gold-standard dataset acquisition and validation tool (`data_tools/download_datasets.py` for Celeb-DF v2, DFDC, and FaceForensics++)
- [x] Comprehensive performance profiler and benchmark report (`scripts/benchmark_suite.py` and `BENCHMARK_REPORT.md` confirming 19.2x real-time audio and 7,358 FPS biometrics)
