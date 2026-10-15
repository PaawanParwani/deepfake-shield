# Scope (LOCKED)

## IN scope
- Audio deepfake detection: ASVspoof2019 LA, fine-tuned wav2vec2
- Video deepfake detection: FaceForensics++, fine-tuned EfficientNet or ViT (timm)
- One adversarial attack per modality, measured
- Threat model document covering both modalities
- Demo: upload an audio clip and/or a video clip, get a verdict
- Dissertation, paper draft, YC application draft

## OUT of scope (do not build, do not suggest)
- Live real-time call integration (Zoom/Meet/WebRTC)
- Custom neural network architectures
- Training from scratch
- Defense mechanisms (STRETCH ONLY, after attacks are measured)
- Mobile apps, user accounts, databases, payments
- Datasets beyond one per modality
- Frontend frameworks

## Cut order if time runs short (cut from top)
1. Fused audio+video decision (ship two separate verdicts instead)
2. Adversarial testing on whichever modality is behind
3. Demo polish
Never cut: working baseline classifiers, and the written dissertation.

## Rule
Each track must be independently submittable. One person's delay must never block the other.
