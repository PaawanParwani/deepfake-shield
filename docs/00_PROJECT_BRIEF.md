# Project Brief

## Problem
Voice clones and face-swap deepfakes are now cheap and convincing. In a widely reported case, a finance worker wired $25M after a video call where every other participant was a deepfake. Traditional trust signals (seeing a face, hearing a voice) no longer work, and most published detectors are rarely tested against attackers who adapt.

## Idea
Build detectors for synthetic **voice** and synthetic **face video**, then rigorously measure how much each degrades under adversarial attack. Combine both signals into one "Verified Human / Suspicious" verdict in a demo.

## Why this is novel enough
Most detection work reports accuracy on clean benchmarks. Our contribution is a **security-minded evaluation**: a threat model plus measured robustness for both modalities side by side.

## Three outputs
| Output | What it needs |
|---|---|
| Dissertation | Full write-up: method, threat model, results, limitations |
| Research paper | Focus on the adversarial robustness findings |
| YC application | Working demo + founder story (security + data science team) |

## Success criteria (minimum)
- [ ] Audio classifier with reported EER on ASVspoof2019 LA eval
- [ ] Video classifier with reported AUC on FaceForensics++ test
- [ ] At least one attack per modality with clean vs attacked numbers
- [ ] Demo that accepts a clip and returns a verdict
- [ ] Dissertation draft complete

## Success criteria (stretch)
- [ ] A defense (adversarial training or preprocessing) per modality
- [ ] Fused audio+video decision
- [ ] Live-call integration
