# Adversarial Evaluation Plan (shared contribution)

## Purpose
Show how much each detector degrades when an attacker tries to make a fake look real. This is the core of the paper.

## Threat model
- **White-box**: attacker has the model weights (worst case, upper bound on risk)
- **Signal-level / black-box**: attacker only manipulates the media (noise, compression, resampling)

## Attacks (pick ONE per modality minimum)
| Modality | Primary | Optional extras |
|---|---|---|
| Audio | PGD (L-inf) on the waveform, targeting "bonafide" on spoofed clips | Gaussian noise at varied SNR, MP3 compression, resampling |
| Video | PGD / FGSM on face crops, targeting "real" on fake frames | JPEG compression, blur, downscaling |

Use `torchattacks` where possible; adapt for the wav2vec2 input if needed.

## Protocol
1. Take only **correctly detected fake samples** from the test/eval set.
2. Attack them at several strengths (e.g. audio SNR 40/30/20 dB or epsilon 1e-4..1e-2; video epsilon 1/255, 2/255, 4/255, 8/255).
3. Re-run the detector. Record attack success rate and metric (EER/AUC) clean vs attacked.
4. Save a results table and a degradation curve (metric vs attack strength).

## Output format
`outputs/adversarial_results.csv` with columns: `modality, attack, strength, metric, clean_value, attacked_value, attack_success_rate`

## Stretch: defenses
- Adversarial training (add attacked samples to training)
- Input preprocessing (denoise, re-encode)
Only start after clean attack numbers exist for both modalities.

## Ethics note
Attacks are run only against our own models, on public research datasets, for defensive research. Do not release attack tooling aimed at deployed third-party systems.
