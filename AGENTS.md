# AGENTS.md - Instructions for the AI coding agent

You are helping two students build a deepfake detection system (audio + video) for a dissertation, a research paper, and a startup application. Read `docs/01_SCOPE.md` before doing anything.

## Hard rules
1. **Stay in scope.** Do not add features outside `docs/01_SCOPE.md`. If a request seems out of scope, say so and ask before building.
2. **Python only** for ML, evaluation and backend. Plain HTML/CSS/JS for the demo frontend. No frontend frameworks.
3. **PyTorch only.** No TensorFlow/Keras.
4. **Pretrained over from-scratch.** Fine-tune wav2vec2 (audio) and EfficientNet/ViT via `timm` (video). Do not design custom architectures.
5. **Keep tracks separate.** Code in `/audio` must not import from `/video` and vice versa. Only `/demo` may use both.
6. **Reproducibility.** Set random seeds (42), log every run to Weights & Biases, save configs next to checkpoints, write metrics to `outputs/metrics.json`.
7. **No data leakage.** Use the official train/dev/eval splits. For video, split by *video identity*, never by frame.
8. **Never commit** datasets, checkpoints, API keys or `.env` files. Respect `.gitignore`.
9. **Datasets are for academic use only.** Do not redistribute them.
10. **Small, working steps.** Always get an end-to-end pipeline running on a tiny subset first, then scale.

## Code style
- Python 3.10+, type hints, docstrings on public functions.
- One responsibility per file. Config via a YAML file or dataclass, not hard-coded constants.
- Every script runnable via CLI with `argparse` and `--help`.
- Write a short test or smoke-run for each module.

## Evaluation conventions
- Audio metrics: **EER** (primary), accuracy, AUC.
- Video metrics: **AUC** (primary), accuracy, per-manipulation-method breakdown.
- Adversarial metrics: metric on clean data vs attacked data, plus attack strength (epsilon / SNR).
- Always report results as tables saved to `outputs/` and also plotted.

## When uncertain
Ask. Do not invent dataset paths, paper results, or citations. If you cite a paper, only cite ones the user supplied or that you can verify.

## Definition of done for any task
Code runs end to end, metrics are logged, results saved to `outputs/`, `TASKS.md` updated.
