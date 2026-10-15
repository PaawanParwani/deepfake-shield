# Timeline

Deadline: **end of October 2026**.

> IMPORTANT: Pick the plan that matches how many days you actually have left. The 13-week plan assumes a start in early August. If you are starting later, use the compressed plan.

## Plan A: 13 weeks (full)
| Weeks | Audio (Partner A) | Video (Partner B) |
|---|---|---|
| 1-2 | Setup, read papers, download ASVspoof2019 LA | Setup, read papers, download FF++ |
| 3-5 | Fine-tune wav2vec2 baseline. **Checkpoint: EER number** | Fine-tune EfficientNet baseline. **Checkpoint: AUC number** |
| 6-8 | 1 attack, measured | 1 attack, measured |
| 9 | Joint: build demo, fuse both models | |
| 10-11 | Demo polish and buffer | |
| 12-13 | Dissertation, YC application, paper skeleton, submit | |

## Plan B: Compressed (about 17 days)
Use this if you have roughly 2-3 weeks.

| Days | Work |
|---|---|
| 1-2 | Setup repo, accounts, Colab. Download datasets (use SUBSETS: ~2-3k audio files, ~100-200 videos per class). Lock scope. |
| 3-7 | Baselines on subsets. Freeze most of the backbone for speed. Get one real metric per modality. |
| 8-11 | One simple attack each (noise/PGD small sweep). Produce clean vs attacked table. |
| 12-14 | Minimal demo (FastAPI + one HTML page). Skip ONNX if short on time, load PyTorch checkpoints directly. |
| 15-17 | Write dissertation from results, YC application draft, submit. Paper skeleton only. |

Under Plan B, strongly consider **dropping to one modality done well** and listing the other as future work. A finished audio-only result beats two half-finished ones.

## Weekly rhythm
- Mon: 20-min sync, review `TASKS.md`
- Fri: both tracks push code and update metrics
- Never block each other: each track must be submittable alone
