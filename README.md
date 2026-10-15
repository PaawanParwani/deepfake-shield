# Human Verification: Audio + Video Deepfake Detection

A dual-modality system that detects AI-cloned voices and deepfake faces, and measures how well each detector holds up against adversarial attackers.

Serves three goals at once: **dissertation**, **research paper**, **YC application**.

## Team and ownership
| Person | Background | Owns |
|---|---|---|
| Partner A | Cybersecurity | `/audio` track + threat model + adversarial evaluation |
| Partner B | Data Science | `/video` track + data pipelines + evaluation metrics |

## Read these first (in order)
1. `AGENTS.md` - rules the AI agent must follow in this repo
2. `docs/00_PROJECT_BRIEF.md` - what and why
3. `docs/01_SCOPE.md` - what is IN and OUT (locked)
4. `docs/02_ARCHITECTURE.md` - how the system fits together
5. `docs/04_TIMELINE.md` - schedule
6. `TASKS.md` - current task checklist

## Repo structure
```
.
├── README.md
├── AGENTS.md
├── TASKS.md
├── docs/               # all planning docs (this folder)
├── audio/
│   ├── data/           # ASVspoof2019 LA (gitignored)
│   ├── src/            # dataset.py, model.py, train.py, eval.py, attack.py
│   ├── notebooks/
│   └── outputs/        # checkpoints, metrics, plots
├── video/
│   ├── data/           # FaceForensics++ (gitignored)
│   ├── src/            # dataset.py, model.py, train.py, eval.py, attack.py
│   ├── notebooks/
│   └── outputs/
├── demo/
│   ├── backend/        # FastAPI app, loads ONNX models
│   └── frontend/       # plain HTML/CSS/JS upload page
├── paper/              # LaTeX or markdown drafts
├── dissertation/       # chapters
└── requirements.txt
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Status
See `TASKS.md`.
