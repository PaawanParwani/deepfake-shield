# Demo Spec

## Goal
A simple, reliable demo that shows both detectors working. This is what you show YC and your examiners. Scrappy and working beats polished and broken.

## Backend (FastAPI)
- `GET /health` -> `{"status":"ok"}`
- `POST /predict/audio` (multipart file) -> `{"p_spoof": float, "verdict": "real|suspicious"}`
- `POST /predict/video` (multipart file) -> `{"p_fake": float, "frames_used": int, "verdict": "real|suspicious"}`
- Load ONNX (or PyTorch) models once at startup
- Reject files over 50 MB, validate extensions, delete temp files after inference

## Frontend (single HTML file)
- Two file inputs (audio, video) and a Submit button each
- Result card per modality: score bar + verdict badge
- Overall badge: "Verified Human" only if every provided modality is below threshold, else "Suspicious"
- No framework, no build step

## Run
```bash
uvicorn demo.backend.app:app --reload
# open demo/frontend/index.html
```

## Demo script (for the YC video, ~90 s)
1. Problem: the $25M deepfake call story (15 s)
2. Upload a real voice clip -> Verified (15 s)
3. Upload a cloned voice clip -> Suspicious (15 s)
4. Upload a real face clip, then a deepfake clip (20 s)
5. Show the robustness chart: "we also test how attackers can fool it" (15 s)
6. One-line vision (10 s)

## Security notes for the demo itself
- Never trust client-side verdicts; verdict is computed server-side
- Don't log uploaded media
- State clearly that this is a research prototype
