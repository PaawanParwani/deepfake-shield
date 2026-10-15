# Tech Stack

## Languages
- Python 3.10+ (everything ML/backend)
- HTML/CSS/vanilla JS (demo page only)

## Core libraries
| Purpose | Package |
|---|---|
| Deep learning | `torch`, `torchaudio`, `torchvision` |
| Pretrained speech models | `transformers` |
| Pretrained vision models | `timm` |
| Audio features | `librosa`, `soundfile` |
| Video/frames | `opencv-python` |
| Face detection | `facenet-pytorch` |
| Metrics | `scikit-learn` |
| Adversarial attacks | `torchattacks` |
| Experiment tracking | `wandb` |
| Inference/export | `onnx`, `onnxruntime` |
| Demo backend | `fastapi`, `uvicorn`, `python-multipart` |
| Utilities | `numpy`, `pandas`, `matplotlib`, `pyyaml`, `tqdm` |

## requirements.txt
```
torch
torchaudio
torchvision
transformers
timm
librosa
soundfile
opencv-python
facenet-pytorch
scikit-learn
torchattacks
wandb
onnx
onnxruntime
fastapi
uvicorn
python-multipart
numpy
pandas
matplotlib
pyyaml
tqdm
```

## Datasets
| Modality | Dataset | Notes |
|---|---|---|
| Audio | ASVspoof2019 LA | Free for research. Use official train/dev/eval protocol files. |
| Video | FaceForensics++ | Requires signing the academic-use form. Use the compressed (c23) version to save space. |

## Accounts and cost
| Tool | Cost |
|---|---|
| GitHub | Free |
| Hugging Face | Free |
| Weights & Biases | Free (academic) |
| Google Colab Pro | ~$10/month, needed mainly for video |
| TTS credits (optional, for extra test spoofs) | ~$5-22 |

Check for free university GPU/HPC access before paying anything.
