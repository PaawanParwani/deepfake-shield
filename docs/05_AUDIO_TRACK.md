# Audio Track Spec (owner: Partner A, cybersecurity)

## Goal
Binary classifier: bonafide vs spoofed speech, evaluated by **EER** on ASVspoof2019 LA eval.

## Steps
1. **Data**: parse protocol files (`ASVspoof2019.LA.cm.{train,dev,eval}.trn.txt`), build a PyTorch `Dataset` returning (waveform, label). Label: bonafide=0, spoof=1.
2. **Preprocess**: resample to 16 kHz, mono, fixed length ~4 s (random crop in train, center crop in eval), normalize.
3. **Model**: wav2vec2 backbone, mean-pool over time, linear head (2 classes). Start by freezing the backbone for a fast sanity run, then unfreeze top layers.
4. **Train**: AdamW, lr ~1e-5 to 3e-5 for backbone, 1e-3 for head, batch 8-16, mixed precision, 3-10 epochs. Log to W&B.
5. **Evaluate**: EER, AUC, accuracy on dev and eval. Save `outputs/metrics.json` and a ROC/DET plot.
6. **Attack** (see `07_ADVERSARIAL_EVAL.md`).
7. **Export** to ONNX with dynamic axis on time length.

## Files to create
```
audio/src/dataset.py   audio/src/model.py   audio/src/train.py
audio/src/eval.py      audio/src/attack.py  audio/src/export_onnx.py
audio/configs/baseline.yaml
```

## Checkpoint
A working classifier with a real EER number before the baseline deadline in `04_TIMELINE.md`. A mediocre number beats no number.

## Pitfalls
- Don't evaluate on train-distribution data only; always report on the official eval split.
- Watch for silence-length shortcuts in ASVspoof; mention as limitation if observed.
- Keep a tiny subset (e.g. 500 files) config for smoke tests.
