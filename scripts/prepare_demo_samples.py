"""Extract and prepare authentic vs deepfake test samples for live verification."""

import os
import shutil
import sys
import pandas as pd
import soundfile as sf

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def main():
    os.makedirs("demo_samples", exist_ok=True)

    # 1. Real vs Deepfake Face Images
    real_crop_src = "video/data/crops/val/real/115/frame_0000.jpg"
    fake_crop_src = "video/data/crops/val/fake/Deepfakes_115_939/frame_0000.jpg"

    if os.path.exists(real_crop_src):
        shutil.copy(real_crop_src, "demo_samples/real_face.jpg")
    if os.path.exists(fake_crop_src):
        shutil.copy(fake_crop_src, "demo_samples/deepfake_face.jpg")

    # 2. Real vs Spoof Audio from ASVspoof2019 dataset
    parquet_path = "audio/data/LA/data/train-00000-of-00001.parquet"
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        real_row = df[df["key"] == 0].iloc[0]
        fake_row = df[df["key"] == 1].iloc[0]

        real_audio = real_row["audio"]
        fake_audio = fake_row["audio"]

        with open("demo_samples/real_voice.flac", "wb") as f:
            f.write(real_audio["bytes"])
        with open("demo_samples/spoof_voice.flac", "wb") as f:
            f.write(fake_audio["bytes"])

    print("=" * 60)
    print("📁 Test Samples Prepared in demo_samples/:")
    print("=" * 60)
    for f in os.listdir("demo_samples"):
        fp = os.path.join("demo_samples", f)
        size_kb = os.path.getsize(fp) / 1024.0
        print(f"  • {f:<22} ({size_kb:.1f} KB)")
    print("=" * 60)

if __name__ == "__main__":
    main()
