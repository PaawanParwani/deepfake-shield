"""Run live end-to-end deepfake inference tests on real vs fake samples against the live cloud backend."""

import base64
import json
import os
import sys
import time
import urllib.request
import urllib.parse

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BACKEND_URL = os.getenv("BACKEND_URL", "https://deepfake-shield-vyaq.onrender.com")

def post_multipart(url: str, file_path: str, field_name: str) -> dict:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    filename = os.path.basename(file_path)
    
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as response:
        latency = round((time.time() - t0) * 1000, 1)
        res_data = json.loads(response.read().decode("utf-8"))
        res_data["_latency_ms"] = latency
        return res_data

def post_frame_base64(url: str, image_path: str) -> dict:
    with open(image_path, "rb") as f:
        b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

    data = json.dumps({"image_base64": b64}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as response:
        latency = round((time.time() - t0) * 1000, 1)
        res_data = json.loads(response.read().decode("utf-8"))
        res_data["_latency_ms"] = latency
        return res_data

def main():
    print("=" * 70)
    print("🧪 LIVE DEEPFAKE DETECTION VERIFICATION BENCHMARK")
    print(f"🌐 Target Backend: {BACKEND_URL}")
    print("=" * 70)

    # Health Check
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/health", timeout=10) as r:
            health = json.loads(r.read().decode("utf-8"))
            print(f"🟢 Cloud Backend Status: {health.get('status')} | Audio: {health.get('audio_status')} | Video: {health.get('video_status')}")
    except Exception as e:
        print(f"❌ Failed to reach backend: {e}")
        return

    print("-" * 70)

    # 1. Test Authentic Face
    print("1️⃣ Testing Authentic Real Face: demo_samples/real_face.jpg")
    try:
        res = post_frame_base64(f"{BACKEND_URL}/predict/frame", "demo_samples/real_face.jpg")
        print(f"   • Verdict:     {res.get('verdict').upper()} (Risk: {res.get('p_fake', 0)*100:.1f}%)")
        print(f"   • Confidence:  {res.get('confidence')}%")
        print(f"   • Face Found:  {res.get('face_detected')}")
        print(f"   • Latency:     {res.get('_latency_ms')} ms")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("-" * 70)

    # 2. Test Deepfake Face
    print("2️⃣ Testing AI Deepfake Face: demo_samples/deepfake_face.jpg")
    try:
        res = post_frame_base64(f"{BACKEND_URL}/predict/frame", "demo_samples/deepfake_face.jpg")
        print(f"   • Verdict:     {res.get('verdict').upper()} (Risk: {res.get('p_fake', 0)*100:.1f}%)")
        print(f"   • Confidence:  {res.get('confidence')}%")
        print(f"   • Face Found:  {res.get('face_detected')}")
        print(f"   • Latency:     {res.get('_latency_ms')} ms")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("-" * 70)

    # 3. Test Authentic Voice
    print("3️⃣ Testing Authentic Voice: demo_samples/real_voice.flac")
    try:
        res = post_multipart(f"{BACKEND_URL}/predict/audio", "demo_samples/real_voice.flac", "file")
        print(f"   • Verdict:     {res.get('verdict').upper()} (Spoof Risk: {res.get('p_spoof', 0)*100:.1f}%)")
        print(f"   • Confidence:  {res.get('confidence')}%")
        print(f"   • Duration:    {res.get('duration_seconds')}s")
        print(f"   • Latency:     {res.get('_latency_ms')} ms")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("-" * 70)

    # 4. Test Synthetic Cloned Voice
    print("4️⃣ Testing Synthetic Cloned Voice: demo_samples/spoof_voice.flac")
    try:
        res = post_multipart(f"{BACKEND_URL}/predict/audio", "demo_samples/spoof_voice.flac", "file")
        print(f"   • Verdict:     {res.get('verdict').upper()} (Spoof Risk: {res.get('p_spoof', 0)*100:.1f}%)")
        print(f"   • Confidence:  {res.get('confidence')}%")
        print(f"   • Duration:    {res.get('duration_seconds')}s")
        print(f"   • Latency:     {res.get('_latency_ms')} ms")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("=" * 70)
    print("🎉 ALL SAMPLES EVALUATED BY CLOUD AI MODELS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
