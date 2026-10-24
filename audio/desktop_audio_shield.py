"""Universal Desktop Audio Deepfake Shield.

Captures real-time meeting and system audio from Google Meet, Microsoft Teams,
Zoom, Discord, or any desktop application via Windows WASAPI Loopback, and classifies
incoming voices using the CRNN-BiGRU neural deepfake audio detector.

Usage:
    python audio/desktop_audio_shield.py [--backend URL] [--mode loopback|mic] [--threshold 0.65]
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import sys
import time
from typing import Any, Optional

import numpy as np
import requests

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    try:
        import pyaudio
    except ImportError:
        pyaudio = None

try:
    import soundfile as sf
except ImportError:
    sf = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AudioShield")

DEFAULT_BACKEND = "http://localhost:8000"
SAMPLE_RATE = 16000  # Target rate for deepfake model
CHUNK_DURATION_SEC = 2.5
TARGET_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
ENERGY_THRESHOLD = 0.008  # RMS threshold for voice activity


def get_wasapi_loopback_device(pa: Any) -> Optional[dict]:
    """Locate the default Windows WASAPI Loopback recording device."""
    if pyaudio is None or pa is None:
        return None
    try:
        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        speaker_name = default_speakers["name"]

        # Search loopback devices for the matching default output
        for loopback in pa.get_loopback_device_info_generator():
            if speaker_name in loopback["name"]:
                return loopback

        # Fallback: return any available loopback device
        for loopback in pa.get_loopback_device_info_generator():
            return loopback
    except Exception as exc:
        logger.warning("Could not auto-detect WASAPI loopback: %s", exc)
    return None


def calculate_rms(audio_data: np.ndarray) -> float:
    """Compute Root-Mean-Square (RMS) amplitude of audio array."""
    if len(audio_data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_data))))


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Quick linear interpolation resample to 16,000 Hz mono."""
    if orig_sr == SAMPLE_RATE:
        return audio
    num_samples = int(len(audio) * SAMPLE_RATE / orig_sr)
    indices = np.linspace(0, len(audio) - 1, num_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def send_audio_chunk(
    backend_url: str,
    wav_bytes: bytes,
    timeout: float = 8.0
) -> Optional[dict]:
    """Send audio WAV chunk to the backend /predict/audio_chunk endpoint."""
    endpoint = f"{backend_url.rstrip('/')}/predict/audio_chunk"
    try:
        resp = requests.post(
            endpoint,
            files={"file": ("meeting_speech_chunk.wav", wav_bytes, "audio/wav")},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()
        logger.warning("Backend returned HTTP %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Backend connection error: %s", exc)
    return None


def run_shield(
    backend_url: str = DEFAULT_BACKEND,
    mode: str = "loopback",
    threshold: float = 0.65,
    notify: bool = False,
):
    """Main real-time audio capture and analysis loop."""
    if pyaudio is None:
        logger.error("pyaudiowpatch is required. Run: pip install pyaudiowpatch")
        return

    pa = pyaudio.PyAudio()
    device_info = None

    if mode == "loopback":
        device_info = get_wasapi_loopback_device(pa)
        if not device_info:
            logger.warning("No WASAPI loopback device found. Falling back to default microphone.")
            mode = "mic"

    if mode == "mic":
        try:
            device_info = pa.get_default_input_device_info()
        except Exception as exc:
            logger.error("No input microphone available: %s", exc)
            pa.terminate()
            return

    channels = int(device_info.get("maxInputChannels", 2))
    native_sr = int(device_info.get("defaultSampleRate", 48000))
    buffer_frames = int(native_sr * 0.25)  # 250ms chunks

    print("\n" + "=" * 65)
    print("   🛡️  HUMAN VERIFICATION — REAL-TIME DESKTOP AUDIO SHIELD")
    print("=" * 65)
    print(f"  Target Device : {device_info['name']}")
    print(f"  Mode          : {mode.upper()} ({'System/Call Audio' if mode == 'loopback' else 'Microphone'})")
    print(f"  Native Rate   : {native_sr} Hz ({channels} channels)")
    print(f"  Backend       : {backend_url}")
    print(f"  Threshold     : {threshold:.2f} P(Fake)")
    print("=" * 65 + "\n")
    print("Listening for speech in Google Meet, Teams, Zoom, Discord, etc...\n")

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=native_sr,
            input=True,
            input_device_index=device_info["index"],
            frames_per_buffer=buffer_frames,
        )
    except Exception as exc:
        logger.error("Failed to open audio stream: %s", exc)
        pa.terminate()
        return

    rolling_audio: list[np.ndarray] = []
    consecutive_silent = 0
    max_rolling_chunks = int(CHUNK_DURATION_SEC / 0.25)

    try:
        while True:
            raw_bytes = stream.read(buffer_frames, exception_on_overflow=False)
            data_16 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Convert multi-channel to mono
            if channels > 1:
                data_16 = data_16.reshape(-1, channels).mean(axis=1)

            # Resample slice to 16kHz
            slice_16k = resample_to_16k(data_16, native_sr)
            rolling_audio.append(slice_16k)

            if len(rolling_audio) > max_rolling_chunks:
                rolling_audio.pop(0)

            rms = calculate_rms(slice_16k)
            vol_bars = int(min(20, rms * 400))
            meter = "█" * vol_bars + "░" * (20 - vol_bars)

            if rms < ENERGY_THRESHOLD:
                consecutive_silent += 1
                sys.stdout.write(f"\r[Vol: {meter}]  Silence... ({rms:.4f})     ")
                sys.stdout.flush()
                continue

            consecutive_silent = 0
            sys.stdout.write(f"\r[Vol: {meter}]  🎙️ Speech Detected! ({rms:.4f})")
            sys.stdout.flush()

            # Analyze whenever we have a full window
            if len(rolling_audio) >= max_rolling_chunks:
                combined_audio = np.concatenate(rolling_audio)
                if len(combined_audio) > TARGET_SAMPLES:
                    combined_audio = combined_audio[-TARGET_SAMPLES:]

                # Write to in-memory WAV
                bio = io.BytesIO()
                if sf is not None:
                    sf.write(bio, combined_audio, SAMPLE_RATE, format="WAV")
                else:
                    import wave
                    with wave.open(bio, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes((combined_audio * 32767).astype(np.int16).tobytes())

                wav_bytes = bio.getvalue()
                result = send_audio_chunk(backend_url, wav_bytes)

                if result:
                    verdict = result.get("verdict", "uncertain")
                    p_fake = float(result.get("p_fake") if result.get("p_fake") is not None else result.get("p_spoof", 0.0))
                    confidence = result.get("confidence", 0.0)
                    model_type = result.get("model", "CRNN-BiGRU")

                    timestamp = time.strftime("%H:%M:%S")

                    if p_fake >= threshold or verdict == "suspicious":
                        badge = "\033[91m⚠️ [SYNTHETIC VOICE CLONE DETECTED]\033[0m"
                        print(f"\n{timestamp} {badge} Risk: {p_fake * 100:.1f}% | Model: {model_type}")
                    elif verdict == "uncertain":
                        badge = "\033[93m🟡 [ANALYZING / UNCERTAIN]\033[0m"
                        print(f"\n{timestamp} {badge} Ambiguous ({confidence:.1f}%) | Model: {model_type}")
                    else:
                        badge = "\033[92m🛡️ [VERIFIED HUMAN VOICE]\033[0m"
                        print(f"\n{timestamp} {badge} Real: {(1.0 - p_fake) * 100:.1f}% | Model: {model_type}")

                # Slide window slightly to avoid hammering the server
                rolling_audio = rolling_audio[int(max_rolling_chunks // 2):]

    except KeyboardInterrupt:
        print("\n\nStopping Audio Shield...")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("Audio stream closed. Shield terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Desktop Audio Deepfake Shield")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="Backend URL")
    parser.add_argument(
        "--mode",
        choices=["loopback", "mic"],
        default="loopback",
        help="Audio capture mode: 'loopback' (captures remote call audio) or 'mic' (microphone)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Probability threshold to flag deepfake voice",
    )
    args = parser.parse_args()

    run_shield(
        backend_url=args.backend,
        mode=args.mode,
        threshold=args.threshold,
    )
