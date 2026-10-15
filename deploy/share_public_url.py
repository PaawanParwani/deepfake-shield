"""Instant Free Public HTTPS Tunnel for Deepfake Shield.

Creates a secure, zero-config public HTTPS URL via Cloudflare Tunnel:
- 100% Free (No credit card, no subscription, no account required)
- Public HTTPS domain (https://*.trycloudflare.com)
- Instant worldwide access for Google Meet extension, mobile apps, or web browser
"""

import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

# Safe encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def ensure_cloudflared(deploy_dir: Path) -> Path:
    exe_path = deploy_dir / "cloudflared.exe"
    if exe_path.exists() and exe_path.stat().st_size > 10000000:
        return exe_path

    print("📥 Downloading Cloudflare Tunnel binary (one-time setup)...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    urllib.request.urlretrieve(url, str(exe_path))
    print("✅ Download complete.")
    return exe_path


def check_backend_running(port: int = 8000) -> bool:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_running_backend() -> int:
    """Find which port the backend is currently running on."""
    for p in [8000, 8001, 8002, 8080]:
        if check_backend_running(p):
            return p
    return 8000


def start_tunnel(port: Optional[int] = None):
    deploy_dir = Path(__file__).resolve().parent
    exe_path = ensure_cloudflared(deploy_dir)

    if port is None:
        port = find_running_backend()

    backend_proc = None
    if not check_backend_running(port):
        print(f"⚙️  Backend server is not running yet. Automatically launching backend on port {port}...")
        root_dir = deploy_dir.parent
        backend_proc = subprocess.Popen(
            [sys.executable, str(root_dir / "run_server.py"), "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(12):
            time.sleep(1)
            if check_backend_running(port):
                break

    if check_backend_running(port):
        print(f"✅ Active backend verified on http://127.0.0.1:{port}")
    else:
        print(f"⚠️  Backend did not report ready yet on http://127.0.0.1:{port}. Starting tunnel...")

    cmd = [str(exe_path), "tunnel", "--url", f"http://127.0.0.1:{port}"]

    print("\n" + "=" * 70)
    print("⚡ GENERATING INSTANT FREE PUBLIC HTTPS URL...")
    print("=" * 70)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    url_found = False
    public_url = None

    try:
        for line in process.stdout:
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match and not url_found:
                url_found = True
                public_url = match.group(0)
                print("\n" + "🔥" * 35)
                print("🎉 YOUR LIVE WEB APP IS NOW ACCESSIBLE WORLDWIDE!")
                print("🔥" * 35)
                print(f"\n👉 Public Web URL: {public_url}")
                print(f"👉 API Health:     {public_url}/health")
                print(f"👉 Multimodal:     {public_url}/predict/multimodal")
                print("\n" + "-" * 70)
                print("💡 Share this URL with your reviewers, friends, or enter it")
                print("   into the Chrome extension options / Web UI settings!")
                print("-" * 70)
                print("\n(Press Ctrl+C anytime to stop sharing)\n")
            elif not url_found:
                if "Registered tunnel connection" in line or "error" in line.lower():
                    print(f"  {line.strip()}")
    except KeyboardInterrupt:
        print("\n🛑 Stopping public tunnel...")
    finally:
        process.terminate()
        process.wait()
        if backend_proc is not None:
            backend_proc.terminate()
            backend_proc.wait()
        print("Tunnel closed.")


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_tunnel(port)
