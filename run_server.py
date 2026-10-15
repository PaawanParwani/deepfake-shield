"""Root launcher for Deepfake Verification Backend Server.

Usage:
    python run_server.py
    python run_server.py --port 8000
"""

import os
import socket
import sys
from pathlib import Path

# Safe encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Guarantee root is in sys.path and PYTHONPATH
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import uvicorn


def find_available_port(start_port: int = 8000) -> int:
    """Find the first open port starting from start_port."""
    for p in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            # If connect fails, port is available to bind
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start_port


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Deepfake Verification Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=None, help="Port number (defaults to first free port from 8000)")
    parser.add_argument("--reload", action="store_true", default=False, help="Auto-reload on code change")
    args = parser.parse_args()

    active_port = args.port or find_available_port(8000)

    print("\n" + "=" * 70)
    print(f">> STARTING DEEPFAKE SHIELD SERVER ON http://localhost:{active_port}")
    if active_port != 8000:
        print(f"   (Port 8000 was in use by another process; automatically switched to port {active_port})")
    print("=" * 70 + "\n")

    uvicorn.run("demo.backend.app:app", host=args.host, port=active_port, reload=args.reload, app_dir=str(ROOT))
