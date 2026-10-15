"""Automated production deployment script for Hugging Face Spaces.

Supports:
1. Free Static Space (100% Free for all Hugging Face accounts, no PRO needed)
2. Docker Space (requires HF PRO subscription)
"""

import argparse
import getpass
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Safe encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from huggingface_hub import HfApi, create_repo, get_token
except ImportError:
    print("❌ Error: 'huggingface_hub' is required. Run: pip install huggingface_hub")
    sys.exit(1)


def get_authenticated_token(cli_token: Optional[str]) -> str:
    """Resolve a valid Hugging Face write token."""
    if cli_token and cli_token.strip() != "hf_yourToken":
        return cli_token.strip()

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        print("🔑 Using Hugging Face token detected from environment variable (HF_TOKEN).")
        return env_token.strip()

    saved_token = get_token()
    if saved_token:
        print("🔑 Using saved Hugging Face token from local login.")
        return saved_token.strip()

    print("\n" + "=" * 70)
    print("🔑 Hugging Face Authentication Required")
    print("=" * 70)
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Click 'Create new token' -> Role: 'Write'.")
    print("3. Copy the generated token (starts with 'hf_...').")
    print("=" * 70 + "\n")

    user_token = getpass.getpass("Paste your Hugging Face Write Token here: ").strip()
    if not user_token:
        print("❌ Error: Hugging Face token cannot be empty.")
        sys.exit(1)
    return user_token


def resolve_repo_id(cli_repo_id: Optional[str], api: HfApi) -> str:
    """Resolve repository ID ensuring real username is used."""
    if cli_repo_id and "your_username" not in cli_repo_id and "/" in cli_repo_id:
        return cli_repo_id.strip()

    try:
        user_info = api.whoami()
        username = user_info.get("name")
    except Exception:
        username = None

    default_name = "deepfake-meeting-shield"
    if username:
        suggested = f"{username}/{default_name}"
        print(f"\n👤 Authenticated as Hugging Face user: '{username}'")
        choice = input(f"Create space at '{suggested}'? [Y/n]: ").strip().lower()
        if choice in ("", "y", "yes"):
            return suggested

    val = input("Enter your Hugging Face Space ID (format: username/space-name): ").strip()
    if not val or "/" not in val or "your_username" in val:
        print("❌ Error: Invalid repository ID format. Must be 'username/space-name'.")
        sys.exit(1)
    return val


def deploy_static_space(repo_id: str, token: str, project_root: Path) -> None:
    """Deploy the frontend as a 100% Free Static Space on Hugging Face."""
    api = HfApi(token=token)
    print("\n📦 Deploying 100% Free Static Space to Hugging Face...")

    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        private=False,
        token=token,
        exist_ok=True,
    )

    readme_content = """---
title: Deepfake Meeting Shield
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
---

# Deepfake Meeting Shield & Biometric Liveness Detector
Production web application for real-time video and audio deepfake verification.
"""

    temp_readme = Path(tempfile.gettempdir()) / f"hf_readme_{os.getpid()}.md"
    temp_readme.write_text(readme_content, encoding="utf-8")
    try:
        api.upload_file(
            path_or_fileobj=str(temp_readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
        )
    finally:
        temp_readme.unlink(missing_ok=True)

    # Upload frontend index.html as root index.html
    frontend_html = project_root / "demo" / "frontend" / "index.html"
    if frontend_html.exists():
        print("  Uploading frontend application (index.html)...")
        api.upload_file(
            path_or_fileobj=str(frontend_html),
            path_in_repo="index.html",
            repo_id=repo_id,
            repo_type="space",
        )

    print("\n" + "=" * 70)
    print("🎉 DEPLOYMENT COMPLETE (100% FREE STATIC SPACE)!")
    print("=" * 70)
    print(f"👉 Your public web app is live at: https://huggingface.co/spaces/{repo_id}")
    print("\n💡 To connect your local AI backend for free worldwide access:")
    print("   Run: python deploy/share_public_url.py")
    print("=" * 70 + "\n")


def deploy_docker_space(repo_id: str, token: str, project_root: Path) -> None:
    """Deploy full Docker container (requires HF PRO)."""
    api = HfApi(token=token)
    deploy_dir = project_root / "deploy" / "huggingface_space"

    print("\n📦 Uploading Docker deployment specifications...")
    for filename in ["Dockerfile", "README.md", "requirements.txt"]:
        file_path = deploy_dir / filename
        if file_path.exists():
            print(f"  Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="space",
            )

    folders_to_upload = [
        ("demo", ["*.pyc", "__pycache__", "temp_*", "*.wav", "test_crop*.jpg"]),
        ("video/src", ["*.pyc", "__pycache__"]),
        ("video/configs", ["*.pyc", "__pycache__"]),
        ("audio/src", ["*.pyc", "__pycache__"]),
        ("audio/configs", ["*.pyc", "__pycache__"]),
    ]

    for rel_folder, ignore in folders_to_upload:
        folder_path = project_root / rel_folder
        if folder_path.exists():
            print(f"  Uploading {rel_folder}...")
            api.upload_folder(
                folder_path=str(folder_path),
                path_in_repo=rel_folder,
                repo_id=repo_id,
                repo_type="space",
                ignore_patterns=ignore,
            )

    model_checkpoints = [
        ("video/outputs/best_model.pt", "video/outputs/best_model.pt"),
        ("audio/outputs/best_model.pt", "audio/outputs/best_model.pt"),
    ]
    for local_rel, repo_rel in model_checkpoints:
        ckpt_path = project_root / local_rel
        if ckpt_path.exists():
            print(f"  Uploading trained weights: {local_rel} ({round(ckpt_path.stat().st_size / 1e6, 1)} MB)...")
            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo=repo_rel,
                repo_id=repo_id,
                repo_type="space",
            )

    for init_rel in ["video/__init__.py", "audio/__init__.py"]:
        init_path = project_root / init_rel
        if init_path.exists():
            api.upload_file(
                path_or_fileobj=str(init_path),
                path_in_repo=init_rel,
                repo_id=repo_id,
                repo_type="space",
            )

    print("\n" + "=" * 70)
    print("🎉 DOCKER DEPLOYMENT COMPLETE!")
    print(f"👉 Live Space: https://huggingface.co/spaces/{repo_id}")
    print("=" * 70 + "\n")


def deploy_space(repo_id: str, token: str, mode: str = "auto") -> None:
    api = HfApi(token=token)
    project_root = Path(__file__).resolve().parent.parent

    print("\n" + "=" * 70)
    print(f"🚀 Deploying to Hugging Face Space: {repo_id}")
    print("=" * 70)

    if mode == "static":
        deploy_static_space(repo_id, token, project_root)
        return

    # Try Docker space first, gracefully fallback to Static Space if user is on Free Tier
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=token,
            exist_ok=True,
        )
        print("✅ Docker Space initialized.")
        deploy_docker_space(repo_id, token, project_root)
    except Exception as e:
        err_msg = str(e)
        if "402" in err_msg or "Payment Required" in err_msg or "PRO subscription" in err_msg:
            print("\n💡 Notice: Hugging Face requires PRO for Docker Spaces on this account tier.")
            print("   Deploying as a 100% FREE Static Space instead (Free for all accounts!)...")
            deploy_static_space(repo_id, token, project_root)
        else:
            print(f"⚠️  Repository notice: {e}")
            print("   Attempting Free Static Space deployment...")
            deploy_static_space(repo_id, token, project_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Deepfake Shield to Hugging Face Spaces")
    parser.add_argument("--repo-id", default=None, help="HF Space ID, e.g. username/deepfake-meeting-shield")
    parser.add_argument("--token", default=None, help="Hugging Face User Access Token (Write permissions)")
    parser.add_argument("--mode", default="auto", choices=["auto", "static", "docker"], help="Deployment mode")
    args = parser.parse_args()

    token = get_authenticated_token(args.token)
    api = HfApi(token=token)
    repo_id = resolve_repo_id(args.repo_id, api)
    deploy_space(repo_id, token, mode=args.mode)
