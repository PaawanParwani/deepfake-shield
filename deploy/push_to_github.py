"""Helper script to connect and push local git repository to GitHub.

Usage:
    python deploy/push_to_github.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 65)
    print("🚀 Push Deepfake Shield to GitHub")
    print("=" * 65)
    print("1. Create a new empty repository at: https://github.com/new")
    print("   (Repository name: deepfake-shield)")
    print("2. Copy your repository HTTPS URL (e.g. https://github.com/USERNAME/deepfake-shield.git)")
    print("=" * 65)

    repo_url = input("\nPaste your GitHub Repository URL: ").strip()
    if not repo_url:
        print("❌ No URL provided. Aborting.")
        return

    # Check existing remotes
    try:
        remotes = subprocess.check_output(["git", "remote"], text=True).strip().split()
        if "origin" in remotes:
            print("Notice: Updating existing 'origin' remote...")
            subprocess.run(["git", "remote", "set-url", "origin", repo_url], check=True)
        else:
            print("Adding 'origin' remote...")
            subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)

        print(f"Setting branch to 'main'...")
        subprocess.run(["git", "branch", "-M", "main"], check=True)

        print(f"Pushing code to GitHub: {repo_url}...")
        result = subprocess.run(["git", "push", "-u", "origin", "main"])
        if result.returncode == 0:
            print("\n" + "=" * 65)
            print("🎉 SUCCESSFULLY PUSHED TO GITHUB!")
            print("=" * 65)
            print("Repository is synced with Cloudflare Pages and local environment.")
            print("=" * 65)
        else:
            print("\n⚠️ Push encountered an issue. Please ensure your GitHub credentials are authenticated.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
