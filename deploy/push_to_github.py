"""Helper script to connect and push local git repository to GitHub for Render deployment.

Usage:
    python deploy/push_to_github.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 65)
    print("🚀 Push Deepfake Shield to GitHub (for Render.com)")
    print("=" * 65)
    print("1. Create a new empty repository at: https://github.com/new")
    print("   (Repository name: deepfake-meeting-shield)")
    print("2. Copy your repository HTTPS URL (e.g. https://github.com/USERNAME/deepfake-meeting-shield.git)")
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
            print("Next step:")
            print("1. Go to https://dashboard.render.com")
            print("2. Click 'New +' -> 'Web Service'")
            print("3. Connect this GitHub repository and select the Free tier!")
            print("=" * 65)
        else:
            print("\n⚠️ Push encountered an issue. Please ensure your GitHub credentials are authenticated.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
