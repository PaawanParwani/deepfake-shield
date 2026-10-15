# 🚀 Deploy Deepfake Shield Backend to Render.com (100% Free Tier)

Render provides free cloud hosting for FastAPI web services with automated HTTPS and GitHub integration.

---

## ⚡ 2-Minute Deployment via Render Dashboard

### Step 1: Create a Free Account
1. Go to [render.com](https://render.com) and click **Get Started for Free** (sign up with GitHub).

### Step 2: Create New Web Service
1. In the Render Dashboard, click **New +** $\rightarrow$ **Web Service**.
2. Select **Build and deploy from a Git repository** $\rightarrow$ connect your repository: `deepfake-meeting-shield`.
3. Fill in the settings:
   - **Name:** `deepfake-shield-backend`
   - **Region:** `Oregon (US West)` or closest to you
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install -r deploy/render/requirements.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn demo.backend.app:app --host 0.0.0.0 --port $PORT --workers 1
     ```
   - **Instance Type:** `Free` (0.1 CPU, 512 MB RAM)
4. Click **Create Web Service**.

---

## 🌐 Your Live Backend API URL
Once the build completes (usually ~2 minutes), Render will give you your permanent public HTTPS URL:
👉 **`https://deepfake-shield-backend.onrender.com`**

You can test it immediately:
- Health check: `https://deepfake-shield-backend.onrender.com/health`
- Connect your Cloudflare Pages frontend or Chrome Extension by setting the Backend URL to this address!
