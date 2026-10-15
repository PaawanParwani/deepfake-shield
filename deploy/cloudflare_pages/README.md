# ☁️ Deploy Deepfake Shield Frontend to Cloudflare Pages (100% Free Forever)

Cloudflare Pages provides free, unlimited bandwidth, global CDN caching, and automated HTTPS for static web applications.

---

## ⚡ Option A: 60-Second Drag & Drop Deployment (No Git / CLI Needed)

1. Log in or create a free account at [dash.cloudflare.com](https://dash.cloudflare.com).
2. In the left sidebar, click **Workers & Pages**.
3. Click **Create Application** $\rightarrow$ select the **Pages** tab.
4. Click **Upload assets**.
5. Set Project Name: `deepfake-meeting-shield` (or any name you choose).
6. Drag and drop this folder (`deploy/cloudflare_pages`) into the dropzone.
7. Click **Deploy site**.
8. **Done!** Your web app is live instantly at:
   👉 `https://deepfake-meeting-shield.pages.dev`

---

## 🔄 Option B: Connect via GitHub (Automated Continuous Deployment)

1. Push your repository to GitHub.
2. In Cloudflare Dashboard, go to **Workers & Pages > Create application > Pages > Connect to Git**.
3. Select your repository `deepfake-meeting-shield`.
4. Configure Build settings:
   - Framework preset: **None**
   - Build output directory: `deploy/cloudflare_pages` (or `demo/frontend`)
5. Click **Save and Deploy**.
   - Every time you `git push`, Cloudflare will build and update your site automatically!
