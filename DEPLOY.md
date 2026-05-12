# Deploying to Streamlit Community Cloud

This guide walks you through publishing the Bridge SHM dashboard so anyone
with the URL can view it — perfect for sharing on LinkedIn.

---

## 1. Clean up local git state (one-time)

Open Terminal, `cd` to this folder, and run:

```bash
cd ~/Desktop/College/DL/project

# Remove the stale Claude/Cursor worktree (safe — it's an abandoned agent worktree)
git worktree remove --force .claude/worktrees/modest-hertz-a20fc7 2>/dev/null || true
rm -rf .git/worktrees/modest-hertz-a20fc7
rm -rf .claude/worktrees/modest-hertz-a20fc7

# Clear any stale git lock
rm -f .git/index.lock

# Untrack cruft that should never have been committed
git rm -r --cached --ignore-unmatch \
  .DS_Store data/.DS_Store src/.DS_Store \
  .cursor \
  src/__pycache__

# Fix the remote so the embedded PAT is no longer in the URL
git remote set-url origin https://github.com/atharva-1910/bridge-shm-gcn.git
```

If `git remote -v` still shows the token, you're done — it will now prompt
for credentials (use a fresh PAT, GitHub CLI `gh auth login`, or SSH).

## 2. Commit the deploy fixes

```bash
git add .gitignore requirements.txt runtime.txt .streamlit/ dashboard/app.py DEPLOY.md
git commit -m "Prepare for Streamlit Cloud deploy

- Pin torch/torch-geometric to CPU-friendly versions
- Pin Python 3.11 via runtime.txt
- Guard hardcoded debug log path so it no-ops in production
- Add .gitignore and Streamlit config
- Untrack .DS_Store / .cursor / __pycache__"
git push origin main
```

If the push prompts for credentials, paste your new GitHub PAT as the
password (username is your GitHub handle).

## 3. Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **"Create app"** → **"Deploy a public app from GitHub"**.
3. Fill in:
   - **Repository:** `atharva-1910/bridge-shm-gcn`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
   - **App URL (optional):** pick something like `bridge-shm-gcn` →
     your URL will be `https://bridge-shm-gcn.streamlit.app`
4. Click **"Deploy"**. The first build takes 3–6 minutes (torch is a
   ~200 MB install). Subsequent pushes redeploy in under a minute.

## 4. While the build runs

Watch the **"Manage app"** logs at the bottom of the deployed page.
Common things to expect:

- A `pip` install log followed by `streamlit run dashboard/app.py`.
- A few warnings about torch CPU wheels — those are normal.
- If you see `ModuleNotFoundError`, the pin in `requirements.txt` needs
  adjusting. Bump the failing package and push again.

## 5. Sharing on LinkedIn

Once the app is live, your post can use:

- The Streamlit Cloud URL: `https://<your-app-slug>.streamlit.app`
- A screenshot or short screen recording of the dashboard (LinkedIn
  posts with media get much more reach).
- A link to the GitHub repo for credibility:
  `https://github.com/atharva-1910/bridge-shm-gcn`

Sample caption:

> Just shipped a live demo of my GCN-based Bridge Structural Health
> Monitoring dashboard. It uses a Graph Convolutional Network over a
> 54-feature digital-twin dataset to classify bridge health states in
> real time. Try the live dashboard 👇
>
> Live demo: https://<your-slug>.streamlit.app
> Code: https://github.com/atharva-1910/bridge-shm-gcn
>
> #GraphNeuralNetworks #StructuralHealthMonitoring #DigitalTwin
> #DeepLearning #PyTorch

---

## Troubleshooting

**Build fails with "torch==X has no wheels with a matching Python ABI tag"** —
Streamlit Community Cloud now defaults to **Python 3.14**, and
`runtime.txt` is no longer honored (Python version is set in the app's
Advanced settings UI). The current pins (`torch==2.9.1`,
`torch-geometric==2.6.1`) are chosen to have Python 3.14 wheels. If you
ever switch the app's Python version in Streamlit Cloud settings, you
may need to re-pin torch to match what's available for that Python.

**Build fails on `torch-geometric`** — usually means torch and
torch-geometric got out of sync. Check the build log for the actual
error and bump `torch-geometric` to the latest if torch is recent.

**App crashes on `FileNotFoundError: data/raw/bridge_digital_twin_dataset.csv`** —
the dataset (37 MB) is tracked in git, so it should be there. If you
ever remove it from the repo, you'll need to fetch it at runtime or
host the file elsewhere (Hugging Face dataset, S3, Google Drive).

**App is too slow / runs out of memory** — Streamlit Community Cloud's
free tier has ~1 GB RAM. The 37 MB CSV plus torch is fine, but heavy
caching of all features may push it. Add `@st.cache_data` /
`@st.cache_resource` to large loaders if you hit limits.

**"This app has been put to sleep"** — free Streamlit apps sleep after
inactivity. Visitors just click "Wake up" and wait ~30 s. Mention this
in your LinkedIn post so people don't think the link is broken.
