# Deploy bundle (Render)

Private repo: only what’s needed to run `gunicorn web.app:app`.

## What you must add before push

1. **`models/best_ppo_twap_gap.zip`**  
   Trained PPO checkpoint (Stable-Baselines3). The app prefers this filename.  
   **Tip:** Remove other `PPO_*.zip` files in `models/` for faster startup (the case study can sweep many checkpoints).

2. **Feature parquet files** under **`deploy_data/features/`**  
   - `features_train.parquet`  
   - `features_val.parquet`  
   - `features_test.parquet`  

   Copy from your local `CFADATA/features/` (or wherever `build_features` wrote them).

3. **Render environment variables**  
   - **`PYTHON_VERSION`** = `3.11.11` (required: Render’s default is 3.14+, which breaks `pandas==2.2.2` wheels and tries a long source build).  
   - **`CFA_DATA_ROOT`** = full path to the **`deploy_data`** directory on the server.  
   - On Render, this is often:  
     `CFA_DATA_ROOT=/opt/render/project/src/deploy_data`  
     (adjust if your service **Root Directory** is not the repo root.)

4. **Python pin in repo** — root **`.python-version`** is set to `3.11.11` so Render installs 3.11 (see [Render Python version](https://render.com/docs/python-version)). **`runtime.txt`** is Heroku-style and is not used by Render for native Python.

## Optional (same as main project)

- **`data/processed/bbo_daily.parquet`** — order imbalance; app works without it (warning in logs).
- **`data/processed/news_daily_SPY.parquet`** — news features; optional.
- **`ANTHROPIC_API_KEY`** — live LLM; without it, governance uses cached/offline text where applicable.

## Render dashboard (summary)

| Setting | Value |
|--------|--------|
| Build command | **`bash build.sh`** (installs **CPU-only** PyTorch, then `requirements.txt` — avoids huge CUDA wheels and saves RAM) |
| Start command | Leave empty to use **`Procfile`**, or paste the `web:` line from it |

**Do not** use plain `pip install -r requirements.txt` as the only build step: that pulls **CUDA** PyTorch from PyPI and often **OOMs** on 512 MB instances.

Use **Python 3.11** (`.python-version` + `PYTHON_VERSION` on Render). Use **1 worker** so case-study precompute runs once per process.

## Local smoke test

```bash
cd /path/to/CFAforDeployment
python3.11 -m venv .venv && source .venv/bin/activate
bash build.sh
export CFA_DATA_ROOT="$(pwd)/deploy_data"
gunicorn --workers 1 --threads 1 --timeout 600 --bind 127.0.0.1:8000 web.app:app
```

Open http://127.0.0.1:8000
