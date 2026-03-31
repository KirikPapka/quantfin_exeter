# QuantFin Exeter — CFA AI Investment Challenge

**University of Exeter · Competition Group 18**

| Kirill Papka | Thomas Nguyen | Harrison Maxwell | Maksim Kitikov (lead) |
|--------------|---------------|------------------|------------------------|

Institutional **optimal execution** with **regime-aware RL** (PPO/SAC), **classical benchmarks** (TWAP / VWAP / Almgren–Chriss), **NASDAQ ITCH BBO–based order imbalance**, and an **LLM governance** layer (Anthropic Claude, cached for reproducibility). Primary judge-facing demo: **Streamlit**; narrative: **`notebooks/main_notebook.ipynb`**.

---

## Requirements

- **Python 3.11** (see `environment.yml`; Torch/SB3 wheels may fail on very new Python versions)
- Git

---

## Quick start

```bash
git clone <your-repo-url>
cd quantfin_exeter

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Point the code at team feature parquet (sibling **`CFADATA`** or custom path):

```bash
export CFA_DATA_ROOT="/absolute/path/to/CFADATA"
```

Smoke test:

```bash
pytest -q
python scripts/train.py --ticker SPY
```

---

## Repository layout

```
quantfin_exeter/
├── README.md                 # this file
├── requirements.txt
├── environment.yml
├── pytest.ini
├── .env.example              # copy to .env (gitignored) for API keys
├── apps/
│   └── streamlit_app.py      # judge UI: regimes, episode, benchmarks, LLM
├── data/
│   ├── raw/                  # yfinance cache (large files → gitignored)
│   ├── processed/            # e.g. bbo_daily.parquet (regenerate locally)
│   └── cached_llm/           # commit JSON caches for judges (no API key)
├── notebooks/
│   └── main_notebook.ipynb   # narrative / figures
├── scripts/
│   ├── train.py              # regime → env → eval [→ optional PPO train]
│   ├── build_bbo_daily.py    # BBO CSV → daily OBI parquet
│   └── llm_demo.py           # sample governance calls
├── src/
│   ├── bbo_pipeline.py       # ITCH BBO 1m → daily imbalance
│   ├── data_pipeline.py      # parquet load + BBO merge
│   ├── regime_detector.py    # HMM (+ vol fallback), optional 3rd feature (OBI)
│   ├── trading_env.py        # Gymnasium execution env
│   ├── rl_agent.py           # SB3 train / evaluate
│   ├── benchmarks.py
│   ├── llm_explainer.py
│   ├── ui_rollout.py
│   └── utils.py
├── models/                   # saved PPO/SAC .zip (gitignored by default)
├── tests/
└── logs/                     # TensorBoard (gitignored)
```

---

## Data setup

### CRSP-style features (required)

Parquet splits under **`$CFA_DATA_ROOT/features/`**:

- `features_train.parquet`, `features_val.parquet`, `features_test.parquet`

If the repo lives next to shared **`CFADATA`**, the default `CFA_DATA_ROOT` is resolved automatically; otherwise set **`export CFA_DATA_ROOT=...`**.

### BBO / order imbalance (optional but recommended)

- Raw file: **`CFADATA/raw/XNAS-*/xnas-itch-*bbo-1m.csv`** (team NASDAQ ITCH **top-of-book 1-minute** extract).
- **Coverage starts 2019-01-02** (no BBO before that). Merged panel uses **neutral OBI (0)** for earlier daily rows.

Build processed daily panel:

```bash
export CFA_DATA_ROOT="/path/to/CFADATA"
python scripts/build_bbo_daily.py --symbols SPY AAPL
```

Writes **`data/processed/bbo_daily.parquet`**. Training/UI merge this when `use_bbo=True` (default in `load_split`; disable with **`python scripts/train.py --no-bbo`** or Streamlit checkbox).

**Do not commit** multi-GB raw CSVs; keep them on shared drive / Box and document paths in team notes.

---

## Commands cheat sheet

| Task | Command |
|------|---------|
| Tests | `pytest` or `pytest -q` |
| Pipeline + random baseline vs benchmarks | `python scripts/train.py --ticker SPY` |
| Train PPO (slow) | `python scripts/train.py --ticker SPY --train --timesteps 50000` |
| Judge UI | `streamlit run apps/streamlit_app.py` |
| TensorBoard | `tensorboard --logdir logs` |
| LLM samples | `python scripts/llm_demo.py` (needs `ANTHROPIC_API_KEY` for live calls) |

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CFA_DATA_ROOT` | Root folder containing `features/` and optionally `raw/XNAS-*/` |
| `ANTHROPIC_API_KEY` | Live Claude calls for `llm_explainer` |
| `ANTHROPIC_MODEL` | Override model id (e.g. match Stage 1 naming) |

Copy **`.env.example` → `.env`** (see `.gitignore`) if using `python-dotenv` with Streamlit/scripts.

---

## What to commit vs keep local

**Commit**

- All **`src/`**, **`scripts/`**, **`apps/`**, **`tests/`**, **`notebooks/`** (strip huge outputs if needed)
- **`data/cached_llm/*.json`** so judges reproduce explanations without API cost
- **`.env.example`**, **`requirements.txt`**, **`README.md`**, **`.gitignore`**

**Do not commit**

- **`.env`**, API keys
- **`.venv/`**, `__pycache__/`, `.pytest_cache/`
- **Large raw data** under `data/raw/` or full BBO CSV
- **Trained weights** if policy is huge (optional: use Git LFS or share checkpoints separately); default `.gitignore` ignores `models/*.zip`

---

## Team workflow (suggested)

1. Branch from `main`, small PRs.
2. Commit messages: **`[module] short description`** (e.g. `[regime] fix HMM fallback`).
3. Run **`pytest`** before pushing.
4. Align notebook kernel with **project `.venv`** (Python 3.11 + `requirements.txt`).

---

## Licence / competition use

Built for the **CFA AI Investment Challenge**. Cite the team Stage 1–2 submission where required. External data vendors (CRSP, market data) remain subject to their own terms.

---

## Support

Primary contact (Stage 1): **mk859@exeter.ac.uk** (Maksim Kitikov). For code questions, use team chat / issues on your shared Git host.
