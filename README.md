# QuantFin Exeter — AI Investment Challenge 2025-26

**University of Exeter · Team 18**

| Kirill Papka | Thomas Nguyen | Harrison Maxwell | Maksim Kitikov (lead) |
|---|---|---|---|

> Institutional **optimal trade execution** using **regime-aware reinforcement learning** (PPO), **classical benchmarks** (TWAP / VWAP / Almgren–Chriss / Immediate), **NASDAQ ITCH order-book imbalance**, and an **LLM governance** layer (Anthropic Claude, cached for reproducibility).

**Live demo (Deployed online):** → [https://cfadeployedteam18.onrender.com](https://cfadeployedteam18.onrender.com)

**Live demo:** `python -m web.app` → [http://localhost:5001](http://localhost:5001)

---

## Table of contents

1. [Quick start (for judges)](#1-quick-start-for-judges)
2. [How to obtain data](#2-how-to-obtain-data)
3. [Repository layout](#3-repository-layout)
4. [Running the web application](#4-running-the-web-application)
5. [Training & evaluation](#5-training--evaluation)
6. [Environment variables](#6-environment-variables)
7. [AI tools disclosure](#7-ai-tools-disclosure)
8. [Licence](#8-licence)

---

## 1. Quick start (for judges)

### Requirements

- **Python 3.11** (Torch / SB3 wheels may fail on newer versions)
- Git, ~4 GB disk (venv + data)

### Setup

```bash
git clone https://github.com/KirikPapka/quantfin_exeter.git
cd quantfin_exeter

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure data path

```bash
# Point to the folder that contains features/ (see section 2)
export CFA_DATA_ROOT="/absolute/path/to/your/data"
```

Or copy `.env.example` → `.env` and fill in `CFA_DATA_ROOT` there (auto-loaded by the app).

### Run

```bash
# Web application (recommended — includes Case Study, Execution Lab, User Manual)
python -m web.app
# Open http://localhost:5001/case-study

# Quick smoke test
pytest -q
```

The web app **pre-computes** the case study at startup (~10 s). It uses the same settings as `scripts/train.py` when evaluating with **`--order-notional-usd` > 0** (see §5): **\$5M** notional, **T = 10**, **TWAP-residual ±15%**, **relative-IS scale 2.0**, test split **SPY**, optional BBO/news merge if files exist.

### Committed model (no training required)

This repository includes:

- **`models/best_ppo_twap_gap.zip`** — trained PPO checkpoint for SPY execution (eval-only / case study).
- **`models/fixed_eval_starts.json`** — path-aligned evaluation windows shared by `train.py` and the web app.

You do **not** need to run `--train` to verify the solution. After `pip install` and `CFA_DATA_ROOT`, use **`python -m web.app`** or **`python scripts/train.py --ticker SPY --order-notional-usd 5e6`** (eval-only loads `best_ppo_twap_gap.zip` automatically when `--load-model` is omitted).

### Matching numbers vs the public deployment

Use the **same** market data and optional microstructure files as the host you compare against:

1. **`CFA_DATA_ROOT`** — identical `features_*.parquet`; for closest match to [the deployed demo](https://cfadeployedteam18.onrender.com/case-study), also mirror optional `data/processed/bbo_daily.parquet` and `news_daily_SPY.parquet` if the server has them.
2. **`models/fixed_eval_starts.json`** — already in-repo; delete it only if you intentionally want new random windows (reported means will change).
3. **Checkpoint choice:** **`python -m web.app`** runs a small **sweep** over recent `models/*.zip` files and picks the best on a validation subset (see `web/precompute.py`). If you add extra checkpoints beside `best_ppo_twap_gap.zip`, the UI might select a different policy than the committed default. For a **deterministic** match to the shipped checkpoint, either keep only the intended `.zip` files in `models/` or rely on **`train.py`** eval, which uses **`best_ppo_twap_gap.zip`** by default when not training.
4. **CLI benchmark table:** `train.py` prints TWAP / VWAP / Almgren–Chriss / Immediate on the **same** row starts as the RL line when `fixed_eval_starts.json` is used (aligned with the case study logic).

Then open **`/case-study`** or run **`python scripts/train.py --ticker SPY --order-notional-usd 5e6`** and compare the printed **Benchmarks** table. Override the checkpoint with **`--load-model path/to/other.zip`** if needed.

---

## 2. How to obtain data

All data sources are **publicly accessible** per competition rules (Rule 4.6). No proprietary terminals required.

### 2.1 Stock price data (required)

**Source:** any free stock API (e.g. Yahoo Finance via `yfinance`, Alpha Vantage, Polygon free tier, or a CRSP educational extract).

The pipeline expects **per-split parquet files** under `$CFA_DATA_ROOT/features/`:

```
features/
├── features_train.parquet   (2018-01-01 to 2022-12-31)
├── features_val.parquet     (2023-01-01 to 2023-12-31)
└── features_test.parquet    (2024-01-01 to 2024-12-31)
```

**Parquet schema** matches `load_features_parquet` in [`src/data_pipeline.py`](src/data_pipeline.py) (that function is the source of truth).

| Column | Required | Description |
|---|---|---|
| `date` | Yes | Trading date (indexed after load) |
| `prc` | Yes | Closing price |
| `high`, `low` | Yes | Daily high / low |
| `vol` | Yes | Daily volume (shares) |
| `real_vol_20d` | Yes* | 20-day realised volatility (annualised, CRSP-style) |
| `realised_vol_20` | Yes* | Alias for the same series if your extract already uses this name |
| `oprc` | No | Opening price (preferred name in loader) |
| `openprc` | No | Alias for opening price; if neither `oprc` nor `openprc` is present, open defaults to close |
| `amihud_20d` | No | Amihud illiquidity; else `amihud_daily`; missing values filled with 0 in-panel |
| `vix` | No | VIX level (forward-filled into `vix_aligned`) |
| `ticker` | No | Filter to one symbol (e.g. `SPY`) when present |
| `permno` | No | Security identifier |
| `shrout` | No | Shares outstanding |
| `ret`, `retx` | No | Total & ex-dividend return |

\*Exactly one of `real_vol_20d` or `realised_vol_20` must be present.

The loader then builds: `realised_vol_20` (internal), `sigma_daily`, `amihud_illiquidity`, `bid_ask_proxy`, `volume_to_spread`, `vix_aligned`.

### 2.2 VIX data (included in features or downloadable)

**Source:** [Cboe VIX historical data](https://www.cboe.com/tradable-products/vix/vix-historical-data/) — free CSV download, 1990–present. The pipeline expects a `vix` column in the parquet or merges it from a separate file.

### 2.3 NASDAQ ITCH BBO order imbalance (optional, recommended)

**Source:** [Databento](https://databento.com/portal/browse) — NASDAQ TotalView-ITCH **BBO-1m** schema.

**How to obtain:**

1. Go to [databento.com/portal/browse](https://databento.com/portal/browse)
2. Register for a free account — new accounts receive **$125 USD free credit**
3. Search for **NASDAQ TotalView-ITCH**, select **BBO-1m** schema
4. Purchase data for **SPY** from **2019-01-02 to 2024-12-31** (optionally also **AAPL**)
5. Download the CSV and place under `$CFA_DATA_ROOT/raw/`

**Build the daily panel:**

```bash
python scripts/build_bbo_daily.py --symbols SPY
# writes data/processed/bbo_daily.parquet
```

Coverage starts **2019-01-02**; earlier rows use neutral OBI (0). The pipeline merges this automatically when `bbo_daily.parquet` exists.

### 2.4 News sentiment counts (optional)

**Source:** [Finnhub](https://finnhub.io/register) — free tier, no credit card required.

1. Register at [finnhub.io/register](https://finnhub.io/register) to get a free API key
2. Add `FINNHUB_API_KEY=your_key` to `.env`
3. Fetch news:

```bash
python scripts/fetch_finnhub_news.py --symbol SPY
# or for ETF holdings-weighted proxy:
python scripts/fetch_finnhub_news.py --symbol SPY --etf-proxy --max-constituents 20
```

Writes `data/processed/news_daily_SPY.parquet`. The RL observation includes z-scored daily news intensity when this file exists.

### 2.5 LLM governance (Anthropic Claude)

**Cost to reproduce: < $20 USD** (well within the competition threshold).

- Cached responses are committed in `data/cached_llm/*.json` — judges can reproduce all governance text **without any API key or cost**.
- For live calls: add `ANTHROPIC_API_KEY=your_key` to `.env`. Model: `claude-sonnet-4-20250514`.
- Without an API key, an offline template fallback generates and caches deterministic explanations.

---

## 3. Repository layout

```
quantfin_exeter/
├── README.md
├── LICENSE                     # MIT License (competition requirement)
├── requirements.txt
├── environment.yml
├── .env.example                # copy to .env, fill in keys
├── web/
│   ├── app.py                  # Flask routes + API
│   ├── precompute.py           # Case study pre-computation
│   ├── templates/              # Jinja2: home, case study, run, user manual
│   └── static/                 # CSS, JS (Plotly charts)
├── data/
│   ├── raw/                    # Large files (gitignored)
│   ├── processed/              # bbo_daily.parquet, news (gitignored)
│   ├── features/               # Train/val/test parquet (via CFA_DATA_ROOT)
│   └── cached_llm/             # Committed JSON caches for judges
├── notebooks/
│   └── main_notebook.ipynb     # Narrative + figures
├── scripts/
│   ├── train.py                # Regime → env → eval [→ PPO train]
│   ├── scenario_benchmarks.py  # Controlled flat/up/down scenarios
│   ├── build_bbo_daily.py      # BBO CSV → daily OBI parquet
│   ├── fetch_finnhub_news.py   # Finnhub → daily news counts
│   └── llm_demo.py             # Governance demo
├── src/
│   ├── data_pipeline.py        # Parquet load + BBO/news merge
│   ├── regime_detector.py      # HMM (+ vol fallback)
│   ├── trading_env.py          # Gymnasium optimal execution env
│   ├── rl_agent.py             # SB3 train / evaluate
│   ├── benchmarks.py           # TWAP, VWAP, A-C, Immediate
│   ├── execution_impact.py     # Participation-style market impact
│   ├── trend_classifier.py     # Rolling return trend labels
│   ├── regime_switching.py     # Trend-based policy routing
│   ├── llm_explainer.py        # Claude governance + cache
│   ├── ui_rollout.py           # Single-episode rollout for UI
│   └── utils.py
├── models/                     # best_ppo_twap_gap.zip + fixed_eval_starts.json (tracked); other .zip gitignored
├── tests/
└── logs/                       # TensorBoard (gitignored)
```

---

## 4. Running the web application

```bash
python -m web.app
# Starts on http://localhost:5001
```

**Pages:**

| URL | Description |
|---|---|
| `/` | Home — project overview and pipeline diagram |
| `/case-study` | Pre-computed SPY sell execution walkthrough with benchmark results |
| `/run` | Interactive Execution Lab — configure and run the full pipeline |
| `/user-manual` | Methodology, formulas (KaTeX), and configuration reference |

---

## 5. Training & evaluation

### Quick eval (no training)

```bash
python scripts/train.py --ticker SPY --order-notional-usd 5e6
```

With **`--order-notional-usd` > 0**, `train.py` automatically applies **`--residual-bound 0.15`** and **`--relative-is-scale 2.0`** if you omit them — matching **`web/precompute.py`** and the **Case Study** page. **Eval-only** loads **`models/best_ppo_twap_gap.zip`** automatically when that file exists (the committed checkpoint). Evaluation uses **`models/fixed_eval_starts.json`** when present (otherwise it creates one under `models/`).

Use another checkpoint explicitly:

```bash
python scripts/train.py --ticker SPY --order-notional-usd 5e6 \
  --load-model models/your_other.zip
```

### Train PPO

```bash
python scripts/train.py --ticker SPY --train --order-notional-usd 5e6 \
  --residual-bound 0.15 --relative-is-scale 2.0 --timesteps 300000
```

(`--residual-bound` / `--relative-is-scale` are redundant here because physical mode defaults them to 0.15 and 2.0; kept explicit for clarity.)

Saves `models/best_ppo_twap_gap.zip` when the path-aligned TWAP gap improves during training.

### Key training flags

| Flag | Default | Purpose |
|---|---|---|
| `--order-notional-usd` | 0 | USD order size (>0 enables physical mode) |
| `--residual-bound` | **0.15** if physical\* | TWAP-residual action (± fraction from TWAP schedule) |
| `--relative-is-scale` | **2.0** if physical\* | Per-bar improvement-over-TWAP reward scaling |
| `--timesteps` | 300000 | Training steps |
| `--n-envs` | 1 | Vectorized rollouts |
| `--eval-freq` | 25000 | Evaluate & save best checkpoint every N steps |

\*When `--order-notional-usd > 0` and the flag is omitted. Pass your own values to override.

### Scenario benchmarks

```bash
python scripts/scenario_benchmarks.py \
  --model models/best_ppo_twap_gap.zip \
  --order-notional-usd 5e6 --T 10 \
  --residual-bound 0.15 --relative-is-scale 2.0
```

### Path-aligned evaluation

**Case Study** and **`train.py`** evaluate RL and benchmarks on the **same** `(row_start, T)` windows when **`models/fixed_eval_starts.json`** is present (up to 200 windows; see `web/precompute.py`). If that file is missing, `train.py` **generates** it once; the web app then reuses it — so local numbers stay consistent across CLI and UI.

For head-to-head RL vs TWAP, the report includes **path-aligned metrics**:

| Metric | Meaning |
|---|---|
| `mean_RL_minus_TWAP_bps` | Mean IS gap on matched windows (> 0 = RL wins) |
| `pct_beat_TWAP_IS` | Share of episodes where RL IS > TWAP IS |
| `IS-gap Sharpe` | Sharpe ratio of the per-episode IS gap |

---

## 6. Environment variables

Copy `.env.example` → `.env`:

| Variable | Required | Purpose |
|---|---|---|
| `CFA_DATA_ROOT` | Yes | Folder containing `features/` parquet splits |
| `ANTHROPIC_API_KEY` | No | Live Claude calls (cached responses work without) |
| `ANTHROPIC_MODEL` | No | Override model (default: `claude-sonnet-4-20250514`) |
| `FINNHUB_API_KEY` | No | News data fetch ([free registration](https://finnhub.io/register)) |

---

## 7. AI tools disclosure

Per competition Rule 4.1 (Transparency) and Rule 4.5 (Disclosure):

### AI as development assistant

- **Cursor IDE with Claude and Composer** — used for code writing, debugging, and refactoring across the codebase.


All generated code was reviewed, understood, and can be explained by the team.

### AI as component of the solution

- **Anthropic Claude** (`claude-sonnet-4-20250514`) — integrated into the system for **LLM governance** (explaining execution decisions in plain English for compliance). Called via API at runtime; responses cached in `data/cached_llm/*.json`.
- **Reproduction cost: < $20 USD.** Cached responses are committed so judges can verify without API cost.
- Without an API key, an **offline template** generates deterministic explanations automatically.

### Data created as part of the solution

- **Derived features** (volatility, Amihud, bid-ask proxy) — computed from public stock data in `data_pipeline.py`.
- **BBO daily order imbalance** — aggregated from NASDAQ ITCH BBO-1m (Databento, publicly purchasable) in `build_bbo_daily.py`.
- **News daily counts** — fetched from Finnhub free API in `fetch_finnhub_news.py`.
- **LLM cache files** — committed JSON responses from Claude governance calls.

---

## 8. Licence

Released under the **MIT License** as required by competition rules (Rule 1.5).

Built for the **CFA Institute AI Investment Challenge 2025–26** (CFA Society United Kingdom).

**Contact:** mk859@exeter.ac.uk (Maksim Kitikov)
