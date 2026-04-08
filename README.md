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
# Open http://localhost:5001

# Quick smoke test
pytest -q
```

The web app pre-computes the case study at startup (~10 s) and requires a trained PPO checkpoint under `models/` (see section 5 or use the provided `best_ppo_twap_gap.zip` if included).

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

Required columns (CRSP naming):

| Column | Description |
|---|---|
| `permno` | Security identifier |
| `date` | Trading date |
| `prc` | Closing price |
| `openprc` | Opening price |
| `high` | Daily high |
| `low` | Daily low |
| `vol` | Daily volume (shares) |
| `shrout` | Shares outstanding |
| `ret`, `retx` | Total & ex-dividend return |
| `ticker` | Ticker symbol (e.g. `SPY`) |

The code builds derived features automatically: `realised_vol_20`, `sigma_daily`, `amihud_illiquidity`, `bid_ask_proxy`, `volume_to_spread`, `vix_aligned`.

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
├── models/                     # PPO .zip checkpoints (gitignored)
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

### Train PPO

```bash
python scripts/train.py --ticker SPY --train --order-notional-usd 5e6 \
  --residual-bound 0.15 --relative-is-scale 2.0 --timesteps 300000
```

Saves `models/best_ppo_twap_gap.zip` when the path-aligned TWAP gap improves during training.

### Key training flags

| Flag | Default | Purpose |
|---|---|---|
| `--order-notional-usd` | 0 | USD order size (>0 enables physical mode) |
| `--residual-bound` | — | TWAP-residual action (e.g. 0.15 = ±15% from TWAP) |
| `--relative-is-scale` | 0 | Per-bar improvement-over-TWAP reward scaling |
| `--timesteps` | 300000 | Training steps |
| `--n-envs` | 1 | Vectorized rollouts |
| `--eval-freq` | 25000 | Evaluate & save best checkpoint every N steps |

### Scenario benchmarks

```bash
python scripts/scenario_benchmarks.py \
  --model models/best_ppo_twap_gap.zip \
  --order-notional-usd 5e6 --T 10 \
  --residual-bound 0.15 --relative-is-scale 2.0
```

### Path-aligned evaluation

The benchmark table uses random episode starts. For head-to-head RL vs TWAP comparison, the system computes **path-aligned metrics** on identical `(start, T)` windows:

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
