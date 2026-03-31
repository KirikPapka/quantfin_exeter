# QuantFin Exeter — CFA AI Investment Challenge

**University of Exeter · Competition Group 18**

| Kirill Papka | Thomas Nguyen | Harrison Maxwell | Maksim Kitikov (lead) |
|--------------|---------------|------------------|------------------------|

Institutional **optimal execution** with **regime-aware RL** (PPO/SAC), **classical benchmarks** (TWAP / VWAP / Almgren–Chriss / **Immediate**; optional **USD notional** + participation / Amihud impact and delayed start bar), **NASDAQ ITCH BBO–based order imbalance**, and an **LLM governance** layer (Anthropic Claude, cached for reproducibility). Primary judge-facing demo: **Streamlit**; narrative: **`notebooks/main_notebook.ipynb`**.

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

**News (Finnhub, free API):** Add **`FINNHUB_API_KEY`** to `.env` (register at [finnhub.io/register](https://finnhub.io/register)), then:

```bash
python scripts/fetch_finnhub_news.py --symbol SPY
```

For **SPY** (and other ETFs), direct company-news is often empty; use **holdings-weighted** constituent news (Finnhub `stock/etf/holdings`, or a built-in fallback slice):

```bash
python scripts/fetch_finnhub_news.py --symbol SPY --etf-proxy --max-constituents 20
```

That issues roughly `(years of chunks) × (constituents)` calls — use **`--sleep 1.2`** and a narrower **`--from-date`** if you hit 429.

Uses **~1 request/second** pacing and **retries on HTTP 429**. If you still hit limits, run again later or use `--sleep 2` / smaller `--from-date` ranges.

Writes **`data/processed/news_daily_SPY.parquet`** (daily article counts). `load_split` merges **`news_count`** when the file exists; the RL observation includes **z-scored news intensity** plus a **TWAP schedule gap** (inventory vs equal schedule). **`--no-news`** on `train.py` skips the merge. **Retrain** policies after this change (observation size **9**).

**Do not commit** multi-GB raw CSVs; keep them on shared drive / Box and document paths in team notes.

---

## Commands cheat sheet

| Task | Command |
|------|---------|
| Tests | `pytest` or `pytest -q` |
| Synthetic **flat / up / down** paths (benchmarks + RL) | `python scripts/scenario_benchmarks.py` · random RL by default · **trained:** `python scripts/scenario_benchmarks.py --model models/PPO_....zip --order-notional-usd 5e8` (match `T` / notional / start-bar to training) |
| Finnhub news → daily counts parquet | `python scripts/fetch_finnhub_news.py --symbol SPY` · ETF proxy: add `--etf-proxy --max-constituents 20` |
| Pipeline + random baseline vs benchmarks | `python scripts/train.py --ticker SPY` |
| Physical USD order + impact (participation / Amihud) | `python scripts/train.py --ticker SPY --order-notional-usd 10e9 --order-start-bar 0` |
| Train PPO (slow) | `python scripts/train.py --ticker SPY --train` (default **300k** steps; wider MLP + **9-dim** obs — **retrain** old checkpoints) |
| Judge UI | `streamlit run apps/streamlit_app.py` |
| TensorBoard | `tensorboard --logdir logs` |
| LLM samples | `python scripts/llm_demo.py` (needs `ANTHROPIC_API_KEY` for live calls) |

**Order size & timing:** With `--order-notional-usd > 0`, the pipeline maps **USD notional → shares** at the arrival (prior close) and applies a shared **participation-style** impact (daily σ, Amihud, bar dollar volume) on each slice for classical benchmarks and for **`OptimalExecutionEnv`** (RL row). `--order-start-bar` is the first bar inside the `T`-day window when trading is allowed; **arrival** for implementation shortfall is the **previous bar’s close**. Notional `0` keeps the legacy abstract unit inventory without that layer. Streamlit exposes the same controls in the sidebar.

**Train / eval parity (physical RL):** When notional is positive, `train.py`, `scenario_benchmarks.py`, and Streamlit all attach the same **institutional** env defaults unless you override: **per-step inventory cap** `max_inventory_fraction_per_step=0.33` (disable with `--no-per-step-cap`), **`--is-reward-scale`** default **1.28** on the dollar IS term, and **`--twap-slice-bonus`** default **0.30** (same-bar TWAP-sized reference vs your slice in **`sell_effective_close`** space; set `0` to turn off). Retrain and evaluate with **matching** `--order-notional-usd`, `--order-start-bar`, `T`, and these flags—otherwise the policy sees a different reward than benchmarks/scenarios.

### RL evaluation (path-aligned)

The **benchmark table** still shows classical strategies averaged over **random episode starts** (same as before). That mixes many different price paths, so **RL mean IS** and **TWAP mean IS** are **not** automatically comparable as “who won head-to-head.”

After each run, logs (and Streamlit **Benchmarks** expander) include **path-aligned diagnostics** from `evaluate_agent(..., bench_params=...)`:

| Field | Meaning |
|--------|--------|
| **mean_episode_return** | Average sum of rewards per episode (use to compare **trained vs random** policy on the same env). |
| **mean_twap_is_bps (paths)** | TWAP implementation shortfall (bps) averaged over the **exact** `(start, T)` windows used for each RL episode. |
| **mean_vwap_is_bps (paths)** | Same for VWAP. |
| **mean_RL_minus_TWAP_bps** | Mean of (RL IS bps − TWAP IS bps) on those **same** windows. **> 0** ⇒ RL achieved a **higher** average sell price vs arrival than TWAP on those paths (under your IS sign). |
| **mean_RL_minus_VWAP_bps** | Same vs VWAP. |
| **pct_beat_TWAP_IS** | Share of episodes where RL IS bps **>** TWAP IS bps on the identical window. |

**How to judge RL:** (1) **Episode return** vs random baseline with the same flags. (2) **pct_beat_TWAP_IS** and **mean_RL_minus_TWAP_bps** for a fair schedule comparison on matched paths. (3) **Completion rate**. The aggregate benchmark row remains useful for judges alongside classical baselines, but **head-to-head** metrics above are the correct read for “did RL beat TWAP?”

**Controlled scenarios:** `src/scenario_paths.py` builds long **flat**, **monotone up**, and **monotone down** daily panels. `scripts/scenario_benchmarks.py` runs the same benchmark machinery on each so you can see how **TWAP/VWAP/AC** IS flips sign with trend (up ⇒ positive mean IS vs arrival in legacy mode; down ⇒ negative; flat ⇒ ~0) while **Immediate (legacy)** stays at zero. Pass **`--model path/to/PPO_*.zip`** to evaluate a **trained** policy on those synthetics (use the same **`--T`**, **`--order-notional-usd`**, **`--order-start-bar`**, and institutional overrides **`--no-per-step-cap` / `--max-inventory-frac-per-step` / `--is-reward-scale` / `--twap-slice-bonus`** as in training; **9-dim** observation must match). Use this before interpreting real-ticker results.

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
