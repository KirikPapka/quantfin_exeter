# Deploy (Cloudflare)

The hosted demo at **https://exeter.quantfin.dev** is a static export of the Flask app, served as Cloudflare Workers static assets. `web/export.py` renders every page and precomputes all Execution Lab configurations into plain HTML fragments, so the deployed site needs no Python runtime, never sleeps, and costs nothing to host.

## How it deploys

Every push to `main` runs [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml):

1. `bash build.sh`: CPU-only PyTorch, then `requirements.txt`
2. `pytest -q`
3. `python -m web.export`: builds `dist/` (~3 min: case study plus ~2,900 lab fragments)
4. `wrangler deploy` via `cloudflare/wrangler-action`

## One-time setup

1. In the Cloudflare dashboard, create an API token from the **Edit Cloudflare Workers** template.
2. Add two repository secrets (**Settings → Secrets and variables → Actions**):
   - `CLOUDFLARE_API_TOKEN`
   - `CLOUDFLARE_ACCOUNT_ID` (dashboard → Workers & Pages → right sidebar)
3. Push to `main`. The first deploy creates the `quantfin-exeter` worker and attaches the custom domain `exeter.quantfin.dev` (the `quantfin.dev` zone must be on the same Cloudflare account).

## Deploy from a laptop

```bash
python -m web.export
npx wrangler deploy        # prompts for a Cloudflare login on first use
```

Preview the build locally with `python -m http.server --directory dist 8000`.

## What is precomputed

The Execution Lab grid covers every valid start date in the test split for:

| Parameter | Values |
|---|---|
| Horizon T | 5, 10, 20 days |
| HMM states | 2, 3 |
| Policy | PPO best checkpoint, random agent |
| Split (episode + benchmarks) | test |
| Split (regime detection) | train, val, test |

Governance text comes from the committed response cache with the offline template as fallback, so no API keys are needed at build or serve time. For arbitrary horizons, other splits, or live LLM calls, run the Flask app locally: `python -m web.app`.

## Data needed in the repo

Both are committed, nothing to add before a deploy:

- `models/best_ppo_twap_gap.zip`: trained PPO checkpoint
- `deploy_data/features/features_{train,val,test}.parquet`: feature panels (auto-detected when `CFA_DATA_ROOT` is unset)

Optional, also committed when present: `data/processed/bbo_daily.parquet` (order imbalance) and `data/processed/news_daily_SPY.parquet` (news features).
