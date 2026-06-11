"""Static site export for the Cloudflare deployment.

Renders every page and precomputes Execution Lab results into plain HTML
under ``dist/``, so the site is served as static assets with no Python
runtime. The fragment URL scheme must stay in sync with
``web/static/js/lab.js``.

Run from the repo root:

    python -m web.export
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DIST = ROOT / "dist"

# Precomputed grid. Every valid start date in the test split is exported
# for each horizon, regime count, and policy.
HORIZONS = [5, 10, 20]
N_REGS = [2, 3]
REGIME_SPLITS = ["train", "val", "test"]
LAB_SPLIT = "test"

PAGES = {
    "index.html": "/",
    "case-study/index.html": "/case-study",
    "user-manual/index.html": "/user-manual",
}


def _redirect_llm_cache() -> None:
    """Send governance cache writes to a temp dir seeded with the committed
    cache, so an export run keeps cache hits but never dirties the repo."""
    committed = ROOT / "data" / "cached_llm"
    tmp = Path(tempfile.mkdtemp(prefix="qf_llm_cache_"))
    if committed.is_dir():
        for f in committed.glob("*.json"):
            shutil.copy2(f, tmp / f.name)
    os.environ["CFA_LLM_CACHE_DIR"] = str(tmp)


def _policies() -> dict[str, dict[str, str]]:
    """Slug -> {form value, display label}."""
    out: dict[str, dict[str, str]] = {}
    best = ROOT / "models" / "best_ppo_twap_gap.zip"
    if best.is_file():
        out["ppo"] = {"value": str(best), "label": "PPO (best checkpoint)"}
    out["random"] = {"value": "random", "label": "Random Agent"}
    return out


def _write(rel_path: str, content: str) -> None:
    path = DIST / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _export_pages(client) -> None:
    for rel_path, url in PAGES.items():
        resp = client.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"GET {url} returned {resp.status_code}")
        _write(rel_path, resp.get_data(as_text=True))
        logger.info("Exported %s", rel_path)


def _export_run_page(app, dates: list[str], policies: dict) -> None:
    from flask import render_template

    from web.app import _split_date_context

    lab_config = {
        "dates": dates,
        "horizons": HORIZONS,
        # Last valid start index per horizon: an episode needs T bars plus
        # one settlement bar after the start (see _resolve_start_date).
        "maxStartIndex": {str(t): len(dates) - t - 1 for t in HORIZONS},
    }
    with app.test_request_context("/run"):
        html = render_template(
            "run.html",
            active_page="run",
            static_mode=True,
            static_horizons=HORIZONS,
            static_policies=[
                {"value": slug, "label": p["label"]} for slug, p in policies.items()
            ],
            lab_config=json.dumps(lab_config),
            **_split_date_context(LAB_SPLIT, T=10),
        )
    _write("run/index.html", html)
    logger.info("Exported run/index.html")


def _export_404(app) -> None:
    from flask import render_template

    with app.test_request_context("/404"):
        _write("404.html", render_template("404.html", active_page=None))
    logger.info("Exported 404.html")


def _export_regime_fragments(client) -> None:
    for split in REGIME_SPLITS:
        for n_reg in N_REGS:
            resp = client.post("/api/regimes", data={"split": split, "n_reg": n_reg})
            if resp.status_code != 200:
                raise RuntimeError(f"/api/regimes {split} r{n_reg} returned {resp.status_code}")
            _write(f"lab/regimes/{split}-r{n_reg}.html", resp.get_data(as_text=True))
    logger.info("Exported %d regime fragments", len(REGIME_SPLITS) * len(N_REGS))


def _export_run_fragments(client, dates: list[str], policies: dict) -> None:
    total = sum(
        (len(dates) - t) * len(N_REGS) * len(policies) for t in HORIZONS
    )
    done = 0
    started = time.time()
    for T in HORIZONS:
        valid_dates = dates[: len(dates) - T]
        for n_reg in N_REGS:
            for slug, policy in policies.items():
                for date in valid_dates:
                    resp = client.post(
                        "/api/run-all",
                        data={
                            "split": LAB_SPLIT,
                            "n_reg": n_reg,
                            "horizon": T,
                            "start_date": date,
                            "policy": policy["value"],
                        },
                    )
                    body = resp.get_data(as_text=True)
                    if resp.status_code != 200 or "renderRegimeChart" not in body:
                        raise RuntimeError(
                            f"/api/run-all failed for {slug} r{n_reg} t{T} {date}: "
                            f"status {resp.status_code}, body: {body[:300]}"
                        )
                    _write(f"lab/run/{slug}-r{n_reg}-t{T}-{date}.html", body)
                    done += 1
                    if done % 200 == 0:
                        rate = done / (time.time() - started)
                        logger.info(
                            "Run fragments: %d/%d (%.1f/s, ~%.0f s left)",
                            done, total, rate, (total - done) / max(rate, 1e-9),
                        )
    logger.info("Exported %d run fragments in %.0f s", done, time.time() - started)


def main() -> None:
    _redirect_llm_cache()

    # Imported here so the cache redirect is in place before the app
    # precomputes the case study.
    from web.app import app, _lab_df_for_split

    df = _lab_df_for_split(LAB_SPLIT)
    if df is None:
        raise RuntimeError(
            f"No '{LAB_SPLIT}' feature panel. Check CFA_DATA_ROOT or deploy_data/features/."
        )
    dates = [d.date().isoformat() for d in df.index]
    policies = _policies()

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    shutil.copytree(ROOT / "web" / "static", DIST / "static")

    client = app.test_client()
    _export_pages(client)
    _export_run_page(app, dates, policies)
    _export_404(app)
    _export_regime_fragments(client)
    _export_run_fragments(client, dates, policies)

    n_files = sum(1 for p in DIST.rglob("*") if p.is_file())
    size_mb = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file()) / 1e6
    logger.info("Done: %d files, %.1f MB in %s", n_files, size_mb, DIST)


if __name__ == "__main__":
    main()
