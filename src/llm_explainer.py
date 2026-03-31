"""Claude governance explanations + cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .utils import project_root

CACHE_DIR = project_root() / "data" / "cached_llm"
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


def _build_prompt(**kw: str | float | int) -> str:
    return f"""You are an AI execution governance system for an institutional trading desk.
Explain the following trade execution decision in clear, professional English suitable for
a portfolio manager or compliance officer. Be concise (3-5 sentences). Be specific about numbers.

Market conditions:
- Detected regime: {kw['regime_name']} (regime {kw['regime']})
- Realised daily volatility: {float(kw['sigma_t']) * 100:.2f}%
- Liquidity indicator (Amihud): {float(kw['liquidity_t']):.4f}

Execution decision:
- Remaining inventory: {float(kw['inventory_remaining']) * 100:.1f}% of original order
- Fraction executed this interval: {float(kw['action_taken']) * 100:.1f}%
- Estimated execution cost this interval: {float(kw['execution_cost_bps']):.2f} bps

Benchmark comparison (estimated full-order cost):
- Our RL strategy: {float(kw['execution_cost_bps']):.2f} bps
- TWAP benchmark: {float(kw['twap_cost_bps']):.2f} bps
- Almgren-Chriss benchmark: {float(kw['ac_cost_bps']):.2f} bps

Explain: (1) what the market regime means for execution, (2) why this execution speed
was chosen relative to benchmarks, (3) what risk is being managed."""


def explain_execution(
    regime: int,
    regime_name: str,
    inventory_remaining: float,
    action_taken: float,
    execution_cost_bps: float,
    twap_cost_bps: float,
    ac_cost_bps: float,
    sigma_t: float,
    liquidity_t: float,
    use_cache: bool = True,
) -> str:
    prompt = _build_prompt(
        regime=regime,
        regime_name=regime_name,
        inventory_remaining=inventory_remaining,
        action_taken=action_taken,
        execution_cost_bps=execution_cost_bps,
        twap_cost_bps=twap_cost_bps,
        ac_cost_bps=ac_cost_bps,
        sigma_t=sigma_t,
        liquidity_t=liquidity_t,
    )
    key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)["explanation"]
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        explanation = (
            f"In the {regime_name} regime (id {regime}), the desk executed "
            f"{action_taken*100:.1f}% of the remaining book this interval with "
            f"~{execution_cost_bps:.1f} bps estimated cost versus TWAP ~{twap_cost_bps:.1f} bps "
            f"and Almgren–Chriss ~{ac_cost_bps:.1f} bps. "
            f"Daily volatility is about {sigma_t*100:.2f}% with Amihud liquidity {liquidity_t:.4f}. "
            f"The pace balances market impact against inventory risk with {inventory_remaining*100:.1f}% "
            "still to work."
        )
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {"model": "offline-template", "explanation": explanation, "prompt_hash": key},
                f,
                indent=2,
            )
        return explanation
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    explanation = message.content[0].text  # type: ignore[index]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "explanation": explanation, "prompt_hash": key}, f, indent=2)
    return explanation
