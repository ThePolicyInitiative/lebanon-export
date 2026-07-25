"""Deterministic yes/no verdict engine.

Turns the LangGraph tool payload into a clear verdict (yes / no / mixed)
with plain-language reasons and next steps. Used two ways:

1. As a hint passed to the LLM so its answer starts with the right verdict.
2. As the full fallback answer when no LLM is reachable, so the chat
   still says "Yes." or "No." and explains — never an error message.
"""
from __future__ import annotations

from typing import Any


def _fmt_money(value: float) -> str:
    value = float(value or 0)
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def compute_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    """Return {'verdict', 'word', 'reasons', 'next_steps', 'facts'}."""
    trend = payload.get("export_trend") or {}
    capacity = payload.get("factory_capacity") or {}
    competitor = payload.get("competitor_snapshot") or {}

    product = trend.get("product_name") or f"HS {trend.get('hs_code', '?')}"
    short_product = str(product).split(";")[0].strip()
    dest = trend.get("destination") or "this market"
    records = trend.get("records") or []
    start = float(trend.get("start_value") or 0)
    end = float(trend.get("end_value") or 0)
    change = float(trend.get("absolute_change") or 0)
    pct = trend.get("percent_change")
    factories = int(capacity.get("matching_factories") or 0)
    top = competitor.get("top_exporters") or []
    first_year = records[0]["year"] if records else None
    last_year = records[-1]["year"] if records else None

    exports_now = end > 0
    growing = change > 0
    declining = change < 0 and start > 0
    has_capacity = factories > 0
    rivals = [str(r.get("exporter", "")) for r in top[:3] if r.get("exporter")]

    reasons: list[str] = []
    next_steps: list[str] = []

    if exports_now:
        line = f"Lebanon sold {_fmt_money(end)} of {short_product} to {dest} in {last_year}"
        if pct is not None and first_year is not None:
            direction = "up" if change >= 0 else "down"
            line += f", {direction} {abs(pct):.0f}% since {first_year}"
        elif start == 0 and change > 0:
            line += f", starting from zero in {first_year}"
        reasons.append(line + ".")
    elif start > 0:
        reasons.append(
            f"Lebanon used to sell {short_product} to {dest} ({_fmt_money(start)} in {first_year}) "
            f"but exports fell to zero by {last_year}."
        )
    else:
        reasons.append(f"Lebanon has no recorded exports of {short_product} to {dest} in the data.")

    if has_capacity:
        reasons.append(f"{factories} Lebanese factories make this kind of product, so supply capacity exists.")
    else:
        reasons.append("No matching Lebanese factories were found for this product, so local supply capacity is unproven.")

    if rivals:
        reasons.append(f"The market is contested — the biggest suppliers are {', '.join(rivals)}.")

    # Decide the verdict
    if exports_now and growing and has_capacity:
        verdict, word = "yes", "Yes"
        next_steps = [
            "Build on what is already working: grow existing buyer relationships before finding new ones.",
            f"Compare Lebanon's prices and quality against {rivals[0] if rivals else 'the top competitors'} in {dest}.",
        ]
    elif not exports_now and not has_capacity:
        verdict, word = "no", "No"
        next_steps = [
            "This product has neither current sales nor identified factories, so it is a weak starting point.",
            "Screen alternative products with demonstrated domestic production and existing regional sales before committing resources.",
        ]
    elif declining or (exports_now and not has_capacity) or (not exports_now and has_capacity):
        verdict, word = "mixed", "Yes, but with caution"
        if declining:
            next_steps.append("Find out why sales fell: lost buyers, price competition, or logistics — talk to past exporters first.")
        if not has_capacity:
            next_steps.append("Verify real production capacity before committing — the factory data shows no clear match.")
        if not exports_now and has_capacity:
            next_steps.append(f"Capacity exists but there are no sales yet — start with a small trial shipment or a trade fair in {dest}.")
    else:
        verdict, word = "no", "No"
        next_steps = ["Screen alternative products with demonstrated domestic capacity and stronger evidence of market demand."]

    return {
        "verdict": verdict,
        "word": word,
        "reasons": reasons,
        "next_steps": next_steps,
        "facts": {
            "product": short_product,
            "destination": dest,
            "latest_value": end,
            "latest_year": last_year,
            "percent_change": pct,
            "matching_factories": factories,
        },
    }


def verdict_answer(payload: dict[str, Any], diagnosis: dict[str, Any] | None = None) -> str:
    """Full plain-language answer built only from the data (no LLM needed)."""
    v = compute_verdict(payload)
    lines = [f"**{v['word']}.**", ""]
    lines += [f"- {r}" for r in v["reasons"]]

    output = (diagnosis or {}).get("output", {})
    extra_causes = [c for c in output.get("main_causes", []) if c not in v["reasons"]]
    if extra_causes:
        lines.append("")
        lines.append("Most likely causes:")
        lines += [f"- {c}" for c in extra_causes[:3]]

    if v["next_steps"]:
        lines.append("")
        lines.append("What to do next:")
        lines += [f"- {s}" for s in v["next_steps"]]
    return "\n".join(lines)


def compute_confidence(payload: dict[str, Any]) -> float:
    """Computed confidence = evidence coverage x signal agreement.

    A formula, not a model's felt number (design decision motivated by the
    2025 Deloitte fabricated-citations incident: numbers that matter are
    computed, and every claim traces to a tool output).
    """
    trend = payload.get("export_trend") or {}
    capacity = payload.get("factory_capacity") or {}
    competitor = payload.get("competitor_snapshot") or {}
    macro = payload.get("macro_logistics") or {}
    signals = payload.get("market_signals") or {}

    # Coverage: how many evidence sources actually returned something
    sources = [
        any(r.get("export_value", 0) > 0 for r in (trend.get("records") or [])),
        bool(competitor.get("top_exporters")),
        (capacity.get("matching_factories") or 0) > 0,
        bool(macro.get("latest_indicators")),
        signals.get("status") == "ok" and bool(signals.get("results")),
    ]
    coverage = sum(sources) / len(sources)

    # Agreement: do the three core signals point the same way?
    end = float(trend.get("end_value") or 0)
    change = float(trend.get("absolute_change") or 0)
    factories = int(capacity.get("matching_factories") or 0)
    positive = [end > 0, change > 0, factories > 0]
    agreement = 1.0 if len(set(positive)) == 1 else 0.5

    return round(min(0.95, max(0.2, 0.25 + 0.45 * coverage + 0.3 * agreement * coverage)), 2)
