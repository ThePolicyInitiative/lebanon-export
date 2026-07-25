from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agents.llm_clients import call_llm


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "diagnostic_agent_v2.md"


def fallback_diagnosis(tool_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a structured evidence-led diagnosis without a hosted model."""
    trend = tool_payload.get("export_trend", {}) or {}
    capacity = tool_payload.get("factory_capacity", {}) or {}
    competitor = tool_payload.get("competitor_snapshot", {}) or {}
    macro = tool_payload.get("macro_logistics", {}) or {}

    product = trend.get("product_name") or f"HS {trend.get('hs_code', '?')}"
    destination = trend.get("destination") or "the selected market"
    records = trend.get("records") or []
    start = float(trend.get("start_value") or 0)
    end = float(trend.get("end_value") or 0)
    change = float(trend.get("absolute_change") or 0)
    pct = trend.get("percent_change")
    first_year = int(records[0].get("year")) if records else None
    last_year = int(records[-1].get("year")) if records else None
    latest_prior = float(records[-2].get("export_value") or 0) if len(records) >= 2 else None
    latest_change = end - latest_prior if latest_prior is not None else None
    latest_pct = latest_change / latest_prior * 100 if latest_prior not in (None, 0) else None
    peak_row = max(records, key=lambda row: float(row.get("export_value") or 0), default={})
    peak_year = peak_row.get("year")
    peak_value = float(peak_row.get("export_value") or 0)
    peak_gap = end - peak_value if peak_value else None

    factories = int(capacity.get("matching_factories") or 0)
    top_exporters = competitor.get("top_exporters") or []
    rivals = [str(row.get("exporter")) for row in top_exporters[:3] if row.get("exporter")]
    latest_indicators = macro.get("latest_indicators") or []
    tariffs = [row for row in latest_indicators if row.get("category") == "tariff" and row.get("value") is not None]
    logistics = [row for row in latest_indicators if row.get("category") == "logistics" and row.get("value") is not None]
    applied_tariff = next((row for row in tariffs if row.get("indicator_code") == "TM.TAX.MRCH.WM.AR.ZS"), None)
    overall_lpi = next((row for row in logistics if row.get("indicator_code") == "LP.LPI.OVRL.XQ"), None)

    if end == 0 and start > 0:
        summary = (
            f"Recorded exports of {product} to {destination} fell from an established flow to zero by {last_year}. "
            "The data establish market exit, but not its cause."
        )
    elif change < 0:
        summary = (
            f"Recorded exports of {product} to {destination} declined over {first_year}–{last_year}. "
            "The immediate diagnostic question is whether the loss reflects specific buyers, product positioning, or supply execution rather than a broad absence of Lebanese capacity."
        )
    elif change > 0:
        summary = (
            f"Recorded exports of {product} to {destination} increased over {first_year}–{last_year}. "
            "The relevant question is how durable and scalable the position is, not whether the flow is declining."
        )
    else:
        summary = (
            f"The recorded export flow of {product} to {destination} is flat or absent. "
            "The evidence is insufficient to label a causal barrier without buyer, price, standards, and firm-level data."
        )

    findings: list[str] = []
    findings.append(
        f"Long-run exports changed from {start:,.0f} in {first_year} to {end:,.0f} in {last_year}"
        + (f" ({float(pct):.1f}%)." if pct is not None else ".")
    )
    if latest_change is not None:
        findings.append(
            f"The latest annual movement was {latest_change:,.0f}"
            + (f" ({latest_pct:.1f}%)" if latest_pct is not None else "")
            + f" from {int(records[-2].get('year'))} to {last_year}."
        )
    if peak_year is not None:
        findings.append(
            f"The series peaked in {int(peak_year)} at {peak_value:,.0f}; the latest value is {abs(peak_gap or 0):,.0f} "
            f"{'below' if (peak_gap or 0) < 0 else 'above'} that peak."
        )
    if factories:
        findings.append(
            f"The factory file returns {factories} HS-linked matches. This indicates a possible domestic supply base, but does not verify current output, quality, certification, spare capacity, or export readiness."
        )
    else:
        findings.append(
            "The factory file returns no HS-linked matches. This may indicate limited mapped capacity or a classification mismatch; it is not proof that production is absent."
        )
    if rivals:
        findings.append(f"The competitor snapshot identifies {', '.join(rivals)} among the leading recorded suppliers.")
    else:
        findings.append("The available competitor file contains no usable rows for this product–market pair, so relative price and market-share pressure cannot be established.")
    if applied_tariff:
        findings.append(
            f"The latest embedded weighted applied tariff indicator for all products is {float(applied_tariff['value']):.2f}% ({int(applied_tariff['year'])}). "
            "This economy-wide rate is context, not the product-specific tariff."
        )
    if overall_lpi:
        findings.append(
            f"The destination's embedded overall Logistics Performance Index is {float(overall_lpi['value']):.1f}/5 ({int(overall_lpi['year'])}). "
            "This does not measure the Lebanon-to-destination route or firm-level freight performance."
        )

    hypotheses: list[str] = []
    if latest_change is not None and latest_change < 0:
        hypotheses.append("A recent buyer loss, order normalization, or competitive displacement is plausible because the latest annual movement is negative; buyer-level transactions are needed to distinguish them.")
    if peak_value and end < peak_value:
        hypotheses.append("The gap from the historical peak may reflect failure to retain temporary orders or expand beyond a narrow customer base; this cannot be confirmed from aggregate trade data.")
    if factories > 0:
        hypotheses.append("Supply quantity alone may not be the binding constraint; certification, consistency, packaging, price, and delivery performance should be tested at the firm level.")
    if not hypotheses:
        hypotheses.append("No single causal mechanism is supported strongly enough by the available evidence; the next step is to test buyer demand, landed price, standards, and supplier readiness separately.")

    recommendations = [
        "Decompose the export series by exporter and buyer to determine whether the change is concentrated in a small number of commercial relationships.",
        "Benchmark landed price, product specification, packaging, certification, payment terms, and lead time against the leading suppliers; do not rely on aggregate RCA or tariff indicators for this step.",
        "Validate the HS-linked factories' current production, spare capacity, quality controls, and export certifications before estimating scalable supply.",
        "Check the product-specific tariff, rules of origin, registration, labeling, sanitary or technical requirements, and current freight quotations for the exact route.",
    ]

    evidence = [
        {
            "source": "Export trend",
            "summary": (
                f"{first_year}: {start:,.0f}; {last_year}: {end:,.0f}; absolute change: {change:,.0f}"
                + (f"; percentage change: {float(pct):.1f}%" if pct is not None else "")
            ),
        },
        {
            "source": "Factory mapping",
            "summary": f"HS-linked factory matches: {factories}; mapping does not verify active capacity or export readiness.",
        },
        {
            "source": "Competitor data",
            "summary": "Leading suppliers: " + ", ".join(rivals) if rivals else "No usable product–market competitor rows in the embedded file.",
        },
    ]
    if applied_tariff or overall_lpi:
        context_bits = []
        if applied_tariff:
            context_bits.append(f"weighted applied tariff {float(applied_tariff['value']):.2f}% ({int(applied_tariff['year'])})")
        if overall_lpi:
            context_bits.append(f"overall LPI {float(overall_lpi['value']):.1f}/5 ({int(overall_lpi['year'])})")
        evidence.append({"source": "Market context", "summary": "; ".join(context_bits)})

    missing_data = [
        "Exporter–buyer transactions, buyer concentration, and contract/order histories.",
        "Product-specific competitor shares and prices in the destination.",
        "Firm-level production, utilization, costs, certifications, rejection rates, and delivery performance.",
        "Current product-specific tariffs, non-tariff requirements, and route-level freight and insurance costs.",
    ]

    return {
        "action": "diagnose_export_performance",
        "reasoning": "Observed trade, mapped capacity, competitor availability, and market context are separated from causal hypotheses. Unsupported mechanisms are not presented as findings.",
        "output": {
            "diagnosis_summary": summary,
            "observed_findings": findings,
            "main_causes": hypotheses,
            "evidence": evidence,
            "recommendations": recommendations,
            "missing_data": missing_data,
        },
        "confidence": 0.55,
    }


def diagnose(tool_payload: dict[str, Any], provider: str, model: str) -> dict[str, Any]:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    if not tool_payload.get("valid"):
        return {
            "action": "invalid_input",
            "reasoning": "Input validation failed before running the diagnostic workflow.",
            "output": {"errors": tool_payload.get("validation", {}).get("errors", [])},
            "confidence": 0.99,
        }
    from agents.verdict import compute_confidence

    computed_conf = compute_confidence(tool_payload)
    if not (
        os.environ.get("GROQ_API_KEY", "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    ):
        result = fallback_diagnosis(tool_payload)
        result["confidence"] = computed_conf
        return result
    try:
        raw = call_llm(provider, system_prompt, tool_payload, model)
        parsed = _validate_schema(raw)
        # Guardrail: confidence is COMPUTED (coverage x agreement), never the
        # model's felt number. The model's own value is kept for audit.
        parsed["output"]["model_reported_confidence"] = parsed.get("confidence")
        parsed["confidence"] = computed_conf
        return parsed
    except Exception as exc:
        result = fallback_diagnosis(tool_payload)
        result["output"]["llm_error"] = str(exc)
        result["confidence"] = computed_conf
        return result


def _validate_schema(raw: str) -> dict[str, Any]:
    """Schema-enforcement guardrail: the diagnosis JSON must match the
    course schema {action, reasoning, output, confidence} with correct
    types, or the deterministic fallback takes over."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model output")
    parsed = json.loads(raw[start:end])
    for key in ["action", "reasoning", "output", "confidence"]:
        if key not in parsed:
            raise ValueError(f"Missing key: {key}")
    if not isinstance(parsed["action"], str) or not isinstance(parsed["reasoning"], str):
        raise ValueError("action and reasoning must be strings")
    if not isinstance(parsed["output"], dict):
        raise ValueError("output must be an object")
    try:
        parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
    except (TypeError, ValueError):
        raise ValueError("confidence must be a float in [0, 1]")
    return parsed
