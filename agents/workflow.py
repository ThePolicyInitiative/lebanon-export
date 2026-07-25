from __future__ import annotations

from typing import Any, TypedDict

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - only used when dependency is missing locally
    END = None
    StateGraph = None

from agents.diagnostic_agent import diagnose
from tools.web_tools import get_market_signals
from tools.data_tools import (
    get_competitor_snapshot,
    get_export_trend,
    get_factory_capacity,
    get_macro_logistics,
    validate_input,
)


class DiagnosticState(TypedDict, total=False):
    hs_code: str
    destination: str
    provider: str
    model: str
    valid: bool
    validation: dict[str, Any]
    export_trend: dict[str, Any]
    competitor_snapshot: dict[str, Any]
    factory_capacity: dict[str, Any]
    macro_logistics: dict[str, Any]
    market_signals: dict[str, Any]
    tool_trace: list[dict[str, Any]]
    final: dict[str, Any]


def validate_node(state: DiagnosticState) -> DiagnosticState:
    validation = validate_input(state["hs_code"], state["destination"])
    return {
        **state,
        "valid": validation["valid"],
        "validation": validation,
        "tool_trace": [{"tool": "validate_input", "status": "ok" if validation["valid"] else "failed", "errors": validation["errors"]}],
    }


def should_continue(state: DiagnosticState) -> str:
    return "continue" if state.get("valid") else "finish"


def export_node(state: DiagnosticState) -> DiagnosticState:
    result = get_export_trend(state["validation"]["hs_code"], state["validation"]["destination"])
    return {**state, "export_trend": result, "tool_trace": state["tool_trace"] + [{"tool": "export_trend", "status": "ok"}]}


def competitor_node(state: DiagnosticState) -> DiagnosticState:
    result = get_competitor_snapshot(state["validation"]["hs_code"], state["validation"]["destination"])
    return {**state, "competitor_snapshot": result, "tool_trace": state["tool_trace"] + [{"tool": "competitor_snapshot", "status": "ok"}]}


def capacity_node(state: DiagnosticState) -> DiagnosticState:
    result = get_factory_capacity(state["validation"]["hs_code"])
    return {**state, "factory_capacity": result, "tool_trace": state["tool_trace"] + [{"tool": "factory_capacity", "status": "ok"}]}


def macro_node(state: DiagnosticState) -> DiagnosticState:
    years = [row["year"] for row in state["export_trend"]["records"]]
    result = get_macro_logistics(state["validation"]["destination"], years)
    return {**state, "macro_logistics": result, "tool_trace": state["tool_trace"] + [{"tool": "macro_logistics", "status": "ok", "source": result.get("source", "local snapshot")}]}


def signals_node(state: DiagnosticState) -> DiagnosticState:
    product = state.get("export_trend", {}).get("product_name")
    result = get_market_signals(product, state["validation"]["destination"])
    return {**state, "market_signals": result, "tool_trace": state["tool_trace"] + [{"tool": "market_signals", "status": result["status"]}]}


def should_search(state: DiagnosticState) -> str:
    """Branch: only call the external web search when a key is configured.

    This keeps the LangGraph run identical and fully offline-capable when
    no key is present — the trace shows the branch that was taken.
    """
    import os
    return "search" if os.getenv("TAVILY_API_KEY") else "skip"


def final_node(state: DiagnosticState) -> DiagnosticState:
    if state.get("valid") and "market_signals" not in state:
        # The graph branched around the web-search node (no API key).
        state = {
            **state,
            "market_signals": {"tool": "market_signals", "status": "skipped", "reason": "TAVILY_API_KEY not set", "results": []},
            "tool_trace": state["tool_trace"] + [{"tool": "market_signals", "status": "skipped (branch: no API key)"}],
        }
    payload = {k: v for k, v in state.items() if k not in {"provider", "model", "final"}}
    final = diagnose(payload, provider=state.get("provider", "ollama"), model=state.get("model", "qwen2.5:7b"))
    return {**state, "final": final}


def run_workflow(hs_code: str, destination: str, provider: str, model: str) -> dict[str, Any]:
    initial: DiagnosticState = {
        "hs_code": hs_code,
        "destination": destination,
        "provider": provider,
        "model": model,
    }
    if StateGraph is None:
        from tools.data_tools import run_all_tools

        payload = run_all_tools(hs_code, destination)
        return {**payload, "final": diagnose(payload, provider=provider, model=model)}

    graph = StateGraph(DiagnosticState)
    graph.add_node("validate", validate_node)
    graph.add_node("export", export_node)
    graph.add_node("competitor", competitor_node)
    graph.add_node("capacity", capacity_node)
    graph.add_node("macro", macro_node)
    graph.add_node("signals", signals_node)
    graph.add_node("final", final_node)
    graph.set_entry_point("validate")
    graph.add_conditional_edges("validate", should_continue, {"continue": "export", "finish": "final"})
    graph.add_edge("export", "competitor")
    graph.add_edge("competitor", "capacity")
    graph.add_edge("capacity", "macro")
    graph.add_conditional_edges("macro", should_search, {"search": "signals", "skip": "final"})
    graph.add_edge("signals", "final")
    graph.add_edge("final", END)
    return graph.compile().invoke(initial)
