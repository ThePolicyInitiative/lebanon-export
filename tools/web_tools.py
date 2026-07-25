"""External web tools.

market_signals: real-world news/market search via Tavily (free tier,
built for LLM agents). Two hard rules, both motivated by the 2025
Deloitte fabricated-citations incident:

1. If the API key is missing or the call fails, the tool reports
   "skipped" or "error" — the agent never invents search results.
2. If the search returns nothing, the tool reports "no_results" and the
   downstream prompt is instructed to say so, not to fill the gap.

Every result carries its URL so any claim in the final answer can be
traced to a source.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def get_market_signals(product_name: str | None, destination: str) -> dict[str, Any]:
    query = f"{(product_name or 'exports').split(';')[0].strip()} market {destination} imports trade news"
    base = {"tool": "market_signals", "query": query, "results": []}
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {**base, "status": "skipped", "reason": "TAVILY_API_KEY not set — running in offline mode"}
    try:
        body = json.dumps({
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
        }).encode("utf-8")
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": str(r.get("content", ""))[:300]}
            for r in data.get("results", [])
        ]
        if not results:
            return {**base, "status": "no_results", "reason": "search returned nothing — reporting that honestly"}
        return {**base, "status": "ok", "results": results}
    except Exception as exc:  # never fabricate around a failed search
        return {**base, "status": "error", "reason": str(exc)}


WB_LIVE_INDICATORS = ["NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "NE.IMP.GNFS.CD"]
_wb_cache: dict[str, list[dict[str, Any]]] = {}


def fetch_worldbank_live(iso3: str) -> list[dict[str, Any]]:
    """Live World Bank API fetch (second external tool call).

    Fetches three headline indicators for the destination. Any failure
    returns [] so the caller falls back to the local snapshot — the agent
    reports which source it used rather than hiding the difference.
    Cached per session per country.
    """
    if iso3 in _wb_cache:
        return _wb_cache[iso3]
    rows: list[dict[str, Any]] = []
    for code in WB_LIVE_INDICATORS:
        url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/{code}?format=json&per_page=3&mrnev=1"
        try:
            with urllib.request.urlopen(url, timeout=4) as response:
                data = json.loads(response.read().decode("utf-8"))
            for entry in (data[1] or []) if isinstance(data, list) and len(data) > 1 else []:
                if entry.get("value") is not None:
                    rows.append({
                        "indicator_name": entry["indicator"]["value"],
                        "indicator_code": code,
                        "category": "live",
                        "year": int(entry["date"]),
                        "value": entry["value"],
                    })
        except Exception:
            _wb_cache[iso3] = []
            return []  # fallback to local snapshot; never partial-and-silent
    _wb_cache[iso3] = rows
    return rows
