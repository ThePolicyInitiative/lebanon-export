from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The deterministic dashboard engine must work without any LLM credential.
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("OPENROUTER_API_KEY", None)

from tools.dashboard_sql_agent import query_dashboard_sql  # noqa: E402


CASES = [
    {
        "question": "What products does Lebanon export to Syria the most in 2025?",
        "contains": ["Top products Lebanon exported to Syria in 2025", "potatoes", "$11.89 million"],
        "excludes": ["Rank of Syria"],
    },
    {
        "question": "How much did Lebanon export to the UAE in 2025?",
        "contains": ["Lebanon's exports to UAE in 2025", "$125.38 million", "Products: 366"],
    },
    {
        "question": "Show the trend in virgin olive oil exports to Saudi Arabia from 2018 to 2025.",
        "contains": ["2018-2025", "2018", "2025", "Export value: $0", "-100.0%"],
        "excludes": ["RCA:", "PCI:", "Unrealized potential"],
    },
    {
        "question": "What are the top markets for jewellery in 2025?",
        "contains": ["Destinations for Jewellery", "UAE", "$29.24 million", "65.4%"],
        "excludes": ["RCA:", "PCI:", "Share of exports to this market"],
    },
    {
        "question": "What share of Lebanon's chocolate exports went to France in 2025?",
        "contains": ["France", "$92.3 thousand", "Share of this product's exports: 0.3%"],
        "excludes": ["Share of exports to this market", "RCA:", "PCI:"],
    },
    {
        "question": "Compare total exports to Saudi Arabia and the UAE in 2025.",
        "contains": ["Market comparison in 2025", "Saudi Arabia", "UAE", "Comparison:", "$125.15 million"],
        "excludes": ["EXPY:", "HHI:", "RCA:", "Unrealized potential", "Performance status", "Priority"],
    },
    {
        "question": "What are the top Agrifood products exported to Qatar in 2025?",
        "contains": ["Products exported by Agrifood to Qatar in 2025", "Nuts and other seeds", "$2.61 million"],
        "excludes": ["RCA:", "PCI:", "Unrealized potential"],
    },
    {
        "question": "What are the most complex products in the Electrical and Machinery sector?",
        "contains": ["Most complex products in Electrical and Machinery, 2025", "PCI: 1.862"],
        "excludes": ["Unrealized potential", "CAGR:", "Trajectory:"],
    },
    {
        "question": "What is the RCA of phosphoric acid in 2025?",
        "contains": ["Phosphoric acid and polyphosphoric acids", "RCA: 564.979", "$48.53 million"],
        "excludes": ["Unrealized potential", "CAGR:", "Growth:", "Trajectory:"],
    },
    {
        "question": "How large was the United States import market for virgin olive oil in 2024?",
        "contains": ["data for 2024 are unavailable", "latest available observation is 2018", "$1.19 billion"],
    },
    {
        "question": "What was Lebanon's market penetration for virgin olive oil in the United States in 2024?",
        "contains": ["calculation below uses the latest available year, 2018", "$4.97 million", "$1.19 billion", "Market penetration: 0.4%"],
    },
    {
        "question": "Which products have the highest unrealized export potential in France?",
        "contains": ["Products with unrealized potential in France", "Jewellery", "$9.19 million"],
    },
    {
        "question": "Which countries have the highest unrealized potential for wooden furniture?",
        "contains": ["Unrealized potential for Furniture", "United States", "$7.69 million"],
    },
    {
        "question": "What are the top sectors Lebanon exports to Iraq in 2025?",
        "contains": ["Sectors exported to Iraq in 2025", "Agrifood", "$21.91 million", "Electrical and Machinery"],
    },
    {
        "question": "Which products newly entered the Egyptian market in 2025?",
        "contains": ["Products newly exported to Egypt in 2025", "silver, unwrought", "$1.80 million"],
        "excludes": ["Products entering Lebanon's exports"],
    },
    {
        "question": "Which products did Lebanon export to France but not Germany in 2025?",
        "contains": ["Products exported to France but not Germany in 2025", "Dresses", "$736.2 thousand"],
    },
    {
        "question": "Which markets are most similar to the UAE?",
        "contains": ["Markets similar to UAE", "United States", "Similarity score: 0.657"],
        "excludes": ["Rank of UAE"],
    },
    {
        "question": "What products drove Lebanon's export growth from 2018 to 2025?",
        "contains": ["Product drivers of export change, 2018-2025", "superphosphates", "Change: $33.73 million"],
        "excludes": ["Products ranked by cagr"],
    },
    {
        "question": "Which export markets are the most diversified in 2025?",
        "contains": ["Most diversified export markets in 2025", "Gabon", "HHI: 0.025"],
        "excludes": ["EXPY:", "RCA:", "Unrealized potential", "Performance status", "Priority"],
    },
    {
        "question": "Which products should a Lebanese exporter prioritize for Saudi Arabia based on current exports and unrealized potential?",
        "contains": ["Product opportunity screen for Saudi Arabia in 2025", "current exports plus recorded unrealized potential", "Jewellery", "Actual exports plus unrealized potential"],
    },
]


def run() -> list[tuple[int, str, str]]:
    failures: list[tuple[int, str, str]] = []
    for index, case in enumerate(CASES, 1):
        result = query_dashboard_sql(case["question"], "groq", "openai/gpt-oss-20b")
        answer = result.answer
        plain = answer.replace("**", "")
        if not result.matched:
            failures.append((index, case["question"], "No dashboard answer was matched."))
            continue
        for expected in case.get("contains", []):
            if expected.lower() not in plain.lower():
                failures.append((index, case["question"], f"Missing expected text: {expected!r}"))
        for forbidden in case.get("excludes", []):
            if forbidden.lower() in plain.lower():
                failures.append((index, case["question"], f"Unexpected text present: {forbidden!r}"))
    return failures


if __name__ == "__main__":
    failures = run()
    if failures:
        for number, question, issue in failures:
            print(f"FAIL {number}: {question}\n  {issue}")
        raise SystemExit(1)
    print(f"PASS: {len(CASES)} exporter questions answered correctly.")
