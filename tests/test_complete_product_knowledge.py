from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("OPENROUTER_API_KEY", None)

from tools.dashboard_sql_agent import (  # noqa: E402
    ProductMatch,
    _build_product_profile_answer,
    _catalog,
    _latest_product_year,
    _product_profile_data,
    execute_sql,
    query_dashboard_sql,
)


def test_all_882_products_have_profile_data():
    year = _latest_product_year()
    _, rows, _ = execute_sql(
        "SELECT printf('%06d',CAST(hs6 AS INTEGER)) AS hs6,name,sector "
        "FROM products_master ORDER BY hs6",
        max_rows=1000,
    )
    assert len(rows) == 882

    failures = []
    for row in rows:
        profile = _product_profile_data(row["hs6"], year)
        if not profile:
            failures.append((row["hs6"], "missing profile"))
            continue
        base = profile.get("base") or {}
        if str(base.get("name")) != str(row["name"]):
            failures.append((row["hs6"], "wrong name"))
        if str(profile.get("sector")) != str(row["sector"]):
            failures.append((row["hs6"], "wrong sector"))
        if len(profile.get("annual") or []) != 8:
            failures.append((row["hs6"], "incomplete annual history"))
        if int(profile.get("actual_year") or 0) != year:
            failures.append((row["hs6"], "wrong latest year"))
        if "destinations" not in profile:
            failures.append((row["hs6"], "missing destinations"))
        if "potential_rows" not in profile:
            failures.append((row["hs6"], "missing potential records"))
        if "market_sizes" not in profile:
            failures.append((row["hs6"], "missing market-size records"))

    assert not failures


def test_every_product_can_render_a_simple_profile():
    year = _latest_product_year()
    catalog = _catalog()
    failures = []
    for product in catalog["products"]:
        match = ProductMatch(
            str(product["hs6"]).zfill(6),
            str(product["name"]),
            str(product["sector"]),
            1.0,
        )
        answer = _build_product_profile_answer(
            match,
            level="simple",
            requested_year=year,
            show_code=False,
        )
        required = (
            str(product["name"]),
            "Current position",
            "Leading destinations",
            "Export trend",
            "What the figures show",
        )
        if not all(item in answer for item in required):
            failures.append((match.hs6, match.name))
    assert not failures


def test_general_and_complete_product_profiles_are_different():
    simple = query_dashboard_sql(
        "Tell me about chocolate",
        "groq",
        "openai/gpt-oss-20b",
    )
    full = query_dashboard_sql(
        "Give me all details about chocolate",
        "groq",
        "openai/gpt-oss-20b",
    )

    assert simple.matched
    assert full.matched
    assert simple.entities.get("profile_level") == "simple"
    assert full.entities.get("profile_level") == "full"

    for expected in (
        "Current position",
        "Leading destinations",
        "Export trend",
        "What the figures show",
    ):
        assert expected in simple.answer

    for forbidden in (
        "Destination concentration (HHI)",
        "RCA in 2025",
        "PCI",
        "CAGR",
        "Recorded unrealized potential",
    ):
        assert forbidden not in simple.answer

    for expected in (
        "Complete annual history",
        "Destination changes",
        "Destination concentration (HHI)",
        "RCA in 2025",
        "PCI",
        "CAGR",
        "Recorded unrealized potential",
        "Largest recorded import markets",
    ):
        assert expected in full.answer


def test_specific_product_questions_do_not_become_profiles():
    cases = (
        ("Where does Lebanon export printed books?", "Destinations for"),
        ("How much did Lebanon export of wooden furniture in 2025?", "Export value"),
        ("What is the RCA of phosphoric acid in 2025?", "RCA"),
        ("What share of Lebanon's chocolate exports went to France in 2025?", "Share of this product's exports"),
        ("Show the trend in virgin olive oil exports from 2018 to 2025.", "comparison across selected years"),
    )
    for question, expected in cases:
        result = query_dashboard_sql(
            question,
            "groq",
            "openai/gpt-oss-20b",
        )
        assert result.matched
        assert result.entities.get("metric") != "product_profile"
        assert expected.lower() in result.answer.lower()


def test_follow_up_can_request_all_details_for_last_product():
    result = query_dashboard_sql(
        "All details related to it. Conversation context: HS6 180690; year 2025",
        "groq",
        "openai/gpt-oss-20b",
    )
    assert result.matched
    assert result.entities.get("metric") == "product_profile"
    assert result.entities.get("profile_level") == "full"
    assert "Chocolate and other food preparations containing cocoa" in result.answer
    assert "Complete annual history" in result.answer


def test_selected_year_controls_profile_position_and_trend_endpoint():
    result = query_dashboard_sql(
        "Tell me about chocolate in 2022",
        "groq",
        "openai/gpt-oss-20b",
    )
    assert result.matched
    assert result.entities.get("metric") == "product_profile"
    assert "Exports in 2022" in result.answer
    assert "**2022:**" in result.answer
    assert "**2025:**" not in result.answer.split("**Export trend**", 1)[1].split("**What the figures show**", 1)[0]

def test_all_882_official_names_work_as_chatbot_prompts():
    _, rows, _ = execute_sql(
        "SELECT printf('%06d',CAST(hs6 AS INTEGER)) AS hs6,name "
        "FROM products_master ORDER BY hs6",
        max_rows=1000,
    )
    failures = []
    for row in rows:
        result = query_dashboard_sql(
            str(row["name"]),
            "groq",
            "openai/gpt-oss-20b",
        )
        if (
            not result.matched
            or result.entities.get("metric") != "product_profile"
            or str(row["name"]) not in result.answer
        ):
            failures.append(
                (
                    row["hs6"],
                    row["name"],
                    result.entities,
                )
            )
    assert not failures


def test_every_product_can_render_a_complete_profile():
    year = _latest_product_year()
    catalog = _catalog()
    failures = []
    required_sections = (
        "Complete annual history",
        "Destination changes",
        "Competitiveness, complexity and growth indicators",
        "Recorded unrealized potential",
        "Largest recorded import markets",
    )
    for product in catalog["products"]:
        match = ProductMatch(
            str(product["hs6"]).zfill(6),
            str(product["name"]),
            str(product["sector"]),
            1.0,
        )
        answer = _build_product_profile_answer(
            match,
            level="full",
            requested_year=year,
            show_code=False,
        )
        if (
            str(product["name"]) not in answer
            or not all(section in answer for section in required_sections)
        ):
            failures.append((match.hs6, match.name))
    assert not failures


def test_product_name_words_are_not_mistaken_for_markets_or_sectors():
    questions = (
        "Tableware and kitchenware; of porcelain or china",
        "Ceramic statuettes and other ornamental ceramic articles, of porcelain or china",
        "Alkali or alkali-earth metals; sodium",
    )
    for question in questions:
        result = query_dashboard_sql(
            question,
            "groq",
            "openai/gpt-oss-20b",
        )
        assert result.matched
        assert result.entities.get("metric") == "product_profile"
        assert result.entities.get("market") is None

