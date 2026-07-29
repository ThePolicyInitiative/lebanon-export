from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("OPENROUTER_API_KEY", None)

from tools.dashboard_sql_agent import (  # noqa: E402
    PRODUCT_ALIASES,
    _catalog,
    _find_products,
    execute_sql,
    query_dashboard_sql,
)


def test_country_product_count_threshold_over_500():
    questions = [
        "to what countries does lebanon export more than 500 products",
        "Which countries receive more than 500 Lebanese products in 2025?",
        "Where does Lebanon export more than 500 products?",
        "What countries import more than 500 products from Lebanon?",
    ]
    for question in questions:
        result = query_dashboard_sql(question, "groq", "openai/gpt-oss-20b")
        assert result.matched
        assert "No destination country received more than 500" in result.answer
        assert "447 products in Syria" in result.answer
        assert "Products ranked by export value" not in result.answer


def test_country_product_count_threshold_returns_matching_countries():
    result = query_dashboard_sql(
        "Which countries receive at least 300 Lebanese products in 2025?",
        "groq",
        "openai/gpt-oss-20b",
    )
    assert result.matched
    for country, count in (
        ("Syria", "447"),
        ("UAE", "366"),
        ("Congo", "361"),
        ("Ivory Coast", "359"),
        ("Iraq", "311"),
    ):
        assert country in result.answer
        assert count in result.answer


def test_all_exact_dashboard_product_names_resolve_to_their_own_record():
    _, rows, _ = execute_sql(
        "SELECT hs6,name FROM products_master ORDER BY hs6",
        max_rows=1000,
    )
    assert len(rows) == 882
    failures = []
    for row in rows:
        expected = str(row["hs6"]).zfill(6)
        question = f"How much did Lebanon export of {row['name']} in 2025?"
        matches = _find_products(question)
        if not matches or matches[0].hs6 != expected:
            failures.append((expected, row["name"], matches))
    assert not failures


def test_common_product_names_resolve_correctly():
    expected = {
        "phosphoric acid": "280920",
        "jewellery": "711319",
        "chocolate": "180690",
        "printed books": "490199",
        "wooden furniture": "940360",
        "concrete mixers": "847431",
        "medicine": "300490",
        "virgin olive oil": "150910",
    }
    for phrase, hs6 in expected.items():
        matches = _find_products(
            f"How much did Lebanon export of {phrase} in 2025?"
        )
        assert matches
        assert matches[0].hs6 == hs6


def test_absent_product_is_not_replaced_with_an_unrelated_product():
    questions = [
        "How much wine did Lebanon export?",
        "Where does Lebanon export perfume?",
        "How much soap did Lebanon export?",
        "How much sparkling wine did Lebanon export?",
    ]
    for question in questions:
        result = query_dashboard_sql(question, "groq", "openai/gpt-oss-20b")
        assert result.matched
        assert "Product not found in the dashboard" in result.answer
        assert "I did not substitute a different product" in result.answer


def test_all_curated_aliases_point_to_existing_dashboard_products():
    catalog = _catalog()
    invalid = [
        (alias, hs6)
        for alias, hs6 in PRODUCT_ALIASES.items()
        if hs6 not in catalog["product_by_hs"]
    ]
    assert not invalid
