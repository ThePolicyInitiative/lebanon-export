from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("OPENROUTER_API_KEY", None)

from tools.dashboard_data import query_dashboard
from tools.dashboard_sql_agent import query_dashboard_sql


CONCEPT_QUESTIONS = [
    "What is scale and diversification?",
    "What do scale and diversification mean?",
    "Tell me about scale and diversification.",
    "Help me understand diversification.",
    "Is more diversification always better?",
    "Why is diversification important?",
    "What is composition and how is it different from diversification?",
    "What is complexity and competitiveness?",
    "What is the difference between scale and growth?",
    "What are scale and diversification in Syria in 2025?",
]

VALUE_QUESTIONS = [
    "What is the RCA of phosphoric acid in 2025?",
    "What is the PCI of phosphoric acid?",
    "What is the HHI of Syria?",
    "What is the market size of phosphoric acid in France?",
]


def test_concepts_use_explanation_route():
    for question in CONCEPT_QUESTIONS:
        direct = query_dashboard(question)
        complete_db = query_dashboard_sql(question, "groq", "openai/gpt-oss-20b")
        assert direct.matched
        assert complete_db.matched
        assert direct.entities.get("metric") == "concept_explanation"
        assert complete_db.entities.get("metric") == "concept_explanation"


def test_named_entity_metrics_use_value_route():
    for question in VALUE_QUESTIONS:
        result = query_dashboard_sql(question, "groq", "openai/gpt-oss-20b")
        assert result.matched
        assert result.entities.get("metric") != "concept_explanation"


def test_scale_and_diversification_are_distinguished_clearly():
    answer = query_dashboard("What is scale and diversification?").answer
    assert "Scale asks how large exports are" in answer
    assert "Diversification asks how widely they are spread" in answer
    assert "high export scale but low diversification" in answer


def test_evaluative_question_has_direct_answer():
    answer = query_dashboard("Is more diversification always better?").answer
    assert "Direct answer" in answer
    assert "Not always" in answer
