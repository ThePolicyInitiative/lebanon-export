from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("OPENROUTER_API_KEY", None)

from tools.dashboard_sql_agent import query_dashboard_sql  # noqa: E402


def _load_conversation_functions():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "_normalise_user_text",
        "_conversation_reply",
        "_looks_like_dashboard_request",
        "_stable_cache_key",
        "_clarification_reply",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    fake_state = SimpleNamespace(
        agent_context={"hs6": "180690"},
        answer_cache={"x": "y"},
    )
    namespace = {
        "re": re,
        "st": SimpleNamespace(session_state=fake_state),
    }
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace, fake_state


def test_greetings_and_small_talk_are_local_and_clear():
    namespace, _ = _load_conversation_functions()
    reply = namespace["_conversation_reply"]

    assert reply("hello").startswith("Hello.")
    assert "doing well" in reply("how are you?")
    assert "doing well" in reply("Hi, how are you?")
    assert reply("thank you") == "You are welcome."
    assert reply("bye") == "Goodbye."


def test_help_and_reset_are_interactive():
    namespace, state = _load_conversation_functions()
    reply = namespace["_conversation_reply"]

    help_answer = reply("what can you do?")
    assert "What I can answer" in help_answer
    assert "Complete profiles for all 882 dashboard products" in help_answer

    reset_answer = reply("start over")
    assert "Context cleared" in reset_answer
    assert state.agent_context == {}
    assert state.answer_cache == {}


def test_dashboard_request_detection_and_clarification():
    namespace, _ = _load_conversation_functions()
    detect = namespace["_looks_like_dashboard_request"]
    clarify = namespace["_clarification_reply"]

    assert detect("What does Lebanon export to Syria?")
    assert detect("What is diversification?")
    assert not detect("hello")
    assert "product name" in clarify("How much did Lebanon export of this product?")
    assert "country" in clarify("What about this market?")


def test_exact_dashboard_question_is_repeatable():
    question = "What products does Lebanon export to Syria the most in 2025?"
    answers = [
        query_dashboard_sql(question, "groq", "openai/gpt-oss-20b").answer
        for _ in range(5)
    ]
    assert len(set(answers)) == 1
    assert "Top products Lebanon exported to Syria in 2025" in answers[0]


def test_product_profile_is_repeatable():
    question = "Tell me about chocolate"
    answers = [
        query_dashboard_sql(question, "groq", "openai/gpt-oss-20b").answer
        for _ in range(5)
    ]
    assert len(set(answers)) == 1
    assert "Product profile for Chocolate" in answers[0]


def test_cache_key_includes_resolved_context():
    namespace, _ = _load_conversation_functions()
    key = namespace["_stable_cache_key"]

    syria = key("what about it?", "what about it. Conversation context: market Syria")
    uae = key("what about it?", "what about it. Conversation context: market UAE")
    assert syria != uae
