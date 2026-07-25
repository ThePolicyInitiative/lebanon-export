from __future__ import annotations

import os
import sys
import re
import hashlib
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
import streamlit as st
from streamlit.components.v1 import declare_component

# 1. MAKE LOCAL PROJECT PACKAGES IMPORTABLE
# Streamlit Cloud can launch the app from a working directory that differs from
# the directory containing app.py. Always put app.py's own directory first.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

# 2. IMPORT CUSTOM MODULES
try:
    from agents import chat_agent
    from agents.llm_clients import chat_completion
    from agents.workflow import run_workflow
    from tools import rag_store
    from tools.data_tools import get_untapped_products, load_export_book, normalize_hs
    from tools.dashboard_data import DashboardAnswer, query_dashboard
    from tools.dashboard_sql_agent import query_dashboard_sql
except ImportError as e:
    missing = []
    for required in ("agents", "tools"):
        if not (APP_DIR / required).is_dir():
            missing.append(required)
    details = (
        f" Missing project folder(s): {', '.join(missing)}."
        if missing else
        " The project folders exist, but Python could not import them."
    )
    st.error(
        "Critical System Import Error: "
        f"{e}.{details} Deploy the complete project with app.py, agents/, "
        "tools/, data/, and dashboard_component/ at the same repository level."
    )
    st.stop()

# 3. PAGE CONFIG
st.set_page_config(
    page_title="Lebanon Industrial Export Dashboard",
    page_icon="🇱🇧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 4. ENVIRONMENT VARIABLES
# Streamlit Community Cloud exposes root-level secrets as environment variables.
# Avoid touching st.secrets so the app starts cleanly when no secrets are configured.
_envfile = Path(__file__).with_name(".env")
if _envfile.exists():
    for _line in _envfile.read_text(encoding="utf-8").splitlines():
        _parts = _line.split("#", 1)
        _clean_line = _parts[0].strip() if _parts else ""
        if "=" in _clean_line:
            _k, _v = _clean_line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

APP_HELP = (
    "Use the complete queryable database generated from every dataset embedded in the dashboard, together with the local diagnostic evidence, to answer the user directly. "
    "Do not propose example questions or suggested prompts."
)


def _dashboard_query(question: str) -> DashboardAnswer:
    """Run the deterministic dashboard query without allowing parser errors to break chat."""
    try:
        return query_dashboard(str(question or ""))
    except Exception as exc:
        print(f"Dashboard data query failed: {exc}")
        return DashboardAnswer(False, 0.0, "", "", {})


@st.cache_data(show_spinner=False)
def available_inputs() -> tuple[list[str], list[str], pd.DataFrame]:
    try:
        years = load_export_book()
        if not years:
            return [], [], pd.DataFrame()
        latest_year = max(years)
        latest = years[latest_year].copy()
        latest["HS Code"] = latest["HS Code"].map(normalize_hs)
        hs_codes = latest["HS Code"].astype(str).sort_values().unique().tolist()
        countries = [c for c in latest.columns if c not in ["HS Code", "Product name"]]
        return hs_codes, countries, latest
    except Exception as e:
        return [], ["UAE", "Saudi Arabia", "Qatar", "Kuwait", "Jordan", "Syria"], pd.DataFrame()

# ------------------------------------------------------------------
# INTENT / ENTITY EXTRACTION HELPERS
# ------------------------------------------------------------------

def looks_like_year(token: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", token)) and 2010 <= int(token) <= 2030

def extract_hs_code(text: str, hs_codes: list[str]) -> str | None:
    if not hs_codes:
        return None
    hs_set = set(hs_codes)
    for token in re.findall(r"\d{4,6}", text):
        if looks_like_year(token):
            continue
        normalized = normalize_hs(token)
        if len(token) == 6 and normalized in hs_set:
            return normalized
        candidates = [code for code in hs_codes if code.startswith(token)]
        if candidates:
            return candidates[0]
    return None

def extract_destination(text: str, countries: list[str]) -> str | None:
    text_lower = text.lower()
    for country in sorted(countries, key=len, reverse=True):
        if country.lower() in text_lower:
            return country
    return None

STOPWORDS = {
    "what", "which", "products", "product", "can", "could", "should", "lebanon",
    "export", "exports", "to", "for", "in", "market", "markets", "diagnose",
    "why", "did", "does", "is", "are", "the", "a", "an", "of", "and", "or",
    "underperform", "underperforming", "drop", "dropped", "problem", "recommend",
    "opportunity", "opportunities", "best", "good", "tell", "me", "about",
    "that", "this", "these", "those", "it", "its", "not", "no", "yes",
    "doesnt", "dont", "isnt", "arent", "wasnt", "werent", "already",
    "currently", "yet", "still", "new", "more", "other", "else", "there",
    "they", "them", "their", "from", "with", "into", "how", "when", "where",
    "who", "will", "would", "has", "have", "had", "was", "were", "been",
    "composition", "complexity", "concentration", "sophistication", "diversification",
    "annual", "history", "trend", "every", "year", "years", "rca", "pci", "hhi",
    "expy", "cagr", "potential", "unrealized", "rank", "ranking", "share", "shares",
    "being", "any", "some", "but", "than", "then",
    "uae", "united", "arab", "emirates", "saudi", "arabia", "qatar", "kuwait",
    "egypt", "jordan", "iraq", "turkey", "france", "germany", "italy", "usa",
    "states", "kingdom", "canada", "australia", "china", "india", "syria"
}

def normalize_words(text: str) -> list[str]:
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", str(text).lower())
    words = []
    for word in text.split():
        if len(word) <= 2 or word in STOPWORDS:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("es") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        words.append(word)
    return words

def product_match_candidates(question: str, latest: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    if latest.empty:
        return pd.DataFrame()
    query_words = normalize_words(question)
    if not query_words:
        return pd.DataFrame()

    products = latest[["HS Code", "Product name"]].dropna().drop_duplicates().copy()
    products["HS Code"] = products["HS Code"].map(normalize_hs)

    aliases = [
        ("virgin olive oil", "150910"), ("olive oil", "150910"), ("olives", "150910"),
        ("wheat cereal", "110311"), ("wheat groats", "110311"), ("cereal groats", "110311"),
        ("whey", "040410"), ("milk powder", "040210"), ("detergent", "340220"),
        ("sparkling wine", "220410"), ("wine", "220410"), ("chocolate", "180690"),
        ("pharma", "300490"), ("pharmaceutical", "300490"), ("medicine", "300490"),
        ("jewelry", "711319"), ("jewellery", "711319"), ("perfume", "330300"),
        ("soap", "340111"), ("furniture", "940360"),
    ]
    lower_question = question.lower()
    alias_rows = []
    for phrase, code in aliases:
        if phrase in lower_question:
            exact = products[products["HS Code"] == code]
            if exact.empty:
                exact = products[products["HS Code"].str.startswith(code[:4])]
            for _, row in exact.head(2).iterrows():
                alias_rows.append({
                    "HS Code": str(row["HS Code"]),
                    "Product name": str(row["Product name"]),
                    "match_score": 99.0,
                    "match_type": f"alias: {phrase}",
                })
    if alias_rows:
        return pd.DataFrame(alias_rows).drop_duplicates("HS Code").head(limit)

    rows = []
    query_text = " ".join(query_words)
    query_set = set(query_words)
    for _, row in products.iterrows():
        name = str(row["Product name"])
        name_words = normalize_words(name)
        if not name_words:
            continue
        name_text = " ".join(name_words)
        name_set = set(name_words)
        overlap = len(query_set & name_set)
        contains_bonus = overlap * 2
        all_terms_bonus = 4 if query_set and query_set.issubset(name_set) else 0
        phrase_bonus = 6 if query_text and query_text in name_text else 0
        ratio = SequenceMatcher(None, query_text, " ".join(name_words)).ratio()
        score = overlap * 3 + contains_bonus + all_terms_bonus + phrase_bonus + ratio
        if score > 0:
            rows.append({
                "HS Code": str(row["HS Code"]),
                "Product name": name,
                "match_score": round(score, 3),
                "match_type": "name similarity",
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("match_score", ascending=False).head(limit)

# ------------------------------------------------------------------
# INTENT CLASSIFICATION HELPERS (for routing)
# ------------------------------------------------------------------

def strong_product_candidate(candidates: pd.DataFrame) -> str | None:
    """Return an HS6 only when name matching is strong enough for diagnostics.

    Alias matches are explicit. Similarity matches require a substantially higher
    score than the old route, preventing ordinary dashboard words from becoming
    arbitrary products.
    """
    if candidates is None or candidates.empty:
        return None
    row = candidates.iloc[0]
    match_type = str(row.get("match_type", ""))
    try:
        score = float(row.get("match_score", 0))
    except (TypeError, ValueError):
        score = 0.0
    if match_type.startswith("alias:") or score >= 12.0:
        return str(row.get("HS Code"))
    return None


def is_clearly_general_question(question: str) -> bool:
    """Detect obvious general questions that should never go to the workflow."""
    q_lower = question.lower()
    general_patterns = [
        "factory", "manufacturer", "company", "producer", "industry", "sector",
        "list of", "what are the", "who produces", "tell me about", "what is",
        "how many", "where is", "when did", "which", "define", "explain",
        "report", "summary", "overview", "introduction", "background"
    ]
    if any(q_lower.startswith(w) for w in ["what", "who", "where", "when", "why", "how", "which"]):
        return True
    if any(pattern in q_lower for pattern in general_patterns):
        return True
    words = q_lower.split()
    if len(words) <= 4 and not any(word in ["export", "exports", "diagnose", "compare"] for word in words):
        return True
    return False

def classify_intent_with_llm(question: str, provider: str, model: str) -> str:
    """Fallback intent classifier using the primary model route."""
    try:
        label = chat_completion(
            provider=provider,
            system_prompt=(
                "Classify the user's question as either 'general' or 'product'. "
                "'product' means the user is asking about a specific product's "
                "export performance to a specific market, or asking to diagnose "
                "or compare exports. All other questions are 'general'. "
                "Reply with one word only."
            ),
            messages=[{"role": "user", "content": question}],
            model=model,
            temperature=0.0,
            max_tokens=20,
        ).strip().lower()
        return "product" if "product" in label else "general"
    except Exception:
        return "general"

# ------------------------------------------------------------------
# RAG WRAPPER
# ------------------------------------------------------------------

def rag_retriever(query: str, k: int = 3) -> list[dict]:
    try:
        return rag_store.search(query, k=k, min_score=0.05)
    except Exception as e:
        print(f"RAG search failed: {e}")
        return []

# ------------------------------------------------------------------
# STREAMLIT USER INTERFACE
# ------------------------------------------------------------------

def resolve_llm_route() -> tuple[str, str]:
    """Select the model route without exposing provider controls in the UI.

    Groq remains preferred whenever its key is configured. If it is absent,
    OpenRouter is used directly. When Groq is present but a request fails, the
    shared LLM client retries silently through OpenRouter when available.
    """
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return "openrouter", os.getenv("OPENROUTER_MODEL", "openrouter/free")
    return "groq", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


PRIMARY_PROVIDER, PRIMARY_MODEL = resolve_llm_route()
DASHBOARD_FILE = APP_DIR / "dashboard_component" / "dashboard.html"
DASHBOARD_COMPONENT_DIR = APP_DIR / "dashboard_component"
render_dashboard_component = declare_component(
    "lebanon_export_dashboard_exact",
    path=str(DASHBOARD_COMPONENT_DIR),
)

# These hashes lock the packaged dashboard and agent data to the verified originals.
EXPECTED_ASSET_SHA256 = {
    "dashboard_component/dashboard.html": "61e5afa6f583fe4550e1e6434acdc7f2c1dc46e0a10ee5b996fc43d0ef43a777",
    "data/dashboard_bundle.json": "ea631ffc2174ca1efc0dba95773d89dbadd429992ac7afbe53b02fbc21ba5ba1",
    "data/dashboard_up_pairs.json": "e4be77d7e512a63355addafe7946608e27ac5efbbe176e89e32e97a4ba77de06",
    "data/dashboard_data.sqlite.gz": "be9e33b92a781364e77974e02c184828fcf68c02df5ced34dc7d19c0f27c4e56",
    "data/TPI_Product_Market_Data.xlsx": "4366fc95d26172a4f6fc66b8f230fb14fb99b1b038e350f9dabce0433cd1b45b",
    "data/all_factories_combined_full_REPLACEMENT.xlsx": "9b1b99d1a37c068c948480e24d434cb2b797be00b783c470dabc5268150b06dc",
    "data/baci_filtered_lebanon_competitors_2018_2024.csv": "f69140fa4dde1e3391d784684c2a53b3c45309bfe1cf15db4da6e2a24bfd343b",
    "data/world_bank_macro_logistics_selected_markets.csv": "174255afb782543086b7f1ed6127d8481866d7c41674e32915be413847fb9add",
    "data/world_bank_extra_trade_backbone_selected_markets.csv": "c2aa3f2d3a5fa5e066130d41522b727d0ed8f7aacd25211f0d31a2ab90a64b49",
}


@st.cache_data(show_spinner=False)
def verify_required_assets() -> list[str]:
    errors: list[str] = []
    for relative_path, expected_hash in EXPECTED_ASSET_SHA256.items():
        path = APP_DIR / relative_path
        if not path.is_file():
            errors.append(f"Missing required file: {relative_path}")
            continue
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            errors.append(f"File differs from the verified original: {relative_path}")
    return errors


# Streamlit chrome is removed so the original dashboard fills the main canvas.
st.markdown(
    """
    <style>
        :root {
            --agent-burgundy: #7a1f1f;
            --agent-green: #1d5f4d;
            --agent-ink: #182235;
            --agent-muted: #667085;
            --agent-line: rgba(24, 34, 53, 0.12);
            --agent-bg: #f7f4ee;
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none !important;
        }

        .stApp {
            background: #f6f1e8;
        }

        .block-container {
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        [data-testid="stSidebar"] {
            min-width: 300px !important;
            max-width: 300px !important;
            width: 300px !important;
            background: #f8f5ef !important;
            border-right: 1px solid var(--agent-line);
            box-shadow: 8px 0 26px rgba(24, 34, 53, 0.08);
        }

        /* The complete agent interface uses one typeface at all times. */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] :where(
            div, span, p, li, ol, ul, label, textarea, input, button, summary,
            code, pre, strong, em, small, h1, h2, h3, h4, h5, h6
        ) {
            font-family: "Times New Roman", Times, serif !important;
        }

        [data-testid="stSidebar"] textarea::placeholder,
        [data-testid="stSidebar"] input::placeholder {
            font-family: "Times New Roman", Times, serif !important;
        }

        /* Preserve Streamlit's icon glyph font while all written text stays Times. */
        [data-testid="stSidebar"] [data-testid="stIconMaterial"] {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            width: 300px !important;
            height: 100vh !important;
            overflow: hidden !important;
            padding: 0 !important;
        }

        [data-testid="stSidebar"] [data-testid="stFragment"],
        [data-testid="stSidebar"] [data-testid="stFragment"] > div,
        [data-testid="stSidebar"] [data-testid="stFragment"] > div > [data-testid="stVerticalBlock"] {
            height: 100vh !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
            padding: 0 !important;
            gap: 0 !important;
        }

        .agent-header {
            flex: 0 0 auto;
            padding: 1.05rem 1rem 0.9rem;
            background: linear-gradient(145deg, #ffffff 0%, #f8f1e8 100%);
            border-bottom: 1px solid var(--agent-line);
            box-shadow: 0 8px 20px rgba(24, 34, 53, 0.06);
            position: relative;
            z-index: 30;
        }

        .agent-title-row {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }

        .agent-logo {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            color: #ffffff;
            font-size: 1.1rem;
            background: linear-gradient(145deg, var(--agent-burgundy), #501313);
            box-shadow: 0 8px 18px rgba(122, 31, 31, 0.22);
        }

        .agent-title {
            color: var(--agent-ink);
            font-size: 1.18rem;
            line-height: 1.2;
            font-weight: 850;
            margin: 0;
        }

        .agent-status {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            color: var(--agent-green);
            font-size: 0.76rem;
            font-weight: 750;
            margin-top: 0.15rem;
        }

        .agent-status-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: var(--agent-green);
            box-shadow: 0 0 0 3px rgba(29, 95, 77, 0.12);
        }

        .st-key-agent_messages {
            height: calc(100vh - 176px) !important;
            min-height: 260px !important;
            box-sizing: border-box !important;
            flex: 1 1 auto !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding: 0.75rem 0.75rem 7.25rem !important;
            scrollbar-width: thin;
            scrollbar-color: rgba(102, 112, 133, 0.38) transparent;
            background:
                radial-gradient(circle at 12% 0%, rgba(29, 95, 77, 0.055), transparent 28%),
                #f8f5ef;
        }

        .st-key-agent_messages::-webkit-scrollbar {
            width: 7px;
        }

        .st-key-agent_messages::-webkit-scrollbar-thumb {
            background: rgba(102, 112, 133, 0.34);
            border-radius: 999px;
        }

        [data-testid="stSidebar"] .stChatMessage {
            margin: 0 0 0.65rem !important;
            padding: 0.82rem 0.88rem !important;
            border-radius: 14px !important;
            border: 1px solid var(--agent-line) !important;
            background: rgba(255, 255, 255, 0.96) !important;
            box-shadow: 0 7px 18px rgba(24, 34, 53, 0.055);
        }

        [data-testid="stSidebar"] .stChatMessage:has([data-testid="chatAvatarIcon-user"]),
        [data-testid="stSidebar"] .stChatMessage:has([data-testid*="AvatarUser"]),
        [data-testid="stSidebar"] .stChatMessage[aria-label*="user" i] {
            background: linear-gradient(145deg, #7a1f1f, #651919) !important;
            border-color: #7a1f1f !important;
            margin-left: 1.35rem !important;
        }

        [data-testid="stSidebar"] .stChatMessage:has([data-testid="chatAvatarIcon-user"]) p,
        [data-testid="stSidebar"] .stChatMessage:has([data-testid="chatAvatarIcon-user"]) li,
        [data-testid="stSidebar"] .stChatMessage:has([data-testid="chatAvatarIcon-user"]) code,
        [data-testid="stSidebar"] .stChatMessage:has([data-testid*="AvatarUser"]) p,
        [data-testid="stSidebar"] .stChatMessage:has([data-testid*="AvatarUser"]) li,
        [data-testid="stSidebar"] .stChatMessage:has([data-testid*="AvatarUser"]) code,
        [data-testid="stSidebar"] .stChatMessage[aria-label*="user" i] p,
        [data-testid="stSidebar"] .stChatMessage[aria-label*="user" i] li,
        [data-testid="stSidebar"] .stChatMessage[aria-label*="user" i] code {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]),
        [data-testid="stSidebar"] .stChatMessage:has([data-testid*="AvatarAssistant"]),
        [data-testid="stSidebar"] .stChatMessage[aria-label*="assistant" i] {
            border-left: 4px solid var(--agent-green) !important;
            margin-right: 0.35rem !important;
        }

        [data-testid="stSidebar"] .stChatMessage p,
        [data-testid="stSidebar"] .stChatMessage li {
            color: var(--agent-ink);
            font-size: 0.88rem !important;
            line-height: 1.58 !important;
        }

        [data-testid="stSidebar"] .stChatMessage p {
            margin-bottom: 0.5rem !important;
        }

        [data-testid="stSidebar"] .stChatMessage ul,
        [data-testid="stSidebar"] .stChatMessage ol {
            padding-left: 1.2rem !important;
            margin: 0.35rem 0 0.55rem !important;
        }

        [data-testid="stSidebar"] .stChatMessage strong {
            color: var(--agent-ink);
            font-weight: 800;
        }

        [data-testid="stSidebar"] .stChatMessage code {
            font-size: 0.82rem !important;
            color: #5a1717;
            background: rgba(122, 31, 31, 0.075);
            border-radius: 5px;
            padding: 0.08rem 0.24rem;
        }

        [data-testid="stSidebar"] [data-testid="stChatInput"] {
            position: fixed !important;
            left: 0 !important;
            bottom: 0 !important;
            width: 300px !important;
            max-width: 300px !important;
            z-index: 2147482500 !important;
            flex: 0 0 auto !important;
            background: #ffffff !important;
            border-top: 3px solid var(--agent-burgundy) !important;
            padding: 0.55rem 0.78rem 0.82rem !important;
            box-shadow: 0 -14px 32px rgba(24, 34, 53, 0.16);
        }

        [data-testid="stSidebar"] [data-testid="stChatInput"]::before {
            content: "Ask me about Lebanon exports";
            display: block;
            color: var(--agent-burgundy);
            font-size: 0.76rem;
            line-height: 1;
            font-weight: 850;
            letter-spacing: 0.09em;
            margin: 0 0 0.45rem 0.15rem;
        }

        [data-testid="stSidebar"] [data-testid="stChatInput"] textarea {
            min-height: 46px !important;
            max-height: 120px !important;
            color: var(--agent-ink) !important;
            background: #f8f5ef !important;
            border: 1px solid rgba(122, 31, 31, 0.34) !important;
            border-radius: 14px !important;
            font-size: 0.88rem !important;
            line-height: 1.45 !important;
            box-shadow: inset 0 1px 2px rgba(24, 34, 53, 0.04) !important;
        }

        [data-testid="stSidebar"] [data-testid="stChatInput"] textarea:focus {
            border-color: var(--agent-burgundy) !important;
            box-shadow: 0 0 0 3px rgba(122, 31, 31, 0.12) !important;
        }

        [data-testid="stSidebar"] [data-testid="stChatInput"] button {
            background: var(--agent-burgundy) !important;
            color: #ffffff !important;
            border-radius: 10px !important;
        }

        [data-testid="stSidebar"] details {
            border: 1px solid var(--agent-line);
            border-radius: 10px;
            background: #fbfaf7;
            margin-top: 0.55rem;
        }

        iframe {
            display: block !important;
            width: 100% !important;
            border: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            background: #f6f1e8 !important;
        }

        [data-testid="stMainBlockContainer"] > div,
        [data-testid="stVerticalBlock"],
        [data-testid="stElementContainer"] {
            width: 100% !important;
            max-width: 100% !important;
            gap: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        @media (max-width: 900px) {
            [data-testid="stSidebar"],
            [data-testid="stSidebar"] > div:first-child {
                min-width: 300px !important;
                max-width: 300px !important;
                width: 300px !important;
            }
            [data-testid="stSidebar"] [data-testid="stChatInput"] {
                width: 300px !important;
                max-width: 300px !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


_asset_errors = verify_required_assets()
if _asset_errors:
    st.error("Critical asset integrity error:\n\n" + "\n".join(f"- {item}" for item in _asset_errors))
    st.stop()

hs_codes, countries, latest = available_inputs()
if not countries:
    countries = ["UAE", "Saudi Arabia", "Qatar", "Kuwait", "Jordan", "Syria"]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_context" not in st.session_state:
    st.session_state.agent_context = {}


def _has_model_key() -> bool:
    return bool(
        os.environ.get("GROQ_API_KEY", "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    )


def _is_diagnostic_request(prompt: str) -> bool:
    q = prompt.lower()
    diagnostic_terms = (
        "diagnose", "diagnosis", "why is", "why are", "underperform",
        "weak in", "barrier", "problem", "recommend", "competitor",
        "capacity", "factory", "logistics", "market access",
    )
    return any(term in q for term in diagnostic_terms)


def _is_follow_up(prompt: str) -> bool:
    """Return True when a concise question is likely to depend on prior context."""
    q = re.sub(r"\s+", " ", str(prompt).strip().lower())
    follow_up_starts = (
        "what about", "how about", "and ", "also ", "then ", "in that case",
        "for the same", "what of", "compared with", "compare that", "same for",
    )
    follow_up_terms = {"it", "that", "this", "there", "same", "those", "them", "its"}
    metric_terms = {
        "rca", "pci", "hhi", "expy", "cagr", "complexity", "potential", "growth",
        "share", "rank", "markets", "destinations", "products", "trend", "drivers",
    }
    words = set(re.findall(r"[a-z]+", q))
    year_only = bool(re.fullmatch(r"(?:in\s+|for\s+)?20\d{2}\??", q))
    metric_only = len(q.split()) <= 4 and bool(words & metric_terms)
    return (
        q.startswith(follow_up_starts)
        or year_only
        or metric_only
        or (len(q.split()) <= 8 and bool(words & follow_up_terms))
    )


def _update_agent_context(entities: dict[str, Any] | None) -> None:
    """Persist only stable dashboard entities for concise follow-up questions."""
    if not entities:
        return
    context = dict(st.session_state.get("agent_context", {}))
    for key in ("hs6", "market", "sector", "year", "years", "metric", "products", "markets", "sectors"):
        value = entities.get(key)
        if value is not None and value != "":
            context[key] = value
    # Keep the most recently named member of comparison lists available to pronoun follow-ups.
    if entities.get("products"):
        context["hs6"] = list(entities["products"])[-1]
    if entities.get("markets"):
        context["market"] = list(entities["markets"])[-1]
    if entities.get("sectors"):
        context["sector"] = list(entities["sectors"])[-1]
    st.session_state.agent_context = context


def _contextual_prompt(prompt: str) -> str:
    """Resolve concise follow-ups while allowing newly named entities to override memory."""
    clean = str(prompt or "").strip()
    if not clean:
        return clean
    context = dict(st.session_state.get("agent_context", {}))
    if not context or not _is_follow_up(clean):
        return clean

    explicit_year = bool(re.search(r"\b20\d{2}\b", clean))
    explicit_hs = extract_hs_code(clean, hs_codes) is not None
    explicit_market = extract_destination(clean, countries) is not None
    direct_entities = _dashboard_query(clean).entities
    explicit_sector = bool(direct_entities.get("sector") or direct_entities.get("sectors"))
    explicit_product = bool(direct_entities.get("hs6") or direct_entities.get("products"))

    details: list[str] = []
    if context.get("hs6") and not (explicit_hs or explicit_product):
        details.append(f"HS6 {context['hs6']}")
    if context.get("market") and not explicit_market:
        details.append(f"market {context['market']}")
    if context.get("sector") and not explicit_sector and not (explicit_hs or explicit_product):
        details.append(f"sector {context['sector']}")
    if not explicit_year:
        if context.get("year"):
            details.append(f"year {context['year']}")
        elif context.get("years"):
            details.append("years " + ", ".join(str(y) for y in context["years"]))
    if not details:
        return clean
    return f"{clean}. Conversation context: {'; '.join(details)}"


def _normalise_markdown(text: str) -> str:
    """Keep every assistant response visually consistent in the sidebar."""
    value = str(text or "").strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE).strip()
    value = re.sub(r"^```(?:markdown|md|text)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    lines: list[str] = []
    section_labels = {
        "most likely causes", "what to do next", "recommendations", "evidence",
        "missing data", "main causes", "observed findings", "diagnosis", "summary", "interpretation",
        "highest-pci active products", "highest export-weighted sector complexity",
    }
    for raw_line in value.replace("\t", " ").splitlines():
        line = raw_line.rstrip()
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading:
            line = f"**{heading.group(1).strip()}**"
        elif line.strip().rstrip(":").lower() in section_labels:
            line = f"**{line.strip().rstrip(':')}**"
        elif re.match(r"^[•▪◦]\s+", line):
            line = re.sub(r"^[•▪◦]\s+", "- ", line)
        lines.append(line)
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value or "I could not produce an answer from the available data."


def _format_diagnostic_dict(data: dict[str, Any]) -> str:
    """Render structured diagnostic objects using one stable answer template."""
    if not data:
        return "No diagnostic result was returned."
    if data.get("errors"):
        errors = data.get("errors") or []
        return "**Input issue**\n\n" + "\n".join(f"- {item}" for item in errors)

    lines: list[str] = []
    summary = data.get("diagnosis_summary") or data.get("summary") or data.get("reasoning")
    if summary:
        lines.extend(["**Diagnosis**", "", str(summary).strip()])
    if data.get("confidence") is not None:
        try:
            confidence = float(data.get("confidence"))
            lines.extend(["", f"**Evidence confidence:** {confidence * 100:.0f}%"] )
        except (TypeError, ValueError):
            pass

    sections = (
        ("Observed findings", data.get("observed_findings")),
        ("Main causes", data.get("main_causes")),
        ("Evidence", data.get("evidence")),
        ("Recommendations", data.get("recommendations")),
        ("Missing data", data.get("missing_data")),
    )
    for label, items in sections:
        if not items:
            continue
        lines.extend(["", f"**{label}**", ""])
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if isinstance(item, dict):
                source = item.get("source") or item.get("label")
                detail = item.get("summary") or item.get("text") or item.get("value")
                if source and detail is not None:
                    lines.append(f"- **{source}:** {detail}")
                else:
                    compact = "; ".join(f"{k}: {v}" for k, v in item.items())
                    lines.append(f"- {compact}")
            else:
                lines.append(f"- {item}")

    snapshot = data.get("dashboard_snapshot")
    if snapshot:
        lines.extend(["", "**Dashboard snapshot**", "", str(snapshot).strip()])

    if not lines:
        lines.append("**Result**")
        lines.append("")
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                lines.append(f"**{str(key).replace('_', ' ').title()}**")
                lines.extend(f"- {item}" for item in value)
            elif isinstance(value, dict):
                lines.append(f"- **{str(key).replace('_', ' ').title()}:** " + "; ".join(f"{k}: {v}" for k, v in value.items()))
            else:
                lines.append(f"- **{str(key).replace('_', ' ').title()}:** {value}")
    return _normalise_markdown("\n".join(lines))


def format_agent_response(response: Any) -> str:
    """Convert every response type to a stable Markdown string before display/storage."""
    if isinstance(response, dict):
        return _format_diagnostic_dict(response)
    if isinstance(response, (list, tuple)):
        return _normalise_markdown("\n".join(f"- {item}" for item in response))
    return _normalise_markdown(str(response))


def _explain_dashboard_with_model(prompt: str, dashboard_answer: str, context: str) -> str:
    """Keep exact dashboard facts intact and add a tightly controlled interpretation."""
    if not _has_model_key():
        return dashboard_answer
    try:
        interpretation = chat_completion(
            provider=PRIMARY_PROVIDER,
            system_prompt=(
                "You are a senior trade-policy analyst interpreting the Lebanon Industrial Export Dashboard. "
                "The exact dashboard result is already displayed and must not be rewritten. Add one analytical paragraph of at most 110 words. "
                "Explain the most decision-relevant implication, compare scale and direction where the context permits, and distinguish descriptive evidence "
                "from causal inference. Do not introduce any number, year, HS code, market, product, or sector absent from the exact result or JSON context. "
                "Do not give generic advice, repeat the list, use headings, tables, emojis, code blocks, or suggested questions.\n\n"
                f"EXACT RESULT:\n{dashboard_answer}\n\nDASHBOARD JSON CONTEXT:\n{context}"
            ),
            messages=[{"role": "user", "content": prompt}],
            model=PRIMARY_MODEL,
            temperature=0.15,
            max_tokens=240,
        ).strip()
        if not interpretation:
            return dashboard_answer

        # Reject model prose that introduces unsupported numerical claims.
        allowed_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", f"{prompt}\n{dashboard_answer}\n{context}"))
        proposed_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", interpretation))
        if proposed_numbers - allowed_numbers:
            return dashboard_answer
        words = interpretation.split()
        if len(words) > 110:
            interpretation = " ".join(words[:110]).rstrip(" ,;:") + "."
        return f"{dashboard_answer}\n\n**Analytical reading**\n\n{interpretation}"
    except Exception:
        return dashboard_answer


def _aggregate_diagnostic_answer(
    prompt: str,
    resolved_prompt: str,
    dashboard_result: Any,
    detected_hs: str | None,
    destination: str | None,
    candidates: pd.DataFrame,
) -> tuple[str, list[dict]] | None:
    """Provide an evidence-based aggregate diagnosis when only one entity is known.

    This prevents the application from silently choosing an arbitrary destination.
    """
    if not _is_diagnostic_request(prompt):
        return None

    resolved_hs = detected_hs or strong_product_candidate(candidates)

    # Product known, destination absent: diagnose the national product trend.
    if resolved_hs and not destination:
        profile = _dashboard_query(f"profile of HS6 {resolved_hs} in 2025")
        drivers = _dashboard_query(f"what drove exports of HS6 {resolved_hs} from 2018 to 2025")
        parts = [
            "**Aggregate product diagnosis**",
            "",
            "The dashboard can identify the product's national export pattern, but a destination-specific diagnosis requires a named market.",
            "",
            profile.answer if profile.matched else dashboard_result.answer,
        ]
        if drivers.matched:
            parts.extend(["", drivers.answer])
        parts.extend([
            "",
            "**What this establishes**",
            "",
            "- The figures show export scale, direction, market reach, competitiveness, complexity, and where the recorded change occurred.",
            "- They do not by themselves prove whether the underlying cause was demand, logistics, pricing, regulation, finance, or productive capacity.",
        ])
        _update_agent_context({"hs6": resolved_hs, "metric": "aggregate_diagnostic"})
        return format_agent_response("\n".join(parts)), [{
            "source": "dashboard_component/dashboard.html",
            "text": "Aggregate product diagnosis calculated from the exact embedded dashboard data.",
        }]

    # Market known, product absent: diagnose the market portfolio rather than invent a product.
    if destination and not resolved_hs:
        profile = _dashboard_query(f"market profile for {destination} in 2025")
        drivers = _dashboard_query(f"what drove exports to {destination} from 2018 to 2025")
        parts = [
            "**Aggregate market diagnosis**",
            "",
            profile.answer if profile.matched else dashboard_result.answer,
        ]
        if drivers.matched:
            parts.extend(["", drivers.answer])
        parts.extend([
            "",
            "**Interpretation limit**",
            "",
            "- This diagnoses the market portfolio as a whole. A product-specific diagnosis requires a named product or HS code.",
        ])
        _update_agent_context({"market": destination, "metric": "aggregate_diagnostic"})
        return format_agent_response("\n".join(parts)), [{
            "source": "dashboard_component/dashboard.html",
            "text": "Aggregate market diagnosis calculated from the exact embedded dashboard data.",
        }]
    return None


def answer_agent_prompt(prompt: str) -> tuple[str, list[dict]]:
    """Answer with exact dashboard evidence first and specialist tools second."""
    provider = PRIMARY_PROVIDER
    model = PRIMARY_MODEL

    resolved_prompt = _contextual_prompt(prompt)
    detected_hs = extract_hs_code(resolved_prompt, hs_codes)
    candidates = product_match_candidates(resolved_prompt, latest) if not latest.empty else pd.DataFrame()
    destination = extract_destination(resolved_prompt, countries)
    matched_candidate_hs = strong_product_candidate(candidates)
    named_diagnostic = bool(
        _is_diagnostic_request(prompt)
        and destination
        and (detected_hs or matched_candidate_hs)
    )

    # Use the complete database before the older entity parser. This guarantees
    # that broad rankings, comparisons, coverage questions, and exact export
    # lookups are answered from all embedded dashboard datasets.
    if not named_diagnostic:
        universal_result = query_dashboard_sql(resolved_prompt, provider, model)
        if universal_result.matched and universal_result.confidence >= 0.90:
            _update_agent_context(universal_result.entities)
            return format_agent_response(universal_result.answer), [{
                "source": "data/dashboard_data.sqlite",
                "text": "Exact query result from the complete database generated from every embedded dashboard dataset.",
            }]

    dashboard_result = _dashboard_query(resolved_prompt)
    aggregate_diagnostic = _aggregate_diagnostic_answer(
        prompt, resolved_prompt, dashboard_result, detected_hs, destination, candidates
    )
    if aggregate_diagnostic is not None:
        return aggregate_diagnostic

    # Exact dashboard questions are answered before any language-model route.
    if dashboard_result.matched and not (
        _is_diagnostic_request(prompt) and (detected_hs or strong_product_candidate(candidates)) and destination
    ):
        _update_agent_context(dashboard_result.entities)
        exact_answer = dashboard_result.answer
        if dashboard_result.confidence < 0.90 and dashboard_result.entities.get("hs6"):
            exact_answer = (
                "**Match note**\n\n"
                "I used the closest product match in the dashboard. Check the HS6 code shown before relying on the result.\n\n"
                + exact_answer
            )
        # Exact dashboard figures are returned directly. The language model is
        # never allowed to rewrite, reinterpret, or replace the data answer.
        return format_agent_response(exact_answer), [{
            "source": "dashboard_component/dashboard.html",
            "text": "Exact embedded dashboard data and calculations used for this answer.",
        }]

    is_product_specific = detected_hs is not None or matched_candidate_hs is not None

    # If the locally resolved complete-database route produced only a low-confidence
    # search result, give the direct parser one chance before general RAG.
    if not named_diagnostic:
        universal_result = query_dashboard_sql(resolved_prompt, provider, model)
        if universal_result.matched:
            _update_agent_context(universal_result.entities)
            return format_agent_response(universal_result.answer), [{
                "source": "data/dashboard_data.sqlite",
                "text": "Query result from the complete database generated from every embedded dashboard dataset.",
            }]

    clearly_general = is_clearly_general_question(prompt)
    if clearly_general and not _is_diagnostic_request(prompt):
        is_product_specific = False
    elif not is_product_specific and _has_model_key():
        is_product_specific = classify_intent_with_llm(prompt, provider, model) == "product"

    if not is_product_specific:
        response_text, rag_chunks = chat_agent.answer_general(
            question=resolved_prompt,
            history=st.session_state.messages[:-1],
            provider=provider,
            model=model,
            app_help=APP_HELP,
            retriever=rag_retriever,
            top_k=5,
            use_web_search=False,
        )
        return format_agent_response(response_text), rag_chunks

    resolved_hs = detected_hs or matched_candidate_hs

    # Destination-specific workflow only runs when the destination is explicit.
    if not destination:
        product_profile = _dashboard_query(f"HS6 {resolved_hs} exports, markets, RCA, PCI, rank and share in 2025")
        if product_profile.matched:
            _update_agent_context(product_profile.entities)
            return format_agent_response(product_profile.answer), [{
                "source": "dashboard_component/dashboard.html",
                "text": "Exact product profile from the embedded dashboard data.",
            }]
        return (
            "**Answer**\n\nI identified the product, but the evidence does not support a destination-specific diagnosis without a named market. "
            "The product profile can describe scale, trend, competitiveness, complexity, and market reach, but it cannot identify market-specific barriers.",
            [],
        )

    response = run_workflow(
        resolved_hs or prompt,
        destination=destination,
        provider=provider,
        model=model,
    )
    if isinstance(response, dict) and "final" in response:
        final_result = response["final"] or {}
        output_data = dict(final_result.get("output") or {})
        if final_result.get("reasoning"):
            output_data.setdefault("reasoning", final_result.get("reasoning"))
        if final_result.get("confidence") is not None:
            output_data["confidence"] = final_result.get("confidence")
    else:
        output_data = response

    if dashboard_result.matched and isinstance(output_data, dict):
        output_data = dict(output_data)
        output_data["dashboard_snapshot"] = dashboard_result.answer
    _update_agent_context({"hs6": resolved_hs, "market": destination, "metric": "diagnostic"})
    return format_agent_response(output_data), []


@st.fragment
def render_agent_sidebar() -> None:
    """Render a fixed bottom composer with the conversation above it."""
    st.markdown(
        """
        <div class="agent-header">
            <div class="agent-title-row">
                <div class="agent-logo">↗</div>
                <div style="flex:1; min-width:0;">
                    <div class="agent-title">Export Agent</div>
                    <div class="agent-status"><span class="agent-status-dot"></span>All dashboard data connected</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="agent_messages"):
        for msg in st.session_state.messages:
            role = msg.get("role", "assistant")
            avatar = "👤" if role == "user" else "↗"
            with st.chat_message(role, avatar=avatar):
                content = str(msg.get("content", ""))
                st.markdown(content)
                if msg.get("sources"):
                    with st.expander("Sources", expanded=False):
                        for i, chunk in enumerate(msg["sources"], 1):
                            source = chunk.get("source", "unknown")
                            preview = str(chunk.get("text", ""))[:300]
                            if len(str(chunk.get("text", ""))) > 300:
                                preview += "…"
                            st.markdown(f"**{i}. {source}**\n\n{preview}")

    prompt = st.chat_input(
        "Write your question…",
        key="sidebar_agent_chat",
    )
    if not prompt:
        return

    clean_prompt = str(prompt).strip()
    if not clean_prompt:
        return
    st.session_state.messages.append({"role": "user", "content": clean_prompt})

    try:
        with st.spinner("Analyzing dashboard data…"):
            response_text, rag_chunks = answer_agent_prompt(clean_prompt)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": format_agent_response(response_text),
                "sources": rag_chunks,
            }
        )
    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": _normalise_markdown(
                    "**Answer unavailable**\n\nI could not complete that request reliably. The dashboard remains available, and no unsupported answer was generated."
                ),
            }
        )
    st.rerun(scope="fragment")


with st.sidebar:
    render_agent_sidebar()

try:
    try:
        dashboard_height = max(680, int(os.getenv("DASHBOARD_HEIGHT", "920")))
    except ValueError:
        dashboard_height = 920

    # The wrapper performs Streamlit's component-ready handshake and then loads
    # the original, byte-for-byte dashboard HTML in a nested same-origin frame.
    render_dashboard_component(
        requested_height=dashboard_height,
        key="lebanon-industrial-export-dashboard",
        default=None,
    )
except Exception as exc:
    st.error(f"Unable to load the industrial export dashboard: {exc}")
