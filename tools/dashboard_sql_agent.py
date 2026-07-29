from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import sqlite3
import statistics
import tempfile
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from agents.llm_clients import chat_completion
from tools.dashboard_data import DashboardAnswer, query_dashboard

ROOT = Path(__file__).resolve().parents[1]
DB_GZIP_PATH = ROOT / "data" / "dashboard_data.sqlite.gz"
DB_PLAIN_PATH = ROOT / "data" / "dashboard_data.sqlite"


# ---------------------------------------------------------------------------
# Database and schema
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _database_path() -> Path:
    if DB_PLAIN_PATH.is_file():
        return DB_PLAIN_PATH
    if not DB_GZIP_PATH.is_file():
        raise FileNotFoundError("The complete dashboard database is missing.")
    target = Path(tempfile.gettempdir()) / "lebanon_export_dashboard_data.sqlite"
    if not target.is_file() or target.stat().st_size == 0:
        temporary = target.with_suffix(".tmp")
        with gzip.open(DB_GZIP_PATH, "rb") as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        temporary.replace(target)
    return target


SCHEMA_CATALOG = """
Read-only SQLite data generated from every numerical/business dataset embedded in the dashboard.

Core tables/views:
- export_overview(year, total_exports_usd, real_exports_2018_usd, active_products, active_markets)
- totals_by_year(year, filtered, filtered_cpi_adj, filtered_nominal_original, filtered_cpi_adj_original, real_ratio_2018_base)
- product_year(hs6, hs4, name, sector, year, export_value, rca, pci, n_countries, unrealized_potential_usd, cagr, growth, trajectory)
- products_master: one row per HS6 product, annual values/RCA 2018-2025, PCI, CAGR, trajectory, reach and potential
- product_market_year(year, hs6, hs4, product_name, sector, country, iso3, continent, value_usd, pci, rca)
- product_market_share: product_market_year plus share_of_market_exports and share_of_product_exports
- market_year(country, iso3, continent, year, export_value, n_products, expy, hhi, rca, unrealized_potential_usd, status, priority, cagr)
- markets_master: annual values, product counts, HHI, EXPY, performance, entry/exit and potential by destination
- sector_year(sector, year, export_value, share, rca, pci_avg, unrealized_potential_usd, n_products_hs6, n_products_hs4, cagr)
- sectors_master: annual values, shares and RCA plus PCI and potential by sector
- market_size_hs6(country, year, hs6, market_size_usd)
- up_pairs(hs6, country, partner_raw, display_country, value, value_usd)
- up_hs6, up_hs4, up_sector, up_partner, up_top_pairs, up_totals
- similar_markets(country, rank, iso3, score, continent, exports_2025)
- market_top_products, product_top_destinations, market_all_products, product_all_markets
- topsis_overperformers, topsis_underperformers
- meta, filter_funnel, up_provenance, geo_features, raw_dataset_records

Definitions:
- Monetary values are USD.
- Default year is 2025 when omitted; market-size data is normally 2018 or 2024.
- RCA >= 1 means revealed comparative advantage.
- PCI is product complexity; sector PCI is export-weighted average PCI.
- EXPY is sophistication of the basket exported to a destination.
- Scale is current export value in USD; it is not growth, diversity, or complexity.
- Diversification has breadth and balance dimensions: product counts and market counts show breadth, while lower HHI shows a more even destination basket.
- HHI is concentration; lower HHI means more diversification.
- Composition means the shares of products, sectors, or markets in total exports.
- Product-market exports must come from product_market_year/product_market_share.
- Product names are the primary identifier; show HS codes only if explicitly requested.
""".strip()

BLOCKED_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|detach|pragma|vacuum|create|replace|reindex|trigger)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Semantic catalog and entity resolution
# ---------------------------------------------------------------------------

GENERIC_WORDS = {
    "a", "about", "above", "all", "also", "amount", "an", "and", "annual", "any", "are", "as", "at",
    "available", "average", "be", "best", "between", "biggest", "bottom", "by", "can", "change", "compare",
    "comparison", "composition", "concentration", "country", "countries", "current", "data", "destination",
    "destinations", "did", "difference", "diversification", "do", "does", "during", "each", "every", "export",
    "exported", "exporting", "exports", "for", "from", "growth", "had", "has", "have", "highest", "history",
    "how", "hscode", "hs", "import", "imports", "in", "into", "is", "it", "largest", "latest", "least",
    "lebanese", "lebanon", "leading", "list", "lowest", "market", "markets", "me", "metric", "more", "most", "size", "similar", "but", "not", "no", "longer", "exiting", "entered", "exit", "entry", "both", "common", "each",
    "much", "name", "new", "number", "of", "or", "overall", "percentage", "product", "products", "rank",
    "ranking", "related", "sector", "sectors", "share", "show", "since", "smallest", "some", "tell", "than",
    "the", "their", "this", "through", "to", "top", "total", "trend", "value", "values", "versus", "vs",
    "was", "were", "what", "when", "where", "which", "who", "why", "with", "year", "years",
    "rca", "pci", "hhi", "expy", "cagr", "complexity", "complex", "potential", "unrealized", "untapped",
    "performance", "overperforming", "underperforming", "sophistication", "real", "nominal", "count", "many",
    "buy", "buys", "buying", "sell", "sells", "selling", "purchases", "imported", "reach", "reached",
}

MARKET_ALIASES = {
    "ksa": "Saudi Arabia", "saudi": "Saudi Arabia", "saudia": "Saudi Arabia",
    "united arab emirates": "UAE", "emirates": "UAE", "u a e": "UAE",
    "america": "United States", "usa": "United States", "u s a": "United States", "us": "United States",
    "uk": "United Kingdom", "u k": "United Kingdom", "britain": "United Kingdom",
    "cote d ivoire": "Ivory Coast", "côte d ivoire": "Ivory Coast",
    "korea": "South Korea", "south korea": "South Korea", "egyptian": "Egypt",
    "russia": "Russian Federation", "uae": "UAE",
}

SECTOR_ALIASES = {
    "agrofood": "Agrifood", "agri food": "Agrifood", "food sector": "Agrifood",
    "machinery": "Electrical and Machinery", "electrical": "Electrical and Machinery",
    "electrical machinery": "Electrical and Machinery",
    "pharma": "Pharma & Parapharma", "pharmaceuticals": "Pharma & Parapharma",
    "pharmaceutical": "Pharma & Parapharma", "parapharma": "Pharma & Parapharma",
    "plastics": "Plastics / Rubbers", "rubber": "Plastics / Rubbers", "rubbers": "Plastics / Rubbers",
    "wood": "Wood & Wood Products", "wood products": "Wood & Wood Products",
    "stone": "Stone / Glass", "glass": "Stone / Glass",
    "chemicals": "Chemicals & Allied Industries", "chemical": "Chemicals & Allied Industries",
    "fertilizers": "Fertilizers & Agri-inputs", "fertiliser": "Fertilizers & Agri-inputs",
    "agri inputs": "Fertilizers & Agri-inputs", "textile": "Textiles", "textiles": "Textiles",
    "metal": "Metals", "metals": "Metals", "furniture sector": "Furniture",
    "transport": "Transportation", "transportation": "Transportation",
    "mineral": "Mineral Products", "minerals": "Mineral Products",
    "leather": "Raw Hides, Skins, Leather, & Furs", "hides": "Raw Hides, Skins, Leather, & Furs",
    "footwear": "Footwear / Headgear", "headgear": "Footwear / Headgear",
}

PRODUCT_ALIASES = {
    "olive oil": "150910", "virgin olive oil": "150910", "olives oil": "150910",
    "jewelry": "711319", "jewellery": "711319", "gold jewelry": "711319",
    "phosphoric acid": "280920",
    "chocolate": "180690", "medicine": "300490", "medicines": "300490",
    "pharmaceutical product": "300490",
    "wooden furniture": "940360", "printed books": "490199", "books and brochures": "490199",
    "children books": "490300", "children s books": "490300", "concrete mixers": "847431",
    "mortar mixers": "847431", "concrete mixer lorries": "870540", "concrete mixer trucks": "870540",
}

METRIC_DEFINITIONS = {
    "rca": "RCA (Revealed Comparative Advantage) compares Lebanon's export specialization in a product or sector with the world pattern. An RCA of 1 is the threshold: values at or above 1 indicate revealed comparative advantage.",
    "pci": "PCI (Product Complexity Index) measures how sophisticated and knowledge-intensive a product is relative to other traded products. Higher PCI means greater complexity.",
    "complexity": "The dashboard measures product complexity with PCI. Sector complexity is an export-weighted average of product PCI, while destination-basket sophistication is represented by EXPY.",
    "hhi": "HHI (Herfindahl-Hirschman Index) measures export concentration. A higher HHI means exports are concentrated in fewer products; a lower HHI means a more diversified basket.",
    "expy": "EXPY measures the sophistication of the products Lebanon exports to a destination. Higher EXPY indicates a more sophisticated export basket.",
    "cagr": "CAGR is the compound annual growth rate over the dashboard's stated period. It summarizes the average annual pace of change between the endpoints.",
    "unrealized potential": "Unrealized export potential is the estimated additional value Lebanon could export to a product-market pair under the dashboard's potential methodology. It is not a guaranteed forecast.",
    "market size": "Market size is the destination country's total imports of the HS6 product in the available market-size year, generally 2018 or 2024.",
    "real exports": "Real exports are nominal export values adjusted to 2018 prices using the dashboard's CPI adjustment.",
}


@dataclass(frozen=True)
class ProductMatch:
    hs6: str
    name: str
    sector: str
    score: float = 1.0


@dataclass
class ParsedQuestion:
    original: str
    norm: str
    years: list[int] = field(default_factory=list)
    products: list[ProductMatch] = field(default_factory=list)
    markets: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    continents: list[str] = field(default_factory=list)
    measures: set[str] = field(default_factory=set)
    group: str | None = None
    operation: str = "lookup"
    limit: int = 10
    ascending: bool = False
    explicit_code: bool = False
    filters: list[tuple[str, str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class QueryPlan:
    title: str
    sql: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.98
    kind: str = "sql"
    direct_answer: str = ""
    notes: tuple[str, ...] = ()


def _norm_text(value: Any) -> str:
    value = str(value or "").lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _term_root(token: str) -> str:
    token = str(token or "").lower()
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("sses") and len(token) > 6:
        return token[:-2]
    if token.endswith("es") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _has_model_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip())


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    with sqlite3.connect(_database_path()) as conn:
        conn.row_factory = sqlite3.Row
        products = [dict(r) for r in conn.execute(
            "SELECT hs6,name,sector,COALESCE(value_2025,0) AS value_2025 FROM products_master ORDER BY value_2025 DESC"
        )]
        markets = [dict(r) for r in conn.execute(
            "SELECT country,iso3,continent,COALESCE(exports_2025,0) AS exports_2025 FROM markets_master ORDER BY exports_2025 DESC"
        )]
        sectors = [str(r[0]) for r in conn.execute("SELECT sector FROM sectors_master ORDER BY sector")]
    product_by_hs = {str(p["hs6"]).zfill(6): p for p in products}
    market_names = [str(m["country"]) for m in markets]
    continent_names = sorted({str(m["continent"]) for m in markets if m.get("continent")})
    product_tokens: dict[str, set[int]] = {}
    for idx, product in enumerate(products):
        for token in {_term_root(t) for t in _norm_text(product["name"]).split() if len(t) > 2}:
            product_tokens.setdefault(token, set()).add(idx)
    return {
        "products": products,
        "product_by_hs": product_by_hs,
        "product_tokens": product_tokens,
        "markets": markets,
        "market_names": market_names,
        "sectors": sectors,
        "continents": continent_names,
    }



@lru_cache(maxsize=1)
def _dashboard_bundle() -> dict[str, Any]:
    path = ROOT / "data" / "dashboard_bundle.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _similar_market_answer(market: str, limit: int) -> str | None:
    bundle = _dashboard_bundle()
    for record in bundle.get("markets", []):
        if str(record.get("country")) != market:
            continue
        rows = list(record.get("similar_countries") or [])[:limit]
        if not rows:
            return None
        lines = [f"**Markets similar to {market}**", ""]
        for idx, row in enumerate(rows, 1):
            lines.append(
                f"{idx}. **{row.get('country', 'Unknown')}** - Similarity score: {float(row.get('score') or 0):.3f}; "
                f"Continent: {row.get('continent', 'No data')}; 2025 exports: {_money(row.get('exports_2025'))}"
            )
        return "\n".join(lines)
    return None

def _phrase_present(norm_question: str, phrase: str) -> bool:
    phrase_norm = _norm_text(phrase)
    return bool(phrase_norm and re.search(rf"(?<![a-z0-9]){re.escape(phrase_norm)}(?![a-z0-9])", norm_question))


def _find_named(question: str, names: Iterable[str], aliases: dict[str, str] | None = None) -> list[str]:
    q = _norm_text(question)
    aliases = aliases or {}
    positioned: list[tuple[int, int, str]] = []
    for alias, canonical in aliases.items():
        alias_norm = _norm_text(alias)
        match = re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", q) if alias_norm else None
        if match:
            positioned.append((match.start(), -len(alias_norm), canonical))
    for name in names:
        norm = _norm_text(name)
        match = re.search(rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])", q) if norm else None
        if match:
            positioned.append((match.start(), -len(norm), str(name)))
    positioned.sort()
    found: list[str] = []
    for _, _, canonical in positioned:
        if canonical not in found:
            found.append(canonical)
    return found


def _find_products(question: str, limit: int = 8) -> list[ProductMatch]:
    catalog = _catalog()
    q = _norm_text(question)
    matches: list[ProductMatch] = []

    # Explicit HS references have the highest priority, but numbers embedded
    # inside official product descriptions (for example "heading no. 1509")
    # are not treated as user-supplied codes.
    code_cue = bool(
        re.search(
            r"\b(?:hs|hs4|hs6|h s|harmonized|harmonised|tariff|customs|product code|code)\b",
            q,
        )
    )
    query_words = q.split()
    numeric_tokens: list[str] = []
    for token in re.findall(r"\b\d{4,6}\b", question):
        if token in {str(year) for year in range(2018, 2026)}:
            continue
        if len(token) == 6 or code_cue or len(query_words) <= 4:
            numeric_tokens.append(token)

    for token in numeric_tokens:
        code = token.zfill(6)
        exact = catalog["product_by_hs"].get(code)
        if exact:
            matches.append(ProductMatch(code, str(exact["name"]), str(exact["sector"]), 1.0))
            continue
        # Four-digit prefixes are accepted only when the question clearly asks
        # about an HS/product code.
        if len(token) == 4 and not code_cue and len(query_words) > 4:
            continue
        prefix = token
        for product in catalog["products"]:
            hs6 = str(product["hs6"]).zfill(6)
            if hs6.startswith(prefix):
                matches.append(ProductMatch(hs6, str(product["name"]), str(product["sector"]), 0.94))
                if len(matches) >= limit:
                    break

    if matches:
        unique: list[ProductMatch] = []
        seen: set[str] = set()
        for match in matches:
            if match.hs6 not in seen:
                unique.append(match)
                seen.add(match.hs6)
        return unique[:limit]

    # A complete dashboard product name must win over any shorter alias that it
    # happens to contain. This guarantees that all 882 exact names resolve to
    # their own record rather than to a broader product category.
    exact_candidates: list[tuple[int, int, ProductMatch]] = []
    for product in catalog["products"]:
        name_norm = _norm_text(product["name"])
        position = q.find(name_norm) if name_norm else -1
        if len(name_norm) >= 5 and position >= 0:
            exact_candidates.append(
                (
                    position,
                    -len(name_norm),
                    ProductMatch(
                        str(product["hs6"]).zfill(6),
                        str(product["name"]),
                        str(product["sector"]),
                        1.0,
                    ),
                )
            )
    if exact_candidates:
        exact_candidates.sort()
        allow_multiple = bool(re.search(r"\b(?:compare|versus|vs|both)\b", q))
        ordered = [item[2] for item in exact_candidates]
        if not allow_multiple:
            return [min(ordered, key=lambda match: -len(_norm_text(match.name)))]
        unique: list[ProductMatch] = []
        seen: set[str] = set()
        for item in ordered:
            if item.hs6 not in seen:
                unique.append(item)
                seen.add(item.hs6)
        return unique[:limit]

    # Curated ordinary-language aliases are used only after exact names.
    for alias, hs6 in sorted(PRODUCT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if not _phrase_present(q, alias):
            continue
        product = catalog["product_by_hs"].get(hs6)
        if product:
            matches.append(ProductMatch(hs6, str(product["name"]), str(product["sector"]), 0.99))

    if matches:
        unique = []
        seen = set()
        for match in matches:
            if match.hs6 not in seen:
                unique.append(match)
                seen.add(match.hs6)
        return unique[:limit]

    # Token-indexed fuzzy search across all 882 product names. Remove dimensions,
    # metrics and ordinary query words first so generic dashboard wording cannot
    # become a product.
    removable = set(GENERIC_WORDS)
    removable.update(_term_root(t) for t in _find_named(question, catalog["market_names"], MARKET_ALIASES) for t in _norm_text(t).split())
    removable.update(_term_root(t) for t in _find_named(question, catalog["sectors"], SECTOR_ALIASES) for t in _norm_text(t).split())
    tokens = [_term_root(t) for t in q.split() if len(t) > 2 and not t.isdigit() and _term_root(t) not in removable]
    if not tokens:
        return []
    query_set = set(tokens)
    candidate_ids: set[int] = set()
    for token in query_set:
        candidate_ids.update(catalog["product_tokens"].get(token, set()))
    if not candidate_ids:
        return []

    phrase = " ".join(tokens)
    scored: list[tuple[float, int]] = []
    incidental_single_token_patterns = (
        "other than {token}",
        "excluding {token}",
        "containing {token}",
        "manufacture of {token}",
        "used in the manufacture of {token}",
        "for {token}",
    )
    for idx in candidate_ids:
        product = catalog["products"][idx]
        name_norm = _norm_text(product["name"])
        if len(query_set) == 1:
            token = next(iter(query_set))
            # Do not substitute a product that only mentions the requested word
            # as an exclusion, ingredient, end use, or manufacturing application.
            if any(pattern.format(token=token) in name_norm for pattern in incidental_single_token_patterns):
                continue
        name_tokens = {_term_root(t) for t in name_norm.split() if len(t) > 2}
        overlap = query_set & name_tokens
        if not overlap:
            continue
        coverage = len(overlap) / len(query_set)
        precision = len(overlap) / max(1, min(len(name_tokens), len(query_set) + 4))
        ratio = SequenceMatcher(None, phrase, name_norm).ratio()
        score = 0.55 * coverage + 0.20 * precision + 0.25 * ratio
        if query_set.issubset(name_tokens):
            score += 0.28
        if phrase in name_norm:
            score += 0.45
        if len(query_set) == 1 and len(next(iter(query_set))) >= 6:
            score += 0.10
        scored.append((score, idx))
    scored.sort(key=lambda x: (x[0], float(catalog["products"][x[1]].get("value_2025") or 0)), reverse=True)
    if not scored:
        return []
    best = scored[0][0]
    threshold = 0.86 if len(query_set) == 1 else 0.74
    result: list[ProductMatch] = []
    allow_multiple = bool(re.search(r"\b(?:compare|versus|vs|both)\b", q))
    for score, idx in scored:
        if score < threshold or score < best - 0.13:
            continue
        product = catalog["products"][idx]
        result.append(ProductMatch(str(product["hs6"]).zfill(6), str(product["name"]), str(product["sector"]), min(score, 0.96)))
        if len(result) >= (limit if allow_multiple else 1):
            break
    return result


def _extract_years(question: str) -> list[int]:
    years = [int(y) for y in re.findall(r"\b20(?:18|19|20|21|22|23|24|25)\b", question)]
    years = list(dict.fromkeys(years))
    q = _norm_text(question)
    if not years and "latest" in q:
        years = [2025]
    return years


def _extract_limit(question: str, default: int = 10) -> int:
    if re.search(r"\b(?:all|every)\s+(?:products|markets|countries|destinations|sectors|records)\b", question, re.I):
        return 100
    patterns = [
        r"\b(?:top|bottom|first|largest|highest|lowest|smallest|best|leading|most|least)\s+(\d{1,3})\b",
        r"\b(\d{1,3})\s+(?:products|markets|countries|destinations|sectors|years|items|records)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if match:
            return max(1, min(100, int(match.group(1))))
    return default


def _parse_measures(q: str) -> set[str]:
    measures: set[str] = set()
    if any(term in q for term in ("export", "value", "amount", "sales", "trade")):
        measures.add("exports")
    if any(term in q for term in ("real export", "constant price", "inflation adjusted", "cpi adjusted")):
        measures.add("real_exports")
    if re.search(r"\brca\b|comparative advantage|speciali[sz]ation", q):
        measures.add("rca")
    if re.search(r"\bpci\b|product complexity|most complex|least complex", q):
        measures.add("pci")
    if "complexity" in q or "complex" in q:
        measures.add("complexity")
    if re.search(r"\bhhi\b|concentration|diversif", q):
        measures.add("hhi")
    if re.search(r"\bexpy\b|sophisticat", q):
        measures.add("expy")
    if re.search(r"\bcagr\b|growth rate|annual growth", q):
        measures.add("cagr")
    if any(term in q for term in ("growth", "grew", "grown", "increase", "decrease", "change", "decline", "rose", "fell")):
        measures.add("growth")
    if any(term in q for term in ("potential", "unrealized", "untapped", "opportunity")):
        measures.add("potential")
    if any(term in q for term in ("market size", "import market", "imports of", "total imports")):
        measures.add("market_size")
    if any(term in q for term in ("market penetration", "penetration rate", "share of the market", "share in the market", "lebanon share of")) or re.search(r"share of (?:the )?.{0,80} market", q):
        measures.add("penetration")
    if any(term in q for term in ("share", "percentage", "percent", "contribution")):
        measures.add("share")
    if any(term in q for term in ("number of products", "how many products", "product count", "count products")):
        measures.add("n_products")
    if any(term in q for term in ("number of markets", "how many markets", "market count", "countries reached", "market reach")):
        measures.add("n_markets")
    if any(term in q for term in ("status", "priority", "overperform", "underperform", "performance")):
        measures.add("performance")
    if any(term in q for term in ("new products", "entered products", "products entered", "product entry", "newly exported", "entering exports", "newly entered", "entered the market", "new to the market")):
        measures.add("entry")
    if any(term in q for term in ("exited products", "products exited", "products exiting", "lost products", "product exit", "discontinued", "leaving exports", "no longer exported")):
        measures.add("exit")
    if any(term in q for term in ("similar market", "similar markets", "similar country", "similar countries", "comparable market")) or re.search(r"\bmarkets?\b.{0,30}\bsimilar\b|\bsimilar\b.{0,30}\bmarkets?\b", q):
        measures.add("similarity")
    if not measures:
        measures.add("exports")
    return measures


def _parse_group(q: str) -> str | None:
    patterns = [
        (r"\bby\s+(?:product|products|hs6|hs code)\b", "product"),
        (r"\bby\s+(?:sector|sectors|industry|industries)\b", "sector"),
        (r"\bby\s+(?:market|markets|country|countries|destination|destinations)\b", "market"),
        (r"\bby\s+(?:continent|region|regions)\b", "continent"),
        (r"\bby\s+(?:year|years|time)\b", "year"),
    ]
    for pattern, group in patterns:
        if re.search(pattern, q):
            return group
    count_context = bool(re.search(r"\b(?:how many|number of|count)\b", q))
    # Plural dimension words are explicit even when an adjective/entity appears
    # between the ranking word and the dimension (e.g. "top Agrifood products").
    if re.search(r"\bproducts\b", q) and not count_context:
        return "product"
    if re.search(r"\b(?:markets|countries|destinations)\b", q) and not count_context:
        return "market"
    if re.search(r"\bsectors\b", q) and not count_context:
        return "sector"
    if re.search(r"\b(?:continents|regions)\b", q):
        return "continent"
    if q.startswith("where "):
        return "market"
    if re.search(r"\b(?:trend|history|over time|year by year|annual series|every year)\b", q):
        return "year"
    return None


def _parse_operation(q: str, years: list[int]) -> tuple[str, bool]:
    ascending = bool(re.search(r"\b(?:bottom|lowest|smallest|least|worst|declined most|fell most)\b", q))
    if re.search(r"\b(?:define|definition|meaning|what does|what is|explain)\b", q) and re.search(r"\b(?:rca|pci|hhi|expy|cagr|complexity|potential|market size|real exports)\b", q):
        return "definition", ascending
    if re.search(r"\b(?:correlation|relationship|associated|linked)\b", q):
        return "correlation", ascending
    if re.search(r"\b(?:compare|comparison|versus|\bvs\b|difference between)\b", q):
        return "compare", ascending
    if re.search(r"\b(?:what drove|drove|driving|drivers|contributed to|contribution to|biggest increase|biggest decline)\b", q):
        return "drivers", ascending
    if re.search(r"\b(?:trend|history|over time|year by year|annual series|every year|evolution)\b", q) or len(years) >= 2:
        return "trend", ascending
    if re.search(r"\b(?:rank|ranking|top|bottom|largest|highest|lowest|smallest|leading|best|most|least)\b", q):
        return "rank", ascending
    if re.search(r"\b(?:how many|number of|count of|count)\b", q):
        return "count", ascending
    if re.search(r"\b(?:average|mean)\b", q):
        return "average", ascending
    if re.search(r"\b(?:share|percentage|percent|contribution)\b", q):
        return "share", ascending
    if re.search(r"\b(?:list|show all|all products|all markets|all countries|all sectors)\b", q):
        return "list", ascending
    return "lookup", ascending


def _parse_numeric_filters(question: str) -> list[tuple[str, str, float]]:
    q = _norm_text(question)
    aliases = {
        "rca": "rca", "pci": "pci", "hhi": "hhi", "expy": "expy", "cagr": "cagr",
        "potential": "unrealized_potential_usd", "market size": "market_size_usd",
    }
    filters: list[tuple[str, str, float]] = []
    for phrase, column in aliases.items():
        patterns = [
            rf"\b{re.escape(phrase)}\s*(>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)",
            rf"\b{re.escape(phrase)}\s+(?:above|over|greater than|higher than)\s+(-?\d+(?:\.\d+)?)",
            rf"\b{re.escape(phrase)}\s+(?:below|under|less than|lower than)\s+(-?\d+(?:\.\d+)?)",
        ]
        for idx, pattern in enumerate(patterns):
            match = re.search(pattern, q)
            if not match:
                continue
            if idx == 0:
                operator, value = match.group(1), float(match.group(2))
            elif idx == 1:
                operator, value = ">", float(match.group(1))
            else:
                operator, value = "<", float(match.group(1))
            filters.append((column, operator, value))
            break
    # Natural count thresholds, such as "countries receiving more than
    # 500 products" or "markets with at least 300 products".
    count_patterns = [
        (r"\b(?:more than|over|above|greater than)\s+(\d+(?:\.\d+)?)\s+(?:(?:distinct|different|lebanese|exported|industrial)\s+)*products\b", "n_products", ">"),
        (r"\b(?:at least|no fewer than)\s+(\d+(?:\.\d+)?)\s+(?:(?:distinct|different|lebanese|exported|industrial)\s+)*products\b", "n_products", ">="),
        (r"\b(?:fewer than|less than|under|below)\s+(\d+(?:\.\d+)?)\s+(?:(?:distinct|different|lebanese|exported|industrial)\s+)*products\b", "n_products", "<"),
        (r"\b(?:at most|no more than)\s+(\d+(?:\.\d+)?)\s+(?:(?:distinct|different|lebanese|exported|industrial)\s+)*products\b", "n_products", "<="),
        (r"\b(?:more than|over|above|greater than)\s+(\d+(?:\.\d+)?)\s+(?:countries|markets|destinations)\b", "n_markets", ">"),
        (r"\b(?:at least|no fewer than)\s+(\d+(?:\.\d+)?)\s+(?:countries|markets|destinations)\b", "n_markets", ">="),
        (r"\b(?:fewer than|less than|under|below)\s+(\d+(?:\.\d+)?)\s+(?:countries|markets|destinations)\b", "n_markets", "<"),
        (r"\b(?:at most|no more than)\s+(\d+(?:\.\d+)?)\s+(?:countries|markets|destinations)\b", "n_markets", "<="),
    ]
    for pattern, column, operator in count_patterns:
        match = re.search(pattern, q)
        if match and not any(existing[0] == column for existing in filters):
            filters.append((column, operator, float(match.group(1))))

    return filters



def _asks_for_products_in_market(q: str) -> bool:
    """Detect natural questions asking what Lebanon sells to a named market.

    These should return the product basket inside that market, not the market's
    overall rank. Explicit requests for the total/value remain market-profile
    questions.
    """
    q = _norm_text(q)
    if re.search(r"\b(?:how much|how many dollars|total|overall|value|amount|worth)\b", q):
        return False
    patterns = (
        r"\bwhat\s+(?:does|do|did)\s+lebanon\s+(?:export|sell|ship)\s+to\b",
        r"\bwhat\s+(?:does|do|did)\s+[^?]+\s+(?:import|buy|purchase)\s+from\s+lebanon\b",
        r"\b(?:what|which)\s+(?:products|goods|items|commodities)\b.*\b(?:export|sell|ship|send|go)\b",
        r"\b(?:top|main|largest|biggest|leading|most important)\s+(?:exports|products|goods|items|commodities)\b",
        r"\b(?:export basket|basket of exports)\b",
        r"\bwhat\s+are\s+lebanon(?:'s| s)?\s+(?:main\s+|top\s+)?exports\s+to\b",
    )
    return any(re.search(pattern, q) for pattern in patterns)


def _requested_market_columns(parsed: ParsedQuestion, include_rank: bool = False) -> list[str]:
    """Return only the market indicators explicitly requested by the user."""
    columns = ["country", "continent", "year", "export_value", "n_products"]
    if "expy" in parsed.measures:
        columns.append("expy")
    if "hhi" in parsed.measures:
        columns.append("hhi")
    if "rca" in parsed.measures:
        columns.append("rca")
    if "potential" in parsed.measures:
        columns.append("unrealized_potential_usd")
    if "performance" in parsed.measures:
        columns.extend(["status", "priority"])
    if "cagr" in parsed.measures or "growth" in parsed.measures:
        columns.append("cagr")
    if include_rank:
        columns.append("metric_rank")
    return list(dict.fromkeys(columns))



def _asks_for_market_by_product_count(q: str) -> bool:
    """Detect country/market lists filtered by the number of exported products."""
    q = _norm_text(q)
    has_market_dimension = bool(
        re.search(r"\b(?:country|countries|market|markets|destination|destinations|where)\b", q)
        or re.search(r"\bto what\b", q)
    )
    has_product_threshold = bool(
        re.search(
            r"\b(?:more than|over|above|greater than|at least|no fewer than|"
            r"fewer than|less than|under|below|at most|no more than)\s+"
            r"\d+(?:\.\d+)?\s+(?:(?:distinct|different|lebanese|exported|industrial)\s+)*products\b",
            q,
        )
    )
    return has_market_dimension and has_product_threshold


def _extract_unresolved_product_phrase(question: str) -> str | None:
    """Extract a likely product phrase when no catalog product was resolved."""
    raw = re.sub(r"\s+", " ", str(question or "").strip())
    patterns = (
        r"how much (?P<item>.+?) (?:did|does|do) lebanon (?:export|sell|ship)(?: in 20\d{2})?\??$",
        r"how much of (?P<item>.+?) (?:did|does|do) lebanon (?:export|sell|ship)(?: in 20\d{2})?\??$",
        r"how much (?:did |does |do )?lebanon (?:export|sell|ship) of (?P<item>.+?)(?: in 20\d{2}| to [A-Z][^?]*|$)",
        r"how much (?:did |does |do )?lebanon (?:export|sell|ship) (?P<item>(?!(?:to|into|toward|towards)\b).+?)(?: in 20\d{2}| to [A-Z][^?]*|$)",
        r"where (?:does|did) lebanon (?:export|sell|ship) (?P<item>.+?)(?: in 20\d{2}|$)",
        r"what countries (?:buy|import|purchase) (?P<item>.+?) from lebanon",
        r"what markets (?:buy|import|purchase) (?P<item>.+?) from lebanon",
        r"exports? of (?P<item>.+?)(?: in 20\d{2}| to [A-Z][^?]*|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        item = match.group("item").strip(" ?.,")
        norm = _norm_text(item)
        if norm.startswith(("to ", "into ", "toward ", "towards ")):
            continue
        if not norm or norm in {
            "products", "goods", "items", "commodities", "industrial products",
            "more than products",
        }:
            continue
        if len(norm.split()) <= 8:
            return item
    return None



def _product_reference_remainder(question: str, product: ProductMatch) -> str:
    """Remove the resolved product reference before parsing other intent."""
    q = _norm_text(question)
    product_norm = _norm_text(product.name)
    if product_norm and product_norm in q:
        return re.sub(r"\s+", " ", q.replace(product_norm, " ", 1)).strip()

    aliases = [
        alias for alias, hs6 in PRODUCT_ALIASES.items()
        if hs6 == product.hs6 and _phrase_present(q, alias)
    ]
    if aliases:
        alias = max(aliases, key=len)
        alias_norm = _norm_text(alias)
        return re.sub(
            r"\s+",
            " ",
            re.sub(
                rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])",
                " ",
                q,
                count=1,
            ),
        ).strip()

    # Remove an explicit HS6 reference while retaining the surrounding command.
    if re.search(rf"\b{re.escape(product.hs6)}\b", q):
        return re.sub(
            r"\s+",
            " ",
            re.sub(rf"\b{re.escape(product.hs6)}\b", " ", q, count=1),
        ).strip()
    return q


def _explicit_code_request(question: str) -> bool:
    """Recognize true code requests without using numbers inside product names."""
    q = _norm_text(question)
    if re.search(
        r"\b(?:hs|hs4|hs6|h s|harmonized|harmonised|tariff|customs|product code|code)\b",
        q,
    ):
        return True
    years = {str(year) for year in range(2018, 2101)}
    tokens = [
        token for token in re.findall(r"\b\d{6}\b", question)
        if token not in years
    ]
    return bool(tokens and len(q.split()) <= 6)


def parse_question(question: str) -> ParsedQuestion:
    original = str(question or "").strip()
    q = _norm_text(original)
    years = _extract_years(original)
    operation, ascending = _parse_operation(q, years)
    continents = _find_named(original, _catalog()["continents"], {"middle east": "Asia"})
    products = _find_products(original)

    parsed = ParsedQuestion(
        original=original,
        norm=q,
        years=years,
        products=products,
        markets=_find_named(original, _catalog()["market_names"], MARKET_ALIASES),
        sectors=_find_named(original, _catalog()["sectors"], SECTOR_ALIASES),
        continents=continents,
        measures=_parse_measures(q),
        group=_parse_group(q),
        operation=operation,
        limit=_extract_limit(original),
        ascending=ascending,
        explicit_code=_explicit_code_request(original),
        filters=_parse_numeric_filters(original),
    )

    # Once one product is resolved, remove that exact product reference before
    # reading the remaining market, sector, metric, and operation words. Long
    # customs descriptions frequently contain terms such as "metals" or
    # "china" that are part of the product name rather than user filters.
    if len(products) == 1:
        remainder = _product_reference_remainder(original, products[0])
        parsed.markets = _find_named(
            remainder,
            _catalog()["market_names"],
            MARKET_ALIASES,
        )
        parsed.sectors = _find_named(
            remainder,
            _catalog()["sectors"],
            SECTOR_ALIASES,
        )
        parsed.continents = _find_named(
            remainder,
            _catalog()["continents"],
            {"middle east": "Asia"},
        )
        parsed.measures = _parse_measures(remainder)
        parsed.group = _parse_group(remainder)
        parsed.operation, parsed.ascending = _parse_operation(remainder, years)
        parsed.filters = _parse_numeric_filters(remainder)

    # Questions such as "to what countries does Lebanon export more than
    # 500 products?" refer to destination markets filtered by product count.
    if _asks_for_market_by_product_count(q):
        parsed.group = "market"
        parsed.measures.add("n_products")
        parsed.operation = "list"

    # Natural market-basket questions such as "what does Lebanon export to
    # Syria the most?" must rank products within Syria. They must not be
    # interpreted as a request for Syria's rank among destination markets.
    if parsed.markets and not parsed.products and _asks_for_products_in_market(q):
        parsed.group = "product"
        if re.search(r"\b(?:most|top|main|largest|biggest|leading|best)\b", q):
            parsed.operation = "rank"
        elif parsed.operation == "lookup":
            parsed.operation = "list"
    return parsed

def resolve_dashboard_entities(question: str) -> dict[str, Any]:
    parsed = parse_question(question)
    entities: dict[str, Any] = {}
    if parsed.products:
        entities["products"] = [p.hs6 for p in parsed.products]
        entities["hs6"] = parsed.products[-1].hs6
        entities["product_names"] = [p.name for p in parsed.products]
    if parsed.markets:
        entities["markets"] = parsed.markets
        entities["market"] = parsed.markets[-1]
    if parsed.sectors:
        entities["sectors"] = parsed.sectors
        entities["sector"] = parsed.sectors[-1]
    if parsed.years:
        entities["years"] = parsed.years
        entities["year"] = parsed.years[-1]
    if parsed.measures:
        entities["metric"] = sorted(parsed.measures)[0]
    return entities



# ---------------------------------------------------------------------------
# Complete product knowledge profiles
# ---------------------------------------------------------------------------

_FULL_PRODUCT_PROFILE_TERMS = (
    "all details", "every detail", "everything about", "everything on",
    "full profile", "complete profile", "complete details", "all information",
    "all data", "full details", "complete information", "detailed profile",
)

_SIMPLE_PRODUCT_PROFILE_TERMS = (
    "product profile", "profile of", "profile for", "overview of",
    "tell me about", "what do you know about", "information about",
    "details about", "details on", "describe", "summary of",
)


@lru_cache(maxsize=1)
def _latest_product_year() -> int:
    with sqlite3.connect(_database_path()) as conn:
        row = conn.execute("SELECT MAX(year) FROM product_year").fetchone()
    return int(row[0]) if row and row[0] is not None else 2025


def _product_profile_level(parsed: ParsedQuestion) -> str | None:
    """Return simple/full for broad product-profile questions.

    The exact product name is removed before interpreting the remaining words.
    This prevents words inside long customs descriptions from being mistaken
    for a separate market, sector, metric, or query instruction.
    """
    if len(parsed.products) != 1:
        return None

    product = parsed.products[0]
    q = parsed.norm
    product_norm = _norm_text(product.name)

    # Remove the authoritative exact product name before reading intent. For a
    # common-language alias, the alias stays in the remainder but does not
    # create false sector or market entities.
    remainder = q
    if product_norm and product_norm in remainder:
        remainder = remainder.replace(product_norm, " ", 1)
    remainder = re.sub(r"\s+", " ", remainder).strip()

    explicit_markets = _find_named(
        remainder,
        _catalog()["market_names"],
        MARKET_ALIASES,
    )
    explicit_sectors = _find_named(
        remainder,
        _catalog()["sectors"],
        SECTOR_ALIASES,
    )
    explicit_continents = _find_named(
        remainder,
        _catalog()["continents"],
        {"middle east": "Asia"},
    )
    if explicit_markets or explicit_sectors or explicit_continents:
        return None

    if any(term in remainder or term in q for term in _FULL_PRODUCT_PROFILE_TERMS):
        return "full"

    # Destination, value, count, ranking and comparison wording must answer the
    # requested product question rather than opening a general profile.
    dedicated_intent_patterns = (
        r"^where\b",
        r"^which\s+(?:countries|markets|destinations)\b",
        r"^what\s+(?:countries|markets|destinations)\b",
        r"\b(?:top|main|largest|leading|best)\s+(?:countries|markets|destinations)\b",
        r"\bhow much\b",
        r"\bhow many\b",
        r"\b(?:export value|amount exported|total exports?|rank|ranking|share|percentage)\b",
        r"\b(?:trend|history|over time|year by year|compare|versus|\bvs\b)\b",
    )
    if any(re.search(pattern, remainder) for pattern in dedicated_intent_patterns):
        return None

    # Interpret metrics and operations only from words outside the exact name.
    remainder_measures = _parse_measures(remainder)
    specific_measures = {
        "rca", "pci", "complexity", "hhi", "expy", "cagr", "growth",
        "potential", "market_size", "penetration", "share", "n_markets",
        "performance", "entry", "exit",
    }
    if remainder_measures & specific_measures:
        return None
    remainder_operation, _ = _parse_operation(remainder, parsed.years)
    if remainder_operation not in {"lookup", "definition"}:
        return None

    if any(term in remainder or term in q for term in _SIMPLE_PRODUCT_PROFILE_TERMS):
        return "simple"

    alias_hit = any(
        _phrase_present(q, alias) and hs6 == product.hs6
        for alias, hs6 in PRODUCT_ALIASES.items()
    )

    # Remove harmless profile phrasing from the instruction remainder. If
    # nothing meaningful remains, the prompt is a general product request.
    reduced = re.sub(
        r"\b(?:what|is|are|the|a|an|product|about|lebanon|lebanese|exports?|"
        r"tell|me|show|give|information|details|profile|overview|of|on|for|"
        r"please|this|that)\b",
        " ",
        remainder,
    )
    reduced = re.sub(r"\s+", " ", reduced).strip()

    if q == product_norm or alias_hit or not reduced:
        return "simple"
    if len(remainder.split()) <= 4 and not remainder_measures:
        return "simple"
    return None

def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _pct_text(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "Not calculable"
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value * 100:.1f}%"


@lru_cache(maxsize=4096)
def _product_profile_data(hs6: str, year: int) -> dict[str, Any]:
    """Load every product-related dashboard fact needed by the answer layer."""
    hs6 = str(hs6).zfill(6)
    with sqlite3.connect(_database_path()) as conn:
        conn.row_factory = sqlite3.Row

        base_row = conn.execute(
            "SELECT * FROM products_master WHERE printf('%06d',CAST(hs6 AS INTEGER))=? LIMIT 1",
            (hs6,),
        ).fetchone()
        if base_row is None:
            return {}
        base = dict(base_row)

        annual = [
            dict(row) for row in conn.execute(
                "SELECT year,export_value,rca,pci,n_countries,"
                "unrealized_potential_usd,cagr,growth,trajectory "
                "FROM product_year WHERE hs6=? ORDER BY year",
                (hs6,),
            )
        ]
        annual_by_year = {int(row["year"]): row for row in annual}
        latest = annual_by_year.get(year) or (annual[-1] if annual else {})
        actual_year = int(latest.get("year") or year)
        latest_value = float(latest.get("export_value") or 0)

        overview = conn.execute(
            "SELECT total_exports_usd,active_products FROM export_overview WHERE year=?",
            (actual_year,),
        ).fetchone()
        total_exports = float(overview["total_exports_usd"] or 0) if overview else 0.0
        active_products = int(overview["active_products"] or 0) if overview else 0

        overall_rank_row = conn.execute(
            "WITH ranked AS ("
            "SELECT hs6,RANK() OVER (ORDER BY export_value DESC) AS rank_value "
            "FROM product_year WHERE year=? AND export_value>0"
            ") SELECT rank_value FROM ranked WHERE hs6=?",
            (actual_year, hs6),
        ).fetchone()
        overall_rank = int(overall_rank_row["rank_value"]) if overall_rank_row else None

        sector = str(base.get("sector") or latest.get("sector") or "")
        sector_total_row = conn.execute(
            "SELECT SUM(export_value) AS sector_total,COUNT(*) AS active_sector_products "
            "FROM product_year WHERE year=? AND sector=? AND export_value>0",
            (actual_year, sector),
        ).fetchone()
        sector_total = float(sector_total_row["sector_total"] or 0) if sector_total_row else 0.0
        active_sector_products = int(sector_total_row["active_sector_products"] or 0) if sector_total_row else 0

        sector_rank_row = conn.execute(
            "WITH ranked AS ("
            "SELECT hs6,RANK() OVER (ORDER BY export_value DESC) AS rank_value "
            "FROM product_year WHERE year=? AND sector=? AND export_value>0"
            ") SELECT rank_value FROM ranked WHERE hs6=?",
            (actual_year, sector, hs6),
        ).fetchone()
        sector_rank = int(sector_rank_row["rank_value"]) if sector_rank_row else None

        destinations = [
            dict(row) for row in conn.execute(
                "SELECT country,continent,value_usd,share_of_product_exports,"
                "share_of_market_exports FROM product_market_share "
                "WHERE hs6=? AND year=? AND value_usd>0 "
                "ORDER BY value_usd DESC,country ASC",
                (hs6, actual_year),
            )
        ]

        market_sets: dict[int, set[str]] = {}
        for row in conn.execute(
            "SELECT year,country FROM product_market_year "
            "WHERE hs6=? AND value_usd>0 ORDER BY year,country",
            (hs6,),
        ):
            market_sets.setdefault(int(row["year"]), set()).add(str(row["country"]))
        first_year = min(market_sets) if market_sets else min(annual_by_year, default=2018)
        first_markets = market_sets.get(first_year, set())
        latest_markets = market_sets.get(actual_year, set())
        entered_markets = sorted(latest_markets - first_markets)
        exited_markets = sorted(first_markets - latest_markets)
        lifetime_markets = sorted(set().union(*market_sets.values())) if market_sets else []

        potential_rows = [
            dict(row) for row in conn.execute(
                "SELECT country,MAX(value_usd) AS unrealized_potential_usd "
                "FROM up_pairs WHERE CAST(hs6 AS INTEGER)=CAST(? AS INTEGER) "
                "GROUP BY country HAVING MAX(value_usd)>0 "
                "ORDER BY unrealized_potential_usd DESC,country ASC",
                (hs6,),
            )
        ]
        total_potential = sum(float(row.get("unrealized_potential_usd") or 0) for row in potential_rows)

        market_sizes = [
            dict(row) for row in conn.execute(
                "WITH latest AS ("
                "SELECT country,MAX(year) AS year FROM market_size_hs6 "
                "WHERE hs6=? GROUP BY country"
                ") "
                "SELECT m.country,m.year,m.market_size_usd,"
                "COALESCE(e.value_usd,0) AS lebanon_exports_same_year,"
                "CASE WHEN m.market_size_usd>0 "
                "THEN COALESCE(e.value_usd,0)/m.market_size_usd ELSE NULL END AS market_penetration "
                "FROM market_size_hs6 m "
                "JOIN latest l ON l.country=m.country AND l.year=m.year "
                "LEFT JOIN product_market_year e "
                "ON e.hs6=m.hs6 AND e.country=m.country AND e.year=m.year "
                "WHERE m.hs6=? "
                "ORDER BY m.market_size_usd DESC,m.country ASC",
                (hs6, hs6),
            )
        ]

    # Trend calculations end in the profile's selected year. The complete
    # annual history remains available separately for full-profile requests.
    values = [
        (int(row["year"]), float(row.get("export_value") or 0))
        for row in annual
        if int(row["year"]) <= actual_year
    ]
    start_year, start_value = values[0] if values else (actual_year, 0.0)
    end_year, end_value = values[-1] if values else (actual_year, latest_value)
    absolute_change = end_value - start_value
    percentage_change = (end_value / start_value - 1) if start_value else None
    peak_year, peak_value = max(values, key=lambda item: item[1]) if values else (actual_year, latest_value)
    low_year, low_value = min(values, key=lambda item: item[1]) if values else (actual_year, latest_value)
    destination_hhi = sum(
        float(row.get("share_of_product_exports") or 0) ** 2
        for row in destinations
    )
    top_destination = destinations[0] if destinations else None

    return {
        "base": base,
        "annual": annual,
        "latest": latest,
        "actual_year": actual_year,
        "latest_value": latest_value,
        "total_exports": total_exports,
        "active_products": active_products,
        "overall_rank": overall_rank,
        "sector": sector,
        "sector_total": sector_total,
        "sector_rank": sector_rank,
        "active_sector_products": active_sector_products,
        "destinations": destinations,
        "market_sets": {year_key: sorted(values_set) for year_key, values_set in market_sets.items()},
        "first_year": first_year,
        "entered_markets": entered_markets,
        "exited_markets": exited_markets,
        "lifetime_markets": lifetime_markets,
        "potential_rows": potential_rows,
        "total_potential": total_potential,
        "market_sizes": market_sizes,
        "start_year": start_year,
        "start_value": start_value,
        "end_year": end_year,
        "end_value": end_value,
        "absolute_change": absolute_change,
        "percentage_change": percentage_change,
        "peak_year": peak_year,
        "peak_value": peak_value,
        "low_year": low_year,
        "low_value": low_value,
        "destination_hhi": destination_hhi,
        "top_destination": top_destination,
    }


def _product_interpretation(profile: dict[str, Any]) -> list[str]:
    """Create compact, factual interpretation without unsupported causality."""
    latest_value = float(profile.get("latest_value") or 0)
    start_value = float(profile.get("start_value") or 0)
    pct_change = profile.get("percentage_change")
    destinations = list(profile.get("destinations") or [])
    top = profile.get("top_destination")
    lines: list[str] = []

    if pct_change is None:
        lines.append(
            "The product has no non-zero starting base for a percentage comparison, so the dollar trend is more informative than a growth percentage."
        )
    elif pct_change > 0.25:
        lines.append(
            f"Export value is materially above its {profile.get('start_year')} level."
        )
    elif pct_change < -0.25:
        lines.append(
            f"Export value is materially below its {profile.get('start_year')} level."
        )
    else:
        lines.append(
            f"Export value is relatively close to its {profile.get('start_year')} level."
        )

    if top:
        top_share = float(top.get("share_of_product_exports") or 0)
        if top_share >= 0.60:
            lines.append(
                f"The product is highly dependent on {top.get('country')}, which accounts for {_pct_text(top_share)} of its exports."
            )
        elif top_share >= 0.35:
            lines.append(
                f"{top.get('country')} is the leading destination at {_pct_text(top_share)}, but other markets also contribute materially."
            )
        else:
            lines.append(
                f"No single destination exceeds 35% of exports; the largest is {top.get('country')} at {_pct_text(top_share)}."
            )
    elif latest_value > 0:
        lines.append("The latest export value is positive, but no destination breakdown is available.")
    else:
        lines.append("The product records no positive exports in the latest dashboard year.")

    if len(destinations) >= 25:
        lines.append(f"The product has broad current reach across {len(destinations):,} destination markets.")
    elif len(destinations) > 0:
        lines.append(f"The product currently reaches {len(destinations):,} destination markets.")
    return lines


def _build_product_profile_answer(
    product: ProductMatch,
    level: str,
    requested_year: int | None,
    show_code: bool,
) -> str:
    year = requested_year or _latest_product_year()
    profile = _product_profile_data(product.hs6, year)
    if not profile:
        return "No product profile is available for that exact dashboard product."

    base = profile["base"]
    latest = profile["latest"]
    actual_year = int(profile["actual_year"])
    name = str(base.get("name") or product.name)
    sector = str(profile.get("sector") or product.sector)
    latest_value = float(profile.get("latest_value") or 0)
    total_share = _safe_ratio(latest_value, float(profile.get("total_exports") or 0))
    sector_share = _safe_ratio(latest_value, float(profile.get("sector_total") or 0))
    destinations = list(profile.get("destinations") or [])
    lines: list[str] = []

    lines.extend([
        "**Current position**",
        f"- **Product:** {name}",
        f"- **Sector:** {sector}",
    ])
    if show_code:
        lines.append(f"- **HS6 code:** {product.hs6}")
    lines.extend([
        f"- **Exports in {actual_year}:** {_money(latest_value)}",
        f"- **Overall product rank:** {('#' + format(profile['overall_rank'], ',')) if profile.get('overall_rank') else 'Not ranked because exports are zero'}"
        + (f" of {int(profile.get('active_products') or 0):,} active products" if profile.get("active_products") else ""),
        f"- **Rank within {sector}:** {('#' + format(profile['sector_rank'], ',')) if profile.get('sector_rank') else 'Not ranked'}"
        + (f" of {int(profile.get('active_sector_products') or 0):,} active sector products" if profile.get("active_sector_products") else ""),
        f"- **Share of Lebanon's industrial exports:** {_pct_text(total_share)}",
        f"- **Share of the {sector} sector:** {_pct_text(sector_share)}",
        f"- **Destination markets in {actual_year}:** {len(destinations):,}",
        f"- **Lifetime destination markets:** {len(profile.get('lifetime_markets') or []):,}",
    ])

    lines.extend(["", f"**Leading destinations in {actual_year}**"])
    if destinations:
        for index, row in enumerate(destinations[:5], 1):
            lines.append(
                f"{index}. **{row.get('country')}** - {_money(row.get('value_usd'))} "
                f"({_pct_text(float(row.get('share_of_product_exports') or 0))} of this product's exports)"
            )
    else:
        lines.append("- No positive destination-level exports are recorded.")

    lines.extend([
        "",
        "**Export trend**",
        f"- **{profile.get('start_year')}:** {_money(profile.get('start_value'))}",
        f"- **{profile.get('end_year')}:** {_money(profile.get('end_value'))}",
        f"- **Dollar change:** {_money(profile.get('absolute_change'))}",
        f"- **Percentage change:** {_pct_text(profile.get('percentage_change'), signed=True)}",
        f"- **Peak year:** {profile.get('peak_year')} at {_money(profile.get('peak_value'))}",
    ])

    interpretations = _product_interpretation(profile)
    if interpretations:
        lines.extend(["", "**What the figures show**"])
        lines.extend(f"- {item}" for item in interpretations)

    if level == "full":
        lines.extend(["", "**Complete annual history**"])
        for row in profile.get("annual") or []:
            lines.append(
                f"- **{int(row.get('year'))}:** {_money(row.get('export_value'))}; "
                f"{int(row.get('n_countries') or 0):,} destination markets"
            )

        first_year = int(profile.get("first_year") or profile.get("start_year") or 2018)
        entered = list(profile.get("entered_markets") or [])
        exited = list(profile.get("exited_markets") or [])
        lines.extend([
            "",
            f"**Destination changes since {first_year}**",
            f"- **Markets present now but not in {first_year}:** {len(entered):,}"
            + (f" - {', '.join(entered[:12])}" if entered else ""),
            f"- **Markets present in {first_year} but not now:** {len(exited):,}"
            + (f" - {', '.join(exited[:12])}" if exited else ""),
            f"- **Destination concentration (HHI):** {float(profile.get('destination_hhi') or 0):.3f}",
        ])

        rca = latest.get("rca")
        pci = latest.get("pci")
        cagr = latest.get("cagr")
        lines.extend(["", "**Competitiveness, complexity and growth indicators**"])
        lines.append(
            f"- **RCA in {actual_year}:** {_format_value(rca, 'rca') if rca is not None else 'Not available'}"
            + (" - above the specialization threshold of 1" if rca is not None and float(rca) >= 1 else "")
        )
        lines.append(
            f"- **PCI:** {_format_value(pci, 'pci') if pci is not None else 'Not available'}"
        )
        lines.append(
            f"- **CAGR:** {_format_value(cagr, 'cagr') if cagr is not None else 'Not calculable'}"
        )
        if latest.get("trajectory"):
            lines.append(f"- **Trajectory:** {latest.get('trajectory')}")

        potential_rows = list(profile.get("potential_rows") or [])
        lines.extend(["", "**Recorded unrealized potential**"])
        lines.append(
            f"- **Total recorded potential:** {_money(profile.get('total_potential'))}"
        )
        lines.append(
            f"- **Markets with recorded potential:** {len(potential_rows):,}"
        )
        for index, row in enumerate(potential_rows[:5], 1):
            lines.append(
                f"{index}. **{row.get('country')}** - {_money(row.get('unrealized_potential_usd'))}"
            )
        if not potential_rows:
            lines.append("- No positive product-country potential pair is recorded.")

        market_sizes = list(profile.get("market_sizes") or [])
        lines.extend(["", "**Largest recorded import markets for this product**"])
        if market_sizes:
            for index, row in enumerate(market_sizes[:5], 1):
                penetration = row.get("market_penetration")
                penetration_text = (
                    f"; Lebanon's share in that observation year: {_pct_text(float(penetration))}"
                    if penetration is not None else ""
                )
                lines.append(
                    f"{index}. **{row.get('country')}** - {_money(row.get('market_size_usd'))} "
                    f"in {int(row.get('year'))}{penetration_text}"
                )
        else:
            lines.append("- No market-size observations are available for this product.")

        lines.extend([
            "",
            "**How to use the complete profile**",
            "- Export values and destination shares describe recorded trade, not profitability.",
            "- Market-size observations may use a year earlier than the latest export year.",
            "- Unrealized potential is an opportunity estimate, not a guaranteed forecast.",
        ])

    return "\n".join(lines)


def _product_profile_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    level = _product_profile_level(parsed)
    if level is None:
        return None
    product = parsed.products[0]
    requested_year = parsed.years[-1] if parsed.years else None
    answer = _build_product_profile_answer(
        product,
        level=level,
        requested_year=requested_year,
        show_code=parsed.explicit_code,
    )
    entities = {
        "products": [product.hs6],
        "hs6": product.hs6,
        "product_names": [product.name],
        "year": requested_year or _latest_product_year(),
        "metric": "product_profile",
        "profile_level": level,
    }
    title = (
        f"Complete profile for {product.name}"
        if level == "full"
        else f"Product profile for {product.name}"
    )
    return QueryPlan(
        title,
        kind="direct",
        direct_answer=answer,
        entities=entities,
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Query planning
# ---------------------------------------------------------------------------

def _year(parsed: ParsedQuestion, market_size: bool = False) -> int:
    if parsed.years:
        return parsed.years[-1]
    return 2024 if market_size else 2025


def _years_condition(column: str, parsed: ParsedQuestion) -> str:
    if not parsed.years:
        return ""
    if len(parsed.years) == 1:
        return f" AND {column}={parsed.years[0]}"
    lo, hi = min(parsed.years), max(parsed.years)
    return f" AND {column} BETWEEN {lo} AND {hi}"


def _metric_order(parsed: ParsedQuestion, scope: str) -> tuple[str, str]:
    direction = "ASC" if parsed.ascending else "DESC"
    if "market_size" in parsed.measures:
        return "market_size_usd", direction
    if "potential" in parsed.measures:
        return "unrealized_potential_usd", direction
    if "rca" in parsed.measures:
        return "rca", direction
    if "pci" in parsed.measures or ("complexity" in parsed.measures and scope in {"product", "sector"}):
        return "pci" if scope == "product" else "pci_avg", direction
    if "expy" in parsed.measures or ("complexity" in parsed.measures and scope == "market"):
        return "expy", direction
    if "hhi" in parsed.measures:
        # "most diversified" means lowest HHI; "most concentrated" means highest.
        if "diversif" in parsed.norm and not parsed.ascending:
            direction = "ASC"
        if "concentrat" in parsed.norm and not parsed.ascending:
            direction = "DESC"
        return "hhi", direction
    if "cagr" in parsed.measures or "growth" in parsed.measures:
        return "cagr", direction
    if "n_products" in parsed.measures:
        return "n_products", direction
    return "export_value", direction


def _threshold_sql(parsed: ParsedQuestion, aliases: dict[str, str] | None = None) -> str:
    aliases = aliases or {}
    conditions: list[str] = []
    for column, operator, value in parsed.filters:
        actual = aliases.get(column, column)
        if operator not in {">", "<", ">=", "<=", "="}:
            continue
        conditions.append(f"{actual} {operator} {float(value)}")
    return (" AND " + " AND ".join(conditions)) if conditions else ""


def _definition_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    if parsed.operation != "definition" or parsed.products or parsed.markets or parsed.sectors or parsed.years:
        return None
    if any(term in parsed.norm for term in ("total", "amount", "value", "how much", "rank", "top", "largest")):
        return None
    q = parsed.norm
    keys = [
        ("real export", "real exports"), ("market size", "market size"),
        ("unrealized", "unrealized potential"), ("potential", "unrealized potential"),
        ("rca", "rca"), ("pci", "pci"), ("hhi", "hhi"), ("expy", "expy"),
        ("cagr", "cagr"), ("complexity", "complexity"),
    ]
    for needle, key in keys:
        if needle in q:
            return QueryPlan(f"Definition of {key.upper() if len(key) <= 5 else key.title()}", kind="direct", direct_answer=METRIC_DEFINITIONS[key], confidence=1.0, entities={"metric": key})
    return None


def _coverage_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    q = parsed.norm
    if any(term in q for term in ("data coverage", "what data", "datasets", "data source", "methodology", "years covered", "available years", "year range")):
        direct = (
            "The dashboard covers Lebanon's industrial exports from 2018 through 2025. Its queryable data include annual totals, 882 HS6 products, 187 destination markets, 16 sectors, 83,676 product-market-year export observations, 16,693 unrealized-potential pairs, 307,368 product-market-size observations, RCA, PCI, HHI, EXPY, growth, entry/exit, performance classifications and source metadata."
        )
        sql = "SELECT key,value FROM meta ORDER BY key"
        return QueryPlan("Dashboard data coverage", direct_answer=direct, confidence=1.0, kind="direct", entities={"metric": "coverage"})
    return None


def _total_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    q = parsed.norm
    no_entity = not (parsed.products or parsed.markets or parsed.sectors or parsed.continents)
    total_words = any(term in q for term in ("total exports", "lebanon exports", "lebanon s exports", "overall exports", "industrial exports", "exports of lebanon"))
    if not no_entity or (parsed.group and parsed.group not in {"year", "continent"}):
        return None
    if "potential" in parsed.measures:
        if parsed.group == "continent":
            sql = "SELECT m.continent,SUM(u.value_usd) AS unrealized_potential_usd,COUNT(DISTINCT u.country) AS n_markets FROM up_pairs u JOIN markets_master m ON m.country=u.country GROUP BY m.continent ORDER BY unrealized_potential_usd DESC"
            return QueryPlan("Unrealized export potential by continent", sql, entities={"metric": "potential", "group": "continent"})
        return QueryPlan("Total unrealized export potential", "SELECT CAST(value AS REAL) AS unrealized_potential_usd FROM up_totals WHERE key='exact_pair_total_usd' LIMIT 1", entities={"metric": "potential"})
    if parsed.group == "continent":
        year = _year(parsed)
        return QueryPlan(f"Exports by continent in {year}", f"SELECT continent,SUM(export_value) AS export_value,COUNT(DISTINCT country) AS n_markets FROM market_year WHERE year={year} GROUP BY continent ORDER BY export_value DESC", entities={"year": year, "metric": "exports", "group": "continent"})
    if parsed.operation == "count":
        if "sector" in parsed.norm:
            return QueryPlan("Number of sectors in the dashboard", "SELECT COUNT(*) AS count_value FROM sectors_master", entities={"metric": "count"})
        if "market" in parsed.norm or "country" in parsed.norm or "destination" in parsed.norm:
            return QueryPlan("Number of destination markets in the dashboard", "SELECT COUNT(*) AS count_value FROM markets_master", entities={"metric": "count"})
        if "product" in parsed.norm:
            return QueryPlan("Number of products in the dashboard", "SELECT COUNT(*) AS count_value FROM products_master", entities={"metric": "count"})
    if parsed.operation == "trend" or parsed.group == "year":
        cols = "year,total_exports_usd,real_exports_2018_usd,active_products,active_markets"
        condition = ""
        if parsed.years:
            condition = f" WHERE year BETWEEN {min(parsed.years)} AND {max(parsed.years)}"
        return QueryPlan("Lebanon industrial export trend", f"SELECT {cols} FROM export_overview{condition} ORDER BY year", entities={"years": parsed.years or list(range(2018, 2026)), "metric": "exports"})
    if total_words or parsed.operation in {"lookup", "count", "compare"}:
        if len(parsed.years) >= 2:
            values = ",".join(str(y) for y in parsed.years)
            return QueryPlan("Lebanon industrial export comparison", f"SELECT year,total_exports_usd,real_exports_2018_usd,active_products,active_markets FROM export_overview WHERE year IN ({values}) ORDER BY year", entities={"years": parsed.years, "metric": "exports"})
        year = _year(parsed)
        return QueryPlan(f"Lebanon industrial exports in {year}", f"SELECT year,total_exports_usd,real_exports_2018_usd,active_products,active_markets FROM export_overview WHERE year={year}", entities={"year": year, "metric": "exports"})
    return None


def _pair_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    if not parsed.products or not parsed.markets:
        return None
    product = parsed.products[0]
    market = parsed.markets[0]
    hs = _sql_quote(product.hs6)
    mk = _sql_quote(market)
    entities = {"hs6": product.hs6, "market": market, "product_names": [product.name], "metric": sorted(parsed.measures)[0]}

    if "penetration" in parsed.measures:
        requested_year = parsed.years[-1] if parsed.years else None
        with sqlite3.connect(_database_path()) as conn:
            row = conn.execute(
                "SELECT MAX(year) FROM market_size_hs6 WHERE hs6=? AND country=?" + (" AND year=?" if requested_year else ""),
                (product.hs6, market, requested_year) if requested_year else (product.hs6, market),
            ).fetchone()
            exact_year = int(row[0]) if row and row[0] else None
            if exact_year is None:
                fallback = conn.execute("SELECT MAX(year) FROM market_size_hs6 WHERE hs6=? AND country=?", (product.hs6, market)).fetchone()
                exact_year = int(fallback[0]) if fallback and fallback[0] else None
        if exact_year is None:
            return QueryPlan(
                f"Market penetration for {product.name} in {market}",
                kind="direct",
                direct_answer="No market-size observation is available for this product-country pair, so market penetration cannot be calculated from the dashboard.",
                entities=entities,
            )
        notes = ()
        if requested_year and exact_year != requested_year:
            notes = (f"Market-size data for {requested_year} are unavailable for this exact product-country pair. The calculation below uses the latest available year, {exact_year}.",)
        sql = (
            "WITH e AS (SELECT COALESCE(SUM(value_usd),0) AS exports FROM product_market_year "
            f"WHERE hs6={hs} AND country={mk} AND year={exact_year}), "
            "m AS (SELECT MAX(market_size_usd) AS market_size FROM market_size_hs6 "
            f"WHERE hs6={hs} AND country={mk} AND year={exact_year}) "
            f"SELECT {_sql_quote(product.name)} AS product_name,{mk} AS country,{exact_year} AS year,e.exports AS value_usd,m.market_size AS market_size_usd,"
            "CASE WHEN m.market_size>0 THEN e.exports/m.market_size ELSE NULL END AS market_penetration FROM e CROSS JOIN m"
        )
        return QueryPlan(f"Lebanon's market penetration for {product.name} in {market}", sql, entities=entities, notes=notes)
    if "potential" in parsed.measures and any(term in parsed.norm for term in ("actual", "gap", "versus", "compare", "realized")):
        year = _year(parsed)
        sql = (
            "WITH e AS (SELECT COALESCE(SUM(value_usd),0) AS actual_exports FROM product_market_year "
            f"WHERE hs6={hs} AND country={mk} AND year={year}), "
            "u AS (SELECT COALESCE(MAX(value_usd),0) AS unrealized_potential FROM up_pairs "
            f"WHERE CAST(hs6 AS INTEGER)=CAST({hs} AS INTEGER) AND country={mk}) "
            f"SELECT {_sql_quote(product.name)} AS product_name,{mk} AS country,{year} AS year,e.actual_exports AS export_value,u.unrealized_potential AS unrealized_potential_usd,"
            "e.actual_exports+u.unrealized_potential AS total_addressable_exports FROM e CROSS JOIN u"
        )
        return QueryPlan(f"Actual exports and unrealized potential for {product.name} in {market}", sql, entities=entities)
    if "market_size" in parsed.measures:
        requested_year = parsed.years[-1] if parsed.years else None
        with sqlite3.connect(_database_path()) as conn:
            if requested_year:
                row = conn.execute("SELECT MAX(year) FROM market_size_hs6 WHERE hs6=? AND country=? AND year=?", (product.hs6, market, requested_year)).fetchone()
                exact_year = int(row[0]) if row and row[0] else None
            else:
                exact_year = None
            if exact_year is None:
                row = conn.execute("SELECT MAX(year) FROM market_size_hs6 WHERE hs6=? AND country=?", (product.hs6, market)).fetchone()
                exact_year = int(row[0]) if row and row[0] else None
        if exact_year is None:
            return QueryPlan(
                f"{product.name} market size in {market}",
                kind="direct",
                direct_answer="No market-size observation is available for this exact product-country pair in the dashboard.",
                entities=entities,
            )
        notes = ()
        if requested_year and exact_year != requested_year:
            notes = (f"Market-size data for {requested_year} are unavailable for this exact product-country pair. The latest available observation is {exact_year}.",)
        sql = (
            "SELECT m.country,m.year,p.name AS product_name,m.market_size_usd "
            "FROM market_size_hs6 m JOIN products_master p ON CAST(p.hs6 AS TEXT)=m.hs6 "
            f"WHERE m.hs6={hs} AND m.country={mk} AND m.year={exact_year} LIMIT 1"
        )
        return QueryPlan(f"{product.name} market size in {market}", sql, entities=entities, notes=notes)
    if "potential" in parsed.measures:
        sql = (
            "SELECT p.name AS product_name,u.country,u.value_usd AS unrealized_potential_usd "
            "FROM up_pairs u JOIN products_master p ON CAST(p.hs6 AS INTEGER)=CAST(u.hs6 AS INTEGER) "
            f"WHERE u.hs6={hs} AND u.country={mk} LIMIT 1"
        )
        return QueryPlan(f"Unrealized potential for {product.name} in {market}", sql, entities=entities)
    if parsed.operation == "trend" or len(parsed.years) >= 2:
        start, end = (min(parsed.years), max(parsed.years)) if parsed.years else (2018, 2025)
        sql = (
            "WITH years AS (SELECT year FROM totals_by_year WHERE year BETWEEN " + str(start) + " AND " + str(end) + "), "
            "series AS (SELECT year,SUM(value_usd) AS value_usd FROM product_market_year "
            f"WHERE hs6={hs} AND country={mk} AND year BETWEEN {start} AND {end} GROUP BY year) "
            f"SELECT y.year,{_sql_quote(product.name)} AS product_name,{mk} AS country,COALESCE(s.value_usd,0) AS value_usd "
            "FROM years y LEFT JOIN series s USING(year) ORDER BY y.year"
        )
        return QueryPlan(f"{product.name} exports to {market}, {start}-{end}", sql, entities=entities)
    year = _year(parsed)
    if parsed.operation == "rank" or "rank" in parsed.norm:
        if "in market" in parsed.norm or "among products" in parsed.norm or "within" in parsed.norm:
            sql = (
                "WITH ranked AS (SELECT hs6,product_name,country,value_usd,share_of_market_exports,"
                "RANK() OVER (ORDER BY value_usd DESC) AS export_rank FROM product_market_share "
                f"WHERE country={mk} AND year={year}) SELECT product_name,country,value_usd,share_of_market_exports,export_rank "
                f"FROM ranked WHERE hs6={hs} LIMIT 1"
            )
        else:
            sql = (
                "WITH ranked AS (SELECT country,value_usd,share_of_product_exports,"
                "RANK() OVER (ORDER BY value_usd DESC) AS destination_rank FROM product_market_share "
                f"WHERE hs6={hs} AND year={year}) SELECT { _sql_quote(product.name) } AS product_name,country,value_usd,share_of_product_exports,destination_rank "
                f"FROM ranked WHERE country={mk} LIMIT 1"
            )
        return QueryPlan(f"Rank of {product.name} in {market}, {year}", sql, entities=entities)
    sql = (
        "SELECT product_name,country,year,value_usd,share_of_product_exports,share_of_market_exports,rca,pci "
        f"FROM product_market_share WHERE hs6={hs} AND country={mk} AND year={year} LIMIT 1"
    )
    return QueryPlan(f"{product.name} exports to {market} in {year}", sql, entities=entities)


def _sector_market_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    if not parsed.sectors or not parsed.markets or parsed.products:
        return None
    sector, market = parsed.sectors[0], parsed.markets[0]
    sec, mk = _sql_quote(sector), _sql_quote(market)
    entities = {"sector": sector, "market": market, "metric": sorted(parsed.measures)[0]}
    if parsed.group == "product" or re.search(r"\b(?:which|what|top|list) products\b", parsed.norm):
        year = _year(parsed)
        order_col, direction = _metric_order(parsed, "product")
        order_map = {"export_value": "value_usd", "unrealized_potential_usd": "value_usd", "cagr": "value_usd"}
        order_col = order_map.get(order_col, order_col)
        sql = (
            "SELECT product_name,sector,country,value_usd,rca,pci FROM product_market_year "
            f"WHERE sector={sec} AND country={mk} AND year={year} AND value_usd>0 "
            f"ORDER BY {order_col} {direction} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Products exported by {sector} to {market} in {year}", sql, entities=entities)
    if parsed.operation == "trend":
        condition = _years_condition("year", parsed)
        sql = (
            "SELECT year,SUM(value_usd) AS export_value,COUNT(DISTINCT hs6) AS n_products "
            f"FROM product_market_year WHERE sector={sec} AND country={mk}{condition} GROUP BY year ORDER BY year"
        )
        return QueryPlan(f"{sector} exports to {market}", sql, entities=entities)
    year = _year(parsed)
    sql = (
        "WITH scoped AS (SELECT SUM(value_usd) AS export_value,COUNT(DISTINCT hs6) AS n_products "
        f"FROM product_market_year WHERE sector={sec} AND country={mk} AND year={year}), "
        "market_total AS (SELECT SUM(value_usd) AS total FROM product_market_year "
        f"WHERE country={mk} AND year={year}) "
        f"SELECT {sec} AS sector,{mk} AS country,{year} AS year,s.export_value,s.n_products,"
        "CASE WHEN m.total>0 THEN s.export_value/m.total ELSE 0 END AS share_of_market_exports "
        "FROM scoped s CROSS JOIN market_total m"
    )
    return QueryPlan(f"{sector} exports to {market} in {year}", sql, entities=entities)


def _product_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    if not parsed.products:
        return None
    products = parsed.products
    entities = {
        "products": [p.hs6 for p in products],
        "hs6": products[-1].hs6,
        "product_names": [p.name for p in products],
        "metric": sorted(parsed.measures)[0] if parsed.measures else "exports",
    }
    if len(products) == 1 and len(parsed.years) >= 2:
        product = products[0]
        values = ",".join(str(y) for y in parsed.years)
        sql = f"SELECT year,name AS product_name,sector,export_value,rca,pci,n_countries,unrealized_potential_usd,cagr,trajectory FROM product_year WHERE hs6={_sql_quote(product.hs6)} AND year IN ({values}) ORDER BY year"
        return QueryPlan(f"{product.name} comparison across selected years", sql, entities=entities)
    if len(products) >= 2 or (parsed.operation == "compare" and len(products) >= 2):
        year = _year(parsed)
        values = ",".join(_sql_quote(p.hs6) for p in products[:20])
        sql = (
            "SELECT name AS product_name,sector,year,export_value,rca,pci,n_countries,unrealized_potential_usd,cagr,trajectory "
            f"FROM product_year WHERE year={year} AND hs6 IN ({values}) ORDER BY export_value DESC"
        )
        return QueryPlan(f"Product comparison in {year}", sql, entities=entities)
    product = products[0]
    hs = _sql_quote(product.hs6)

    if parsed.operation == "count" or "n_markets" in parsed.measures:
        year = _year(parsed)
        sql = f"SELECT {_sql_quote(product.name)} AS product_name,{year} AS year,COUNT(DISTINCT country) AS n_markets FROM product_market_year WHERE hs6={hs} AND year={year} AND value_usd>0"
        return QueryPlan(f"Markets reached by {product.name} in {year}", sql, entities=entities)
    if parsed.operation == "share" and not parsed.markets:
        year = _year(parsed)
        sql = (
            f"WITH p AS (SELECT export_value FROM product_year WHERE hs6={hs} AND year={year}), "
            f"t AS (SELECT total_exports_usd FROM export_overview WHERE year={year}) "
            f"SELECT {_sql_quote(product.name)} AS product_name,{year} AS year,p.export_value,CASE WHEN t.total_exports_usd>0 THEN p.export_value/t.total_exports_usd ELSE 0 END AS share_of_total_exports FROM p CROSS JOIN t"
        )
        return QueryPlan(f"Share of total exports accounted for by {product.name} in {year}", sql, entities=entities)
    if "market_size" in parsed.measures:
        if parsed.years:
            year = parsed.years[-1]
            sql = (
                "SELECT m.country,m.year,p.name AS product_name,m.market_size_usd "
                "FROM market_size_hs6 m JOIN products_master p ON CAST(p.hs6 AS TEXT)=m.hs6 "
                f"WHERE m.hs6={hs} AND m.year={year} ORDER BY m.market_size_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
            )
        else:
            sql = (
                "WITH latest AS (SELECT country,MAX(year) AS year FROM market_size_hs6 WHERE hs6=" + hs + " GROUP BY country) "
                "SELECT m.country,m.year,p.name AS product_name,m.market_size_usd FROM market_size_hs6 m "
                "JOIN latest l ON l.country=m.country AND l.year=m.year "
                "JOIN products_master p ON CAST(p.hs6 AS TEXT)=m.hs6 "
                f"WHERE m.hs6={hs} ORDER BY m.market_size_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
            )
        return QueryPlan(f"Largest recorded markets for {product.name}", sql, entities=entities)
    if "potential" in parsed.measures and (parsed.operation == "rank" or parsed.group == "market"):
        sql = (
            "SELECT u.country,p.name AS product_name,u.value_usd AS unrealized_potential_usd "
            "FROM up_pairs u JOIN products_master p ON CAST(p.hs6 AS INTEGER)=CAST(u.hs6 AS INTEGER) "
            f"WHERE CAST(u.hs6 AS INTEGER)=CAST({hs} AS INTEGER) ORDER BY u.value_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Unrealized potential for {product.name}", sql, entities=entities)
    if parsed.group in {"market", "continent"} or parsed.norm.startswith("where ") or re.search(r"\b(?:where|which|what|top|list).*(?:markets|countries|destinations)\b", parsed.norm):
        year = _year(parsed)
        if parsed.group == "continent":
            order_col, direction = _metric_order(parsed, "market")
            order_col = "export_value" if order_col not in {"export_value"} else order_col
            sql = (
                "SELECT continent,SUM(value_usd) AS export_value,COUNT(DISTINCT country) AS n_markets "
                f"FROM product_market_year WHERE hs6={hs} AND year={year} GROUP BY continent ORDER BY {order_col} {direction} LIMIT {parsed.limit}"
            )
        else:
            order_col, direction = _metric_order(parsed, "market")
            order_map = {"export_value": "value_usd", "hhi": "value_usd", "expy": "value_usd", "cagr": "value_usd", "unrealized_potential_usd": "value_usd"}
            order_col = order_map.get(order_col, order_col)
            sql = (
                "SELECT country,value_usd,share_of_product_exports,share_of_market_exports,rca,pci "
                f"FROM product_market_share WHERE hs6={hs} AND year={year} AND value_usd>0 ORDER BY {order_col} {direction} LIMIT {parsed.limit}"
            )
        return QueryPlan(f"Destinations for {product.name} in {year}", sql, entities=entities)
    if parsed.operation == "trend":
        condition = _years_condition("year", parsed)
        sql = (
            "SELECT year,name AS product_name,sector,export_value,rca,pci,n_countries,unrealized_potential_usd,cagr,trajectory "
            f"FROM product_year WHERE hs6={hs}{condition} ORDER BY year"
        )
        return QueryPlan(f"Annual exports of {product.name}", sql, entities=entities)
    if parsed.operation == "rank":
        year = _year(parsed)
        order_col, direction = _metric_order(parsed, "product")
        sql = (
            f"WITH ranked AS (SELECT hs6,name AS product_name,sector,export_value,rca,pci,n_countries,unrealized_potential_usd,cagr,"
            f"RANK() OVER (ORDER BY {order_col} {direction}) AS metric_rank FROM product_year WHERE year={year} AND {order_col} IS NOT NULL AND export_value>0) "
            f"SELECT product_name,sector,{year} AS year,export_value,rca,pci,n_countries,unrealized_potential_usd,cagr,metric_rank FROM ranked WHERE hs6={hs} LIMIT 1"
        )
        return QueryPlan(f"Rank of {product.name} in {year}", sql, entities=entities)
    year = _year(parsed)
    sql = (
        "SELECT name AS product_name,sector,year,export_value,rca,pci,n_countries,unrealized_potential_usd,cagr,growth,trajectory "
        f"FROM product_year WHERE hs6={hs} AND year={year} LIMIT 1"
    )
    return QueryPlan(f"{product.name} in {year}", sql, entities=entities)


def _market_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    if not parsed.markets:
        return None
    markets = parsed.markets
    entities = {"markets": markets, "market": markets[-1], "metric": sorted(parsed.measures)[0]}
    if len(markets) == 1 and len(parsed.years) >= 2:
        market = markets[0]
        values = ",".join(str(y) for y in parsed.years)
        sql = f"SELECT country,continent,year,export_value,n_products,expy,hhi,rca,unrealized_potential_usd,status,priority,cagr FROM market_year WHERE country={_sql_quote(market)} AND year IN ({values}) ORDER BY year"
        return QueryPlan(f"{market} comparison across selected years", sql, entities=entities)
    if len(markets) >= 2 or (parsed.operation == "compare" and len(markets) >= 2):
        year = _year(parsed)
        values = ",".join(_sql_quote(m) for m in markets[:30])
        sql = (
            "SELECT country,continent,year,export_value,n_products,expy,hhi,rca,unrealized_potential_usd,status,priority,cagr "
            f"FROM market_year WHERE year={year} AND country IN ({values}) ORDER BY export_value DESC"
        )
        return QueryPlan(f"Market comparison in {year}", sql, entities=entities)
    market = markets[0]
    mk = _sql_quote(market)
    if "potential" in parsed.measures and "exports" in parsed.measures and any(term in parsed.norm for term in ("prioritize", "priority", "based on", "screen", "opportunity")):
        year = _year(parsed)
        sql = (
            "WITH actual AS (SELECT hs6,product_name,sector,SUM(value_usd) AS export_value FROM product_market_year "
            f"WHERE country={mk} AND year={year} GROUP BY hs6,product_name,sector), "
            "potential AS (SELECT printf('%06d',CAST(hs6 AS INTEGER)) AS hs6,MAX(value_usd) AS unrealized_potential_usd FROM up_pairs "
            f"WHERE country={mk} GROUP BY CAST(hs6 AS INTEGER)), "
            "keys AS (SELECT hs6 FROM actual UNION SELECT hs6 FROM potential) "
            "SELECT COALESCE(a.product_name,p.name) AS product_name,COALESCE(a.sector,p.sector) AS sector,"
            "COALESCE(a.export_value,0) AS export_value,COALESCE(u.unrealized_potential_usd,0) AS unrealized_potential_usd,"
            "COALESCE(a.export_value,0)+COALESCE(u.unrealized_potential_usd,0) AS total_addressable_exports "
            "FROM keys k LEFT JOIN actual a USING(hs6) LEFT JOIN potential u USING(hs6) "
            "LEFT JOIN products_master p ON CAST(p.hs6 AS TEXT)=k.hs6 "
            "WHERE COALESCE(a.export_value,0)+COALESCE(u.unrealized_potential_usd,0)>0 "
            "ORDER BY total_addressable_exports DESC LIMIT " + str(parsed.limit)
        )
        entities.update({"year": year, "metric": "export_opportunity_screen"})
        return QueryPlan(f"Product opportunity screen for {market} in {year}", sql, entities=entities, notes=("Products are ranked by current exports plus recorded unrealized potential. This is a transparent screening measure, not a forecast.",))
    if "similarity" in parsed.measures:
        direct = _similar_market_answer(market, parsed.limit)
        if direct:
            return QueryPlan(f"Markets similar to {market}", kind="direct", direct_answer=direct.replace(f"**Markets similar to {market}**\n\n", ""), confidence=1.0, entities=entities)
        return QueryPlan(f"Markets similar to {market}", f"SELECT rank,country,iso3,score,continent,exports_2025 AS export_value FROM similar_markets WHERE country={mk} ORDER BY rank LIMIT {parsed.limit}", entities=entities)
    if "market_size" in parsed.measures and (parsed.group == "product" or "product" in parsed.norm):
        if parsed.years:
            year_condition = f"m.year={parsed.years[-1]}"
        else:
            year_condition = f"m.year=(SELECT MAX(year) FROM market_size_hs6 WHERE country={mk})"
        sql = (
            "SELECT p.name AS product_name,m.year,m.market_size_usd FROM market_size_hs6 m "
            "JOIN products_master p ON CAST(p.hs6 AS TEXT)=m.hs6 "
            f"WHERE m.country={mk} AND {year_condition} ORDER BY m.market_size_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Largest product markets in {market}", sql, entities=entities)
    if "potential" in parsed.measures and (parsed.group == "product" or "product" in parsed.norm):
        sql = (
            "SELECT p.name AS product_name,p.sector,u.value_usd AS unrealized_potential_usd FROM up_pairs u "
            "JOIN products_master p ON CAST(p.hs6 AS INTEGER)=CAST(u.hs6 AS INTEGER) "
            f"WHERE u.country={mk} ORDER BY u.value_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Products with unrealized potential in {market}", sql, entities=entities)
    if parsed.operation == "count" and (parsed.group == "sector" or "sectors" in parsed.norm):
        year = _year(parsed)
        sql = f"SELECT {mk} AS country,{year} AS year,COUNT(DISTINCT sector) AS count_value FROM product_market_year WHERE country={mk} AND year={year} AND value_usd>0"
        return QueryPlan(f"Sectors exported to {market} in {year}", sql, entities=entities)
    if parsed.operation == "count" or "n_products" in parsed.measures:
        year = _year(parsed)
        sql = f"SELECT country,year,n_products FROM market_year WHERE country={mk} AND year={year} LIMIT 1"
        return QueryPlan(f"Products exported to {market} in {year}", sql, entities=entities)
    if parsed.operation == "share":
        year = _year(parsed)
        sql = (
            f"WITH m AS (SELECT export_value FROM market_year WHERE country={mk} AND year={year}), "
            f"t AS (SELECT total_exports_usd FROM export_overview WHERE year={year}) "
            f"SELECT {mk} AS country,{year} AS year,m.export_value,CASE WHEN t.total_exports_usd>0 THEN m.export_value/t.total_exports_usd ELSE 0 END AS share_of_total_exports FROM m CROSS JOIN t"
        )
        return QueryPlan(f"Share of total exports going to {market} in {year}", sql, entities=entities)
    if parsed.group == "product" or re.search(r"\b(?:what|which|top|list).*(?:products|goods|items)\b", parsed.norm):
        year = _year(parsed)
        order_col, direction = _metric_order(parsed, "product")
        order_map = {"export_value": "value_usd", "unrealized_potential_usd": "value_usd", "cagr": "value_usd"}
        order_col = order_map.get(order_col, order_col)
        selected = ["product_name", "sector", "value_usd", "share_of_market_exports"]
        if "rca" in parsed.measures:
            selected.append("rca")
        if "pci" in parsed.measures or "complexity" in parsed.measures:
            selected.append("pci")
        sql = (
            f"SELECT {','.join(selected)} FROM product_market_share "
            f"WHERE country={mk} AND year={year} AND value_usd>0{_threshold_sql(parsed)} "
            f"ORDER BY {order_col} {direction} LIMIT {parsed.limit}"
        )
        title = (
            f"Top products Lebanon exported to {market} in {year}"
            if parsed.operation == "rank"
            else f"Products Lebanon exported to {market} in {year}"
        )
        product_entities = dict(entities)
        product_entities.update({"group": "product", "year": year, "query_type": "market_product_ranking"})
        return QueryPlan(title, sql, entities=product_entities)
    if parsed.group == "sector" or re.search(r"\b(?:what|which|top|list).*(?:sectors|industries)\b", parsed.norm):
        year = _year(parsed)
        sql = (
            "SELECT sector,SUM(value_usd) AS export_value,COUNT(DISTINCT hs6) AS n_products "
            f"FROM product_market_year WHERE country={mk} AND year={year} GROUP BY sector ORDER BY export_value {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Sectors exported to {market} in {year}", sql, entities=entities)
    if "potential" in parsed.measures and parsed.group == "product":
        sql = (
            "SELECT p.name AS product_name,p.sector,u.value_usd AS unrealized_potential_usd "
            "FROM up_pairs u JOIN products_master p ON CAST(p.hs6 AS INTEGER)=CAST(u.hs6 AS INTEGER) "
            f"WHERE u.country={mk} ORDER BY u.value_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Unrealized product opportunities in {market}", sql, entities=entities)
    if parsed.operation == "trend":
        condition = _years_condition("year", parsed)
        sql = (
            "SELECT country,year,export_value,n_products,expy,hhi,rca,unrealized_potential_usd,status,priority,cagr "
            f"FROM market_year WHERE country={mk}{condition} ORDER BY year"
        )
        return QueryPlan(f"Annual exports to {market}", sql, entities=entities)
    if parsed.operation == "rank":
        year = _year(parsed)
        order_col, direction = _metric_order(parsed, "market")
        base_columns = _requested_market_columns(parsed, include_rank=False)
        rank_source = list(dict.fromkeys(base_columns + [order_col]))
        output_columns = list(dict.fromkeys(base_columns + ["metric_rank"]))
        sql = (
            f"WITH ranked AS (SELECT {','.join(rank_source)},"
            f"RANK() OVER (ORDER BY {order_col} {direction}) AS metric_rank FROM market_year WHERE year={year} AND {order_col} IS NOT NULL) "
            f"SELECT {','.join(output_columns)} FROM ranked WHERE country={mk} LIMIT 1"
        )
        return QueryPlan(f"Rank of {market} in {year}", sql, entities=entities)
    year = _year(parsed)
    lookup_columns = _requested_market_columns(parsed, include_rank=False)
    sql = f"SELECT {','.join(lookup_columns)} FROM market_year WHERE country={mk} AND year={year} LIMIT 1"
    return QueryPlan(f"Lebanon's exports to {market} in {year}", sql, entities=entities)


def _sector_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    if not parsed.sectors:
        return None
    sectors = parsed.sectors
    entities = {"sectors": sectors, "sector": sectors[-1], "metric": sorted(parsed.measures)[0]}
    if len(sectors) == 1 and len(parsed.years) >= 2:
        sector = sectors[0]
        values = ",".join(str(y) for y in parsed.years)
        sql = f"SELECT sector,year,export_value,share,rca,pci_avg,n_products_hs6,n_products_hs4,unrealized_potential_usd,cagr FROM sector_year WHERE sector={_sql_quote(sector)} AND year IN ({values}) ORDER BY year"
        return QueryPlan(f"{sector} comparison across selected years", sql, entities=entities)
    if len(sectors) >= 2 or (parsed.operation == "compare" and len(sectors) >= 2):
        year = _year(parsed)
        values = ",".join(_sql_quote(s) for s in sectors[:20])
        sql = (
            "SELECT sector,year,export_value,share,rca,pci_avg,n_products_hs6,n_products_hs4,unrealized_potential_usd,cagr "
            f"FROM sector_year WHERE year={year} AND sector IN ({values}) ORDER BY export_value DESC"
        )
        return QueryPlan(f"Sector comparison in {year}", sql, entities=entities)
    sector = sectors[0]
    sec = _sql_quote(sector)
    if parsed.operation == "count" and (parsed.group == "market" or "markets" in parsed.norm or "countries" in parsed.norm):
        year = _year(parsed)
        sql = f"SELECT {sec} AS sector,{year} AS year,COUNT(DISTINCT country) AS n_markets FROM product_market_year WHERE sector={sec} AND year={year} AND value_usd>0"
        return QueryPlan(f"Markets reached by {sector} in {year}", sql, entities=entities)
    if parsed.operation == "count" or "n_products" in parsed.measures:
        year = _year(parsed)
        sql = f"SELECT sector,year,n_products_hs6,n_products_hs4 FROM sector_year WHERE sector={sec} AND year={year} LIMIT 1"
        return QueryPlan(f"Products in {sector} in {year}", sql, entities=entities)
    if parsed.operation == "share":
        year = _year(parsed)
        sql = f"SELECT sector,year,export_value,share AS share_of_total_exports FROM sector_year WHERE sector={sec} AND year={year} LIMIT 1"
        return QueryPlan(f"Share of total exports accounted for by {sector} in {year}", sql, entities=entities)
    if "potential" in parsed.measures and (parsed.group == "market" or "markets" in parsed.norm or "countries" in parsed.norm or "destinations" in parsed.norm):
        sql = (
            "SELECT u.country,SUM(u.value_usd) AS unrealized_potential_usd,COUNT(DISTINCT u.hs6) AS n_products FROM up_pairs u "
            "JOIN products_master p ON CAST(p.hs6 AS INTEGER)=CAST(u.hs6 AS INTEGER) "
            f"WHERE p.sector={sec} GROUP BY u.country ORDER BY unrealized_potential_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Markets with unrealized potential for {sector}", sql, entities=entities)
    if "potential" in parsed.measures and (parsed.group == "product" or "products" in parsed.norm):
        sql = (
            "SELECT p.name AS product_name,p.sector,SUM(u.value_usd) AS unrealized_potential_usd,COUNT(DISTINCT u.country) AS n_destinations FROM up_pairs u "
            "JOIN products_master p ON CAST(p.hs6 AS INTEGER)=CAST(u.hs6 AS INTEGER) "
            f"WHERE p.sector={sec} GROUP BY u.hs6,p.name,p.sector ORDER BY unrealized_potential_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Products with unrealized potential in {sector}", sql, entities=entities)
    if parsed.group == "product" or re.search(r"\b(?:what|which|top|list).*(?:products|goods|items)\b", parsed.norm):
        year = _year(parsed)
        order_col, direction = _metric_order(parsed, "product")
        sql = (
            "SELECT name AS product_name,sector,export_value,rca,pci,n_countries,unrealized_potential_usd,cagr,trajectory "
            f"FROM product_year WHERE sector={sec} AND year={year} AND export_value>0{_threshold_sql(parsed)} "
            f"ORDER BY {order_col} {direction} LIMIT {parsed.limit}"
        )
        title = f"Most complex products in {sector}, {year}" if ("pci" in parsed.measures or "complexity" in parsed.measures) and not parsed.ascending else f"Products in {sector}, {year}"
        return QueryPlan(title, sql, entities=entities)
    if "market_size" in parsed.measures:
        if parsed.years:
            year = parsed.years[-1]
            sql = (
                "SELECT m.country,m.year,p.name AS product_name,m.market_size_usd "
                "FROM market_size_hs6 m JOIN products_master p ON CAST(p.hs6 AS TEXT)=m.hs6 "
                f"WHERE m.hs6={hs} AND m.year={year} ORDER BY m.market_size_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
            )
        else:
            sql = (
                "WITH latest AS (SELECT country,MAX(year) AS year FROM market_size_hs6 WHERE hs6=" + hs + " GROUP BY country) "
                "SELECT m.country,m.year,p.name AS product_name,m.market_size_usd FROM market_size_hs6 m "
                "JOIN latest l ON l.country=m.country AND l.year=m.year "
                "JOIN products_master p ON CAST(p.hs6 AS TEXT)=m.hs6 "
                f"WHERE m.hs6={hs} ORDER BY m.market_size_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
            )
        return QueryPlan(f"Largest recorded markets for {product.name}", sql, entities=entities)
    if "potential" in parsed.measures and (parsed.operation == "rank" or parsed.group == "market"):
        sql = (
            "SELECT u.country,p.name AS product_name,u.value_usd AS unrealized_potential_usd "
            "FROM up_pairs u JOIN products_master p ON CAST(p.hs6 AS INTEGER)=CAST(u.hs6 AS INTEGER) "
            f"WHERE CAST(u.hs6 AS INTEGER)=CAST({hs} AS INTEGER) ORDER BY u.value_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Unrealized potential for {product.name}", sql, entities=entities)
    if parsed.group in {"market", "continent"} or parsed.norm.startswith("where ") or re.search(r"\b(?:where|which|what|top|list).*(?:markets|countries|destinations)\b", parsed.norm):
        year = _year(parsed)
        group = "continent" if parsed.group == "continent" else "country"
        sql = (
            f"SELECT {group},SUM(value_usd) AS export_value,COUNT(DISTINCT hs6) AS n_products "
            f"FROM product_market_year WHERE sector={sec} AND year={year} GROUP BY {group} ORDER BY export_value {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Destinations for {sector} in {year}", sql, entities=entities)
    if parsed.operation == "trend":
        condition = _years_condition("year", parsed)
        sql = (
            "SELECT sector,year,export_value,share,rca,pci_avg,n_products_hs6,n_products_hs4,unrealized_potential_usd,cagr "
            f"FROM sector_year WHERE sector={sec}{condition} ORDER BY year"
        )
        return QueryPlan(f"Annual exports of {sector}", sql, entities=entities)
    if parsed.operation == "rank":
        year = _year(parsed)
        order_col, direction = _metric_order(parsed, "sector")
        sql = (
            f"WITH ranked AS (SELECT sector,export_value,share,rca,pci_avg,n_products_hs6,n_products_hs4,unrealized_potential_usd,cagr,"
            f"RANK() OVER (ORDER BY {order_col} {direction}) AS metric_rank FROM sector_year WHERE year={year} AND {order_col} IS NOT NULL) "
            f"SELECT sector,{year} AS year,export_value,share,rca,pci_avg,n_products_hs6,n_products_hs4,unrealized_potential_usd,cagr,metric_rank FROM ranked WHERE sector={sec} LIMIT 1"
        )
        return QueryPlan(f"Rank of {sector} in {year}", sql, entities=entities)
    year = _year(parsed)
    sql = (
        "SELECT sector,year,export_value,share,rca,pci_avg,n_products_hs6,n_products_hs4,unrealized_potential_usd,cagr "
        f"FROM sector_year WHERE sector={sec} AND year={year} LIMIT 1"
    )
    return QueryPlan(f"{sector} in {year}", sql, entities=entities)


def _drivers_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    if parsed.operation != "drivers":
        return None
    start, end = (min(parsed.years), max(parsed.years)) if len(parsed.years) >= 2 else (2018, 2025)
    group = parsed.group or "product"
    entities = {"years": [start, end], "metric": "growth_drivers"}
    if parsed.markets:
        market = parsed.markets[0]
        entities["market"] = market
        mk = _sql_quote(market)
        if group == "sector":
            sql = (
                "WITH a AS (SELECT sector,SUM(value_usd) v FROM product_market_year WHERE country=" + mk + f" AND year={start} GROUP BY sector), "
                "b AS (SELECT sector,SUM(value_usd) v FROM product_market_year WHERE country=" + mk + f" AND year={end} GROUP BY sector), "
                "keys AS (SELECT sector FROM a UNION SELECT sector FROM b) "
                f"SELECT k.sector,COALESCE(a.v,0) AS value_{start},COALESCE(b.v,0) AS value_{end},COALESCE(b.v,0)-COALESCE(a.v,0) AS change_usd "
                "FROM keys k LEFT JOIN a USING(sector) LEFT JOIN b USING(sector) ORDER BY change_usd " + ("ASC" if parsed.ascending else "DESC") + f" LIMIT {parsed.limit}"
            )
            return QueryPlan(f"Sector drivers of exports to {market}, {start}-{end}", sql, entities=entities)
        sql = (
            "WITH a AS (SELECT hs6,product_name,SUM(value_usd) v FROM product_market_year WHERE country=" + mk + f" AND year={start} GROUP BY hs6,product_name), "
            "b AS (SELECT hs6,product_name,SUM(value_usd) v FROM product_market_year WHERE country=" + mk + f" AND year={end} GROUP BY hs6,product_name), "
            "keys AS (SELECT hs6,product_name FROM a UNION SELECT hs6,product_name FROM b) "
            f"SELECT k.product_name,COALESCE(a.v,0) AS value_{start},COALESCE(b.v,0) AS value_{end},COALESCE(b.v,0)-COALESCE(a.v,0) AS change_usd "
            "FROM keys k LEFT JOIN a USING(hs6,product_name) LEFT JOIN b USING(hs6,product_name) ORDER BY change_usd " + ("ASC" if parsed.ascending else "DESC") + f" LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Product drivers of exports to {market}, {start}-{end}", sql, entities=entities)
    if group == "market":
        sql = (
            f"SELECT country,MAX(CASE WHEN year={start} THEN export_value END) AS value_{start},MAX(CASE WHEN year={end} THEN export_value END) AS value_{end},"
            f"COALESCE(MAX(CASE WHEN year={end} THEN export_value END),0)-COALESCE(MAX(CASE WHEN year={start} THEN export_value END),0) AS change_usd "
            f"FROM market_year WHERE year IN ({start},{end}) GROUP BY country ORDER BY change_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Market drivers of export change, {start}-{end}", sql, entities=entities)
    if group == "sector":
        sql = (
            f"SELECT sector,MAX(CASE WHEN year={start} THEN export_value END) AS value_{start},MAX(CASE WHEN year={end} THEN export_value END) AS value_{end},"
            f"COALESCE(MAX(CASE WHEN year={end} THEN export_value END),0)-COALESCE(MAX(CASE WHEN year={start} THEN export_value END),0) AS change_usd "
            f"FROM sector_year WHERE year IN ({start},{end}) GROUP BY sector ORDER BY change_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Sector drivers of export change, {start}-{end}", sql, entities=entities)
    sql = (
        f"SELECT name AS product_name,MAX(CASE WHEN year={start} THEN export_value END) AS value_{start},MAX(CASE WHEN year={end} THEN export_value END) AS value_{end},"
        f"COALESCE(MAX(CASE WHEN year={end} THEN export_value END),0)-COALESCE(MAX(CASE WHEN year={start} THEN export_value END),0) AS change_usd "
        f"FROM product_year WHERE year IN ({start},{end}) GROUP BY hs6,name ORDER BY change_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
    )
    return QueryPlan(f"Product drivers of export change, {start}-{end}", sql, entities=entities)


def _entry_exit_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    if not ({"entry", "exit"} & parsed.measures):
        return None
    year = _year(parsed)
    entering = "entry" in parsed.measures
    entities = {"year": year, "metric": "entry" if entering else "exit"}
    if parsed.markets:
        market = parsed.markets[0]
        entities["market"] = market
        mk = _sql_quote(market)
        if entering:
            sql = (
                f"WITH curr AS (SELECT hs6,product_name,value_usd FROM product_market_year WHERE country={mk} AND year={year} AND value_usd>0), "
                f"prev AS (SELECT hs6,value_usd FROM product_market_year WHERE country={mk} AND year={year-1} AND value_usd>0) "
                "SELECT curr.product_name,curr.value_usd FROM curr LEFT JOIN prev USING(hs6) WHERE prev.hs6 IS NULL ORDER BY curr.value_usd DESC LIMIT " + str(parsed.limit)
            )
            return QueryPlan(f"Products newly exported to {market} in {year}", sql, entities=entities)
        sql = (
            f"WITH prev AS (SELECT hs6,product_name,value_usd FROM product_market_year WHERE country={mk} AND year={year-1} AND value_usd>0), "
            f"curr AS (SELECT hs6,value_usd FROM product_market_year WHERE country={mk} AND year={year} AND value_usd>0) "
            "SELECT prev.product_name,prev.value_usd AS previous_export_value FROM prev LEFT JOIN curr USING(hs6) WHERE curr.hs6 IS NULL ORDER BY prev.value_usd DESC LIMIT " + str(parsed.limit)
        )
        return QueryPlan(f"Products no longer exported to {market} in {year}", sql, entities=entities)
    if entering:
        sql = (
            f"SELECT p.name AS product_name,p.sector,p.export_value FROM product_year p LEFT JOIN product_year prev ON prev.hs6=p.hs6 AND prev.year={year-1} "
            f"WHERE p.year={year} AND p.export_value>0 AND COALESCE(prev.export_value,0)=0 ORDER BY p.export_value DESC LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Products entering Lebanon's exports in {year}", sql, entities=entities)
    sql = (
        f"SELECT prev.name AS product_name,prev.sector,prev.export_value AS previous_export_value FROM product_year prev LEFT JOIN product_year p ON p.hs6=prev.hs6 AND p.year={year} "
        f"WHERE prev.year={year-1} AND prev.export_value>0 AND COALESCE(p.export_value,0)=0 ORDER BY prev.export_value DESC LIMIT {parsed.limit}"
    )
    return QueryPlan(f"Products exiting Lebanon's exports in {year}", sql, entities=entities)


def _ranking_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    group = parsed.group
    q = parsed.norm
    if not group:
        if "sector" in q:
            group = "sector"
        elif any(t in q for t in ("market", "country", "destination")):
            group = "market"
        elif "continent" in q or "region" in q:
            group = "continent"
        elif "product" in q:
            group = "product"
    if parsed.operation not in {"rank", "list", "average", "count"} and not group:
        return None
    year = _year(parsed, market_size="market_size" in parsed.measures)
    entities = {"year": year, "metric": sorted(parsed.measures)[0], "group": group or "overview"}

    if group == "product":
        if "market_size" in parsed.measures:
            sql = (
                "SELECT p.name AS product_name,SUM(m.market_size_usd) AS market_size_usd,COUNT(DISTINCT m.country) AS n_markets "
                "FROM market_size_hs6 m JOIN products_master p ON CAST(p.hs6 AS TEXT)=m.hs6 "
                f"WHERE m.year={year} GROUP BY m.hs6,p.name ORDER BY market_size_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
            )
            return QueryPlan(f"Products by total recorded market size in {year}", sql, entities=entities)
        if "potential" in parsed.measures:
            sql = (
                "SELECT name AS product_name,sector,value_usd AS unrealized_potential_usd,pci,n_destinations,top_destination "
                f"FROM up_hs6 ORDER BY value_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
            )
            return QueryPlan("Products by unrealized export potential", sql, entities=entities)
        order_col, direction = _metric_order(parsed, "product")
        condition = f"year={year} AND {order_col} IS NOT NULL"
        # Product rankings in the dashboard refer to active exports unless the user explicitly asks for inactive/zero products.
        if not any(term in parsed.norm for term in ("inactive", "zero export", "not exported")):
            condition += " AND export_value>0"
        condition += _threshold_sql(parsed)
        sql = (
            "SELECT name AS product_name,sector,export_value,rca,pci,n_countries,unrealized_potential_usd,cagr,trajectory "
            f"FROM product_year WHERE {condition} ORDER BY {order_col} {direction} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Products ranked by {order_col.replace('_', ' ')} in {year}", sql, entities=entities)

    if group == "market":
        if "potential" in parsed.measures:
            sql = f"SELECT country,value_usd AS unrealized_potential_usd,n_hs6 FROM up_partner ORDER BY value_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
            return QueryPlan("Markets by unrealized export potential", sql, entities=entities)
        order_col, direction = _metric_order(parsed, "market")
        condition = f"year={year} AND {order_col} IS NOT NULL" + _threshold_sql(parsed)
        sql = (
            "SELECT country,continent,export_value,n_products,expy,hhi,rca,unrealized_potential_usd,status,priority,cagr "
            f"FROM market_year WHERE {condition} ORDER BY {order_col} {direction} LIMIT {parsed.limit}"
        )
        if order_col == "hhi" and "diversif" in parsed.norm:
            title = f"Most diversified export markets in {year}"
        elif order_col == "hhi" and "concentrat" in parsed.norm:
            title = f"Most concentrated export markets in {year}"
        else:
            title = f"Markets ranked by {order_col.replace('_', ' ')} in {year}"
        return QueryPlan(title, sql, entities=entities)

    if group == "sector":
        if "potential" in parsed.measures:
            sql = f"SELECT sector,value_usd AS unrealized_potential_usd,n_hs6,n_partners,n_destinations FROM up_sector ORDER BY value_usd {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
            return QueryPlan("Sectors by unrealized export potential", sql, entities=entities)
        order_col, direction = _metric_order(parsed, "sector")
        condition = f"year={year} AND {order_col} IS NOT NULL" + _threshold_sql(parsed)
        sql = (
            "SELECT sector,export_value,share,rca,pci_avg,n_products_hs6,n_products_hs4,unrealized_potential_usd,cagr "
            f"FROM sector_year WHERE {condition} ORDER BY {order_col} {direction} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Sectors ranked by {order_col.replace('_', ' ')} in {year}", sql, entities=entities)

    if group == "continent":
        order_col = "export_value"
        direction = "ASC" if parsed.ascending else "DESC"
        sql = (
            "SELECT continent,SUM(export_value) AS export_value,COUNT(DISTINCT country) AS n_markets,SUM(n_products) AS product_market_count "
            f"FROM market_year WHERE year={year} GROUP BY continent ORDER BY {order_col} {direction} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Exports by continent in {year}", sql, entities=entities)
    return None


def _performance_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    q = parsed.norm
    if "overperform" in q:
        return QueryPlan("Overperforming markets", f"SELECT rank,value AS country FROM topsis_overperformers ORDER BY rank LIMIT {parsed.limit}", entities={"metric": "performance"})
    if "underperform" in q:
        return QueryPlan("Underperforming markets", f"SELECT rank,value AS country FROM topsis_underperformers ORDER BY rank LIMIT {parsed.limit}", entities={"metric": "performance"})
    return None




def _set_relation_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    q = parsed.norm
    year = _year(parsed)
    # Products present in one market but not another, or common to both.
    if len(parsed.markets) >= 2 and (parsed.group == "product" or "products" in q):
        a, b = parsed.markets[:2]
        qa, qb = _sql_quote(a), _sql_quote(b)
        if "but not" in q or "not in" in q or "excluding" in q:
            sql = (
                f"SELECT a.product_name,a.sector,a.value_usd AS export_value FROM product_market_year a "
                f"WHERE a.country={qa} AND a.year={year} AND a.value_usd>0 AND NOT EXISTS (SELECT 1 FROM product_market_year b WHERE b.hs6=a.hs6 AND b.country={qb} AND b.year={year} AND b.value_usd>0) "
                f"ORDER BY a.value_usd DESC LIMIT {parsed.limit}"
            )
            return QueryPlan(f"Products exported to {a} but not {b} in {year}", sql, entities={"markets": [a,b], "year": year, "metric": "set_difference"})
        if "both" in q or "common" in q or "each" in q:
            sql = (
                f"SELECT a.product_name,a.sector,a.value_usd AS value_{_norm_text(a).replace(' ','_')},b.value_usd AS value_{_norm_text(b).replace(' ','_')} "
                f"FROM product_market_year a JOIN product_market_year b ON b.hs6=a.hs6 AND b.year=a.year "
                f"WHERE a.country={qa} AND b.country={qb} AND a.year={year} AND a.value_usd>0 AND b.value_usd>0 ORDER BY a.value_usd+b.value_usd DESC LIMIT {parsed.limit}"
            )
            return QueryPlan(f"Products exported to both {a} and {b} in {year}", sql, entities={"markets": [a,b], "year": year, "metric": "intersection"})
    # Destinations common to two named products.
    if len(parsed.products) >= 2 and (parsed.group == "market" or "countries" in q or "markets" in q or "both" in q):
        a, b = parsed.products[:2]
        ha, hb = _sql_quote(a.hs6), _sql_quote(b.hs6)
        sql = (
            f"SELECT x.country,x.value_usd AS first_product_exports,y.value_usd AS second_product_exports FROM product_market_year x "
            f"JOIN product_market_year y ON y.country=x.country AND y.year=x.year WHERE x.hs6={ha} AND y.hs6={hb} AND x.year={year} AND x.value_usd>0 AND y.value_usd>0 "
            f"ORDER BY x.value_usd+y.value_usd DESC LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Markets buying both {a.name} and {b.name} in {year}", sql, entities={"products": [a.hs6,b.hs6], "year": year, "metric": "intersection"})
    return None


def _continent_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    if not parsed.continents:
        return None
    continents = parsed.continents
    values = ",".join(_sql_quote(c) for c in continents)
    year = _year(parsed)
    entities = {"continents": continents, "year": year, "metric": sorted(parsed.measures)[0]}
    if parsed.group == "market" or "countries" in parsed.norm or "markets" in parsed.norm:
        order_col, direction = _metric_order(parsed, "market")
        sql = f"SELECT country,continent,export_value,n_products,expy,hhi,rca,unrealized_potential_usd,status,cagr FROM market_year WHERE continent IN ({values}) AND year={year} ORDER BY {order_col} {direction} LIMIT {parsed.limit}"
        return QueryPlan(f"Markets in {', '.join(continents)} in {year}", sql, entities=entities)
    if parsed.group == "product" or "products" in parsed.norm:
        sql = (
            f"SELECT product_name,sector,SUM(value_usd) AS export_value,COUNT(DISTINCT country) AS n_markets FROM product_market_year WHERE continent IN ({values}) AND year={year} "
            f"GROUP BY hs6,product_name,sector ORDER BY export_value {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        )
        return QueryPlan(f"Products exported to {', '.join(continents)} in {year}", sql, entities=entities)
    if parsed.group == "sector" or "sectors" in parsed.norm:
        sql = f"SELECT sector,SUM(value_usd) AS export_value,COUNT(DISTINCT country) AS n_markets FROM product_market_year WHERE continent IN ({values}) AND year={year} GROUP BY sector ORDER BY export_value {'ASC' if parsed.ascending else 'DESC'} LIMIT {parsed.limit}"
        return QueryPlan(f"Sectors exported to {', '.join(continents)} in {year}", sql, entities=entities)
    if parsed.operation == "trend":
        sql = f"SELECT year,SUM(export_value) AS export_value,COUNT(DISTINCT country) AS n_markets FROM market_year WHERE continent IN ({values}) GROUP BY year ORDER BY year"
        return QueryPlan(f"Exports to {', '.join(continents)} over time", sql, entities=entities)
    sql = (
        f"WITH c AS (SELECT SUM(export_value) AS export_value,COUNT(DISTINCT country) AS n_markets FROM market_year WHERE continent IN ({values}) AND year={year}), "
        f"t AS (SELECT total_exports_usd FROM export_overview WHERE year={year}) SELECT {_sql_quote(', '.join(continents))} AS continent,{year} AS year,c.export_value,c.n_markets,CASE WHEN t.total_exports_usd>0 THEN c.export_value/t.total_exports_usd ELSE 0 END AS share_of_total_exports FROM c CROSS JOIN t"
    )
    return QueryPlan(f"Exports to {', '.join(continents)} in {year}", sql, entities=entities)


def _average_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    if parsed.operation != "average" or parsed.products or parsed.markets or parsed.sectors:
        return None
    year = _year(parsed)
    q = parsed.norm
    if parsed.group == "market" or "countries" in q or "markets" in q:
        metric, _ = _metric_order(parsed, "market")
        return QueryPlan(f"Average {metric.replace('_',' ')} across markets in {year}", f"SELECT AVG({metric}) AS average_{metric},COUNT(*) AS count_value FROM market_year WHERE year={year} AND {metric} IS NOT NULL", entities={"year": year, "metric": metric})
    if parsed.group == "sector" or "sectors" in q:
        metric, _ = _metric_order(parsed, "sector")
        return QueryPlan(f"Average {metric.replace('_',' ')} across sectors in {year}", f"SELECT AVG({metric}) AS average_{metric},COUNT(*) AS count_value FROM sector_year WHERE year={year} AND {metric} IS NOT NULL", entities={"year": year, "metric": metric})
    metric, _ = _metric_order(parsed, "product")
    return QueryPlan(f"Average {metric.replace('_',' ')} across exported products in {year}", f"SELECT AVG({metric}) AS average_{metric},COUNT(*) AS count_value FROM product_year WHERE year={year} AND export_value>0 AND {metric} IS NOT NULL", entities={"year": year, "metric": metric})


def _market_product_count_threshold_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    """List destination countries satisfying a product-count threshold."""
    if parsed.products or parsed.markets or parsed.sectors:
        return None
    if parsed.group != "market":
        return None
    count_filters = [item for item in parsed.filters if item[0] == "n_products"]
    if not count_filters:
        return None

    year = _year(parsed)
    operator = count_filters[-1][1]
    threshold = count_filters[-1][2]
    condition = f"n_products {operator} {float(threshold)}"
    sql = (
        "SELECT country,continent,year,n_products,export_value "
        f"FROM market_year WHERE year={year} AND {condition} "
        "ORDER BY n_products DESC,country ASC"
    )
    columns, rows, _ = execute_sql(sql, max_rows=100)
    threshold_text = f"{threshold:,.0f}"
    relation = {
        ">": "more than",
        ">=": "at least",
        "<": "fewer than",
        "<=": "at most",
        "=": "exactly",
    }.get(operator, operator)

    if not rows:
        _, maximum_rows, _ = execute_sql(
            "SELECT country,n_products,export_value FROM market_year "
            f"WHERE year={year} ORDER BY n_products DESC LIMIT 1",
            max_rows=1,
        )
        if maximum_rows:
            top = maximum_rows[0]
            answer = (
                f"No destination country received {relation} {threshold_text} distinct Lebanese products in {year}. "
                f"The highest count was {int(top.get('n_products') or 0):,} products in {top.get('country')}."
            )
        else:
            answer = f"No matching destination record is available for {year}."
        return QueryPlan(
            f"Countries receiving {relation} {threshold_text} Lebanese products in {year}",
            kind="direct",
            direct_answer=answer,
            entities={"year": year, "metric": "n_products", "group": "market"},
            confidence=1.0,
        )

    return QueryPlan(
        f"Countries receiving {relation} {threshold_text} Lebanese products in {year}",
        sql,
        entities={"year": year, "metric": "n_products", "group": "market"},
        confidence=1.0,
    )


def _unresolved_product_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    """Prevent unsupported product substitutions when no exact product resolved."""
    if parsed.products:
        return None
    phrase = _extract_unresolved_product_phrase(parsed.original)
    if not phrase:
        return None
    if parsed.group in {"market", "sector", "continent", "year"} and re.search(
        r"\b(?:products|goods|items|commodities)\b", parsed.norm
    ):
        return None
    direct = (
        f"I could not find an exact dashboard product matching **{phrase}**. "
        "I did not substitute a different product, so no export value was returned."
    )
    return QueryPlan(
        "Product not found in the dashboard",
        kind="direct",
        direct_answer=direct,
        entities={"metric": "product_lookup", "unresolved_product": phrase},
        confidence=1.0,
    )


def _global_count_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    if parsed.operation != "count" or parsed.products or parsed.markets or parsed.sectors:
        return None
    q = parsed.norm
    if "sector" in q:
        return QueryPlan("Number of sectors in the dashboard", "SELECT COUNT(*) AS count_value FROM sectors_master", entities={"metric": "count"})
    if any(term in q for term in ("market", "country", "countries", "destination")):
        return QueryPlan("Number of destination markets in the dashboard", "SELECT COUNT(*) AS count_value FROM markets_master", entities={"metric": "count"})
    if "product" in q:
        return QueryPlan("Number of products in the dashboard", "SELECT COUNT(*) AS count_value FROM products_master", entities={"metric": "count"})
    return None

def build_plan(parsed: ParsedQuestion) -> QueryPlan | None:
    for builder in (
        _definition_plan, _coverage_plan, _market_product_count_threshold_plan,
        _unresolved_product_plan, _global_count_plan, _entry_exit_plan, _drivers_plan, _set_relation_plan,
        _pair_plan, _sector_market_plan, _continent_plan, _product_profile_plan,
        _product_plan, _market_plan, _sector_plan, _average_plan,
        _performance_plan, _ranking_plan, _total_plan,
    ):
        plan = builder(parsed)
        if plan:
            return plan
    return None


# ---------------------------------------------------------------------------
# Safe SQL and answer rendering
# ---------------------------------------------------------------------------

def _safe_sql(sql: str) -> str:
    cleaned = str(sql or "").strip().rstrip(";")
    if not cleaned:
        raise ValueError("No SQL was generated.")
    if BLOCKED_SQL.search(cleaned):
        raise ValueError("Only read-only dashboard queries are allowed.")
    if not re.match(r"^(select|with)\b", cleaned, re.I):
        raise ValueError("The query must be SELECT-only.")
    # Reject statement separators outside quoted string literals. Product names
    # legitimately contain semicolons, so a plain string check is too strict.
    unquoted = re.sub(r"'(?:''|[^'])*'", "''", cleaned)
    if ";" in unquoted:
        raise ValueError("Multiple SQL statements are not allowed.")
    return cleaned


def execute_sql(sql: str, max_rows: int = 100) -> tuple[list[str], list[dict[str, Any]], bool]:
    sql = _safe_sql(sql)
    with sqlite3.connect(_database_path()) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        fetched = cursor.fetchmany(max_rows + 1)
    truncated = len(fetched) > max_rows
    rows = [dict(row) for row in fetched[:max_rows]]
    return columns, rows, truncated


def _money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "No data"
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number >= 1_000_000_000:
        return f"{sign}${number / 1_000_000_000:.2f} billion"
    if number >= 1_000_000:
        return f"{sign}${number / 1_000_000:.2f} million"
    if number >= 1_000:
        return f"{sign}${number / 1_000:.1f} thousand"
    return f"{sign}${number:,.0f}"


def _format_value(value: Any, column: str) -> str:
    if value is None:
        return "No data"
    col = column.lower()
    if col in {"hs6", "hs4", "code"}:
        return str(value).zfill(6 if col == "hs6" else 4)
    if col == "year":
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)
    if col.endswith("_rank") or col in {"rank", "n_products", "n_products_hs6", "n_products_hs4", "n_markets", "active_products", "active_markets", "n_countries", "count_value", "n_destinations", "n_partners"}:
        try:
            return f"{int(float(value)):,}"
        except (TypeError, ValueError):
            return str(value)
    if any(token in col for token in ("value", "exports", "potential", "market_size", "change_usd")) and not any(token in col for token in ("share", "ratio")):
        return _money(value)
    if "share" in col or col.endswith("ratio") or "penetration" in col:
        try:
            number = float(value)
            if abs(number) <= 1.5:
                number *= 100
            return f"{number:.1f}%"
        except (TypeError, ValueError):
            return str(value)
    if col in {"rca", "pci", "pci_avg", "hhi", "expy", "score"}:
        try:
            return f"{float(value):,.3f}"
        except (TypeError, ValueError):
            return str(value)
    if col in {"cagr", "growth"} or "change_pct" in col:
        try:
            number = float(value)
            if abs(number) <= 2:
                number *= 100
            return f"{number:+.1f}%"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float):
        return f"{value:,.2f}"
    text = str(value)
    if col in {"record_json", "geometry_json", "properties_json"} and len(text) > 700:
        return text[:700] + "…"
    return text


def _label(column: str) -> str:
    labels = {
        "product_name": "Product", "name": "Product", "country": "Country", "sector": "Sector", "continent": "Continent",
        "total_exports_usd": "Total exports", "real_exports_2018_usd": "Real exports (2018 USD)", "export_value": "Export value",
        "value_usd": "Export value", "previous_export_value": "Previous export value", "market_size_usd": "Market size",
        "unrealized_potential_usd": "Unrealized potential", "share_of_market_exports": "Share of exports to this market",
        "share_of_product_exports": "Share of this product's exports", "share_of_total_exports": "Share of total exports",
        "n_products": "Products", "n_products_hs6": "Products", "n_products_hs4": "Product groups", "n_markets": "Markets",
        "n_countries": "Markets reached", "active_products": "Active products", "active_markets": "Active markets",
        "metric_rank": "Rank", "export_rank": "Export rank", "destination_rank": "Destination rank",
        "pci": "PCI", "pci_avg": "Average PCI", "rca": "RCA", "hhi": "HHI", "expy": "EXPY", "cagr": "CAGR",
        "change_usd": "Change", "market_penetration": "Market penetration", "total_addressable_exports": "Actual exports plus unrealized potential",
        "trajectory": "Trajectory", "status": "Performance status", "priority": "Priority",
    }
    return labels.get(column.lower(), column.replace("_", " ").title())


def _user_wants_code(question: str) -> bool:
    q = _norm_text(question)
    years = {str(y) for y in range(2018, 2026)}
    return bool(re.search(r"\b(?:hs|h s|harmonized|harmonised|tariff|customs|product code)\b", q) or any(t not in years for t in re.findall(r"\b\d{4,6}\b", q)))


def _lead_column(columns: list[str]) -> str | None:
    for col in ("product_name", "name", "country", "sector", "continent", "year", "item", "label"):
        if col in columns:
            return col
    return None


def _summarize_series(rows: list[dict[str, Any]], value_col: str) -> str | None:
    dated = [r for r in rows if r.get("year") is not None and r.get(value_col) is not None]
    if len(dated) < 2 or len({int(r["year"]) for r in dated}) < 2:
        return None
    dated.sort(key=lambda r: int(r["year"]))
    first, last = dated[0], dated[-1]
    a, b = float(first[value_col]), float(last[value_col])
    if a == 0:
        return f"The series moves from {_format_value(a, value_col)} in {int(first['year'])} to {_format_value(b, value_col)} in {int(last['year'])}."
    change = (b / a - 1) * 100
    direction = "increased" if change >= 0 else "decreased"
    return f"{_label(value_col)} {direction} from {_format_value(a, value_col)} in {int(first['year'])} to {_format_value(b, value_col)} in {int(last['year'])} ({change:+.1f}%)."


def _filter_visible_columns(columns: list[str], question: str) -> list[str]:
    q = _norm_text(question)
    requested = _parse_measures(q)
    destination_ranking = bool(re.search(r"\b(?:top|main|largest|leading|best)\s+(?:markets|countries|destinations)\b|\bwhere\b", q))
    product_share_question = bool(re.search(r"share of .{0,80}(?:product|exports).{0,40}(?:went|go|going|to)", q))
    hidden_by_default = {
        "rca": "rca" not in requested,
        "pci": not ({"pci", "complexity"} & requested),
        "pci_avg": not ({"pci", "complexity"} & requested),
        "hhi": "hhi" not in requested,
        "expy": "expy" not in requested,
        "unrealized_potential_usd": "potential" not in requested,
        "status": "performance" not in requested,
        "priority": "performance" not in requested,
        "cagr": not ({"cagr", "growth"} & requested),
        "growth": "growth" not in requested,
        "trajectory": "growth" not in requested,
        "share_of_market_exports": destination_ranking or product_share_question,
    }
    result = []
    for col in columns:
        if hidden_by_default.get(col.lower(), False):
            continue
        result.append(col)
    return result


def deterministic_answer(plan: QueryPlan, columns: list[str], rows: list[dict[str, Any]], truncated: bool, question: str) -> str:
    if plan.kind == "direct":
        return f"**{plan.title}**\n\n{plan.direct_answer}"
    if not rows:
        # Bilateral export lookups are meaningful at zero; potential and market-size
        # lookups remain "no record" because absence does not necessarily mean zero.
        if plan.title and "exports to" in plan.title.lower() and "potential" not in plan.title.lower() and "market size" not in plan.title.lower():
            extra = "\n- **Share:** 0.0%" if any(term in _norm_text(question) for term in ("share", "percent", "percentage")) else ""
            return f"**{plan.title}**\n\n- **Export value:** $0{extra}"
        return f"**{plan.title}**\n\nNo matching record exists in the dashboard for that exact scope."

    show_codes = _user_wants_code(question)
    visible = [c for c in columns if show_codes or c.lower() not in {"hs6", "hs4", "code"}]
    visible = _filter_visible_columns(visible, question)
    lines = [f"**{plan.title}**", ""]
    if plan.notes:
        for note in plan.notes:
            lines.append(f"- {note}")
        lines.append("")
    if plan.direct_answer:
        lines.extend([plan.direct_answer, ""])

    value_col = next((c for c in ("total_exports_usd", "export_value", "value_usd", "market_size_usd", "unrealized_potential_usd", "change_usd") if c in columns), None)
    if "year" in columns and value_col:
        summary = _summarize_series(rows, value_col)
        if summary:
            lines.extend([f"- **Overall:** {summary}", ""])

    if len(rows) == 1:
        row = rows[0]
        ordered: list[str] = []
        for col in ("product_name", "name", "country", "sector", "continent", "year"):
            if col in visible and col not in ordered:
                ordered.append(col)
        ordered.extend(c for c in visible if c not in ordered)
        for col in ordered:
            if row.get(col) is not None:
                lines.append(f"- **{_label(col)}:** {_format_value(row[col], col)}")
        return "\n".join(lines).strip()

    lead = _lead_column(visible)
    if len(rows) > 1 and "year" in visible and len({row.get("year") for row in rows}) > 1:
        lead = "year"
    elif len(rows) > 1 and lead in {"product_name", "name"} and "country" in visible:
        product_values = {str(row.get(lead)) for row in rows if row.get(lead) is not None}
        country_values = {str(row.get("country")) for row in rows if row.get("country") is not None}
        if len(product_values) == 1 and len(country_values) > 1:
            lead = "country"
    time_series = lead == "year"
    constant_cols = {
        col for col in visible
        if col != lead and col in {"product_name", "name", "country", "sector", "continent"}
        and len({str(row.get(col)) for row in rows if row.get(col) is not None}) == 1
    }
    if len(rows) == 2 and value_col and lead in {"country", "sector", "product_name", "name"}:
        try:
            first_label, second_label = str(rows[0].get(lead)), str(rows[1].get(lead))
            first_value, second_value = float(rows[0].get(value_col) or 0), float(rows[1].get(value_col) or 0)
            gap = abs(first_value - second_value)
            higher = first_label if first_value >= second_value else second_label
            lines.extend([f"- **Comparison:** {higher} is higher by {_format_value(gap, value_col)}.", ""])
        except Exception:
            pass
    for idx, row in enumerate(rows, 1):
        lead_value = row.get(lead) if lead else None
        if lead == "year" and lead_value is not None:
            lead_value = str(int(float(lead_value)))
        if lead_value is None:
            lead_value = f"Record {idx}"
        used = {lead} if lead else set()
        parts: list[str] = []
        for col in visible:
            if col in used or col in constant_cols or row.get(col) is None:
                continue
            parts.append(f"{_label(col)}: {_format_value(row[col], col)}")
        prefix = "- " if time_series else f"{idx}. "
        line = f"{prefix}**{lead_value}**"
        if parts:
            line += " - " + "; ".join(parts[:9])
        lines.append(line)
    if truncated:
        lines.extend(["", "- The display is limited to the first 100 matching records."])
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Correlation and LLM fallback
# ---------------------------------------------------------------------------

def _correlation_answer(parsed: ParsedQuestion) -> DashboardAnswer | None:
    if parsed.operation != "correlation":
        return None
    metric_columns = []
    q = parsed.norm
    mappings = [
        ("exports", "export_value"), ("rca", "rca"), ("pci", "pci"), ("complexity", "pci"),
        ("cagr", "cagr"), ("growth", "cagr"), ("potential", "unrealized_potential_usd"),
        ("hhi", "hhi"), ("expy", "expy"),
    ]
    for word, column in mappings:
        if word in q and column not in metric_columns:
            metric_columns.append(column)
    if len(metric_columns) < 2:
        return None
    x, y = metric_columns[:2]
    year = _year(parsed)
    scope = parsed.group or ("market" if x in {"hhi", "expy"} or y in {"hhi", "expy"} else "product")
    if scope == "market":
        table, label = "market_year", "markets"
    elif scope == "sector":
        table, label = "sector_year", "sectors"
        x = "pci_avg" if x == "pci" else x
        y = "pci_avg" if y == "pci" else y
    else:
        table, label = "product_year", "products"
    sql = f"SELECT {x} AS x,{y} AS y FROM {table} WHERE year={year} AND {x} IS NOT NULL AND {y} IS NOT NULL"
    _, rows, _ = execute_sql(sql, max_rows=2000)
    pairs = [(float(r["x"]), float(r["y"])) for r in rows if r.get("x") is not None and r.get("y") is not None]
    if len(pairs) < 3:
        return DashboardAnswer(True, 0.95, "**Correlation**\n\nThere are not enough observations in the dashboard to calculate this correlation.", json.dumps(rows), {"year": year, "metric": "correlation"})
    xs, ys = zip(*pairs)
    try:
        corr = statistics.correlation(xs, ys)
    except Exception:
        mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((a-mean_x)*(b-mean_y) for a,b in pairs)
        den = math.sqrt(sum((a-mean_x)**2 for a in xs) * sum((b-mean_y)**2 for b in ys))
        corr = num / den if den else 0.0
    strength = "very weak" if abs(corr) < .2 else "weak" if abs(corr) < .4 else "moderate" if abs(corr) < .7 else "strong"
    direction = "positive" if corr > 0 else "negative" if corr < 0 else "zero"
    answer = (
        f"**Correlation across {label} in {year}**\n\n"
        f"- **Pearson correlation:** {corr:+.3f}\n"
        f"- **Reading:** The relationship is {strength} and {direction}.\n"
        f"- **Observations:** {len(pairs):,}\n\n"
        "This is a descriptive association in the dashboard data, not evidence of causation."
    )
    return DashboardAnswer(True, 0.99, answer, json.dumps({"sql": sql, "n": len(pairs), "correlation": corr}), {"year": year, "metric": "correlation"})


def _llm_plan(question: str, provider: str, model: str, repair: str = "") -> dict[str, Any]:
    prompt = f"""
Create one read-only SQLite query that answers the user from the dashboard database.
Return JSON only: {{"title":"...","sql":"SELECT ...","not_dashboard_data":false}}.
If the question cannot be answered from the listed dashboard data, return {{"title":"","sql":"","not_dashboard_data":true}}.

{SCHEMA_CATALOG}

Rules:
- Use SELECT/WITH only; no PRAGMA, DDL or writes.
- Preserve the requested product, sector, country and year scope.
- Prefer product names in output; include HS codes only if requested.
- Default to 2025 if year omitted; market size defaults to 2024.
- Use case-insensitive matching or joins to master tables for product names.
- Limit ranked/list results to the number requested, otherwise 10.
- Return enough fields to answer directly, including units and shares when relevant.
- Never infer a bilateral value from separate product and market totals.

User question: {question}
{('Repair instruction: ' + repair) if repair else ''}
""".strip()
    raw = chat_completion(
        provider=provider,
        system_prompt="You are a conservative SQLite query planner for a fixed trade dashboard. Output exactly one JSON object.",
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=1000,
    )
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("The query planner did not return JSON.")
    plan = json.loads(match.group(0))
    if plan.get("sql"):
        plan["sql"] = _safe_sql(str(plan["sql"]))
    return plan


def _semantic_record_search(parsed: ParsedQuestion) -> QueryPlan | None:
    # Last-resort local search. It exposes matching exact records instead of
    # replacing an unrecognised question with an unrelated export overview.
    terms = [t for t in parsed.norm.split() if len(t) >= 4 and t not in GENERIC_WORDS]
    if not terms:
        return None
    conditions = " AND ".join(f"lower(record_json) LIKE '%{t.replace("'", "''")}% '" for t in [])
    # Use OR to maximise recall; rank by number of term hits.
    score_parts = [f"CASE WHEN lower(record_json) LIKE '%{t.replace("'", "''")}%' THEN 1 ELSE 0 END" for t in terms[:8]]
    where = " OR ".join(f"lower(record_json) LIKE '%{t.replace("'", "''")}%'" for t in terms[:8])
    if not where:
        return None
    score = " + ".join(score_parts)
    sql = (
        f"SELECT dataset,record_key,record_json,({score}) AS relevance FROM raw_dataset_records "
        f"WHERE {where} ORDER BY relevance DESC LIMIT {parsed.limit}"
    )
    return QueryPlan("Matching dashboard records", sql, confidence=0.72, entities={"metric": "raw_records"})



def _looks_like_concept_explanation(question: str) -> bool:
    q = re.sub(r"\s+", " ", str(question or "").strip().lower())
    concepts = (
        "scale", "diversification", "diversified", "diversify", "composition",
        "concentration", "complexity", "competitiveness", "comparative advantage",
        "sophistication", "growth", "reach", "market size", "potential",
        "performance", "priority", "hhi", "pci", "rca", "expy", "cagr",
    )
    explanation = (
        "what is", "what are", "what does", "what do", "define", "definition",
        "meaning", "explain", "tell me about", "help me understand", "how is",
        "how are", "how do", "how should", "how to read", "interpret",
        "difference between", "different from", "is higher", "is lower",
        "is more", "is less", "always better", "good or bad", "why is",
        "why are", "why does", "versus", " vs ",
    )
    return any(term in q for term in concepts) and (
        any(term in q for term in explanation) or len(q.split()) <= 6
    )


def query_dashboard_sql(question: str, provider: str, model: str) -> DashboardAnswer:
    question = str(question or "").strip()
    if not question or not (DB_PLAIN_PATH.is_file() or DB_GZIP_PATH.is_file()):
        return DashboardAnswer(False, 0.0, "", "", {})

    # Concept and interpretation questions should never be forced into a SQL
    # ranking or aggregation. Use the dashboard glossary first.
    if _looks_like_concept_explanation(question):
        concept = query_dashboard(question)
        if concept.matched and concept.entities.get("metric") == "concept_explanation":
            return concept

    parsed = parse_question(question)

    corr = _correlation_answer(parsed)
    if corr:
        return corr

    plan = build_plan(parsed)
    if plan:
        if plan.kind == "direct":
            return DashboardAnswer(True, plan.confidence, deterministic_answer(plan, [], [], False, question), json.dumps({"direct": plan.direct_answer}), plan.entities)
        try:
            columns, rows, truncated = execute_sql(plan.sql)
            answer = deterministic_answer(plan, columns, rows, truncated, question)
            return DashboardAnswer(True, plan.confidence, answer, json.dumps({"sql": plan.sql, "rows": rows, "truncated": truncated}, ensure_ascii=False), plan.entities)
        except Exception as exc:
            print(f"Deterministic dashboard plan failed: {exc}")

    if _has_model_key():
        try:
            llm_plan = _llm_plan(question, provider, model)
            if not llm_plan.get("not_dashboard_data") and llm_plan.get("sql"):
                title = str(llm_plan.get("title") or "Dashboard result")
                sql = str(llm_plan["sql"])
                try:
                    columns, rows, truncated = execute_sql(sql)
                except Exception as first_exc:
                    repaired = _llm_plan(question, provider, model, repair=f"The prior query failed: {first_exc}. Previous SQL: {sql}")
                    sql = str(repaired.get("sql") or "")
                    title = str(repaired.get("title") or title)
                    columns, rows, truncated = execute_sql(sql)
                plan = QueryPlan(title, sql, resolve_dashboard_entities(question), 0.97)
                answer = deterministic_answer(plan, columns, rows, truncated, question)
                return DashboardAnswer(True, 0.97, answer, json.dumps({"sql": sql, "rows": rows, "truncated": truncated}, ensure_ascii=False), plan.entities)
        except Exception as exc:
            print(f"LLM dashboard planner failed: {exc}")

    fallback = _semantic_record_search(parsed)
    if fallback:
        try:
            columns, rows, truncated = execute_sql(fallback.sql)
            if rows:
                answer = deterministic_answer(fallback, columns, rows, truncated, question)
                return DashboardAnswer(True, fallback.confidence, answer, json.dumps({"sql": fallback.sql, "rows": rows}, ensure_ascii=False), fallback.entities)
        except Exception as exc:
            print(f"Raw dashboard record search failed: {exc}")

    return DashboardAnswer(False, 0.0, "", "", resolve_dashboard_entities(question))
