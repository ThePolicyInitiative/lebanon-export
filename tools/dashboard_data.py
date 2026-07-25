from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_HTML = ROOT / "dashboard_component" / "dashboard.html"
DASHBOARD_DATA_DIR = ROOT / "data"
DASHBOARD_DATA_FILES = {
    "bundle-data": DASHBOARD_DATA_DIR / "dashboard_bundle.json",
    "up-pair-data": DASHBOARD_DATA_DIR / "dashboard_up_pairs.json",
}


@dataclass(frozen=True)
class DashboardAnswer:
    matched: bool
    confidence: float
    answer: str
    context: str
    entities: dict[str, Any]


def _money(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "No data"
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1_000_000_000:
        return f"{sign}${amount / 1_000_000_000:.2f} billion"
    if amount >= 1_000_000:
        return f"{sign}${amount / 1_000_000:.2f} million"
    if amount >= 1_000:
        return f"{sign}${amount / 1_000:.1f} thousand"
    return f"{sign}${amount:,.0f}"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "No data"


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _hs6(value: Any) -> str:
    try:
        return f"{int(value):06d}"
    except (TypeError, ValueError):
        return str(value).zfill(6)


@lru_cache(maxsize=8)
def _load_script(script_id: str) -> Any:
    """Load the exact dashboard dataset from a packaged JSON mirror.

    The JSON files are generated directly from the matching <script> blocks in
    dashboard.html. Reading them avoids reparsing the 22 MB HTML file for every
    Streamlit process while preserving the same source data. If a mirror is
    missing, the original embedded script remains the fallback.
    """
    data_file = DASHBOARD_DATA_FILES.get(script_id)
    if data_file and data_file.is_file():
        return json.loads(data_file.read_text(encoding="utf-8", errors="strict"))

    text = DASHBOARD_HTML.read_text(encoding="utf-8", errors="strict")
    pattern = re.compile(
        rf'<script(?=[^>]*\bid=["\']{re.escape(script_id)}["\'])[^>]*>(.*?)</script>',
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise KeyError(f"Embedded dashboard dataset not found: {script_id}")
    return json.loads(match.group(1))


@lru_cache(maxsize=1)
def bundle() -> dict[str, Any]:
    return _load_script("bundle-data")


@lru_cache(maxsize=1)
def up_pairs() -> list[dict[str, Any]]:
    return _load_script("up-pair-data")


@lru_cache(maxsize=1)
def indices() -> dict[str, Any]:
    data = bundle()
    products = data.get("products", [])
    markets = data.get("markets", [])
    sectors = data.get("sectors", [])

    products_by_hs = {_hs6(row.get("hs6")): row for row in products}
    markets_by_norm = {_norm(row.get("country")): row for row in markets}
    sectors_by_norm = {_norm(row.get("sector")): row for row in sectors}

    product_market: dict[tuple[str, str, int], float] = {}
    for row in data.get("product_market_yearly", []):
        key = (_hs6(row.get("hs6")), _norm(row.get("country")), int(row.get("year")))
        product_market[key] = float(row.get("value") or 0)

    aliases = {
        "uae": "UAE",
        "united arab emirates": "UAE",
        "usa": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "russia": "Russian Federation",
        "south korea": "South Korea",
        "ivory coast": "Ivory Coast",
        "cote d ivoire": "Ivory Coast",
        "turkiye": "Turkey",
    }
    product_aliases = {
        "olive oil": "150910",
        "virgin olive oil": "150910",
        "pharmaceutical": "300490",
        "pharmaceuticals": "300490",
        "medicine": "300490",
        "medicaments": "300490",
        "jewelry": "711319",
        "jewellery": "711319",
        "wine": "220410",
        "chocolate": "180690",
        "perfume": "330300",
        "soap": "340111",
        "furniture": "940360",
        "detergent": "340220",
    }

    return {
        "products": products,
        "markets": markets,
        "sectors": sectors,
        "products_by_hs": products_by_hs,
        "markets_by_norm": markets_by_norm,
        "sectors_by_norm": sectors_by_norm,
        "product_market": product_market,
        "market_aliases": aliases,
        "product_aliases": product_aliases,
    }


def _years(question: str) -> list[int]:
    valid = set(bundle().get("meta", {}).get("years", range(2018, 2026)))
    return [int(x) for x in re.findall(r"\b20\d{2}\b", question) if int(x) in valid]


def _find_market(question: str) -> dict[str, Any] | None:
    idx = indices()
    q = _norm(question)
    extra_aliases = {
        "ksa": "Saudi Arabia", "saudia": "Saudi Arabia", "saudi": "Saudi Arabia",
        "emirates": "UAE", "u a e": "UAE", "cote divoire": "Ivory Coast",
        "cote d ivoire": "Ivory Coast", "britain": "United Kingdom",
        "america": "United States", "korea": "South Korea",
    }
    aliases = {**idx["market_aliases"], **extra_aliases}
    for alias, canonical in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return idx["markets_by_norm"].get(_norm(canonical))
    for name, row in sorted(idx["markets_by_norm"].items(), key=lambda item: len(item[0]), reverse=True):
        if name and re.search(rf"\b{re.escape(name)}\b", q):
            return row

    # Conservative typo tolerance: only consider one- or two-token phrases and
    # require a high similarity score to a country name.
    tokens = q.split()
    phrases = tokens + [" ".join(tokens[i:i + 2]) for i in range(max(0, len(tokens) - 1))]
    best_score, best_row = 0.0, None
    for phrase in phrases:
        if len(phrase) < 4:
            continue
        for name, row in idx["markets_by_norm"].items():
            score = SequenceMatcher(None, phrase, name).ratio()
            if score > best_score:
                best_score, best_row = score, row
    return best_row if best_score >= 0.88 else None


def _find_sector(question: str) -> dict[str, Any] | None:
    q = _norm(question)
    idx = indices()
    aliases = {
        "agrofood": "Agrifood", "food sector": "Agrifood", "food industry": "Agrifood",
        "machinery sector": "Electrical and Machinery", "electrical sector": "Electrical and Machinery",
        "pharma": "Pharma & Parapharma", "pharmaceutical sector": "Pharma & Parapharma",
        "plastics": "Plastics / Rubbers", "rubber sector": "Plastics / Rubbers",
        "wood sector": "Wood & Wood Products", "stone sector": "Stone / Glass",
        "glass sector": "Stone / Glass", "chemical sector": "Chemicals & Allied Industries",
        "fertilizer sector": "Fertilizers & Agri-inputs", "textile sector": "Textiles",
        "metal sector": "Metals", "furniture sector": "Furniture",
    }
    for alias, canonical in aliases.items():
        if alias in q:
            return idx["sectors_by_norm"].get(_norm(canonical))
    for name, row in sorted(idx["sectors_by_norm"].items(), key=lambda item: len(item[0]), reverse=True):
        if name and name in q:
            return row
    return None


def _find_product(question: str) -> tuple[dict[str, Any] | None, float]:
    idx = indices()
    q = _norm(question)
    for token in re.findall(r"\b\d{4,6}\b", question):
        if len(token) == 6 and token in idx["products_by_hs"]:
            return idx["products_by_hs"][token], 1.0
        candidates = [row for code, row in idx["products_by_hs"].items() if code.startswith(token)]
        if candidates:
            return max(candidates, key=lambda row: float(row.get("value_2025") or 0)), 0.95
    for alias, code in idx["product_aliases"].items():
        if alias in q and code in idx["products_by_hs"]:
            return idx["products_by_hs"][code], 0.98

    stop = {
        "what", "which", "how", "much", "many", "did", "does", "do", "is", "are", "was", "were",
        "lebanon", "lebanese", "export", "exports", "exported", "market", "markets", "destination",
        "destinations", "country", "countries", "product", "products", "sector", "sectors", "growth",
        "cagr", "rca", "comparative", "advantage", "top", "highest", "largest", "biggest", "leading",
        "best", "show", "tell", "about", "the", "and", "for", "from", "into", "with", "year", "years",
        "dashboard", "data", "potential", "unrealized", "untapped", "opportunity", "opportunities",
        "overall", "total", "totals", "summary", "headline", "value", "values", "share", "shares",
        "composition", "complexity", "concentration", "sophistication", "industrial", "base", "basket",
        "structure", "distribution", "annual", "every", "history", "new", "entered", "entry", "exited",
        "exit", "lost", "drivers", "driver", "contribution", "rank", "ranking", "position", "performance",
        "change", "changed", "increase", "increased", "decrease", "decreased", "rose", "fell", "evolution",
        "grew", "declined", "weak", "strong", "best", "worst", "target", "prioritize", "priority",
        "in", "on", "of", "to", "by", "at", "a", "an", "me", "please"
    }
    words = [w for w in q.split() if len(w) > 2 and not w.isdigit() and w not in stop]
    if not words:
        return None, 0.0
    query = " ".join(words)
    query_set = set(words)
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for row in idx["products"]:
        name = _norm(row.get("name"))
        name_words = set(name.split())
        overlap = len(query_set & name_words)
        if overlap == 0:
            continue
        ratio = SequenceMatcher(None, query, name).ratio()
        score = min(1.0, overlap / max(1, len(query_set)) * 0.75 + ratio * 0.25)
        if score > best[0]:
            best = (score, row)
    if best[0] < 0.58:
        return None, best[0]
    return best[1], best[0]


def _number(value: Any, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "No data"


def _value_of(row: dict[str, Any], year: int, prefix: str = "value") -> float:
    candidates = (
        row.get(f"{prefix}_{year}"),
        row.get(f"value_{year}"),
        row.get(f"exports_{year}"),
        (row.get("values") or {}).get(str(year)),
    )
    for value in candidates:
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _weighted_pci(products: list[dict[str, Any]], year: int) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for product in products:
        try:
            pci = float(product.get("pci"))
            exports = _value_of(product, year)
        except (TypeError, ValueError):
            continue
        if exports > 0:
            numerator += pci * exports
            denominator += exports
    return numerator / denominator if denominator else None


def _top_lines(
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
    n: int = 5,
    formatter: Any = _money,
) -> str:
    ordered = sorted(rows, key=lambda row: float(row.get(value_key) or 0), reverse=True)[:n]
    return "\n".join(
        f"{i}. **{row.get(label_key)}** — {formatter(row.get(value_key))}"
        for i, row in enumerate(ordered, 1)
    )


def _top_product_lines(rows: list[dict[str, Any]], year: int, n: int = 5) -> str:
    ordered = sorted(rows, key=lambda row: _value_of(row, year), reverse=True)[:n]
    return "\n".join(
        f"{i}. **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — {_money(_value_of(row, year))}"
        for i, row in enumerate(ordered, 1)
    )


def _complexity_summary(year: int) -> DashboardAnswer:
    idx = indices()
    active = [
        row for row in idx["products"]
        if row.get("pci") is not None and _value_of(row, year) > 0
    ]
    complex_products = [row for row in active if float(row.get("pci") or 0) >= 1]
    complex_exports = sum(_value_of(row, year) for row in complex_products)
    total_exports = sum(_value_of(row, year) for row in active)
    highest = sorted(active, key=lambda row: float(row.get("pci") or -999), reverse=True)[:5]

    sector_rows = []
    for sector in idx["sectors"]:
        products = [row for row in active if row.get("sector") == sector.get("sector")]
        weighted = _weighted_pci(products, year)
        if weighted is not None:
            sector_rows.append({"sector": sector.get("sector"), "pci": weighted})
    sector_rows.sort(key=lambda row: row["pci"], reverse=True)

    answer = (
        f"### Export complexity in {year}\n"
        "The dashboard measures product complexity with **PCI (Product Complexity Index)**. "
        "Higher PCI values indicate products that are generally more knowledge- and capability-intensive.\n\n"
        f"- **Active products with a PCI value:** {len(active):,}\n"
        f"- **Products with PCI ≥ 1:** {len(complex_products):,}\n"
        f"- **Exports from PCI ≥ 1 products:** {_money(complex_exports)}\n"
        f"- **Share of measured exports from PCI ≥ 1 products:** {_pct(complex_exports / total_exports if total_exports else 0)}\n\n"
        "**Highest-PCI active products:**\n"
        + "\n".join(
            f"{i}. **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — PCI {_number(row.get('pci'))}; exports {_money(_value_of(row, year))}"
            for i, row in enumerate(highest, 1)
        )
    )
    if sector_rows:
        answer += "\n\n**Highest export-weighted sector complexity:**\n" + "\n".join(
            f"{i}. **{row['sector']}** — weighted PCI {_number(row['pci'])}"
            for i, row in enumerate(sector_rows[:5], 1)
        )
    context = json.dumps(
        {
            "year": year,
            "active_product_count": len(active),
            "pci_ge_1_count": len(complex_products),
            "pci_ge_1_exports": complex_exports,
            "total_measured_exports": total_exports,
            "top_products": highest,
            "top_sectors": sector_rows[:10],
        },
        ensure_ascii=False,
    )
    return DashboardAnswer(True, 0.99, answer, context, {"year": year, "metric": "complexity"})


def _competitiveness_complexity_summary(year: int) -> DashboardAnswer:
    products = [row for row in indices()["products"] if _value_of(row, year) > 0]
    both = [row for row in products if row.get("pci") is not None and row.get(f"rca_{year}") is not None]
    categories = {
        "Star (RCA ≥ 1, PCI ≥ 1)": [],
        "Established (RCA ≥ 1, PCI < 1)": [],
        "Aspirational (RCA < 1, PCI ≥ 1)": [],
        "Marginal (RCA < 1, PCI < 1)": [],
    }
    for row in both:
        rca = float(row.get(f"rca_{year}") or 0)
        pci = float(row.get("pci") or 0)
        if rca >= 1 and pci >= 1:
            categories["Star (RCA ≥ 1, PCI ≥ 1)"].append(row)
        elif rca >= 1:
            categories["Established (RCA ≥ 1, PCI < 1)"].append(row)
        elif pci >= 1:
            categories["Aspirational (RCA < 1, PCI ≥ 1)"].append(row)
        else:
            categories["Marginal (RCA < 1, PCI < 1)"].append(row)

    measured_exports = sum(_value_of(row, year) for row in both)
    summary: dict[str, Any] = {}
    lines = [
        f"### Competitiveness and complexity in {year}",
        "The dashboard combines **RCA** (current revealed competitiveness) and **PCI** (product complexity).",
        "",
    ]
    for label, rows in categories.items():
        exports = sum(_value_of(row, year) for row in rows)
        share = exports / measured_exports if measured_exports else 0
        summary[label] = {"products": len(rows), "exports": exports, "share": share}
        lines.append(f"- **{label}:** {len(rows):,} products; {_money(exports)}; {_pct(share)} of measured exports")

    star_rows = sorted(categories["Star (RCA ≥ 1, PCI ≥ 1)"], key=lambda row: _value_of(row, year), reverse=True)[:5]
    aspirational_rows = sorted(categories["Aspirational (RCA < 1, PCI ≥ 1)"], key=lambda row: _value_of(row, year), reverse=True)[:5]
    if star_rows:
        lines.extend(["", "**Largest star products**"])
        lines.extend(
            f"- **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — {_money(_value_of(row, year))}; RCA {_number(row.get(f'rca_{year}'))}; PCI {_number(row.get('pci'))}"
            for row in star_rows
        )
    if aspirational_rows:
        lines.extend(["", "**Largest high-complexity products without RCA ≥ 1**"])
        lines.extend(
            f"- **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — {_money(_value_of(row, year))}; RCA {_number(row.get(f'rca_{year}'))}; PCI {_number(row.get('pci'))}"
            for row in aspirational_rows
        )
    lines.extend([
        "",
        "**Key implication**",
        "- The comparison separates products where Lebanon is already competitive from higher-complexity products that may represent capability-building opportunities. It does not by itself establish commercial feasibility.",
    ])
    return DashboardAnswer(
        True,
        0.99,
        "\n".join(lines),
        json.dumps({"year": year, "categories": summary, "star_products": star_rows, "aspirational_products": aspirational_rows}, ensure_ascii=False),
        {"year": year, "metric": "competitiveness_complexity"},
    )


def _diversification_summary(year: int) -> DashboardAnswer:
    idx = indices()
    active_products = [row for row in idx["products"] if _value_of(row, year) > 0]
    active_markets = [row for row in idx["markets"] if _value_of(row, year, "exports") > 0]
    widest_products = sorted(
        active_products,
        key=lambda row: int((row.get("n_countries_by_year") or {}).get(str(year)) or row.get("n_countries") or 0),
        reverse=True,
    )[:5]
    broadest_markets = sorted(
        active_markets,
        key=lambda row: int(row.get(f"products_{year}") or row.get("n_products_hs6") or 0),
        reverse=True,
    )[:5]
    answer = (
        f"### Export diversification in {year}\n"
        "Diversification is shown through the number of products and destination markets. "
        "For individual markets, the dashboard also reports HHI, where a higher value means exports are more concentrated in fewer products.\n\n"
        f"- **Active exported products:** {len(active_products):,}\n"
        f"- **Active destination markets:** {len(active_markets):,}\n\n"
        "**Products reaching the most markets:**\n"
        + "\n".join(
            f"{i}. **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — "
            f"{int((row.get('n_countries_by_year') or {}).get(str(year)) or row.get('n_countries') or 0):,} markets"
            for i, row in enumerate(widest_products, 1)
        )
        + "\n\n**Markets receiving the broadest product range:**\n"
        + "\n".join(
            f"{i}. **{row.get('country')}** — {int(row.get(f'products_{year}') or row.get('n_products_hs6') or 0):,} products"
            for i, row in enumerate(broadest_markets, 1)
        )
    )
    return DashboardAnswer(
        True,
        0.98,
        answer,
        json.dumps({"year": year, "widest_products": widest_products, "broadest_markets": broadest_markets}, ensure_ascii=False),
        {"year": year, "metric": "diversification"},
    )


def _query_dashboard_single(question: str) -> DashboardAnswer:
    data = bundle()
    idx = indices()
    q = _norm(question)
    years = _years(question)
    year = years[-1] if years else 2025
    market = _find_market(question)
    sector = _find_sector(question)
    product, product_score = _find_product(question)

    asks_potential = any(term in q for term in ("potential", "unrealized", "untapped", "opportunity"))
    asks_top = any(term in q for term in ("top", "largest", "biggest", "leading", "best", "highest"))
    asks_lowest = any(term in q for term in ("lowest", "least", "bottom"))
    asks_rca = "rca" in q or "comparative advantage" in q or "competitiveness" in q
    asks_complexity = any(term in q for term in (
        "complexity", "complex", "most complex", "least complex", "economic complexity",
        "knowledge intensive", "knowledge intensive products", "capability intensive",
        "productive capabilities", "product complexity index",
    )) or re.search(r"\bpci\b", q) is not None
    asks_sophistication = any(term in q for term in (
        "sophistication", "sophisticated basket", "export sophistication", "basket sophistication"
    )) or re.search(r"\bexpy\b", q) is not None
    asks_diversification = any(term in q for term in (
        "diversification", "diversified", "diversify", "breadth", "variety",
        "range of products", "range of markets",
    ))
    asks_concentration = any(term in q for term in (
        "concentration", "concentrated", "concentrate", "dependence", "dependent on few products"
    )) or re.search(r"\bhhi\b", q) is not None
    asks_composition = any(term in q for term in (
        "composition", "industrial base", "export basket", "sector mix", "export structure",
        "distribution by sector", "sectoral distribution",
    ))
    asks_scale = any(term in q for term in (
        "scale", "size of exports", "export size", "how large", "how big", "export value"
    ))
    asks_growth = any(term in q for term in (
        "growth", "cagr", "change", "trend", "increase", "decrease", "rose", "fell",
        "evolution", "grew", "declined",
    ))
    asks_performance = any(term in q for term in (
        "overperform", "underperform", "outperform", "lagging", "performance map", "performance"
    ))
    asks_new_exited = any(term in q for term in ("new products", "exited products", "product entry", "product exit"))
    asks_definition = any(term in q for term in ("what is", "define", "meaning", "explain"))
    asks_compare = any(term in q for term in ("compare", "versus", " vs ", "difference between", "from ")) or len(years) >= 2

    explicit_product = product is not None and product_score >= 0.9
    dashboard_terms = ("dashboard", "total export", "overall export", "industrial export", "export total", "headline")

    # Multi-year comparisons preserve the exact yearly values visible in the dashboard.
    comparison_years = sorted(set(years))
    if asks_compare and len(comparison_years) >= 2:
        first_year, last_year = comparison_years[0], comparison_years[-1]
        if product:
            first_value = _value_of(product, first_year)
            last_value = _value_of(product, last_year)
            change = (last_value / first_value - 1) if first_value else None
            hs = _hs6(product.get("hs6"))
            lines = [
                f"### {product.get('name')}: {first_year} compared with {last_year}",
                f"- **HS6:** `{hs}`",
                f"- **Exports in {first_year}:** {_money(first_value)}",
                f"- **Exports in {last_year}:** {_money(last_value)}",
                f"- **Change:** {_pct(change) if change is not None else 'Not calculable from a zero base'}",
            ]
            if asks_rca:
                lines.extend([
                    f"- **RCA in {first_year}:** {_number(product.get(f'rca_{first_year}'))}",
                    f"- **RCA in {last_year}:** {_number(product.get(f'rca_{last_year}'))}",
                ])
            return DashboardAnswer(
                True, 0.99, "\n".join(lines),
                json.dumps({"product": product, "years": comparison_years}, ensure_ascii=False),
                {"hs6": hs, "years": comparison_years, "metric": "comparison"},
            )
        if market:
            first_value = _value_of(market, first_year, "exports")
            last_value = _value_of(market, last_year, "exports")
            change = (last_value / first_value - 1) if first_value else None
            answer = (
                f"### {market.get('country')}: {first_year} compared with {last_year}\n"
                f"- **Exports in {first_year}:** {_money(first_value)}\n"
                f"- **Exports in {last_year}:** {_money(last_value)}\n"
                f"- **Change:** {_pct(change) if change is not None else 'Not calculable from a zero base'}\n"
                f"- **Products in {first_year}:** {int(market.get(f'products_{first_year}') or 0):,}\n"
                f"- **Products in {last_year}:** {int(market.get(f'products_{last_year}') or 0):,}"
            )
            return DashboardAnswer(True, 0.99, answer, json.dumps(market, ensure_ascii=False), {"market": market.get("country"), "years": comparison_years, "metric": "comparison"})
        if sector:
            first_value = _value_of(sector, first_year)
            last_value = _value_of(sector, last_year)
            change = (last_value / first_value - 1) if first_value else None
            answer = (
                f"### {sector.get('sector')}: {first_year} compared with {last_year}\n"
                f"- **Exports in {first_year}:** {_money(first_value)}\n"
                f"- **Exports in {last_year}:** {_money(last_value)}\n"
                f"- **Change:** {_pct(change) if change is not None else 'Not calculable from a zero base'}\n"
                f"- **RCA in {first_year}:** {_number(sector.get(f'rca_{first_year}'))}\n"
                f"- **RCA in {last_year}:** {_number(sector.get(f'rca_{last_year}'))}"
            )
            return DashboardAnswer(True, 0.99, answer, json.dumps(sector, ensure_ascii=False), {"sector": sector.get("sector"), "years": comparison_years, "metric": "comparison"})
        totals = {int(row["year"]): float(row.get("filtered") or 0) for row in data.get("totals_by_year", [])}
        if first_year in totals and last_year in totals:
            first_value, last_value = totals[first_year], totals[last_year]
            change = (last_value / first_value - 1) if first_value else None
            answer = (
                f"### Total industrial exports: {first_year} compared with {last_year}\n"
                f"- **Exports in {first_year}:** {_money(first_value)}\n"
                f"- **Exports in {last_year}:** {_money(last_value)}\n"
                f"- **Change:** {_pct(change) if change is not None else 'Not calculable from a zero base'}"
            )
            return DashboardAnswer(True, 0.98, answer, json.dumps(totals, ensure_ascii=False), {"years": comparison_years, "metric": "comparison"})

    # Product- and sector-specific complexity questions should use their exact PCI values.
    if asks_complexity and product:
        hs = _hs6(product.get("hs6"))
        answer = (
            f"### Complexity of {product.get('name')}\n"
            f"- **HS6:** `{hs}`\n"
            f"- **Product Complexity Index (PCI):** {_number(product.get('pci'))}\n"
            f"- **Sector:** {product.get('sector')}\n"
            f"- **Exports in {year}:** {_money(_value_of(product, year))}\n"
            f"- **RCA in {year}:** {_number(product.get(f'rca_{year}'))}\n\n"
            "A higher PCI indicates a product associated with a broader and less commonly available capability base."
        )
        return DashboardAnswer(True, 0.99, answer, json.dumps(product, ensure_ascii=False), {"hs6": hs, "year": year, "metric": "pci"})

    if asks_complexity and sector:
        products = [row for row in idx["products"] if row.get("sector") == sector.get("sector")]
        weighted = _weighted_pci(products, year)
        active = [row for row in products if _value_of(row, year) > 0 and row.get("pci") is not None]
        high = sorted(active, key=lambda row: float(row.get("pci") or -999), reverse=True)[:5]
        answer = (
            f"### Complexity of {sector.get('sector')} in {year}\n"
            f"- **Export-weighted PCI:** {_number(weighted)}\n"
            f"- **Active products with PCI data:** {len(active):,}\n"
            f"- **Exports:** {_money(_value_of(sector, year))}\n\n"
            "**Highest-PCI products in the sector:**\n"
            + "\n".join(
                f"{i}. **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — PCI {_number(row.get('pci'))}"
                for i, row in enumerate(high, 1)
            )
        )
        return DashboardAnswer(True, 0.99, answer, json.dumps({"sector": sector, "products": high}, ensure_ascii=False), {"sector": sector.get("sector"), "year": year, "metric": "pci"})

    if asks_complexity and market:
        expy_year = 2018 if year == 2018 else 2025
        answer = (
            f"### Export sophistication of Lebanon's basket to {market.get('country')}\n"
            f"- **EXPY {expy_year}:** {_number(market.get(f'expy_{expy_year}'), 0)}\n"
            f"- **Products in {year}:** {int(market.get(f'products_{year}') or market.get('n_products_hs6') or 0):,}\n"
            f"- **Exports in {year}:** {_money(_value_of(market, year, 'exports'))}\n"
            f"- **HHI concentration:** {_number(market.get('hhi'), 3)}\n\n"
            "The dashboard uses EXPY for the sophistication of a destination-specific export basket and PCI for individual products."
        )
        return DashboardAnswer(True, 0.98, answer, json.dumps(market, ensure_ascii=False), {"market": market.get("country"), "year": year, "metric": "expy"})

    if asks_complexity and asks_rca:
        return _competitiveness_complexity_summary(year)
    if asks_complexity:
        return _complexity_summary(year)

    if asks_sophistication:
        if asks_definition and not market and not asks_top and not asks_lowest:
            answer = (
                "### EXPY in the dashboard\n"
                "**EXPY** is the export sophistication index. It summarizes the sophistication level associated with a destination-specific export basket. "
                "Higher EXPY generally means that basket is concentrated in products associated with higher productive capabilities."
            )
            return DashboardAnswer(True, 0.98, answer, json.dumps({"definition": answer}, ensure_ascii=False), {"metric": "expy"})
        if market:
            expy_year = 2018 if year == 2018 else 2025
            answer = (
                f"### Export sophistication for {market.get('country')}\n"
                f"- **EXPY {expy_year}:** {_number(market.get(f'expy_{expy_year}'), 0)}\n"
                f"- **Exports in {year}:** {_money(_value_of(market, year, 'exports'))}\n"
                f"- **Products in {year}:** {int(market.get(f'products_{year}') or market.get('n_products_hs6') or 0):,}\n\n"
                "EXPY summarizes the sophistication level associated with the export basket sent to that market."
            )
            return DashboardAnswer(True, 0.98, answer, json.dumps(market, ensure_ascii=False), {"market": market.get("country"), "metric": "expy"})
        rows = [row for row in idx["markets"] if row.get("expy_2025") is not None]
        rows.sort(key=lambda row: float(row.get("expy_2025") or 0), reverse=not asks_lowest)
        answer = "### Destination baskets by EXPY in 2025\n" + "\n".join(
            f"{i}. **{row.get('country')}** — EXPY {_number(row.get('expy_2025'), 0)}; exports {_money(row.get('exports_2025'))}"
            for i, row in enumerate(rows[:5], 1)
        )
        return DashboardAnswer(True, 0.96, answer, json.dumps(rows[:10], ensure_ascii=False), {"metric": "expy", "year": 2025})

    if asks_diversification and market:
        answer = (
            f"### Diversification in {market.get('country')} in {year}\n"
            f"- **Products exported:** {int(market.get(f'products_{year}') or market.get('n_products_hs6') or 0):,}\n"
            f"- **HHI concentration:** {_number(market.get('hhi'), 3)}\n"
            f"- **New products:** {int(market.get('new_products') or 0):,}\n"
            f"- **Exited products:** {int(market.get('exited_products') or 0):,}\n"
            f"- **Exports:** {_money(_value_of(market, year, 'exports'))}\n\n"
            "A lower HHI indicates a more diversified product mix; a higher HHI indicates greater concentration."
        )
        return DashboardAnswer(True, 0.99, answer, json.dumps(market, ensure_ascii=False), {"market": market.get("country"), "year": year, "metric": "diversification"})
    if asks_diversification:
        return _diversification_summary(year)

    if asks_concentration:
        if asks_definition and not market and not asks_top and not asks_lowest:
            answer = (
                "### HHI in the dashboard\n"
                "**HHI (Herfindahl–Hirschman Index)** measures how concentrated a destination's Lebanese import basket is across products. "
                "Values closer to 1 indicate concentration in fewer products; lower values indicate a more diversified basket."
            )
            return DashboardAnswer(True, 0.98, answer, json.dumps({"definition": answer}, ensure_ascii=False), {"metric": "hhi"})
        if market:
            answer = (
                f"### Export concentration in {market.get('country')}\n"
                f"- **HHI:** {_number(market.get('hhi'), 3)}\n"
                f"- **Products in {year}:** {int(market.get(f'products_{year}') or market.get('n_products_hs6') or 0):,}\n"
                f"- **Exports in {year}:** {_money(_value_of(market, year, 'exports'))}\n\n"
                "HHI is higher when exports are concentrated in a smaller number of products."
            )
            return DashboardAnswer(True, 0.99, answer, json.dumps(market, ensure_ascii=False), {"market": market.get("country"), "metric": "hhi"})
        rows = [row for row in idx["markets"] if row.get("hhi") is not None and _value_of(row, year, "exports") > 0]
        rows.sort(key=lambda row: float(row.get("hhi") or 0), reverse=not asks_lowest)
        label = "Most concentrated" if not asks_lowest else "Least concentrated"
        answer = f"### {label} destination baskets in {year}\n" + "\n".join(
            f"{i}. **{row.get('country')}** — HHI {_number(row.get('hhi'), 3)}; {int(row.get(f'products_{year}') or 0):,} products"
            for i, row in enumerate(rows[:5], 1)
        )
        return DashboardAnswer(True, 0.96, answer, json.dumps(rows[:10], ensure_ascii=False), {"year": year, "metric": "hhi"})

    if asks_composition:
        sectors = sorted(idx["sectors"], key=lambda row: _value_of(row, year), reverse=True)
        total = sum(_value_of(row, year) for row in sectors)
        answer = f"### Composition of Lebanon's industrial export base in {year}\n" + "\n".join(
            f"{i}. **{row.get('sector')}** — {_money(_value_of(row, year))} ({_pct(_value_of(row, year) / total if total else 0)})"
            for i, row in enumerate(sectors[:8], 1)
        )
        return DashboardAnswer(True, 0.98, answer, json.dumps(sectors, ensure_ascii=False), {"year": year, "metric": "composition"})

    if asks_scale and market:
        answer = (
            f"### Export scale in {market.get('country')} in {year}\n"
            f"- **Exports:** {_money(_value_of(market, year, 'exports'))}\n"
            f"- **Products:** {int(market.get(f'products_{year}') or market.get('n_products_hs6') or 0):,}\n"
            f"- **CAGR:** {_pct(market.get('cagr'))}\n"
        )
        return DashboardAnswer(True, 0.98, answer, json.dumps(market, ensure_ascii=False), {"market": market.get("country"), "year": year, "metric": "scale"})

    if asks_performance and not market:
        topsis = data.get("topsis", {})
        answer = (
            "### Market performance groups in the dashboard\n"
            "**Overperformers:** " + ", ".join(topsis.get("overperformers", [])) + "\n\n"
            "**Underperformers:** " + ", ".join(topsis.get("underperformers", [])) + "\n\n"
            "These groups come from the dashboard's market-performance classification."
        )
        return DashboardAnswer(True, 0.96, answer, json.dumps(topsis, ensure_ascii=False), {"metric": "performance"})

    if asks_new_exited and market:
        answer = (
            f"### Product entry and exit in {market.get('country')}\n"
            f"- **New products:** {int(market.get('new_products') or 0):,}\n"
            f"- **Exited products:** {int(market.get('exited_products') or 0):,}\n"
            f"- **Products in 2018:** {int(market.get('products_2018') or 0):,}\n"
            f"- **Products in 2025:** {int(market.get('products_2025') or 0):,}\n"
        )
        return DashboardAnswer(True, 0.99, answer, json.dumps(market, ensure_ascii=False), {"market": market.get("country"), "metric": "entry_exit"})

    if asks_rca and product:
        hs = _hs6(product.get("hs6"))
        answer = (
            f"### Competitiveness of {product.get('name')} in {year}\n"
            f"- **HS6:** `{hs}`\n"
            f"- **RCA:** {_number(product.get(f'rca_{year}'))}\n"
            f"- **PCI:** {_number(product.get('pci'))}\n"
            f"- **Exports:** {_money(_value_of(product, year))}\n\n"
            "RCA ≥ 1 indicates revealed comparative advantage in the dashboard."
        )
        return DashboardAnswer(True, 0.99, answer, json.dumps(product, ensure_ascii=False), {"hs6": hs, "year": year, "metric": "rca"})

    if asks_rca and sector:
        products = [row for row in idx["products"] if row.get("sector") == sector.get("sector") and _value_of(row, year) > 0]
        competitive = [row for row in products if float(row.get(f"rca_{year}") or 0) >= 1]
        exports = sum(_value_of(row, year) for row in competitive)
        answer = (
            f"### Competitiveness of {sector.get('sector')} in {year}\n"
            f"- **Sector RCA:** {_number(sector.get(f'rca_{year}'))}\n"
            f"- **Products with RCA ≥ 1:** {len(competitive):,}\n"
            f"- **Exports from RCA ≥ 1 products:** {_money(exports)}\n"
        )
        return DashboardAnswer(True, 0.98, answer, json.dumps({"sector": sector, "competitive_products": competitive}, ensure_ascii=False), {"sector": sector.get("sector"), "year": year, "metric": "rca"})

    if asks_rca and not explicit_product and market is None and sector is None:
        if asks_definition and not asks_top:
            answer = (
                "### RCA in the dashboard\n"
                "**RCA (Revealed Comparative Advantage)** compares a product's importance in Lebanon's export basket with its importance in world trade. "
                "An RCA of **1 or more** is treated as revealed competitiveness."
            )
            return DashboardAnswer(True, 0.98, answer, json.dumps({"definition": answer}, ensure_ascii=False), {"metric": "rca"})
        rows = [row for row in idx["products"] if row.get(f"rca_{year}") is not None and _value_of(row, year) > 0]
        rows.sort(key=lambda row: float(row.get(f"rca_{year}") or 0), reverse=not asks_lowest)
        answer = f"### {'Highest' if not asks_lowest else 'Lowest'} active product RCA values in {year}\n" + "\n".join(
            f"{i}. **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — {_number(row.get(f'rca_{year}'))}"
            for i, row in enumerate(rows[:5], 1)
        )
        return DashboardAnswer(True, 0.96, answer, json.dumps(rows[:10], ensure_ascii=False), {"year": year, "metric": "rca"})

    if asks_top and not explicit_product and market is None and sector is None:
        if "sector" in q:
            rows = sorted(idx["sectors"], key=lambda row: _value_of(row, year), reverse=True)
            answer = f"### Top sectors in {year}\n" + "\n".join(
                f"{i}. **{row.get('sector')}** — {_money(_value_of(row, year))}"
                for i, row in enumerate(rows[:5], 1)
            )
            return DashboardAnswer(True, 0.96, answer, json.dumps(rows[:20], ensure_ascii=False), {"year": year})
        if "market" in q or "destination" in q or "country" in q:
            rows = sorted(idx["markets"], key=lambda row: _value_of(row, year, "exports"), reverse=True)
            answer = f"### Top destination markets in {year}\n" + "\n".join(
                f"{i}. **{row.get('country')}** — {_money(_value_of(row, year, 'exports'))}"
                for i, row in enumerate(rows[:5], 1)
            )
            return DashboardAnswer(True, 0.96, answer, json.dumps(rows[:20], ensure_ascii=False), {"year": year})
        rows = sorted(idx["products"], key=lambda row: _value_of(row, year), reverse=True)
        answer = f"### Top products in {year}\n" + _top_product_lines(rows, year)
        return DashboardAnswer(True, 0.96, answer, json.dumps(rows[:20], ensure_ascii=False), {"year": year})

    if asks_growth and not explicit_product and market is None and sector is None:
        totals = {int(row["year"]): float(row.get("filtered") or 0) for row in data.get("totals_by_year", [])}
        start = min(totals)
        end = year if year in totals else max(totals)
        start_value = totals[start]
        end_value = totals[end]
        years_elapsed = max(1, end - start)
        cagr = (end_value / start_value) ** (1 / years_elapsed) - 1 if start_value > 0 else 0
        answer = (
            f"### Overall industrial export trend, {start}–{end}\n"
            f"- **Exports in {start}:** {_money(start_value)}\n"
            f"- **Exports in {end}:** {_money(end_value)}\n"
            f"- **Total change:** {_pct(end_value / start_value - 1 if start_value else 0)}\n"
            f"- **Calculated CAGR:** {_pct(cagr)}\n"
        )
        return DashboardAnswer(True, 0.96, answer, json.dumps(totals, ensure_ascii=False), {"start_year": start, "end_year": end, "metric": "growth"})

    if (any(term in q for term in dashboard_terms) or ("export" in q and years) or (asks_scale and not market)) and not explicit_product and market is None and sector is None:
        totals = {int(row["year"]): row for row in data.get("totals_by_year", [])}
        row = totals.get(year) or totals[max(totals)]
        nominal = float(row.get("filtered") or 0)
        real = float(row.get("filtered_cpi_adj") or 0)
        meta = data.get("meta", {})
        answer = (
            f"### Dashboard summary for {year}\n"
            f"- **Nominal industrial exports:** {_money(nominal)}\n"
            f"- **Real exports (2018 prices):** {_money(real)}\n"
            f"- **Products covered:** {meta.get('n_products')}\n"
            f"- **Markets covered:** {meta.get('n_markets')}\n"
            f"- **Sectors covered:** {meta.get('n_sectors')}\n"
            f"- **Total unrealized potential:** {_money(float(meta.get('up_eu_total_million_usd') or 0) * 1_000_000)}\n"
        )
        return DashboardAnswer(True, 0.96, answer, json.dumps({"total": row, "meta": meta}, ensure_ascii=False), {"year": year})

    if product and market:
        hs = _hs6(product.get("hs6"))
        country = str(market.get("country"))
        value = idx["product_market"].get((hs, _norm(country), year), 0.0)
        total_product = _value_of(product, year)
        share = value / total_product if total_product else 0
        potential = 0.0
        if asks_potential:
            potential = sum(
                float(row.get("value_usd") or row.get("value") or 0)
                for row in up_pairs()
                if _hs6(row.get("hs6")) == hs and _norm(row.get("country")) == _norm(country)
            )
        answer = (
            f"### {product.get('name')} → {country}\n"
            f"- **HS6:** `{hs}`\n"
            f"- **Exports in {year}:** {_money(value)}\n"
            f"- **Total Lebanese exports of this product in {year}:** {_money(total_product)}\n"
            f"- **Destination share:** {_pct(share)}\n"
            f"- **Product RCA in {year}:** {_number(product.get(f'rca_{year}'))}\n"
            f"- **Product PCI:** {_number(product.get('pci'))}\n"
        )
        if asks_potential:
            answer += f"- **Recorded unrealized potential:** {_money(potential)}\n"
        context = json.dumps({"product": product, "market": market, "year": year, "product_market_exports": value, "unrealized_potential": potential}, ensure_ascii=False)
        return DashboardAnswer(True, 0.99, answer, context, {"hs6": hs, "market": country, "year": year})

    if product:
        hs = _hs6(product.get("hs6"))
        value = _value_of(product, year)
        previous_year = 2018 if year != 2018 else 2019
        previous = _value_of(product, previous_year)
        destinations = product.get("top_destinations", [])[:5]
        answer = (
            f"### {product.get('name')}\n"
            f"- **HS6:** `{hs}`\n"
            f"- **Sector:** {product.get('sector')}\n"
            f"- **Exports in {year}:** {_money(value)}\n"
            f"- **Exports in {previous_year}:** {_money(previous)}\n"
            f"- **Change, 2018–2025:** {_pct(product.get('change_2018_2025'))}\n"
            f"- **CAGR used by dashboard:** {_pct(product.get('cagr'))}\n"
            f"- **RCA in {year}:** {_number(product.get(f'rca_{year}'))}\n"
            f"- **PCI complexity:** {_number(product.get('pci'))}\n"
            f"- **Markets in {year}:** {int((product.get('n_countries_by_year') or {}).get(str(year)) or product.get('n_countries') or 0)}\n"
        )
        if destinations:
            answer += "- **Top destinations:** " + "; ".join(
                f"{row.get('country')} ({_money(row.get('value_2025'))})" for row in destinations
            ) + "\n"
        if asks_potential:
            answer += f"- **Total unrealized potential:** {_money(product.get('unrealized_potential_eu'))}\n"
        return DashboardAnswer(True, max(0.82, product_score), answer, json.dumps(product, ensure_ascii=False), {"hs6": hs, "year": year})

    if market:
        country = str(market.get("country"))
        value = _value_of(market, year, "exports")
        expy_year = 2018 if year == 2018 else 2025
        answer = (
            f"### {country}\n"
            f"- **Lebanese exports in {year}:** {_money(value)}\n"
            f"- **Growth, 2018–2025:** {_pct(market.get('growth'))}\n"
            f"- **Dashboard CAGR:** {_pct(market.get('cagr'))}\n"
            f"- **Products in {year}:** {int(market.get(f'products_{year}') or market.get('n_products_hs6') or 0)}\n"
            f"- **HHI concentration:** {_number(market.get('hhi'), 3)}\n"
            f"- **EXPY {expy_year}:** {_number(market.get(f'expy_{expy_year}'), 0)}\n"
            f"- **Market status:** {market.get('status')}\n"
            f"- **Unrealized potential:** {_money(market.get('unrealized_potential_exact') or market.get('unrealized_potential'))}\n"
        )
        top_products = market.get("top_products", [])[:5]
        if top_products:
            answer += "- **Top products:** " + "; ".join(
                f"{row.get('name')} [{_hs6(row.get('hs6'))}] ({_money(row.get('value_2025'))})"
                for row in top_products
            ) + "\n"
        return DashboardAnswer(True, 0.97, answer, json.dumps(market, ensure_ascii=False), {"market": country, "year": year})

    if sector:
        name = str(sector.get("sector"))
        value = _value_of(sector, year)
        products = [row for row in idx["products"] if row.get("sector") == name]
        weighted_pci = _weighted_pci(products, year)
        answer = (
            f"### {name}\n"
            f"- **Exports in {year}:** {_money(value)}\n"
            f"- **Products (HS6):** {int(sector.get('n_products_hs6') or sector.get('n_products') or 0)}\n"
            f"- **Change, 2018–2025:** {_pct(sector.get('change_2018_2025'))}\n"
            f"- **Dashboard CAGR:** {_pct(sector.get('cagr'))}\n"
            f"- **Export share in {year}:** {_pct(sector.get(f'share_{year}'))}\n"
            f"- **Sector RCA in {year}:** {_number(sector.get(f'rca_{year}'))}\n"
            f"- **Export-weighted PCI:** {_number(weighted_pci)}\n"
            f"- **Unrealized potential:** {_money(sector.get('unrealized_potential_eu'))}\n"
        )
        if asks_top and products:
            answer += "\n**Top products:**\n" + _top_product_lines(products, year)
        return DashboardAnswer(True, 0.96, answer, json.dumps(sector, ensure_ascii=False), {"sector": name, "year": year})

    return DashboardAnswer(False, 0.0, "", "", {})

# ---------------------------------------------------------------------------
# ENHANCED QUERY PLANNER
# ---------------------------------------------------------------------------

_METRIC_TERMS = {
    "exports", "export", "value", "share", "rank", "ranking", "top", "largest",
    "growth", "cagr", "trend", "change", "driver", "drivers", "drove", "contribution",
    "complexity", "pci", "rca", "competitiveness", "potential", "unrealized", "untapped",
    "diversification", "hhi", "concentration", "expy", "sophistication", "composition",
    "products", "markets", "sectors", "performance", "overperform", "underperform",
    "similar", "comparable", "new products", "exited products",
}


def _total_for_year(year: int) -> float:
    for row in bundle().get("totals_by_year", []):
        if int(row.get("year")) == int(year):
            return float(row.get("filtered") or 0)
    return 0.0


def _rank_of(rows: list[dict[str, Any]], target: dict[str, Any], value_fn: Any) -> tuple[int, int]:
    ordered = sorted(rows, key=value_fn, reverse=True)
    for position, row in enumerate(ordered, 1):
        if row is target or row == target:
            return position, len(ordered)
    return 0, len(ordered)




def _metric_definition_answer(question: str) -> DashboardAnswer | None:
    """Explain dashboard metrics without accidentally returning a ranking."""
    q = _norm(question)
    definitions = {
        "rca": (
            "Revealed Comparative Advantage (RCA)",
            "RCA compares a product's importance in Lebanon's export basket with its importance in world trade. "
            "An RCA of 1 is the usual threshold: values above 1 indicate revealed specialization; values below 1 do not.",
            "RCA is descriptive. It does not prove profitability, productive capacity, or future demand. Very large values should be read together with export value because a small world-trade denominator can magnify the ratio.",
        ),
        "pci": (
            "Product Complexity Index (PCI)",
            "PCI measures how capability-intensive and uncommon a product is in the global productive system. Higher values are associated with products requiring a broader, less widely available capability base.",
            "PCI describes the product, not Lebanon's ability to scale it. Read it together with export value, RCA, market reach, and growth.",
        ),
        "hhi": (
            "Herfindahl-Hirschman Index (HHI)",
            "HHI measures how concentrated Lebanon's exports to a destination are across products. It is the sum of squared product shares, so higher values mean greater reliance on a small number of products.",
            "A lower HHI usually signals a more diversified basket, but it does not by itself indicate higher value added or stronger competitiveness.",
        ),
        "expy": (
            "EXPY export sophistication",
            "EXPY summarizes the income or sophistication level associated with the products in a destination-specific export basket, weighted by their export shares.",
            "It describes basket sophistication, not market profitability or domestic productive capacity.",
        ),
        "cagr": (
            "Compound Annual Growth Rate (CAGR)",
            "CAGR is the constant annual rate that would connect a starting export value to an ending value over a period.",
            "The dashboard notes that its 2025 CAGR uses 2018–2024 endpoints, while the displayed growth measure can still use 2025.",
        ),
        "potential": (
            "Unrealized export potential",
            "Unrealized potential is the positive product–destination opportunity value embedded in the dashboard's uploaded potential matrix after matching it to the dashboard HS6 universe.",
            "It is an opportunity estimate, not a forecast or guaranteed additional export value. A missing positive pair should be described as no recorded potential, not as proof that opportunity is zero.",
        ),
    }
    metric = None
    if re.search(r"\brca\b", q) or "comparative advantage" in q:
        metric = "rca"
    elif re.search(r"\bpci\b", q) or "product complexity index" in q:
        metric = "pci"
    elif re.search(r"\bhhi\b", q) or "herfindahl" in q:
        metric = "hhi"
    elif re.search(r"\bexpy\b", q) or "export sophistication" in q:
        metric = "expy"
    elif re.search(r"\bcagr\b", q) or "compound annual growth" in q:
        metric = "cagr"
    elif any(term in q for term in ("unrealized potential", "untapped potential", "export potential")):
        metric = "potential"
    if not metric:
        return None

    # A genuinely named entity or an explicit ranking request should use a data route instead.
    candidate_product, candidate_score = _find_product(question)
    if _find_market(question) or _find_sector(question) or (candidate_product is not None and candidate_score >= 0.90):
        return None
    ranking_words = ("top", "highest", "lowest", "rank", "compare", "versus", " vs ")
    if any(word in q for word in ranking_words):
        return None
    asks_definition = any(term in q for term in ("what is", "what does", "define", "meaning", "explain", "how is", "what about"))
    if not asks_definition and len(q.split()) > 4:
        return None
    title, definition, limit = definitions[metric]
    answer = f"### {title}\n{definition}\n\n**How to read it**\n- {limit}"
    return DashboardAnswer(True, 0.99, answer, json.dumps({"metric": metric}, ensure_ascii=False), {"metric": metric})


def _annual_series_answer(question: str) -> DashboardAnswer | None:
    """Return a complete year-by-year series rather than only endpoints."""
    q = _norm(question)
    if not any(term in q for term in ("every year", "year by year", "annual series", "annual trend", "annual history", "each year", "all years")):
        return None
    valid_years = list(bundle().get("meta", {}).get("years", range(2018, 2026)))
    product, score = _find_product(question)
    market = _find_market(question)
    sector = _find_sector(question)

    if product and score >= 0.58:
        values = [(year, _value_of(product, year)) for year in valid_years]
        title = f"### Annual exports of {product.get('name')}"
        entities = {"hs6": _hs6(product.get("hs6")), "years": valid_years, "metric": "annual_series"}
    elif market:
        values = [(year, _value_of(market, year, "exports")) for year in valid_years]
        title = f"### Annual Lebanese exports to {market.get('country')}"
        entities = {"market": market.get("country"), "years": valid_years, "metric": "annual_series"}
    elif sector:
        values = [(year, _value_of(sector, year)) for year in valid_years]
        title = f"### Annual exports of {sector.get('sector')}"
        entities = {"sector": sector.get("sector"), "years": valid_years, "metric": "annual_series"}
    else:
        totals = {int(row.get("year")): float(row.get("filtered") or 0) for row in bundle().get("totals_by_year", [])}
        values = [(year, totals.get(year, 0.0)) for year in valid_years]
        title = "### Annual industrial exports"
        entities = {"years": valid_years, "metric": "annual_series"}

    peak_year, peak_value = max(values, key=lambda item: item[1])
    low_year, low_value = min(values, key=lambda item: item[1])
    lines = [title]
    lines.extend(f"- **{year}:** {_money(value)}" for year, value in values)
    lines.extend([
        "",
        "**Series summary**",
        f"- **Peak:** {peak_year} at {_money(peak_value)}",
        f"- **Lowest:** {low_year} at {_money(low_value)}",
        f"- **Change from {values[0][0]} to {values[-1][0]}:** {_pct(values[-1][1] / values[0][1] - 1) if values[0][1] else 'Not calculable from a zero base'}",
    ])
    return DashboardAnswer(True, 0.99, "\n".join(lines), json.dumps(dict(values), ensure_ascii=False), entities)


def _entry_exit_answer(question: str) -> DashboardAnswer | None:
    """Calculate product entry and exit from the exact annual export series."""
    q = _norm(question)
    asks_new = any(term in q for term in ("new products", "entered products", "products entered", "product entry", "new export products"))
    asks_exit = any(term in q for term in ("exited products", "products exited", "product exit", "lost products", "disappeared products"))
    if not (asks_new or asks_exit):
        return None
    years = _years(question)
    year = years[-1] if years else 2025
    valid = list(bundle().get("meta", {}).get("years", range(2018, 2026)))
    previous = max([item for item in valid if item < year], default=None)
    if previous is None:
        return DashboardAnswer(True, 0.95, f"Entry and exit cannot be calculated for {year} because no earlier dashboard year is available.", "{}", {"year": year, "metric": "entry_exit"})

    market = _find_market(question)
    sector = _find_sector(question)
    rows = list(indices()["products"])
    if sector:
        rows = [row for row in rows if row.get("sector") == sector.get("sector")]

    # For a market, use exact product-market-year records; otherwise use product totals.
    if market:
        country = str(market.get("country"))
        def value_fn(row: dict[str, Any], yr: int) -> float:
            return float(indices()["product_market"].get((_hs6(row.get("hs6")), _norm(country), yr), 0.0))
        scope = f" in {country}"
        entities = {"market": country, "year": year, "metric": "entry_exit"}
    else:
        value_fn = _value_of
        scope = f" in {sector.get('sector')}" if sector else ""
        entities = {"year": year, "metric": "entry_exit"}
        if sector:
            entities["sector"] = sector.get("sector")

    entered = [row for row in rows if value_fn(row, previous) <= 0 and value_fn(row, year) > 0]
    exited = [row for row in rows if value_fn(row, previous) > 0 and value_fn(row, year) <= 0]
    entered.sort(key=lambda row: value_fn(row, year), reverse=True)
    exited.sort(key=lambda row: value_fn(row, previous), reverse=True)
    n = _requested_n(question, default=10, maximum=25)

    lines = [f"### Product entry and exit{scope}, {previous}–{year}"]
    if asks_new:
        lines.extend(["", f"**Newly recorded products: {len(entered):,}**"])
        if entered:
            lines.extend(
                f"{i}. **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — {_money(value_fn(row, year))} in {year}"
                for i, row in enumerate(entered[:n], 1)
            )
        else:
            lines.append("- No products moved from zero exports to positive exports in this comparison.")
    if asks_exit:
        lines.extend(["", f"**Exited products: {len(exited):,}**"])
        if exited:
            lines.extend(
                f"{i}. **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — {_money(value_fn(row, previous))} in {previous}; $0 in {year}"
                for i, row in enumerate(exited[:n], 1)
            )
        else:
            lines.append("- No products moved from positive exports to zero exports in this comparison.")
    lines.extend(["", "**Method**", f"- Entry means zero in {previous} and positive in {year}; exit means positive in {previous} and zero in {year}."])
    return DashboardAnswer(True, 0.99, "\n".join(lines), json.dumps({"entered": entered[:n], "exited": exited[:n]}, ensure_ascii=False), entities)


def _product_market_answer(question: str) -> DashboardAnswer | None:
    """Answer product–market questions before generic rank/share routes."""
    product, score = _find_product(question)
    market = _find_market(question)
    if not product or score < 0.58 or not market:
        return None
    years = _years(question)
    year = years[-1] if years else 2025
    idx = indices()
    hs = _hs6(product.get("hs6"))
    country = str(market.get("country"))
    value = float(idx["product_market"].get((hs, _norm(country), year), 0.0))
    product_total = _value_of(product, year)
    market_total = _value_of(market, year, "exports")
    product_share = value / product_total if product_total else 0.0
    market_share = value / market_total if market_total else 0.0

    destination_values = []
    for row in idx["markets"]:
        destination_values.append((str(row.get("country")), float(idx["product_market"].get((hs, _norm(row.get("country")), year), 0.0))))
    active_destinations = sorted((item for item in destination_values if item[1] > 0), key=lambda item: item[1], reverse=True)
    destination_rank = next((i for i, item in enumerate(active_destinations, 1) if _norm(item[0]) == _norm(country)), 0)

    market_products = []
    for row in idx["products"]:
        code = _hs6(row.get("hs6"))
        market_products.append((code, row, float(idx["product_market"].get((code, _norm(country), year), 0.0))))
    active_products = sorted((item for item in market_products if item[2] > 0), key=lambda item: item[2], reverse=True)
    product_rank = next((i for i, item in enumerate(active_products, 1) if item[0] == hs), 0)

    asks_potential = any(term in _norm(question) for term in ("potential", "unrealized", "untapped", "opportunity"))
    pair_rows = [
        row for row in up_pairs()
        if _hs6(row.get("hs6")) == hs and _norm(row.get("country")) == _norm(country)
    ]
    potential = sum(float(row.get("value_usd") or row.get("value") or 0) for row in pair_rows)

    lines = [
        f"### {product.get('name')} → {country} in {year}",
        f"- **HS6:** `{hs}`",
        f"- **Exports:** {_money(value)}",
        f"- **Share of Lebanon's exports of this product:** {_pct(product_share)}",
        f"- **Share of Lebanon's export basket to {country}:** {_pct(market_share)}",
        f"- **Destination rank for this product:** {destination_rank:,} of {len(active_destinations):,} active destinations" if destination_rank else "- **Destination rank for this product:** not active in this year",
        f"- **Product rank within exports to {country}:** {product_rank:,} of {len(active_products):,} active products" if product_rank else f"- **Product rank within exports to {country}:** not active in this year",
        f"- **Product RCA:** {_number(product.get(f'rca_{year}'))}",
        f"- **Product PCI:** {_number(product.get('pci'))}",
    ]
    if asks_potential:
        if pair_rows and potential > 0:
            lines.append(f"- **Recorded unrealized potential:** {_money(potential)}")
        else:
            lines.append("- **Recorded unrealized potential:** no positive product–destination potential is embedded for this pair")
    return DashboardAnswer(
        True, 0.995, "\n".join(lines),
        json.dumps({"product": product, "market": market, "year": year, "exports": value, "potential": potential}, ensure_ascii=False),
        {"hs6": hs, "market": country, "year": year, "metric": "product_market"},
    )


def _entity_list_answer(question: str) -> DashboardAnswer | None:
    """Respect requests for top products in a market or destinations for a product."""
    q = _norm(question)
    year_list = _years(question)
    year = year_list[-1] if year_list else 2025
    n = _requested_n(question, default=10, maximum=25)
    market = _find_market(question)
    product, score = _find_product(question)
    idx = indices()

    asks_products = any(term in q for term in ("what products", "which products", "top products", "products exported", "export basket")) \
        or re.search(r"\btop\s+\d{1,2}\s+products\b", q) is not None
    asks_destinations = any(term in q for term in ("top destinations", "which markets", "what markets", "where is", "where does", "destinations for")) \
        or re.search(r"\btop\s+\d{1,2}\s+(?:markets|destinations|countries)\b", q) is not None
    if market and asks_products and not product:
        country = str(market.get("country"))
        rows = []
        for row in idx["products"]:
            hs = _hs6(row.get("hs6"))
            value = float(idx["product_market"].get((hs, _norm(country), year), 0.0))
            if value > 0:
                rows.append((row, value))
        rows.sort(key=lambda item: item[1], reverse=True)
        total = sum(value for _, value in rows)
        lines = [f"### Largest Lebanese export products in {country}, {year}"]
        lines.extend(
            f"{i}. **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — {_money(value)}; {_pct(value / total if total else 0)} of the basket"
            for i, (row, value) in enumerate(rows[:n], 1)
        )
        lines.extend(["", f"- **Active products:** {len(rows):,}", f"- **Total exports:** {_money(total)}"])
        return DashboardAnswer(True, 0.99, "\n".join(lines), json.dumps([row for row, _ in rows[:n]], ensure_ascii=False), {"market": country, "year": year, "metric": "product_list"})

    if product and score >= 0.58 and asks_destinations and not market:
        hs = _hs6(product.get("hs6"))
        rows = []
        for market_row in idx["markets"]:
            country = str(market_row.get("country"))
            value = float(idx["product_market"].get((hs, _norm(country), year), 0.0))
            if value > 0:
                rows.append((country, value))
        rows.sort(key=lambda item: item[1], reverse=True)
        total = sum(value for _, value in rows)
        lines = [f"### Largest destinations for {product.get('name')}, {year}"]
        lines.extend(
            f"{i}. **{country}** — {_money(value)}; {_pct(value / total if total else 0)} of product exports"
            for i, (country, value) in enumerate(rows[:n], 1)
        )
        lines.extend(["", f"- **Active destinations:** {len(rows):,}", f"- **Total product exports:** {_money(total)}"])
        return DashboardAnswer(True, 0.99, "\n".join(lines), json.dumps(rows[:n], ensure_ascii=False), {"hs6": hs, "year": year, "metric": "destination_list"})
    return None



def _methodology_answer(question: str) -> DashboardAnswer | None:
    q = _norm(question)
    if not any(term in q for term in (
        "what data", "data source", "data sources", "methodology", "coverage", "what years",
        "how many products", "how many markets", "how many sectors", "what does the dashboard cover",
        "how was this calculated", "source of the data"
    )):
        return None
    meta = bundle().get("meta", {})
    years = meta.get("years") or []
    lines = [
        "### Dashboard data and coverage",
        f"- **Years:** {min(years)}–{max(years)}" if years else "- **Years:** not specified",
        f"- **Products:** {int(meta.get('n_products') or 0):,} HS6 products",
        f"- **Markets:** {int(meta.get('n_markets') or 0):,}",
        f"- **Sectors:** {int(meta.get('n_sectors') or 0):,}",
        f"- **Build date:** {meta.get('build_date')}",
        f"- **Real-term base:** {meta.get('real_terms_base')}",
        "",
        "**Key calculation notes**",
        f"- {meta.get('validation_note')}",
        f"- {meta.get('market_size_rule')}",
        f"- {meta.get('cagr_2025_rule')}",
        f"- Unrealized potential source: {meta.get('up_eu_source_file')}; {meta.get('up_eu_scope')}",
    ]
    return DashboardAnswer(True, 0.99, "\n".join(lines), json.dumps(meta, ensure_ascii=False), {"metric": "methodology"})


def _strategy_screen_answer(question: str) -> DashboardAnswer | None:
    """Provide transparent opportunity screens rather than a hidden recommendation score."""
    q = _norm(question)
    if not any(term in q for term in (
        "should lebanon prioritize", "should lebanon focus", "what should lebanon focus",
        "best opportunities", "most promising", "priority products", "priority sectors",
        "target markets", "markets should lebanon target", "structural transformation",
        "policy priorities", "export priorities"
    )):
        return None
    idx = indices()
    year_list = _years(question)
    year = year_list[-1] if year_list else 2025
    n = _requested_n(question, default=5, maximum=10)

    asks_market = any(term in q for term in ("market", "markets", "destination", "destinations"))
    asks_sector = "sector" in q or "sectors" in q
    if asks_market:
        rows = [row for row in idx["markets"] if float(row.get("unrealized_potential_exact") or row.get("unrealized_potential") or 0) > 0]
        rows.sort(key=lambda row: float(row.get("unrealized_potential_exact") or row.get("unrealized_potential") or 0), reverse=True)
        lines = [f"### Market-opportunity screen in {year}", "This is a transparent potential screen, not a forecast."]
        for i, row in enumerate(rows[:n], 1):
            value = _value_of(row, year, "exports")
            potential = float(row.get("unrealized_potential_exact") or row.get("unrealized_potential") or 0)
            lines.append(
                f"{i}. **{row.get('country')}** — exports {_money(value)}; recorded potential {_money(potential)}; "
                f"status {row.get('status')}; HHI {_number(row.get('hhi'), 3)}; EXPY {_number(row.get('expy_2025'), 0)}"
            )
        lines.extend(["", "**How to use this screen**", "- Validate demand, tariffs, logistics, standards, competitors, and firm capacity before treating a market as a priority."])
        return DashboardAnswer(True, 0.98, "\n".join(lines), json.dumps(rows[:n], ensure_ascii=False), {"year": year, "metric": "market_priority_screen"})

    if asks_sector:
        rows = []
        for sector in idx["sectors"]:
            products = [p for p in idx["products"] if p.get("sector") == sector.get("sector")]
            rows.append((sector, _weighted_pci(products, year)))
        rows.sort(key=lambda item: float(item[0].get("unrealized_potential_eu") or 0), reverse=True)
        lines = [f"### Sector-priority screen in {year}", "The screen keeps potential, current scale, competitiveness, and complexity visible separately."]
        for i, (row, pci) in enumerate(rows[:n], 1):
            lines.append(
                f"{i}. **{row.get('sector')}** — exports {_money(_value_of(row, year))}; potential {_money(row.get('unrealized_potential_eu'))}; "
                f"RCA {_number(row.get(f'rca_{year}'))}; weighted PCI {_number(pci)}"
            )
        lines.extend(["", "**Interpretation**", "- High potential supports market-expansion screening; high PCI supports capability-upgrading screening; RCA shows current specialization. No single metric is sufficient on its own."])
        return DashboardAnswer(True, 0.98, "\n".join(lines), json.dumps([row for row, _ in rows[:n]], ensure_ascii=False), {"year": year, "metric": "sector_priority_screen"})

    products = [row for row in idx["products"] if _value_of(row, year) > 0]
    scale = sorted([row for row in products if float(row.get(f"rca_{year}") or 0) >= 1], key=lambda row: _value_of(row, year), reverse=True)[:n]
    upgrading = sorted([row for row in products if float(row.get("pci") or -999) >= 1 and float(row.get(f"rca_{year}") or 0) < 1], key=lambda row: float(row.get("unrealized_potential_eu") or 0), reverse=True)[:n]
    expansion = sorted(products, key=lambda row: float(row.get("unrealized_potential_eu") or 0), reverse=True)[:n]
    lines = [
        f"### Product-priority screen in {year}",
        "The dashboard supports three distinct screens rather than one opaque ranking.",
        "",
        "**Scale existing strengths**",
    ]
    lines.extend(f"- **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — exports {_money(_value_of(row, year))}; RCA {_number(row.get(f'rca_{year}'))}" for row in scale)
    lines.extend(["", "**Build higher-complexity capabilities**"])
    lines.extend(f"- **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — PCI {_number(row.get('pci'))}; RCA {_number(row.get(f'rca_{year}'))}; potential {_money(row.get('unrealized_potential_eu'))}" for row in upgrading)
    lines.extend(["", "**Expand products with recorded market potential**"])
    lines.extend(f"- **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — potential {_money(row.get('unrealized_potential_eu'))}; exports {_money(_value_of(row, year))}" for row in expansion)
    lines.extend(["", "**Decision rule**", "- Treat these as screening lists. Final prioritization requires firm capability, employment, domestic value added, input dependence, standards, logistics, and market-access evidence not contained in the dashboard."])
    context = {"scale": scale, "upgrading": upgrading, "expansion": expansion}
    return DashboardAnswer(True, 0.98, "\n".join(lines), json.dumps(context, ensure_ascii=False), {"year": year, "metric": "product_priority_screen"})


def _cross_metric_answer(question: str) -> DashboardAnswer | None:
    q = _norm(question)
    has_complexity = "complexity" in q or "pci" in q or "complex" in q
    has_competitiveness = "competitiveness" in q or "rca" in q or "comparative advantage" in q
    if has_complexity and has_competitiveness:
        product, score = _find_product(question)
        if (product and score >= 0.58) or _find_market(question) or _find_sector(question):
            return _query_dashboard_single(question)
        years = _years(question)
        return _competitiveness_complexity_summary(years[-1] if years else 2025)
    return None


def _requested_n(question: str, default: int = 5, maximum: int = 20) -> int:
    q = _norm(question)
    match = re.search(r"\b(?:top|bottom|highest|lowest|largest|smallest|most|least)\s+(\d{1,2})\b", q)
    if not match:
        match = re.search(r"\b(\d{1,2})\s+(?:most|least|highest|lowest|largest|smallest|top|bottom)\b", q)
    if not match:
        return default
    return max(1, min(maximum, int(match.group(1))))


def _find_all_markets(question: str) -> list[dict[str, Any]]:
    idx = indices()
    q = _norm(question)
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for alias, canonical in idx["market_aliases"].items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            row = idx["markets_by_norm"].get(_norm(canonical))
            if row and row.get("country") not in seen:
                seen.add(str(row.get("country")))
                found.append(row)
    for name, row in sorted(idx["markets_by_norm"].items(), key=lambda item: len(item[0]), reverse=True):
        if name and re.search(rf"\b{re.escape(name)}\b", q) and row.get("country") not in seen:
            seen.add(str(row.get("country")))
            found.append(row)
    return found


def _find_all_sectors(question: str) -> list[dict[str, Any]]:
    q = _norm(question)
    found: list[dict[str, Any]] = []
    for name, row in sorted(indices()["sectors_by_norm"].items(), key=lambda item: len(item[0]), reverse=True):
        if name and name in q:
            found.append(row)
    return found


def _find_all_products(question: str) -> list[dict[str, Any]]:
    idx = indices()
    q = _norm(question)
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for token in re.findall(r"\b\d{4,6}\b", question):
        candidates = []
        if len(token) == 6 and token in idx["products_by_hs"]:
            candidates = [idx["products_by_hs"][token]]
        else:
            candidates = [row for code, row in idx["products_by_hs"].items() if code.startswith(token)]
            candidates = sorted(candidates, key=lambda row: _value_of(row, 2025), reverse=True)[:1]
        for row in candidates:
            code = _hs6(row.get("hs6"))
            if code not in seen:
                seen.add(code)
                found.append(row)
    for alias, code in idx["product_aliases"].items():
        if alias in q and code in idx["products_by_hs"] and code not in seen:
            seen.add(code)
            found.append(idx["products_by_hs"][code])
    return found


def _entity_comparison_answer(question: str) -> DashboardAnswer | None:
    q = _norm(question)
    if not any(term in q for term in ("compare", "versus", " vs ", "difference between", "against")):
        return None
    year_list = _years(question)
    year = year_list[-1] if year_list else 2025
    total = _total_for_year(year)
    markets = _find_all_markets(question)
    sectors = _find_all_sectors(question)
    products = _find_all_products(question)

    if len(markets) >= 2:
        rows = []
        for row in markets[:4]:
            value = _value_of(row, year, "exports")
            rows.append(
                f"- **{row.get('country')}:** {_money(value)}; share {_pct(value / total if total else 0)}; "
                f"{int(row.get(f'products_{year}') or row.get('n_products_hs6') or 0):,} products; "
                f"HHI {_number(row.get('hhi'), 3)}; EXPY {_number(row.get('expy_2025'), 0)}; "
                f"potential {_money(row.get('unrealized_potential_exact') or row.get('unrealized_potential'))}"
            )
        answer = f"### Market comparison in {year}\n" + "\n".join(rows)
        return DashboardAnswer(True, 0.99, answer, json.dumps(markets[:4], ensure_ascii=False), {
            "markets": [row.get("country") for row in markets[:4]], "year": year, "metric": "comparison"
        })

    if len(sectors) >= 2:
        rows = []
        for row in sectors[:4]:
            value = _value_of(row, year)
            products_in_sector = [p for p in indices()["products"] if p.get("sector") == row.get("sector")]
            rows.append(
                f"- **{row.get('sector')}:** {_money(value)}; share {_pct(value / total if total else 0)}; "
                f"RCA {_number(row.get(f'rca_{year}'))}; weighted PCI {_number(_weighted_pci(products_in_sector, year))}; "
                f"potential {_money(row.get('unrealized_potential_eu'))}"
            )
        answer = f"### Sector comparison in {year}\n" + "\n".join(rows)
        return DashboardAnswer(True, 0.99, answer, json.dumps(sectors[:4], ensure_ascii=False), {
            "sectors": [row.get("sector") for row in sectors[:4]], "year": year, "metric": "comparison"
        })

    if len(products) >= 2:
        rows = []
        for row in products[:4]:
            value = _value_of(row, year)
            rows.append(
                f"- **{row.get('name')}** [`{_hs6(row.get('hs6'))}`]: {_money(value)}; "
                f"share {_pct(value / total if total else 0)}; RCA {_number(row.get(f'rca_{year}'))}; "
                f"PCI {_number(row.get('pci'))}; {int((row.get('n_countries_by_year') or {}).get(str(year)) or row.get('n_countries') or 0):,} markets"
            )
        answer = f"### Product comparison in {year}\n" + "\n".join(rows)
        return DashboardAnswer(True, 0.99, answer, json.dumps(products[:4], ensure_ascii=False), {
            "products": [_hs6(row.get("hs6")) for row in products[:4]], "year": year, "metric": "comparison"
        })
    return None


def _metric_label(metric: str, entity_type: str) -> str:
    labels = {
        ("exports", "product"): "products by export value",
        ("exports", "market"): "markets by export value",
        ("exports", "sector"): "sectors by export value",
        ("pci", "product"): "product-complexity (PCI) products",
        ("pci", "market"): "markets by export sophistication (EXPY)",
        ("pci", "sector"): "sectors by export-weighted complexity",
        ("rca", "product"): "products by RCA",
        ("rca", "market"): "markets by RCA",
        ("rca", "sector"): "sectors by RCA",
        ("potential", "product"): "products by unrealized potential",
        ("potential", "market"): "markets by unrealized potential",
        ("potential", "sector"): "sectors by unrealized potential",
        ("growth", "product"): "products by CAGR",
        ("growth", "market"): "markets by CAGR",
        ("growth", "sector"): "sectors by CAGR",
    }
    return labels.get((metric, entity_type), f"{metric} {entity_type}s")


def _top_metric_answer(question: str) -> DashboardAnswer | None:
    q = _norm(question)
    asks_top = any(term in q for term in ("top", "most", "highest", "largest", "biggest", "leading", "lowest", "bottom", "least", "smallest"))
    if not asks_top:
        return None
    candidate_product, candidate_score = _find_product(question)
    if _find_market(question) or _find_sector(question) or (candidate_product is not None and candidate_score >= 0.90):
        # Explicit entity questions are handled by the profile/rank routes.
        return None
    year_list = _years(question)
    year = year_list[-1] if year_list else 2025
    n = _requested_n(question)
    reverse = not any(term in q for term in ("lowest", "bottom", "least", "smallest"))
    idx = indices()
    total = _total_for_year(year)

    entity_type = "product"
    if "sector" in q:
        entity_type = "sector"
    elif any(term in q for term in ("market", "destination", "country")):
        entity_type = "market"

    metric = "exports"
    if any(term in q for term in ("complexity", "pci", "complex")):
        metric = "pci"
    elif "rca" in q or "comparative advantage" in q or "competitiveness" in q:
        metric = "rca"
    elif any(term in q for term in ("potential", "unrealized", "untapped")):
        metric = "potential"
    elif any(term in q for term in ("growth", "cagr")):
        metric = "growth"

    if entity_type == "sector":
        rows = list(idx["sectors"])
        if metric == "pci":
            scored = [(row, _weighted_pci([p for p in idx["products"] if p.get("sector") == row.get("sector")], year)) for row in rows]
        elif metric == "rca":
            scored = [(row, row.get(f"rca_{year}")) for row in rows]
        elif metric == "potential":
            scored = [(row, row.get("unrealized_potential_eu")) for row in rows]
        elif metric == "growth":
            scored = [(row, row.get("cagr")) for row in rows]
        else:
            scored = [(row, _value_of(row, year)) for row in rows]
        scored = [(row, float(value)) for row, value in scored if value is not None]
        scored.sort(key=lambda item: item[1], reverse=reverse)
        lines = []
        for i, (row, value) in enumerate(scored[:n], 1):
            if metric == "exports":
                rendered = f"{_money(value)}; share {_pct(value / total if total else 0)}"
            elif metric in {"pci", "rca"}:
                rendered = _number(value)
            elif metric == "growth":
                rendered = _pct(value)
            else:
                rendered = _money(value)
            lines.append(f"{i}. **{row.get('sector')}** — {rendered}")
        return DashboardAnswer(True, 0.98, f"### {'Highest' if reverse else 'Lowest'} {_metric_label(metric, 'sector')} in {year}\n" + "\n".join(lines), json.dumps([row for row, _ in scored[:n]], ensure_ascii=False), {"year": year, "metric": metric})

    if entity_type == "market":
        rows = list(idx["markets"])
        key_map = {
            "exports": lambda row: _value_of(row, year, "exports"),
            "pci": lambda row: row.get("expy_2025"),
            "rca": lambda row: row.get("rca"),
            "potential": lambda row: row.get("unrealized_potential_exact") or row.get("unrealized_potential"),
            "growth": lambda row: row.get("cagr"),
        }
        scored = [(row, key_map[metric](row)) for row in rows]
        scored = [(row, float(value)) for row, value in scored if value is not None and (metric != "exports" or value > 0)]
        scored.sort(key=lambda item: item[1], reverse=reverse)
        lines = []
        for i, (row, value) in enumerate(scored[:n], 1):
            if metric == "exports":
                rendered = f"{_money(value)}; share {_pct(value / total if total else 0)}"
            elif metric == "pci":
                rendered = f"EXPY {_number(value, 0)}"
            elif metric == "growth":
                rendered = _pct(value)
            elif metric == "potential":
                rendered = _money(value)
            else:
                rendered = _number(value)
            lines.append(f"{i}. **{row.get('country')}** — {rendered}")
        return DashboardAnswer(True, 0.98, f"### {'Highest' if reverse else 'Lowest'} {_metric_label(metric, 'market')} in {year}\n" + "\n".join(lines), json.dumps([row for row, _ in scored[:n]], ensure_ascii=False), {"year": year, "metric": metric})

    rows = [row for row in idx["products"] if _value_of(row, year) > 0]
    key_map = {
        "exports": lambda row: _value_of(row, year),
        "pci": lambda row: row.get("pci"),
        "rca": lambda row: row.get(f"rca_{year}"),
        "potential": lambda row: row.get("unrealized_potential_eu"),
        "growth": lambda row: row.get("cagr"),
    }
    scored = [(row, key_map[metric](row)) for row in rows]
    scored = [(row, float(value)) for row, value in scored if value is not None]
    scored.sort(key=lambda item: item[1], reverse=reverse)
    lines = []
    for i, (row, value) in enumerate(scored[:n], 1):
        if metric == "exports":
            rendered = f"{_money(value)}; share {_pct(value / total if total else 0)}"
        elif metric in {"pci", "rca"}:
            rendered = f"{_number(value)}; exports {_money(_value_of(row, year))}"
        elif metric == "growth":
            rendered = _pct(value)
        else:
            rendered = _money(value)
        lines.append(f"{i}. **{row.get('name')}** [`{_hs6(row.get('hs6'))}`] — {rendered}")
    answer = f"### {'Highest' if reverse else 'Lowest'} {_metric_label(metric, 'product')} in {year}\n" + "\n".join(lines)
    if metric == "rca":
        answer += (
            "\n\n**Interpretation limit**\n"
            "- Very large RCA values can occur when a product is unusually prominent in Lebanon's basket relative to a small world-trade denominator. "
            "Read RCA together with the export value shown."
        )
    return DashboardAnswer(True, 0.98, answer, json.dumps([row for row, _ in scored[:n]], ensure_ascii=False), {"year": year, "metric": metric})



def _parallel_entity_answer(question: str) -> DashboardAnswer | None:
    """Answer multi-part requests that ask for different entity rankings."""
    segments = _question_segments(question)
    if len(segments) < 2:
        return None

    def group(segment: str) -> str | None:
        q = _norm(segment)
        if "sector" in q:
            return "sector"
        if any(term in q for term in ("market", "destination", "country")):
            return "market"
        if "product" in q:
            return "product"
        return None

    groups = [group(segment) for segment in segments]
    distinct = {item for item in groups if item}
    if len(distinct) < 2:
        return None

    suffix = _context_suffix(question)
    answers: list[DashboardAnswer] = []
    for segment in segments[:4]:
        enriched = f"{segment}. Context: {suffix}" if suffix else segment
        result = query_dashboard(enriched)
        if result.matched:
            answers.append(result)
    if len(answers) < 2:
        return None
    combined = "\n\n---\n\n".join(result.answer for result in answers)
    entities: dict[str, Any] = {"metric": "multi_part"}
    for result in answers:
        entities.update({key: value for key, value in result.entities.items() if value is not None})
    return DashboardAnswer(
        True,
        min(result.confidence for result in answers),
        combined,
        json.dumps([json.loads(result.context) if result.context else {} for result in answers], ensure_ascii=False),
        entities,
    )


def _rank_share_answer(question: str) -> DashboardAnswer | None:
    q = _norm(question)
    asks_rank = any(term in q for term in ("rank", "ranking", "position", "where does", "how does it rank"))
    asks_share = any(term in q for term in ("share", "percentage of exports", "portion of exports", "contribution to exports"))
    if not (asks_rank or asks_share):
        return None

    years = _years(question)
    year = years[-1] if years else 2025
    idx = indices()
    product, score = _find_product(question)
    market = _find_market(question)
    sector = _find_sector(question)
    total = _total_for_year(year)

    if product and score >= 0.58:
        active = [row for row in idx["products"] if _value_of(row, year) > 0]
        rank, count = _rank_of(active, product, lambda row: _value_of(row, year))
        value = _value_of(product, year)
        share = value / total if total else 0
        sector_products = [row for row in active if row.get("sector") == product.get("sector")]
        sector_rank, sector_count = _rank_of(sector_products, product, lambda row: _value_of(row, year))
        answer = (
            f"### Rank and share of {product.get('name')} in {year}\n"
            f"- **HS6:** `{_hs6(product.get('hs6'))}`\n"
            f"- **Export value:** {_money(value)}\n"
            f"- **Share of total industrial exports:** {_pct(share)}\n"
            f"- **National product rank:** {rank:,} of {count:,} active products\n"
            f"- **Rank within {product.get('sector')}:** {sector_rank:,} of {sector_count:,} active products"
        )
        return DashboardAnswer(True, 0.99, answer, json.dumps(product, ensure_ascii=False), {
            "hs6": _hs6(product.get("hs6")), "year": year, "metric": "rank_share"
        })

    if market:
        active = [row for row in idx["markets"] if _value_of(row, year, "exports") > 0]
        rank, count = _rank_of(active, market, lambda row: _value_of(row, year, "exports"))
        value = _value_of(market, year, "exports")
        share = value / total if total else 0
        answer = (
            f"### Rank and share of {market.get('country')} in {year}\n"
            f"- **Lebanese exports:** {_money(value)}\n"
            f"- **Share of total industrial exports:** {_pct(share)}\n"
            f"- **Destination rank:** {rank:,} of {count:,} active markets\n"
            f"- **Products exported:** {int(market.get(f'products_{year}') or market.get('n_products_hs6') or 0):,}"
        )
        return DashboardAnswer(True, 0.99, answer, json.dumps(market, ensure_ascii=False), {
            "market": market.get("country"), "year": year, "metric": "rank_share"
        })

    if sector:
        active = [row for row in idx["sectors"] if _value_of(row, year) > 0]
        rank, count = _rank_of(active, sector, lambda row: _value_of(row, year))
        value = _value_of(sector, year)
        share = value / total if total else 0
        answer = (
            f"### Rank and share of {sector.get('sector')} in {year}\n"
            f"- **Export value:** {_money(value)}\n"
            f"- **Share of total industrial exports:** {_pct(share)}\n"
            f"- **Sector rank:** {rank:,} of {count:,} sectors\n"
            f"- **Active HS6 products:** {int(sector.get('n_products_hs6') or sector.get('n_products') or 0):,}"
        )
        return DashboardAnswer(True, 0.99, answer, json.dumps(sector, ensure_ascii=False), {
            "sector": sector.get("sector"), "year": year, "metric": "rank_share"
        })
    return None


def _change_period(question: str) -> tuple[int, int]:
    years = sorted(set(_years(question)))
    valid = list(bundle().get("meta", {}).get("years", range(2018, 2026)))
    if len(years) >= 2:
        return years[0], years[-1]
    if len(years) == 1:
        end = years[0]
        prior = max([year for year in valid if year < end], default=min(valid))
        return prior, end
    return min(valid), max(valid)


def _delta_lines(rows: list[dict[str, Any]], start: int, end: int, value_fn: Any, label_fn: Any, n: int = 4) -> tuple[list[str], list[str]]:
    calculated = []
    for row in rows:
        start_value = float(value_fn(row, start) or 0)
        end_value = float(value_fn(row, end) or 0)
        delta = end_value - start_value
        if delta:
            calculated.append((delta, row, start_value, end_value))
    gains = sorted((item for item in calculated if item[0] > 0), key=lambda item: item[0], reverse=True)[:n]
    losses = sorted((item for item in calculated if item[0] < 0), key=lambda item: item[0])[:n]
    gain_lines = [
        f"- **{label_fn(row)}:** {_money(start_value)} → {_money(end_value)} (**+{_money(delta).replace('$', '$')}**)"
        for delta, row, start_value, end_value in gains
    ]
    loss_lines = [
        f"- **{label_fn(row)}:** {_money(start_value)} → {_money(end_value)} (**{_money(delta)}**)"
        for delta, row, start_value, end_value in losses
    ]
    return gain_lines, loss_lines


def _drivers_answer(question: str) -> DashboardAnswer | None:
    q = _norm(question)
    asks_drivers = any(term in q for term in (
        "what drove", "what is driving", "what drives", "drivers", "driver", "contributed",
        "contribution", "why did exports", "why exports", "explain the increase", "explain the decline",
        "source of growth", "source of decline"
    ))
    if not asks_drivers:
        return None

    idx = indices()
    start, end = _change_period(question)
    product, score = _find_product(question)
    market = _find_market(question)
    sector = _find_sector(question)

    if product and score >= 0.58:
        hs = _hs6(product.get("hs6"))
        country_rows: list[dict[str, Any]] = []
        country_names = {row.get("country") for row in idx["markets"]}
        for country in country_names:
            if not country:
                continue
            country_rows.append({
                "country": country,
                "start": idx["product_market"].get((hs, _norm(country), start), 0.0),
                "end": idx["product_market"].get((hs, _norm(country), end), 0.0),
            })
        gain_lines, loss_lines = _delta_lines(
            country_rows, start, end,
            lambda row, yr: row["start"] if yr == start else row["end"],
            lambda row: str(row["country"]),
        )
        start_value, end_value = _value_of(product, start), _value_of(product, end)
        title = f"### Export-change drivers for {product.get('name')}, {start}–{end}"
        context_entity = {"hs6": hs}
    elif market:
        country = str(market.get("country"))
        product_rows = []
        for row in idx["products"]:
            hs = _hs6(row.get("hs6"))
            product_rows.append({
                "name": row.get("name"), "hs6": hs,
                "start": idx["product_market"].get((hs, _norm(country), start), 0.0),
                "end": idx["product_market"].get((hs, _norm(country), end), 0.0),
            })
        gain_lines, loss_lines = _delta_lines(
            product_rows, start, end,
            lambda row, yr: row["start"] if yr == start else row["end"],
            lambda row: f"{row['name']} [`{row['hs6']}`]",
        )
        start_value, end_value = _value_of(market, start, "exports"), _value_of(market, end, "exports")
        title = f"### Export-change drivers in {country}, {start}–{end}"
        context_entity = {"market": country}
    elif sector:
        name = str(sector.get("sector"))
        rows = [row for row in idx["products"] if row.get("sector") == name]
        gain_lines, loss_lines = _delta_lines(
            rows, start, end, _value_of,
            lambda row: f"{row.get('name')} [`{_hs6(row.get('hs6'))}`]",
        )
        start_value, end_value = _value_of(sector, start), _value_of(sector, end)
        title = f"### Export-change drivers in {name}, {start}–{end}"
        context_entity = {"sector": name}
    else:
        rows = idx["sectors"]
        gain_lines, loss_lines = _delta_lines(rows, start, end, _value_of, lambda row: str(row.get("sector")))
        start_value, end_value = _total_for_year(start), _total_for_year(end)
        title = f"### Drivers of total industrial export change, {start}–{end}"
        context_entity = {}

    delta = end_value - start_value
    pct = delta / start_value if start_value else None
    lines = [
        title,
        f"- **Exports in {start}:** {_money(start_value)}",
        f"- **Exports in {end}:** {_money(end_value)}",
        f"- **Net change:** {_money(delta)} ({_pct(pct) if pct is not None else 'not calculable'})",
    ]
    if gain_lines:
        lines.extend(["", "**Largest positive contributors**", *gain_lines])
    if loss_lines:
        lines.extend(["", "**Largest negative contributors**", *loss_lines])
    lines.extend([
        "",
        "**Interpretation limit**",
        "- These are accounting contributions from the dashboard data. They identify where the change occurred, not the causal reason it occurred."
    ])
    entities = {**context_entity, "years": [start, end], "metric": "drivers"}
    return DashboardAnswer(True, 0.99, "\n".join(lines), json.dumps(entities, ensure_ascii=False), entities)


def _similar_markets_answer(question: str) -> DashboardAnswer | None:
    q = _norm(question)
    if not any(term in q for term in ("similar market", "similar markets", "comparable market", "comparable markets", "markets like", "are similar", "similar to")):
        return None
    market = _find_market(question)
    if not market:
        return None
    rows = market.get("similar_countries") or []
    if not rows:
        return DashboardAnswer(True, 0.9, f"No similar-market records are embedded for {market.get('country')}.", "{}", {"market": market.get("country"), "metric": "similar_markets"})
    answer = f"### Markets most similar to {market.get('country')}\n" + "\n".join(
        f"{i}. **{row.get('country')}** — similarity {_number(row.get('score'), 3)}; Lebanese exports in 2025 {_money(row.get('exports_2025'))}"
        for i, row in enumerate(rows[:8], 1)
    )
    return DashboardAnswer(True, 0.98, answer, json.dumps(rows[:8], ensure_ascii=False), {"market": market.get("country"), "metric": "similar_markets"})


def _question_segments(question: str) -> list[str]:
    # Strong separators always indicate separate requests.
    raw = [part.strip(" ,") for part in re.split(r"[;\n]+|\?\s*", question) if part.strip(" ,")]
    if len(raw) > 1:
        return raw[:4]

    # Split "and/also" only when both sides contain a dashboard metric term.
    parts = [part.strip(" ,") for part in re.split(r"\s+(?:and|also)\s+", question, flags=re.I) if part.strip(" ,")]
    if len(parts) <= 1:
        return [question]
    metric_hits = [sum(1 for term in _METRIC_TERMS if term in _norm(part)) for part in parts]
    return parts[:4] if sum(hit > 0 for hit in metric_hits) >= 2 else [question]


def _context_suffix(question: str) -> str:
    bits: list[str] = []
    years = _years(question)
    if years:
        bits.append("years " + ", ".join(str(y) for y in years))
    market = _find_market(question)
    if market:
        bits.append("market " + str(market.get("country")))
    sector = _find_sector(question)
    if sector:
        bits.append("sector " + str(sector.get("sector")))
    product, score = _find_product(question)
    if product and score >= 0.8:
        bits.append("product HS6 " + _hs6(product.get("hs6")))
    return "; ".join(bits)


def query_dashboard(question: str) -> DashboardAnswer:
    """Answer dashboard questions with advanced planning and deterministic grounding.

    Enhancements over the original single-route parser:
    - rank and share calculations;
    - descriptive change-driver decomposition;
    - similar-market queries;
    - multi-part questions with shared entity/year context.
    """
    question = str(question or "").strip()
    if not question:
        return DashboardAnswer(False, 0.0, "", "", {})

    # Advanced exact calculations take precedence.
    for resolver in (
        _methodology_answer,
        _strategy_screen_answer,
        _metric_definition_answer,
        _product_market_answer,
        _annual_series_answer,
        _entry_exit_answer,
        _entity_list_answer,
        _cross_metric_answer,
        _entity_comparison_answer,
        _drivers_answer,
        _rank_share_answer,
        _similar_markets_answer,
        _parallel_entity_answer,
        _top_metric_answer,
    ):
        result = resolver(question)
        if result is not None:
            return result

    direct = _query_dashboard_single(question)
    if direct.matched:
        return direct

    segments = _question_segments(question)
    if len(segments) > 1:
        suffix = _context_suffix(question)
        answers: list[DashboardAnswer] = []
        seen: set[str] = set()
        for segment in segments:
            enriched = segment
            if suffix:
                enriched = f"{segment}. Context: {suffix}"
            result = _query_dashboard_single(enriched)
            if result.matched:
                key = re.sub(r"\s+", " ", result.answer).strip().lower()
                if key not in seen:
                    seen.add(key)
                    answers.append(result)
        if len(answers) >= 2:
            combined = "\n\n---\n\n".join(answer.answer for answer in answers[:4])
            contexts = [json.loads(answer.context) if answer.context else {} for answer in answers[:4]]
            entities: dict[str, Any] = {"metric": "multi_part"}
            for answer in answers:
                entities.update({key: value for key, value in answer.entities.items() if value is not None})
            return DashboardAnswer(True, min(answer.confidence for answer in answers), combined, json.dumps(contexts, ensure_ascii=False), entities)

    return _query_dashboard_single(question)

