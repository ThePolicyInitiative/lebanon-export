from __future__ import annotations

import json
import os
import re
import sqlite3
import gzip
import shutil
import tempfile
from functools import lru_cache
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from agents.llm_clients import chat_completion
from tools.dashboard_data import DashboardAnswer

ROOT = Path(__file__).resolve().parents[1]
DB_GZIP_PATH = ROOT / "data" / "dashboard_data.sqlite.gz"
DB_PLAIN_PATH = ROOT / "data" / "dashboard_data.sqlite"

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

# This catalog covers every numerical/business dataset embedded in dashboard.html.
SCHEMA_CATALOG = """
SQLite tables/views available:
- export_overview(year, total_exports_usd, real_exports_2018_usd, active_products, active_markets): annual headline totals, 2018-2025.
- totals_by_year(year, filtered, filtered_cpi_adj, filtered_nominal_original, filtered_cpi_adj_original, real_ratio_2018_base).
- product_year(hs6, hs4, name, sector, year, export_value, rca, pci, n_countries, unrealized_potential_usd, cagr, growth, trajectory): one product-year row.
- products_master: one row per HS6 product, with value_2018...value_2025, rca_2018...rca_2025, PCI, growth, CAGR, trajectory, market reach and unrealized potential.
- product_market_year(year, hs6, hs4, product_name, sector, country, iso3, continent, value_usd, pci, rca): Lebanon's product-destination exports.
- product_market_share: product_market_year plus share_of_market_exports and share_of_product_exports.
- product_top_destinations(hs6, rank, country, iso3, value_2025).
- product_all_markets(hs6, country).
- market_year(country, iso3, continent, year, export_value, n_products, expy, hhi, rca, unrealized_potential_usd, status, priority, cagr): one destination-year row.
- markets_master: one row per destination with annual exports/products, HHI, EXPY, growth, status, priority, potential, entry/exit and performance fields.
- market_top_products(country, rank, hs6, name, sector, value_2025).
- market_all_products(country, hs6).
- similar_markets(country, rank, iso3, score, continent, exports_2025).
- sector_year(sector, year, export_value, share, rca, pci_avg, unrealized_potential_usd, n_products_hs6, n_products_hs4, cagr).
- sectors_master: one row per sector with annual values, annual shares, annual RCA, PCI average, growth, CAGR and potential.
- up_hs6, up_hs4, up_sector, up_partner, up_pairs, up_top_pairs, up_totals: exact unrealized-potential datasets.
- market_size_hs6(country, year, hs6, market_size_usd): destination import-market size for HS6 in available 2018/2024 observations.
- topsis_overperformers, topsis_underperformers: market performance classifications.
- filter_funnel and meta: dashboard construction and coverage metadata.
- up_provenance: potential-data source and validation metadata.
- geo_features: country map metadata and geometry JSON.
- raw_dataset_records(dataset, record_key, record_json): lossless access to every embedded JSON record, including world geography/topology.

Rules and definitions:
- Monetary values are USD unless a column says otherwise.
- Default to 2025 when the user omits a year, except market_size_hs6 which generally has 2018/2024.
- HS codes are TEXT and must remain zero-padded.
- RCA >= 1 indicates revealed comparative advantage. PCI is product complexity. EXPY is destination-basket sophistication. Higher HHI means more concentration.
- Use product_market_year or product_market_share for a product in a destination; do not infer it from product or market totals.
- For product names use case-insensitive LIKE, e.g. lower(product_name) LIKE '%olive oil%'.
- For Lebanon's total exports use export_overview or totals_by_year.
""".strip()

BLOCKED_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|detach|pragma|vacuum|create|replace|reindex|trigger)\b",
    re.IGNORECASE,
)


def _has_model_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip())


def _norm_text(value: Any) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@lru_cache(maxsize=1)
def _entity_catalog() -> dict[str, Any]:
    with sqlite3.connect(_database_path()) as conn:
        products = [
            {"hs6": str(row[0]).zfill(6), "name": str(row[1]), "sector": str(row[2])}
            for row in conn.execute("SELECT hs6,name,sector FROM products_master")
        ]
        markets = [str(row[0]) for row in conn.execute("SELECT country FROM markets_master")]
        sectors = [str(row[0]) for row in conn.execute("SELECT sector FROM sectors_master")]
    return {
        "products": products,
        "markets": markets,
        "sectors": sectors,
        "product_by_hs": {item["hs6"]: item for item in products},
    }


def _extract_years(question: str) -> list[int]:
    years = [int(y) for y in re.findall(r"\b20(?:18|19|20|21|22|23|24|25)\b", question)]
    return list(dict.fromkeys(years))


def _extract_limit(question: str, default: int = 10) -> int:
    match = re.search(r"\b(?:top|first|largest|highest|best|leading)\s+(\d{1,2})\b", question, re.I)
    if not match:
        match = re.search(r"\b(\d{1,2})\s+(?:products|markets|sectors|destinations|countries)\b", question, re.I)
    return max(1, min(30, int(match.group(1)))) if match else default


def _all_named_matches(question: str, names: list[str], aliases: dict[str, str] | None = None) -> list[str]:
    q = _norm_text(question)
    found: list[str] = []
    aliases = aliases or {}
    for alias, canonical in aliases.items():
        if re.search(rf"\b{re.escape(_norm_text(alias))}\b", q):
            if canonical not in found:
                found.append(canonical)
    for name in sorted(names, key=len, reverse=True):
        norm = _norm_text(name)
        if norm and re.search(rf"\b{re.escape(norm)}\b", q) and name not in found:
            found.append(name)
    return found


def _find_markets(question: str) -> list[str]:
    aliases = {
        "ksa": "Saudi Arabia", "saudi": "Saudi Arabia", "saudia": "Saudi Arabia",
        "emirates": "UAE", "united arab emirates": "UAE", "america": "United States",
        "usa": "United States", "us": "United States", "uk": "United Kingdom",
        "britain": "United Kingdom", "cote d ivoire": "Ivory Coast", "korea": "South Korea",
    }
    return _all_named_matches(question, _entity_catalog()["markets"], aliases)


def _find_sectors(question: str) -> list[str]:
    aliases = {
        "agrofood": "Agrifood", "food sector": "Agrifood",
        "machinery": "Electrical and Machinery", "electrical": "Electrical and Machinery",
        "pharma": "Pharma & Parapharma", "pharmaceuticals": "Pharma & Parapharma",
        "plastics": "Plastics / Rubbers", "rubber": "Plastics / Rubbers",
        "wood": "Wood & Wood Products", "stone": "Stone / Glass", "glass": "Stone / Glass",
        "chemicals": "Chemicals & Allied Industries", "fertilizers": "Fertilizers & Agri-inputs",
        "textile": "Textiles", "textiles": "Textiles", "metal": "Metals", "metals": "Metals",
    }
    return _all_named_matches(question, _entity_catalog()["sectors"], aliases)


def _find_products(question: str, limit: int = 3) -> list[dict[str, str]]:
    catalog = _entity_catalog()
    q = _norm_text(question)
    found: list[dict[str, str]] = []
    for token in re.findall(r"\b\d{4,6}\b", question):
        if token in {str(y) for y in range(2018, 2026)}:
            continue
        exact = catalog["product_by_hs"].get(token.zfill(6))
        if exact:
            found.append(exact)
        else:
            pref = [p for p in catalog["products"] if p["hs6"].startswith(token)]
            found.extend(pref[:limit])
    aliases = {
        "olive oil": "150910", "virgin olive oil": "150910", "jewelry": "711319",
        "jewellery": "711319", "phosphoric acid": "280920", "wine": "220410",
        "chocolate": "180690", "medicine": "300490", "pharmaceutical": "300490",
        "perfume": "330300", "soap": "340111", "furniture": "940360",
    }
    for alias, hs6 in aliases.items():
        if alias in q and catalog["product_by_hs"].get(hs6):
            found.append(catalog["product_by_hs"][hs6])
    if found:
        unique = []
        seen = set()
        for item in found:
            if item["hs6"] not in seen:
                unique.append(item); seen.add(item["hs6"])
        return unique[:limit]

    # Exact phrase and conservative fuzzy matching for named products.
    candidates: list[tuple[float, dict[str, str]]] = []
    stop = {"what","which","how","much","many","who","where","when","why","are","is","was","were","do","does","did","have","has","had","can","could","should","would","lebanon","lebanese","export","exports","exported","product","products","market","markets","sector","sectors","top","largest","highest","biggest","best","leading","most","least","year","years","data","coverage","value","values","share","shares","growth","trend","rca","pci","complexity","complex","potential","unrealized","untapped","destination","destinations","country","countries","total","overall","annual","history","ranking","rank","performance","to","from","in","of","and","the","a","an","for","about","show","tell","compare"}
    terms = [t for t in q.split() if len(t) > 2 and t not in stop]
    if not terms:
        return []
    phrase = " ".join(terms)
    for item in catalog["products"]:
        name = _norm_text(item["name"])
        overlap = sum(1 for t in terms if t in name)
        if overlap == 0:
            continue
        score = overlap / max(1, len(terms)) + 0.35 * SequenceMatcher(None, phrase, name).ratio()
        if phrase in name:
            score += 0.7
        candidates.append((score, item))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in candidates[:limit] if score >= 0.86]


def _is_any(q: str, *terms: str) -> bool:
    return any(term in q for term in terms)


def _local_plan(question: str) -> dict[str, str] | None:
    """Resolve common dashboard questions locally, without an LLM or API key."""
    q = _norm_text(question)
    years = _extract_years(question)
    year = years[-1] if years else 2025
    limit = _extract_limit(question)
    markets = _find_markets(question)
    sectors = _find_sectors(question)
    products = _find_products(question)
    market = markets[0] if markets else None
    sector = sectors[0] if sectors else None
    product = products[0] if products else None
    asks_trend = _is_any(q, "trend", "history", "over time", "annual", "every year", "evolution", "from 2018", "between 2018") or len(years) > 1
    asks_top = _is_any(q, "top", "largest", "highest", "leading", "biggest", "rank")
    asks_share = "share" in q or "percent" in q or "percentage" in q
    asks_compare = "compare" in q or "versus" in q or " vs " in f" {q} "

    if _is_any(q, "data years", "years covered", "available years", "year range"):
        return {"title": "Dashboard years", "sql": "SELECT key,value FROM meta WHERE key IN ('year_range','build_date') ORDER BY key DESC"}
    if _is_any(q, "data coverage", "coverage", "methodology", "source data", "what data"):
        return {"title": "Dashboard data coverage", "sql": "SELECT key,value FROM meta WHERE key IN ('year_range','build_date','headline_total','final_number_recheck','market_size_rule','cagr_2025_rule') ORDER BY key"}

    # Multiple named entities must be compared before a single-entity profile is selected.
    if asks_compare and len(markets) >= 2 and not product:
        values = ",".join(_sql_quote(m) for m in markets[:6])
        return {"title": f"Market comparison in {year}", "sql": f"SELECT country,export_value,n_products,expy,hhi,unrealized_potential_usd,status,cagr FROM market_year WHERE year={year} AND country IN ({values}) ORDER BY export_value DESC"}
    if asks_compare and len(sectors) >= 2 and not product:
        values = ",".join(_sql_quote(sec) for sec in sectors[:6])
        return {"title": f"Sector comparison in {year}", "sql": f"SELECT sector,export_value,share,rca,pci_avg,n_products_hs6,unrealized_potential_usd,cagr FROM sector_year WHERE year={year} AND sector IN ({values}) ORDER BY export_value DESC"}

    # Product × destination questions use the exact bilateral observations.
    if product and market:
        hs = _sql_quote(product["hs6"]); mk = _sql_quote(market)
        if _is_any(q, "market size", "import market", "size of the market"):
            requested = year if years else 2024
            return {"title": f"{market} market size for HS6 {product['hs6']}", "sql": f"SELECT country,year,hs6,market_size_usd FROM market_size_hs6 WHERE country={mk} AND hs6={hs} ORDER BY ABS(year-{requested}), year DESC LIMIT 2"}
        if _is_any(q, "potential", "unrealized", "untapped", "opportunity"):
            return {"title": f"Unrealized potential for {product['name']} in {market}", "sql": f"SELECT hs6,country,value_usd AS unrealized_potential_usd FROM up_pairs WHERE printf('%06d',hs6)={hs} AND country={mk} LIMIT 1"}
        if asks_trend:
            return {"title": f"Exports of {product['name']} to {market}", "sql": f"SELECT year,value_usd,share_of_market_exports,share_of_product_exports,rca,pci FROM product_market_share WHERE hs6={hs} AND country={mk} ORDER BY year"}
        extra = ",COALESCE(pm.share_of_market_exports,0) AS share_of_market_exports,COALESCE(pm.share_of_product_exports,0) AS share_of_product_exports,pm.rca,pm.pci" if asks_share or _is_any(q,"rca","pci","complex") else ""
        sql = f"SELECT {year} AS year,p.hs6,p.name AS product_name,{mk} AS country,COALESCE(pm.value_usd,0) AS value_usd{extra} FROM products_master p LEFT JOIN product_market_share pm ON pm.hs6=printf('%06d',p.hs6) AND pm.country={mk} AND pm.year={year} WHERE printf('%06d',p.hs6)={hs} LIMIT 1"
        return {"title": f"{product['name']} exports to {market} in {year}", "sql": sql}

    if product:
        hs = _sql_quote(product["hs6"])
        if _is_any(q, "destination", "country", "countries", "markets") and (asks_top or "where" in q):
            return {"title": f"Top destinations for {product['name']} in {year}", "sql": f"SELECT country,value_usd,share_of_product_exports FROM product_market_share WHERE hs6={hs} AND year={year} ORDER BY value_usd DESC LIMIT {limit}"}
        if asks_trend:
            return {"title": f"Annual exports of {product['name']}", "sql": f"SELECT year,export_value,rca,pci,n_countries FROM product_year WHERE hs6={hs} ORDER BY year"}
        metric_cols = ["year","hs6","name","sector","export_value"]
        if _is_any(q,"rca","comparative advantage"): metric_cols.append("rca")
        if _is_any(q,"pci","complexity","complex"): metric_cols.append("pci")
        if _is_any(q,"market reach","countries","destinations"): metric_cols.append("n_countries")
        if _is_any(q,"potential","unrealized","untapped"): metric_cols.append("unrealized_potential_usd")
        if _is_any(q,"growth","cagr","trajectory"): metric_cols += ["cagr","growth","trajectory"]
        return {"title": f"{product['name']} in {year}", "sql": f"SELECT {','.join(dict.fromkeys(metric_cols))} FROM product_year WHERE hs6={hs} AND year={year} LIMIT 1"}

    if market:
        mk = _sql_quote(market)
        if "similar" in q:
            return {"title": f"Markets similar to {market}", "sql": f"SELECT rank,country,iso3,score,continent,exports_2025 FROM similar_markets WHERE country={mk} ORDER BY rank LIMIT {limit}"}
        asks_market_composition = _is_any(q,"product","products") or re.search(r"\bwhat (?:did|does|is) lebanon export(?:ed)? to\b", q) is not None
        if asks_market_composition and (asks_top or "what" in q or "which" in q):
            return {"title": f"Top products exported to {market} in {year}", "sql": f"SELECT hs6,product_name,sector,value_usd,share_of_market_exports,rca,pci FROM product_market_share WHERE country={mk} AND year={year} ORDER BY value_usd DESC LIMIT {limit}"}
        if asks_trend:
            return {"title": f"Annual exports to {market}", "sql": f"SELECT year,export_value,n_products,expy,hhi,unrealized_potential_usd,status FROM market_year WHERE country={mk} ORDER BY year"}
        cols = ["country","year","export_value","n_products"]
        if _is_any(q,"hhi","concentration","diversif"): cols.append("hhi")
        if _is_any(q,"expy","sophistication","complexity"): cols.append("expy")
        if _is_any(q,"rca","comparative advantage"): cols.append("rca")
        if _is_any(q,"potential","unrealized","untapped"): cols.append("unrealized_potential_usd")
        if _is_any(q,"performance","overperform","underperform","priority"): cols += ["status","priority"]
        if _is_any(q,"growth","cagr"): cols.append("cagr")
        return {"title": f"Lebanon's exports to {market} in {year}", "sql": f"SELECT {','.join(dict.fromkeys(cols))} FROM market_year WHERE country={mk} AND year={year} LIMIT 1"}

    if sector:
        sec = _sql_quote(sector)
        if _is_any(q,"product","products") and asks_top:
            return {"title": f"Top products in {sector} in {year}", "sql": f"SELECT hs6,name,export_value,rca,pci,n_countries FROM product_year WHERE sector={sec} AND year={year} ORDER BY export_value DESC LIMIT {limit}"}
        if asks_trend:
            return {"title": f"Annual exports of {sector}", "sql": f"SELECT year,export_value,share,rca,pci_avg,n_products_hs6,unrealized_potential_usd FROM sector_year WHERE sector={sec} ORDER BY year"}
        cols = ["sector","year","export_value","share","n_products_hs6"]
        if _is_any(q,"rca","comparative advantage"): cols.append("rca")
        if _is_any(q,"pci","complexity","complex"): cols.append("pci_avg")
        if _is_any(q,"potential","unrealized","untapped"): cols.append("unrealized_potential_usd")
        if _is_any(q,"growth","cagr"): cols.append("cagr")
        return {"title": f"{sector} in {year}", "sql": f"SELECT {','.join(dict.fromkeys(cols))} FROM sector_year WHERE sector={sec} AND year={year} LIMIT 1"}

    if _is_any(q,"overperform","over performing"):
        return {"title": "Overperforming markets", "sql": f"SELECT rank,value AS country FROM topsis_overperformers ORDER BY rank LIMIT {limit}"}
    if _is_any(q,"underperform","under performing"):
        return {"title": "Underperforming markets", "sql": f"SELECT rank,value AS country FROM topsis_underperformers ORDER BY rank LIMIT {limit}"}
    if _is_any(q,"new products","entered products","product entry"):
        return {"title": f"Products entering exports in {year}", "sql": f"SELECT p.hs6,p.name,p.sector,p.export_value FROM product_year p LEFT JOIN product_year prev ON prev.hs6=p.hs6 AND prev.year=p.year-1 WHERE p.year={year} AND p.export_value>0 AND COALESCE(prev.export_value,0)=0 ORDER BY p.export_value DESC LIMIT {limit}"}
    if _is_any(q,"exited products","lost products","product exit"):
        return {"title": f"Products exiting exports in {year}", "sql": f"SELECT prev.hs6,prev.name,prev.sector,prev.export_value AS previous_export_value FROM product_year prev LEFT JOIN product_year p ON p.hs6=prev.hs6 AND p.year=prev.year+1 WHERE prev.year={year-1} AND prev.export_value>0 AND COALESCE(p.export_value,0)=0 ORDER BY prev.export_value DESC LIMIT {limit}"}

    if asks_trend and _is_any(q,"export","exports","total"):
        return {"title": "Lebanon industrial export trend", "sql": "SELECT year,total_exports_usd,real_exports_2018_usd,active_products,active_markets FROM export_overview ORDER BY year"}
    if asks_top and _is_any(q,"product","products"):
        is_complex_rank = _is_any(q,"complex","pci")
        order = "pci DESC" if is_complex_rank else "rca DESC" if "rca" in q else "unrealized_potential_usd DESC" if _is_any(q,"potential","unrealized") else "export_value DESC"
        condition = " AND pci IS NOT NULL AND export_value>0" if is_complex_rank else ""
        title = f"Most complex exported products in {year}" if is_complex_rank else f"Top products in {year}"
        return {"title": title, "sql": f"SELECT hs6,name,sector,export_value,rca,pci,n_countries,unrealized_potential_usd FROM product_year WHERE year={year}{condition} ORDER BY {order} LIMIT {limit}"}
    if asks_top and _is_any(q,"market","markets","country","countries","destination","destinations"):
        order = "expy DESC" if _is_any(q,"expy","sophistication","complex") else "hhi ASC" if _is_any(q,"diversif","least concentrated") else "unrealized_potential_usd DESC" if _is_any(q,"potential","unrealized") else "export_value DESC"
        return {"title": f"Top export markets in {year}", "sql": f"SELECT country,continent,export_value,n_products,expy,hhi,unrealized_potential_usd,status FROM market_year WHERE year={year} ORDER BY {order} LIMIT {limit}"}
    if asks_top and _is_any(q,"sector","sectors"):
        order = "pci_avg DESC" if _is_any(q,"complex","pci") else "rca DESC" if "rca" in q else "unrealized_potential_usd DESC" if _is_any(q,"potential","unrealized") else "export_value DESC"
        return {"title": f"Top sectors in {year}", "sql": f"SELECT sector,export_value,share,rca,pci_avg,n_products_hs6,unrealized_potential_usd,cagr FROM sector_year WHERE year={year} ORDER BY {order} LIMIT {limit}"}

    if _is_any(q,"complexity","pci"):
        return {"title": f"Most complex exported products in {year}", "sql": f"SELECT hs6,name,sector,export_value,pci,rca FROM product_year WHERE year={year} AND pci IS NOT NULL AND export_value>0 ORDER BY pci DESC LIMIT {limit}"}
    if _is_any(q,"rca","comparative advantage"):
        return {"title": f"Products with the highest RCA in {year}", "sql": f"SELECT hs6,name,sector,export_value,rca,pci FROM product_year WHERE year={year} AND rca IS NOT NULL ORDER BY rca DESC LIMIT {limit}"}
    if _is_any(q,"potential","unrealized","untapped"):
        return {"title": "Largest unrealized product-market opportunities", "sql": f"SELECT printf('%06d',u.hs6) AS hs6,p.name,u.country,u.value_usd AS unrealized_potential_usd FROM up_pairs u LEFT JOIN products_master p ON printf('%06d',u.hs6)=printf('%06d',p.hs6) ORDER BY u.value_usd DESC LIMIT {limit}"}
    if _is_any(q,"export","exports","industrial exports","total"):
        return {"title": f"Lebanon industrial exports in {year}", "sql": f"WITH o AS (SELECT * FROM export_overview WHERE year={year}), p AS (SELECT name,hs6,export_value FROM product_year WHERE year={year} ORDER BY export_value DESC LIMIT 3), m AS (SELECT country,export_value FROM market_year WHERE year={year} ORDER BY export_value DESC LIMIT 3), s AS (SELECT sector,export_value FROM sector_year WHERE year={year} ORDER BY export_value DESC LIMIT 3) SELECT 'Headline' AS category,'Total exports' AS item,total_exports_usd AS value_usd,NULL AS count_value FROM o UNION ALL SELECT 'Headline','Active products',NULL,active_products FROM o UNION ALL SELECT 'Headline','Active markets',NULL,active_markets FROM o UNION ALL SELECT 'Top product',name || ' (HS6 ' || hs6 || ')',export_value,NULL FROM p UNION ALL SELECT 'Top market',country,export_value,NULL FROM m UNION ALL SELECT 'Top sector',sector,export_value,NULL FROM s"}
    return None


def _generic_local_search(question: str, limit: int = 10) -> dict[str, str] | None:
    q = _norm_text(question)
    tokens = [t for t in q.split() if len(t) >= 4 and t not in {"what","which","about","show","tell","from","with","lebanon","exports","export","data","dashboard"}]
    if not tokens:
        return None
    token = max(tokens, key=len)
    like = _sql_quote(f"%{token}%")
    return {
        "title": "Closest matching dashboard records",
        "sql": f"SELECT 'product' AS record_type,hs6 AS code,name AS label,sector AS detail,value_2025 AS value_usd FROM products_master WHERE lower(name) LIKE {like} UNION ALL SELECT 'market','',country,continent,exports_2025 FROM markets_master WHERE lower(country) LIKE {like} UNION ALL SELECT 'sector','',sector,'',value_2025 FROM sectors_master WHERE lower(sector) LIKE {like} LIMIT {limit}",
    }


def _extract_json(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_sql(sql: str) -> str:
    sql = str(sql or "").strip().rstrip(";")
    if not re.match(r"^(select|with)\b", sql, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are permitted.")
    if BLOCKED_SQL.search(sql):
        raise ValueError("Unsafe SQL keyword detected.")
    if ";" in sql:
        raise ValueError("Multiple SQL statements are not permitted.")
    return sql


@lru_cache(maxsize=1)
def _table_names() -> set[str]:
    with sqlite3.connect(_database_path()) as conn:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'"
            )
        }


def _validate_tables(sql: str) -> None:
    # Catch common hallucinated tables early. Subqueries/aliases are allowed.
    named = set(re.findall(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)", sql, re.IGNORECASE))
    unknown = sorted(name for name in named if name.lower() not in {n.lower() for n in _table_names()})
    if unknown:
        raise ValueError("Unknown table(s): " + ", ".join(unknown))


def execute_sql(sql: str, max_rows: int = 60) -> tuple[list[str], list[dict[str, Any]], bool]:
    sql = _safe_sql(sql)
    wrapped = f"SELECT * FROM ({sql}) AS dashboard_query LIMIT {int(max_rows) + 1}"
    with sqlite3.connect(_database_path()) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(wrapped)
        rows = [dict(row) for row in cursor.fetchall()]
        columns = [item[0] for item in cursor.description or []]
    truncated = len(rows) > max_rows
    return columns, rows[:max_rows], truncated


def _planner_prompt(question: str) -> str:
    return f"""
You are the data-query planner for the Lebanon Industrial Export Dashboard.
Convert the user's question into ONE valid SQLite SELECT query using the catalog below.
Return JSON only with this exact shape:
{{"sql":"SELECT ...","title":"brief answer title","reason":"what the query measures"}}
If the question is not about data in this dashboard, return {{"not_dashboard_data":true}}.

Hard rules:
- Use only tables and columns in the catalog.
- Never invent a table, field, country, product or statistic.
- Default to 2025 when no year is specified.
- Preserve HS codes as text.
- Use LIMIT 20 for rankings unless the user asks for another count.
- For a vague question such as "tell me about exports", return a useful dashboard overview with annual total, active products and active markets, plus the leading products/markets/sectors if possible using CTEs or UNION ALL.
- Query exact values. Do not provide an answer in the JSON, only SQL.

{SCHEMA_CATALOG}

User question: {question}
""".strip()


def _plan_sql(question: str, provider: str, model: str, repair: str = "") -> dict[str, Any]:
    prompt = _planner_prompt(question)
    if repair:
        prompt += "\n\nThe previous plan failed with this error. Correct it:\n" + repair
    output = chat_completion(
        provider=provider,
        system_prompt="You generate safe read-only SQLite queries and output JSON only.",
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=1200,
    )
    return _extract_json(output)


def _format_value(value: Any, column: str = "") -> str:
    if value is None:
        return "—"
    col = column.lower()
    if col in {"year", "rank"}:
        try:
            return str(int(value))
        except Exception:
            return str(value)
    if "count" in col or col.startswith("active_") or col in {"n_products", "n_products_hs6", "n_products_hs4", "n_countries"}:
        try:
            return f"{int(float(value)):,}"
        except Exception:
            return str(value)
    if col in {"hs6", "hs4", "code"}:
        text = str(value)
        if text.endswith(".0"):
            text = text[:-2]
        return text.zfill(6 if col == "hs6" else 4)
    if "share" in col or "ratio" in col:
        try:
            return f"{float(value) * 100:.1f}%"
        except Exception:
            return str(value)
    if isinstance(value, float):
        if any(token in col for token in ("usd", "export", "value", "potential", "market_size", "filtered")):
            return f"${value:,.0f}"
        return f"{value:,.4g}"
    if isinstance(value, int):
        if any(token in col for token in ("usd", "export", "value", "potential", "market_size", "filtered")):
            return f"${value:,.0f}"
        return f"{value:,}"
    return str(value)


def _human_label(column: str) -> str:
    replacements = {
        "hs6": "HS6", "hs4": "HS4", "pci": "PCI", "rca": "RCA", "hhi": "HHI",
        "expy": "EXPY", "cagr": "CAGR", "value_usd": "Export value",
        "export_value": "Export value", "total_exports_usd": "Total exports",
        "real_exports_2018_usd": "Real exports (2018 USD)",
        "unrealized_potential_usd": "Unrealized potential", "market_size_usd": "Market size",
        "share_of_market_exports": "Share of exports to this market",
        "share_of_product_exports": "Share of this product's exports",
        "n_products": "Products", "n_products_hs6": "HS6 products", "n_countries": "Markets reached",
        "count_value": "Count",
    }
    return replacements.get(column.lower(), column.replace("_", " ").title())


def _deterministic_answer(title: str, columns: list[str], rows: list[dict[str, Any]], truncated: bool) -> str:
    if not rows:
        return "**No matching dashboard records**\n\nThe dashboard database contains no record matching that exact scope."
    lines = [f"**{title or 'Dashboard result'}**", ""]
    if len(rows) == 1:
        for col in columns:
            value = rows[0].get(col)
            if value is not None:
                lines.append(f"- **{_human_label(col)}:** {_format_value(value, col)}")
        return "\n".join(lines)

    # Add exact, deterministic analytical summaries where the query shape permits.
    value_col = next((c for c in ("total_exports_usd", "export_value", "value_usd", "market_size_usd", "unrealized_potential_usd") if c in columns), None)
    if "year" in columns and value_col:
        dated = [r for r in rows if r.get("year") is not None and r.get(value_col) is not None]
        if len(dated) >= 2:
            first, last = dated[0], dated[-1]
            first_value, last_value = float(first[value_col]), float(last[value_col])
            change = ((last_value / first_value) - 1) * 100 if first_value else None
            if change is not None:
                direction = "increased" if change >= 0 else "decreased"
                lines.extend([
                    f"- **Overall change:** {_human_label(value_col)} {direction} from {_format_value(first_value, value_col)} in {int(first['year'])} to {_format_value(last_value, value_col)} in {int(last['year'])} ({change:+.1f}%).",
                    "",
                ])
    if len(rows) == 2 and value_col and any(c in columns for c in ("country", "sector", "name", "product_name")):
        label_col = next(c for c in ("country", "sector", "name", "product_name") if c in columns)
        ordered = sorted(rows, key=lambda r: float(r.get(value_col) or 0), reverse=True)
        gap = float(ordered[0].get(value_col) or 0) - float(ordered[1].get(value_col) or 0)
        lines.extend([
            f"- **Direct comparison:** {ordered[0].get(label_col)} is higher by {_format_value(gap, value_col)} on {_human_label(value_col).lower()}.",
            "",
        ])

    # Mixed overview queries are grouped into readable sections.
    if "category" in columns and "item" in columns:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("category") or "Results"), []).append(row)
        for category, group in grouped.items():
            lines.extend([f"**{category}**", ""])
            for idx, row in enumerate(group, 1):
                parts = []
                for col in columns:
                    if col in {"category", "item"} or row.get(col) is None:
                        continue
                    parts.append(f"{_human_label(col)}: {_format_value(row.get(col), col)}")
                prefix = f"{idx}. " if category.lower().startswith("top") else "- "
                line = f"{prefix}**{row.get('item')}**"
                if parts:
                    line += " — " + "; ".join(parts)
                lines.append(line)
            lines.append("")
    else:
        # Sidebar-friendly ranked or chronological bullets instead of wide tables.
        for idx, row in enumerate(rows[:30], 1):
            parts = []
            leading = None
            for preferred in ("year", "rank", "name", "product_name", "country", "sector", "item", "label", "hs6"):
                if preferred in row and row.get(preferred) not in (None, ""):
                    leading = f"{_human_label(preferred)} {_format_value(row.get(preferred), preferred)}" if preferred in {"year","rank"} else str(row.get(preferred))
                    break
            used = {"year","rank","name","product_name","country","sector","item","label","category"}
            for col in columns:
                if col in used or row.get(col) is None:
                    continue
                parts.append(f"{_human_label(col)}: {_format_value(row.get(col), col)}")
            prefix = f"{idx}. " if "rank" in columns or not (columns and columns[0] == "year") else "- "
            line = f"{prefix}**{leading or 'Record'}**"
            if parts:
                line += " — " + "; ".join(parts[:6])
            lines.append(line)
    if truncated:
        lines.extend(["", "- Results are limited to the first 60 matching records."])
    return "\n".join(lines).strip()

def _synthesise(question: str, title: str, sql: str, rows: list[dict[str, Any]], truncated: bool, provider: str, model: str) -> str:
    # Keep the payload bounded while exposing exact query output.
    payload = json.dumps(rows[:30], ensure_ascii=False, indent=2)
    prompt = f"""
Answer the user's question using ONLY the exact SQL result below.
State the direct answer first. Preserve every number, year, HS code, product, market and unit exactly.
Do not add facts, causal explanations or statistics absent from the result.
If the result is a ranking, include the requested entries. If it is a time series, describe the direction and give endpoints.
Use compact bold labels and hyphen bullets. Do not use large headings, code blocks, suggested questions or raw SQL.
Mention a limitation only if the query result itself is insufficient.

Question: {question}
Title: {title}
Rows truncated: {truncated}
SQL result JSON:
{payload}
""".strip()
    return chat_completion(
        provider=provider,
        system_prompt="You are a precise trade-data analyst. Never alter supplied figures.",
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.1,
        max_tokens=1200,
    ).strip()


def _offline_overview(question: str) -> DashboardAnswer:
    q = question.lower()
    year_match = re.findall(r"\b20(?:18|19|20|21|22|23|24|25)\b", q)
    year = int(year_match[-1]) if year_match else 2025
    # Broad but exact fallback for simple export questions when no model is configured.
    sql = f"""
    WITH overview AS (
      SELECT year,total_exports_usd,real_exports_2018_usd,active_products,active_markets
      FROM export_overview WHERE year={year}
    ), top_product AS (
      SELECT name,hs6,export_value FROM product_year WHERE year={year} ORDER BY export_value DESC LIMIT 1
    ), top_market AS (
      SELECT country,export_value FROM market_year WHERE year={year} ORDER BY export_value DESC LIMIT 1
    ), top_sector AS (
      SELECT sector,export_value FROM sector_year WHERE year={year} ORDER BY export_value DESC LIMIT 1
    )
    SELECT 'Total exports' AS metric, CAST(total_exports_usd AS TEXT) AS value, CAST(year AS TEXT) AS detail FROM overview
    UNION ALL SELECT 'Active products', CAST(active_products AS TEXT), CAST({year} AS TEXT) FROM overview
    UNION ALL SELECT 'Active markets', CAST(active_markets AS TEXT), CAST({year} AS TEXT) FROM overview
    UNION ALL SELECT 'Top product', name || ' (HS6 ' || hs6 || ')', CAST(export_value AS TEXT) FROM top_product
    UNION ALL SELECT 'Top market', country, CAST(export_value AS TEXT) FROM top_market
    UNION ALL SELECT 'Top sector', sector, CAST(export_value AS TEXT) FROM top_sector
    """
    columns, rows, truncated = execute_sql(sql)
    lines = [f"**Lebanon industrial exports in {year}**", ""]
    for row in rows:
        metric = row.get("metric")
        value = row.get("value")
        detail = row.get("detail")
        if metric == "Total exports":
            lines.append(f"- **Total exports:** ${float(value):,.0f}")
        elif metric in {"Active products", "Active markets"}:
            lines.append(f"- **{metric}:** {int(float(value)):,}")
        else:
            lines.append(f"- **{metric}:** {value} — ${float(detail):,.0f}")
    return DashboardAnswer(True, 0.93, "\n".join(lines), json.dumps(rows, ensure_ascii=False), {"year": year, "metric": "overview"})


def query_dashboard_sql(question: str, provider: str, model: str) -> DashboardAnswer:
    """Query the complete dashboard database for questions missed by direct routes."""
    question = str(question or "").strip()
    if not question or not (DB_PLAIN_PATH.is_file() or DB_GZIP_PATH.is_file()):
        return DashboardAnswer(False, 0.0, "", "", {})

    # Resolve common analytical questions locally first. This path is exact, fast,
    # and works even when no external model key is configured.
    local_plan = _local_plan(question)
    if local_plan:
        try:
            columns, rows, truncated = execute_sql(local_plan["sql"])
            return DashboardAnswer(
                True, 0.99,
                _deterministic_answer(local_plan.get("title", "Dashboard answer"), columns, rows, truncated),
                json.dumps({"sql": local_plan["sql"], "rows": rows, "truncated": truncated}, ensure_ascii=False),
                {"metric": "local_sql_query", "row_count": len(rows)},
            )
        except Exception as exc:
            print(f"Local dashboard query failed: {exc}")

    if not _has_model_key():
        fallback_plan = _generic_local_search(question)
        if fallback_plan:
            try:
                columns, rows, truncated = execute_sql(fallback_plan["sql"])
                if rows:
                    return DashboardAnswer(True, 0.85, _deterministic_answer(fallback_plan["title"], columns, rows, truncated), json.dumps({"sql": fallback_plan["sql"], "rows": rows}, ensure_ascii=False), {"metric": "local_search"})
            except Exception as exc:
                print(f"Local dashboard search failed: {exc}")
        export_terms = ("export", "product", "market", "sector", "rca", "pci", "complex", "potential", "hhi", "expy", "cagr")
        if any(term in question.lower() for term in export_terms):
            return _offline_overview(question)
        return DashboardAnswer(False, 0.0, "", "", {})

    try:
        plan = _plan_sql(question, provider, model)
        if plan.get("not_dashboard_data"):
            return DashboardAnswer(False, 0.0, "", "", {})
        sql = str(plan.get("sql") or "").strip()
        title = str(plan.get("title") or "Dashboard answer").strip()
        if not sql:
            return DashboardAnswer(False, 0.0, "", "", {})
        try:
            columns, rows, truncated = execute_sql(sql)
        except Exception as first_exc:
            repaired = _plan_sql(question, provider, model, repair=f"{first_exc}\nPrevious SQL: {sql}")
            sql = str(repaired.get("sql") or "").strip()
            title = str(repaired.get("title") or title).strip()
            columns, rows, truncated = execute_sql(sql)

        if not rows:
            # Give the planner one chance to broaden an exact-name match.
            broadened = _plan_sql(
                question,
                provider,
                model,
                repair=(
                    "The SQL executed successfully but returned zero rows. "
                    "Use case-insensitive LIKE matching, aliases, or a broader relevant table without changing the user's requested scope. "
                    f"Previous SQL: {sql}"
                ),
            )
            retry_sql = str(broadened.get("sql") or "").strip()
            if retry_sql and retry_sql != sql:
                sql = retry_sql
                title = str(broadened.get("title") or title).strip()
                columns, rows, truncated = execute_sql(sql)
        if not rows:
            return DashboardAnswer(
                True,
                0.92,
                "**Answer**\n\nNo matching records were found in the dashboard data for that wording or entity.",
                json.dumps({"sql": sql, "rows": []}, ensure_ascii=False),
                {"metric": "sql_query"},
            )
        # Keep SQL answers deterministic so figures and formatting never change randomly.
        answer = _deterministic_answer(title, columns, rows, truncated)
        return DashboardAnswer(
            True,
            0.98,
            answer,
            json.dumps({"sql": sql, "rows": rows, "truncated": truncated}, ensure_ascii=False),
            {"metric": "sql_query", "row_count": len(rows)},
        )
    except Exception as exc:
        print(f"Universal dashboard query failed: {exc}")
        # A broad export overview is preferable to falling through to unrelated RAG.
        if any(term in question.lower() for term in ("export", "product", "market", "sector")):
            return _offline_overview(question)
        return DashboardAnswer(False, 0.0, "", "", {})
