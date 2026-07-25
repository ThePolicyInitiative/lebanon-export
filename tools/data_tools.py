from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


COUNTRY_ALIASES = {
    "uae": "UAE",
    "united arab emirates": "UAE",
    "saudi": "Saudi Arabia",
    "saudi arabia": "Saudi Arabia",
    "usa": "United States",
    "us": "United States",
    "united states": "United States",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "qatar": "Qatar",
    "iraq": "Iraq",
    "jordan": "Jordan",
    "egypt": "Egypt",
    "france": "France",
    "germany": "Germany",
    "italy": "Italy",
    "turkey": "Turkey",
    "oman": "Oman",
    "kuwait": "Kuwait",
}

COUNTRY_TO_ISO3 = {
    "UAE": "ARE",
    "United Arab Emirates": "ARE",
    "Saudi Arabia": "SAU",
    "Qatar": "QAT",
    "Kuwait": "KWT",
    "Egypt": "EGY",
    "Jordan": "JOR",
    "Iraq": "IRQ",
    "Turkey": "TUR",
    "Greece": "GRC",
    "Italy": "ITA",
    "France": "FRA",
    "Germany": "DEU",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Canada": "CAN",
    "Australia": "AUS",
    "China": "CHN",
    "India": "IND",
    "Lebanon": "LBN",
}

BACI_COUNTRIES = {
    422: "Lebanon",
    784: "United Arab Emirates",
    682: "Saudi Arabia",
    634: "Qatar",
    414: "Kuwait",
    818: "Egypt",
    400: "Jordan",
    368: "Iraq",
    792: "Turkey",
    300: "Greece",
    380: "Italy",
    250: "France",
    276: "Germany",
    826: "United Kingdom",
    842: "United States",
    124: "Canada",
    36: "Australia",
    156: "China",
    699: "India",
}


def normalize_hs(hs_code: str | int) -> str:
    digits = re.sub(r"\D", "", str(hs_code))
    if len(digits) < 6:
        digits = digits.zfill(6)
    return digits[:6]


def normalize_country(country: str) -> str:
    value = str(country).strip()
    return COUNTRY_ALIASES.get(value.lower(), value)


@lru_cache(maxsize=1)
def load_export_book() -> dict[int, pd.DataFrame]:
    path = DATA_DIR / "TPI_Product_Market_Data.xlsx"
    xl = pd.ExcelFile(path)
    frames: dict[int, pd.DataFrame] = {}
    for sheet in xl.sheet_names:
        match = re.search(r"(\d{4})", sheet)
        if not match:
            continue
        year = int(match.group(1))
        df = pd.read_excel(path, sheet_name=sheet)
        df["HS Code"] = df["HS Code"].apply(normalize_hs)
        frames[year] = df
    return frames


@lru_cache(maxsize=1)
def load_baci() -> pd.DataFrame:
    path = DATA_DIR / "baci_filtered_lebanon_competitors_2018_2024.csv"
    df = pd.read_csv(path)
    df["hs6"] = df["k"].apply(normalize_hs)
    df["exporter"] = df["i"].map(BACI_COUNTRIES).fillna(df["i"].astype(str))
    df["destination"] = df["j"].map(BACI_COUNTRIES).fillna(df["j"].astype(str))
    return df


@lru_cache(maxsize=1)
def load_factories() -> pd.DataFrame:
    path = DATA_DIR / "all_factories_combined_full_REPLACEMENT.xlsx"
    return pd.read_excel(path)


@lru_cache(maxsize=1)
def load_world_bank() -> pd.DataFrame:
    macro = pd.read_csv(DATA_DIR / "world_bank_macro_logistics_selected_markets.csv")
    extra = pd.read_csv(DATA_DIR / "world_bank_extra_trade_backbone_selected_markets.csv")
    return pd.concat([macro, extra], ignore_index=True).drop_duplicates()


def validate_input(hs_code: str, destination: str) -> dict[str, Any]:
    hs = normalize_hs(hs_code)
    country = normalize_country(destination)
    years = load_export_book()
    available = any(hs in set(df["HS Code"].astype(str)) for df in years.values())
    country_available = any(country in df.columns for df in years.values())
    errors = []
    if len(hs) < 4:
        errors.append("HS code must contain at least 4 digits; 6 digits is recommended.")
    if not available:
        errors.append(f"HS code {hs} was not found in the export dataset.")
    if not country_available:
        errors.append(f"Destination '{country}' was not found in the export dataset columns.")
    return {
        "valid": not errors,
        "hs_code": hs,
        "destination": country,
        "errors": errors,
    }


def get_export_trend(hs_code: str, destination: str) -> dict[str, Any]:
    hs = normalize_hs(hs_code)
    country = normalize_country(destination)
    records = []
    product_name = None
    for year, df in sorted(load_export_book().items()):
        row = df[df["HS Code"] == hs]
        if row.empty or country not in df.columns:
            value = 0.0
        else:
            product_name = product_name or str(row.iloc[0].get("Product name", ""))
            parsed = pd.to_numeric(row.iloc[0][country], errors="coerce")
            value = 0.0 if pd.isna(parsed) else float(parsed)
        records.append({"year": year, "export_value": value})
    trend = pd.DataFrame(records)
    start = float(trend.iloc[0]["export_value"]) if not trend.empty else 0.0
    end = float(trend.iloc[-1]["export_value"]) if not trend.empty else 0.0
    change = end - start
    pct_change = None if start == 0 else (change / start) * 100
    peak_row = trend.loc[trend["export_value"].idxmax()].to_dict() if trend["export_value"].sum() else None
    return {
        "tool": "export_trend",
        "hs_code": hs,
        "product_name": product_name,
        "destination": country,
        "records": records,
        "start_value": start,
        "end_value": end,
        "absolute_change": change,
        "percent_change": pct_change,
        "peak": peak_row,
    }


def get_competitor_snapshot(hs_code: str, destination: str) -> dict[str, Any]:
    hs = normalize_hs(hs_code)
    country = normalize_country(destination)
    df = load_baci()
    subset = df[df["hs6"] == hs].copy()
    if country in set(df["destination"]):
        subset = subset[subset["destination"] == country]
    latest_year = int(subset["t"].max()) if not subset.empty else None
    latest = subset[subset["t"] == latest_year].copy() if latest_year else pd.DataFrame()
    if not latest.empty:
        latest["trade_value"] = pd.to_numeric(latest["v"], errors="coerce").fillna(0)
        top = latest.groupby("exporter", as_index=False)["trade_value"].sum().sort_values("trade_value", ascending=False).head(8)
        top_records = top.to_dict("records")
    else:
        top_records = []
    return {
        "tool": "competitor_snapshot",
        "hs_code": hs,
        "destination_filter_used": country if country in set(df["destination"]) else "all destinations in filtered BACI",
        "latest_year": latest_year,
        "top_exporters": top_records,
        "rows_used": int(len(subset)),
    }


def get_factory_capacity(hs_code: str) -> dict[str, Any]:
    hs = normalize_hs(hs_code)
    df = load_factories()
    hs_cols = [c for c in df.columns if "hs_code" in str(c).lower()]
    sector_col = "leb__sector" if "leb__sector" in df.columns else None
    province_col = "leb__province" if "leb__province" in df.columns else None
    name_col = "leb__name" if "leb__name" in df.columns else df.columns[0]

    def row_match(row: pd.Series) -> bool:
        for col in hs_cols:
            code = normalize_hs(row.get(col, ""))
            if code and (code.startswith(hs[:4]) or hs.startswith(code[:4])):
                return True
        return False

    matches = df[df.apply(row_match, axis=1)] if hs_cols else pd.DataFrame()
    samples = []
    if not matches.empty:
        cols = [name_col]
        if sector_col:
            cols.append(sector_col)
        if province_col:
            cols.append(province_col)
        samples = matches[cols].head(10).fillna("").to_dict("records")
    return {
        "tool": "factory_capacity",
        "hs_code": hs,
        "matching_factories": int(len(matches)),
        "sample_factories": samples,
        "match_rule": "matched by HS prefix using factory HS code columns",
    }


def get_macro_logistics(destination: str, years: list[int] | None = None) -> dict[str, Any]:
    import os as _os

    country = normalize_country(destination)
    iso3 = COUNTRY_TO_ISO3.get(country, country[:3].upper())
    live_rows: list[dict[str, Any]] = []
    if _os.getenv("USE_LIVE_WORLDBANK") == "1":
        from tools.web_tools import fetch_worldbank_live
        live_rows = fetch_worldbank_live(iso3)
    df = load_world_bank()
    subset = df[df["country_code"] == iso3].copy()
    if years:
        subset = subset[subset["year"].isin(years)]
    latest = subset.sort_values("year").groupby("indicator_code", as_index=False).tail(1)
    keep = latest[["indicator_name", "indicator_code", "category", "year", "value"]]
    indicators = live_rows + keep.head(40).to_dict("records")
    return {
        "tool": "macro_logistics",
        "destination": country,
        "country_code": iso3,
        "latest_indicators": indicators,
        "rows_used": int(len(subset)) + len(live_rows),
        "source": "live World Bank API + local snapshot" if live_rows else "local snapshot",
    }


@lru_cache(maxsize=1)
def _factory_hs_prefixes() -> list[str]:
    """4-digit HS prefixes present in the factory registry (for fast counting)."""
    df = load_factories()
    hs_cols = [c for c in df.columns if "hs_code" in str(c).lower()]
    prefixes: list[str] = []
    for col in hs_cols:
        for value in df[col].dropna():
            code = normalize_hs(value)
            if code and code != "000000":
                prefixes.append(code[:4])
    return prefixes


def count_factories(hs_code: str) -> int:
    prefix = normalize_hs(hs_code)[:4]
    return sum(1 for p in _factory_hs_prefixes() if p == prefix)


def get_untapped_products(destination: str, limit: int = 10) -> pd.DataFrame:
    """Products Lebanon exports successfully elsewhere but not (or barely) to
    this destination: proven supply, missing market. Where BACI covers the
    product, the destination's own import demand is checked so that products
    nobody sells to this market are not presented as opportunities."""
    country = normalize_country(destination)
    years = load_export_book()
    latest = years[max(years)].copy()
    if country not in latest.columns:
        return pd.DataFrame()
    latest["HS Code"] = latest["HS Code"].map(normalize_hs)
    value_cols = [c for c in latest.columns if c not in ["HS Code", "Product name"]]
    for col in value_cols:
        latest[col] = pd.to_numeric(latest[col], errors="coerce").fillna(0.0)
    other_cols = [c for c in value_cols if c != country]
    latest = latest.copy()
    latest["exports_elsewhere"] = latest[other_cols].sum(axis=1)
    latest["exports_to_destination"] = latest[country]
    # "Doesn't already export" = zero or negligible (<1% of what it sells elsewhere)
    untapped = latest[
        (latest["exports_elsewhere"] > 0)
        & (latest["exports_to_destination"] <= 0.01 * latest["exports_elsewhere"])
    ].copy()
    if untapped.empty:
        return pd.DataFrame()
    untapped["biggest_current_market"] = untapped[other_cols].idxmax(axis=1)

    # Demand-side check: does the destination import this product from ANYONE?
    # BACI values are in thousand USD; use the latest BACI year.
    baci = load_baci()
    dest_rows = baci[baci["destination"].map(normalize_country) == country]
    demand_usd: dict[str, float] = {}
    if not dest_rows.empty:
        latest_year = dest_rows["t"].max()
        grp = dest_rows[dest_rows["t"] == latest_year].groupby("hs6")["v"].sum() * 1000.0
        demand_usd = grp.to_dict()
    covered = set(baci["hs6"].unique())

    def _demand(hs: str) -> float | None:
        if hs not in covered:
            return None  # no data for this product at all
        return float(demand_usd.get(hs, 0.0))

    untapped["destination_demand_usd"] = untapped["HS Code"].map(_demand)

    def _signal(v) -> str:
        if v is None or pd.isna(v):
            return "no data"
        return "confirmed" if v >= 1_000_000 else "weak"

    untapped["demand_signal"] = untapped["destination_demand_usd"].map(_signal)
    # Confirmed demand first, unknown next, measured-and-weak last; within each
    # band, biggest proven Lebanese supply first. A confirmed-demand product
    # with only token Lebanese supply (<$1M elsewhere) is not a headline
    # opportunity, so it ranks after the unknowns.
    rank = {"confirmed": 0, "no data": 1, "weak": 3}
    untapped["_rank"] = untapped["demand_signal"].map(rank)
    thin = (untapped["demand_signal"] == "confirmed") & (untapped["exports_elsewhere"] < 1_000_000)
    untapped.loc[thin, "_rank"] = 2
    untapped = untapped.sort_values(["_rank", "exports_elsewhere"], ascending=[True, False])
    selected = untapped.head(limit)
    # Keep visible any big-supply product that was demoted for weak measured
    # demand — the demotion itself is the insight worth showing.
    big_but_weak = untapped[
        (untapped["demand_signal"] == "weak")
        & (untapped["exports_elsewhere"] >= 1_000_000)
        & (~untapped.index.isin(selected.index))
    ]
    untapped = pd.concat([selected, big_but_weak])
    untapped["matching_factories"] = untapped["HS Code"].map(count_factories)
    out = untapped[[
        "HS Code", "Product name", "exports_elsewhere", "exports_to_destination",
        "biggest_current_market", "matching_factories",
        "destination_demand_usd", "demand_signal",
    ]].reset_index(drop=True)
    return out.round({"exports_elsewhere": 0, "exports_to_destination": 0,
                      "destination_demand_usd": 0})


def run_all_tools(hs_code: str, destination: str) -> dict[str, Any]:
    validation = validate_input(hs_code, destination)
    if not validation["valid"]:
        return {
            "valid": False,
            "validation": validation,
            "tool_trace": [{"tool": "validate_input", "status": "failed", "errors": validation["errors"]}],
        }
    export_trend = get_export_trend(validation["hs_code"], validation["destination"])
    years = [row["year"] for row in export_trend["records"]]
    competitor = get_competitor_snapshot(validation["hs_code"], validation["destination"])
    capacity = get_factory_capacity(validation["hs_code"])
    macro = get_macro_logistics(validation["destination"], years)
    from tools.web_tools import get_market_signals
    signals = get_market_signals(export_trend.get("product_name"), validation["destination"]) if __import__("os").getenv("TAVILY_API_KEY") else {"tool": "market_signals", "status": "skipped", "reason": "TAVILY_API_KEY not set", "results": []}
    return {
        "valid": True,
        "validation": validation,
        "export_trend": export_trend,
        "competitor_snapshot": competitor,
        "factory_capacity": capacity,
        "macro_logistics": macro,
        "market_signals": signals,
        "tool_trace": [
            {"tool": "validate_input", "status": "ok"},
            {"tool": "export_trend", "status": "ok", "records": len(export_trend["records"])},
            {"tool": "competitor_snapshot", "status": "ok", "rows_used": competitor["rows_used"]},
            {"tool": "factory_capacity", "status": "ok", "matches": capacity["matching_factories"]},
            {"tool": "macro_logistics", "status": "ok", "rows_used": macro["rows_used"]},
            {"tool": "market_signals", "status": signals["status"]},
        ],
    }
