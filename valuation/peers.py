"""Peer groups and peer multiples for relative valuation.

Yahoo Finance does not publish peer lists for Indian stocks, so peers are
picked from a curated map keyed on Yahoo's industry label. Users can always
type their own peer list in the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_IT = ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT", "COFORGE", "MPHASIS"]
_BANKS = ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK", "BANKBARODA",
          "FEDERALBNK", "IDFCFIRSTB", "PNB"]
_POWER = ["NTPC", "POWERGRID", "TATAPOWER", "ADANIPOWER", "NHPC", "JSWENERGY", "TORNTPOWER"]
_INSURANCE_GEN = ["ICICIGI", "NIACL", "GICRE", "STARHEALTH"]

PEER_MAP: dict[str, list[str]] = {
    "Information Technology Services": _IT,
    "Software - Application": ["OFSS", "TATAELXSI", "KPITTECH", "PERSISTENT", "COFORGE", "NAUKRI"],
    "Software - Infrastructure": ["OFSS", "TATAELXSI", "KPITTECH", "PERSISTENT", "COFORGE"],
    "Banks - Regional": _BANKS,
    "Banks - Diversified": _BANKS,
    "Credit Services": ["BAJFINANCE", "SHRIRAMFIN", "CHOLAFIN", "MUTHOOTFIN", "M&MFIN", "LICHSGFIN",
                        "PFC", "RECLTD"],
    "Insurance - Life": ["HDFCLIFE", "SBILIFE", "ICICIPRULI", "LICI"],
    "Insurance - Diversified": _INSURANCE_GEN,
    "Insurance - Property & Casualty": _INSURANCE_GEN,
    "Capital Markets": ["HDFCAMC", "CAMS", "CDSL", "BSE", "ANGELONE", "MCX"],
    "Asset Management": ["HDFCAMC", "NAM-INDIA", "UTIAMC", "CAMS"],
    "Oil & Gas Refining & Marketing": ["RELIANCE", "BPCL", "IOC", "HINDPETRO", "MRPL"],
    "Oil & Gas Integrated": ["ONGC", "OIL", "RELIANCE", "GAIL"],
    "Household & Personal Products": ["HINDUNILVR", "DABUR", "GODREJCP", "MARICO", "COLPAL",
                                      "EMAMILTD", "PGHH"],
    "Packaged Foods": ["NESTLEIND", "BRITANNIA", "TATACONSUM", "HATSUN", "DABUR"],
    "Tobacco": ["ITC", "GODFRYPHLP", "VSTIND"],
    "Beverages - Non-Alcoholic": ["VBL", "TATACONSUM"],
    "Beverages - Wineries & Distilleries": ["UNITDSPR", "RADICO", "UBL"],
    "Specialty Chemicals": ["ASIANPAINT", "BERGEPAINT", "KANSAINER", "PIDILITIND", "SRF",
                            "AARTIIND", "DEEPAKNTR", "NAVINFLUOR"],
    "Auto Manufacturers": ["MARUTI", "M&M", "HYUNDAI", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
                           "TVSMOTOR", "ASHOKLEY"],
    "Auto Parts": ["BOSCHLTD", "MOTHERSON", "SONACOMS", "UNOMINDA", "BHARATFORG", "EXIDEIND",
                   "MRF", "APOLLOTYRE"],
    "Drug Manufacturers - Specialty & Generic": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN",
                                                 "AUROPHARMA", "ZYDUSLIFE", "TORNTPHARM", "ALKEM",
                                                 "DIVISLAB"],
    "Drug Manufacturers - General": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "TORNTPHARM"],
    "Medical Care Facilities": ["APOLLOHOSP", "MAXHEALTH", "FORTIS", "NH"],
    "Building Materials": ["ULTRACEMCO", "SHREECEM", "AMBUJACEM", "ACC", "DALBHARAT", "RAMCOCEM",
                           "JKCEMENT"],
    "Steel": ["TATASTEEL", "JSWSTEEL", "JINDALSTEL", "SAIL"],
    "Other Industrial Metals & Mining": ["HINDALCO", "VEDL", "NMDC", "HINDZINC", "NATIONALUM"],
    "Aluminum": ["HINDALCO", "NATIONALUM", "VEDL"],
    "Thermal Coal": ["COALINDIA", "NMDC"],
    "Utilities - Regulated Electric": _POWER,
    "Utilities - Independent Power Producers": _POWER,
    "Utilities - Renewable": _POWER,
    "Telecom Services": ["BHARTIARTL", "IDEA", "INDUSTOWER", "TATACOMM"],
    "Engineering & Construction": ["LT", "IRB", "KEC", "KPIL", "NCC"],
    "Specialty Industrial Machinery": ["SIEMENS", "ABB", "CUMMINSIND", "BHEL", "THERMAX", "CGPOWER"],
    "Aerospace & Defense": ["HAL", "BEL", "BDL", "MAZDOCK", "COCHINSHIP"],
    "Electrical Equipment & Parts": ["POLYCAB", "HAVELLS", "KEI", "SIEMENS", "CGPOWER"],
    "Furnishings, Fixtures & Appliances": ["HAVELLS", "VOLTAS", "CROMPTON", "WHIRLPOOL", "BLUESTARCO"],
    "Real Estate - Development": ["DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "LODHA"],
    "Restaurants": ["JUBLFOOD", "DEVYANI", "WESTLIFE"],
    "Luxury Goods": ["TITAN", "KALYANKJIL", "SENCO"],
    "Apparel Retail": ["TRENT", "ABFRL", "DMART"],
    "Department Stores": ["TRENT", "DMART", "ABFRL"],
    "Grocery Stores": ["DMART", "TRENT"],
    "Agricultural Inputs": ["UPL", "PIIND", "COROMANDEL", "CHAMBLFERT", "SUMICHEM"],
    "Internet Content & Information": ["NAUKRI", "ETERNAL", "SWIGGY", "PAYTM"],
}

PEER_COLUMNS = ["Ticker", "Name", "Market cap", "P/E", "P/B", "EV/EBITDA", "EV/Sales", "ROE",
                "Revenue growth"]


def suggest_peers(ticker: str, industry: str, n: int = 6) -> list[str]:
    """Curated peers for a ticker (with exchange suffix), excluding itself."""
    base, _, suffix = ticker.partition(".")
    suffix = f".{suffix}" if suffix else config.DEFAULT_SUFFIX
    # Peer symbols are NSE codes; BSE numeric codes still get NSE peers.
    peer_suffix = ".NS" if suffix == ".BO" else suffix
    return [f"{p}{peer_suffix}" for p in PEER_MAP.get(industry, []) if p != base][:n]


def _num(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def fetch_peer_multiples(tickers: list[str]) -> pd.DataFrame:
    """Download trailing multiples for each peer from Yahoo Finance."""
    import yfinance as yf

    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
        except Exception:  # one bad peer should not sink the table
            continue
        if not (info.get("currentPrice") or info.get("regularMarketPrice")):
            continue
        rows.append({
            "Ticker": t,
            "Name": info.get("shortName") or info.get("longName") or t,
            "Market cap": _num(info.get("marketCap")),
            "P/E": _num(info.get("trailingPE")),
            "P/B": _num(info.get("priceToBook")),
            "EV/EBITDA": _num(info.get("enterpriseToEbitda")),
            "EV/Sales": _num(info.get("enterpriseToRevenue")),
            "ROE": _num(info.get("returnOnEquity")),
            "Revenue growth": _num(info.get("revenueGrowth")),
        })
    return pd.DataFrame(rows, columns=PEER_COLUMNS)


def demo_peers() -> pd.DataFrame:
    return pd.DataFrame([
        ("PEER1", "Alpha Engineering (synthetic)", 9.5e11, 24.0, 4.1, 14.5, 2.6, 0.18, 0.11),
        ("PEER2", "Beta Machines (synthetic)", 6.2e11, 29.5, 5.2, 17.0, 3.1, 0.19, 0.14),
        ("PEER3", "Gamma Systems (synthetic)", 3.8e11, 18.5, 2.9, 11.0, 1.8, 0.15, 0.09),
        ("PEER4", "Delta Industrial (synthetic)", 2.1e11, 33.0, 6.0, 19.5, 3.6, 0.20, 0.17),
        ("PEER5", "Epsilon Tech (synthetic)", 1.4e11, 21.0, 3.4, 12.5, 2.1, 0.16, 0.10),
    ], columns=PEER_COLUMNS)


def clean_multiple(s: pd.Series, key: str) -> pd.Series:
    lo, hi = config.MULTIPLE_BOUNDS[key]
    s = pd.to_numeric(s, errors="coerce")
    return s[(s >= lo) & (s <= hi)].dropna()
