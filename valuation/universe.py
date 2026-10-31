"""The searchable list of listed Indian companies.

``data/nse_symbols.csv`` (and optionally ``data/bse_symbols.csv``) are
refreshed weekly by a GitHub Action running ``scripts/update_symbols.py``.
Any ticker not in the list can still be typed directly or found through
Yahoo Finance's name search.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_universe() -> pd.DataFrame:
    """Return columns: symbol (Yahoo format), name, exchange, label."""
    frames = []
    for fname, suffix, exch in [("nse_symbols.csv", ".NS", "NSE"), ("bse_symbols.csv", ".BO", "BSE")]:
        path = DATA_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).dropna(subset=["symbol"])
        df["symbol"] = df["symbol"].str.strip().str.upper() + suffix
        df["name"] = df["name"].fillna("").str.strip()
        df["exchange"] = exch
        frames.append(df[["symbol", "name", "exchange"]])
    if not frames:
        return pd.DataFrame(columns=["symbol", "name", "exchange", "label"])
    u = pd.concat(frames, ignore_index=True)
    # Keep the BSE listing only for companies not on NSE.
    nse_names = set(u.loc[u.exchange == "NSE", "name"].str.lower())
    u = u[(u.exchange == "NSE") | ~u["name"].str.lower().isin(nse_names)]
    u["label"] = u["name"] + " · " + u["symbol"].str.replace(r"\.(NS|BO)$", "", regex=True) + \
        " (" + u["exchange"] + ")"
    return u.sort_values(["exchange", "name"]).reset_index(drop=True)
