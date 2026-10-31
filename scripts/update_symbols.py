"""Refresh the list of listed Indian companies used by the app's search box.

Run by .github/workflows/update-symbols.yml every week. NSE's equity list is
the primary source; BSE's active-scrip list adds BSE-only companies.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36", "Accept": "*/*"}


def nse() -> pd.DataFrame:
    s = requests.Session()
    s.headers.update(UA)
    s.get("https://www.nseindia.com", timeout=20)          # sets cookies
    r = s.get("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv", timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().upper() for c in df.columns]
    df = df[df["SERIES"].str.strip().isin(["EQ", "BE", "BZ", "SM", "ST"])]
    return pd.DataFrame({"symbol": df["SYMBOL"].str.strip(), "name": df["NAME OF COMPANY"].str.strip()})


def bse() -> pd.DataFrame:
    url = ("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry="
           "&segment=Equity&status=Active")
    r = requests.get(url, headers={**UA, "Referer": "https://www.bseindia.com/", "Origin":
                                   "https://www.bseindia.com"}, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    return pd.DataFrame({"symbol": df["SCRIP_CD"].astype(str).str.strip(),
                         "name": df["Scrip_Name"].str.strip()})


def main() -> int:
    OUT.mkdir(exist_ok=True)
    ok = False
    for name, fn in [("nse", nse), ("bse", bse)]:
        try:
            df = fn().drop_duplicates("symbol").sort_values("symbol")
            if len(df) < 500:
                raise ValueError(f"only {len(df)} rows")
            df.to_csv(OUT / f"{name}_symbols.csv", index=False)
            print(f"{name}: {len(df)} companies")
            ok = True
        except Exception as exc:  # keep the previous file if a source is down
            print(f"{name}: failed ({exc}); keeping existing file", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
