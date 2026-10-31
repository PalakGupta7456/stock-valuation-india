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
    """BSE scrip code -> trading symbol -> name, from the latest daily bhavcopy.

    Yahoo lists BSE stocks as SYMBOL.BO, so the symbol column is what lets the
    app turn a numeric scrip code (e.g. 500325) into a Yahoo ticker.
    """
    import datetime as dt

    headers = {**UA, "Referer": "https://www.bseindia.com/"}
    errors = []
    day = dt.date.today()
    for _ in range(10):                     # walk back over weekends / holidays
        stamp = day.strftime("%Y%m%d")
        url = f"https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{stamp}_F_0000.CSV"
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.ok and len(r.content) > 10_000:
                df = pd.read_csv(io.StringIO(r.text))
                df = df[df["FinInstrmTp"].astype(str).str.strip() == "STK"] if "FinInstrmTp" in df else df
                return pd.DataFrame({"symbol": df["FinInstrmId"].astype(str).str.strip(),
                                     "ticker": df["TckrSymb"].astype(str).str.strip(),
                                     "name": df["FinInstrmNm"].astype(str).str.strip()})
            errors.append(f"{stamp}: HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{stamp}: {exc}")
        day -= dt.timedelta(days=1)
    # Fallback: BSE's JSON API (codes and names only; symbols resolved via search)
    url = ("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry="
           "&segment=Equity&status=Active")
    try:
        r = requests.get(url, headers={**headers, "Origin": "https://www.bseindia.com"}, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        return pd.DataFrame({"symbol": df["SCRIP_CD"].astype(str).str.strip(),
                             "ticker": df.get("scrip_id", df["SCRIP_CD"]).astype(str).str.strip(),
                             "name": df["Scrip_Name"].str.strip()})
    except Exception as exc:  # noqa: BLE001
        errors.append(f"api: {exc}")
    raise RuntimeError("; ".join(errors[-4:]))


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
            print(f"::warning title={name.upper()} list not refreshed::{type(exc).__name__}: {exc}")
            print(f"{name}: failed ({exc}); keeping existing file", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
