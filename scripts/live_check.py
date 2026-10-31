"""Value a broad basket of real Indian stocks with live Yahoo Finance data.

Run on GitHub Actions (``.github/workflows/live-check.yml``) on every push and
every week, so data-source breakages show up quickly. Writes a Markdown report
to ``reports/live_check.md`` and to the job summary.

    python scripts/live_check.py                 # default basket
    python scripts/live_check.py TCS ITC 500325  # any tickers
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from valuation import Assumptions, fetch_company, normalise_ticker, value_company  # noqa: E402
from valuation.peers import fetch_peer_multiples, suggest_peers  # noqa: E402

# Large, mid and small caps across sectors, lenders, insurers, a loss-maker,
# a recent IPO and BSE-only style codes.
BASKET = [
    "TCS", "INFY", "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "HDFCLIFE",
    "ITC", "HINDUNILVR", "ASIANPAINT", "MARUTI", "SUNPHARMA", "LT", "ULTRACEMCO", "TATASTEEL",
    "NTPC", "BHARTIARTL", "TITAN", "DMART", "COALINDIA", "HAL", "PIDILITIND", "POLYCAB",
    "PERSISTENT", "IRCTC", "IDEA", "ZYDUSLIFE", "KPITTECH", "SWIGGY",
    "500325.BO", "532540.BO",
]

OUT = Path(__file__).resolve().parent.parent / "reports"


def run_one(raw: str) -> dict:
    t0 = time.time()
    ticker = normalise_ticker(raw)
    row = {"input": raw, "ticker": ticker, "ok": False}
    try:
        c = fetch_company(ticker)
        peers = fetch_peer_multiples(suggest_peers(c.ticker, c.industry))
        rep = value_company(c, Assumptions(mc_runs=1000), peers)
        v = rep.verdict
        row.update(
            ok=bool(np.isfinite(v.fair_value)), ticker=c.ticker, name=c.name, sector=c.sector,
            industry=c.industry, financial=c.is_financial, price=round(c.price, 2),
            fair_value=round(v.fair_value, 2) if np.isfinite(v.fair_value) else None,
            upside=round(v.upside, 4) if np.isfinite(v.upside) else None, rating=v.rating,
            confidence=v.confidence, methods=int(rep.method_table["Applies"].sum()),
            peers=len(peers), years=int(c.revenue.dropna().shape[0]),
            notes=" ".join(c.notes),
        )
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["trace"] = traceback.format_exc(limit=3)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def to_markdown(rows: list[dict]) -> str:
    ok = sum(r["ok"] for r in rows)
    lines = [f"## Live valuation check: {ok}/{len(rows)} tickers valued",
             "", "| Ticker | Company | Price | Fair value | Upside | Rating | Conf. | Methods | Peers | Yrs |",
             "|---|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["ok"]:
            lines.append(f"| {r['ticker']} | {r['name'][:32]} | ₹{r['price']:,.0f} | ₹{r['fair_value']:,.0f} | "
                         f"{r['upside']:+.0%} | {r['rating']} | {r['confidence']} | {r['methods']}/12 | "
                         f"{r['peers']} | {r['years']} |")
        else:
            lines.append(f"| {r['ticker']} | ❌ {r.get('error', 'no value')[:90]} | | | | | | | | |")
    lines += ["", "_Educational output from an automated model, not investment advice._"]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    tickers = argv or BASKET
    rows = []
    for t in tickers:
        r = run_one(t)
        status = "ok " if r["ok"] else "ERR"
        print(f"[{status}] {r['ticker']:<14} {r.get('rating', r.get('error', ''))!s:<40} {r['seconds']}s",
              flush=True)
        rows.append(r)
        time.sleep(0.5)   # be gentle with Yahoo
    OUT.mkdir(exist_ok=True)
    md = to_markdown(rows)
    (OUT / "live_check.md").write_text(md, encoding="utf-8")
    (OUT / "live_check.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    rate = sum(r["ok"] for r in rows) / len(rows)
    print(f"\nSuccess rate: {rate:.0%}")
    return 0 if rate >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
