"""A synthetic company so the app (and tests) work without internet access."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CompanyData


def demo_company() -> CompanyData:
    years = pd.to_datetime(["2023-03-31", "2024-03-31", "2025-03-31", "2026-03-31"])
    rev = pd.Series([4.00e11, 4.52e11, 5.03e11, 5.62e11], index=years)
    ebit = rev * np.array([0.172, 0.178, 0.181, 0.184])
    interest = pd.Series([6.0e9, 6.3e9, 6.1e9, 5.9e9], index=years)
    pretax = ebit - interest + 4.0e9
    tax = pretax * 0.255
    ni = pretax - tax

    rng = np.random.default_rng(7)
    weeks = pd.date_range(end="2026-10-30", periods=260, freq="W-FRI")
    mkt_ret = rng.normal(0.0025, 0.022, len(weeks))
    stk_ret = 0.0035 + 0.9 * mkt_ret + rng.normal(0, 0.025, len(weeks))
    index = pd.Series(24000 * np.cumprod(1 + mkt_ret), index=weeks)
    stock = pd.Series(1415.0 * np.cumprod(1 + stk_ret), index=weeks)

    return CompanyData(
        ticker="DEMO", name="Demo Industries Ltd (synthetic)", sector="Industrials",
        industry="Specialty Industrial Machinery", currency="INR", financial_currency="INR",
        price=float(stock.iloc[-1]), shares=1.2e9,
        revenue=rev, ebit=ebit, net_income=ni, da=rev * 0.04, capex=rev * 0.065,
        delta_nwc=rev.diff().fillna(0) * 0.12, interest_expense=interest,
        pretax_income=pretax, tax_provision=tax,
        total_debt=8.0e10, cash=4.5e10, minority_interest=2.0e9, book_equity=3.1e11,
        beta_reported=0.92, price_history=stock, index_history=index,
        dividend_per_share=14.0, is_demo=True,
        ebitda=ebit + rev * 0.04,
        book_equity_hist=pd.Series([2.30e11, 2.52e11, 2.80e11, 3.10e11], index=years),
        payout_ratio=14.0 * 1.2e9 / float(ni.iloc[-1]),
        analyst_target_mean=1420.0, analyst_target_low=1080.0, analyst_target_high=1750.0,
        analyst_count=18, analyst_view="buy",
        week52_low=float(stock.iloc[-52:].min()), week52_high=float(stock.iloc[-52:].max()),
    )
