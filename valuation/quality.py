"""Data-quality and model-suitability checks.

Each flag carries a penalty that reduces the overall confidence score, so the
recommendation is honest about how much the numbers can be trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .data import CompanyData, _latest
from .dcf import DCFInputs


@dataclass
class Flag:
    level: str      # "error" | "warning" | "info"
    title: str
    detail: str
    penalty: int


def run_checks(c: CompanyData, inp: DCFInputs, risk_free: float) -> list[Flag]:
    flags: list[Flag] = []
    n_years = int(c.revenue.dropna().shape[0])

    if c.is_demo:
        flags.append(Flag("info", "Demo data",
                          "Synthetic company used to show how the app works. Not a real stock.", 0))
    if c.is_financial:
        flags.append(Flag("warning", "Financial company",
                          "Banks, NBFCs and insurers use debt as raw material, so cash-flow DCFs and "
                          "EV multiples are switched off. The blend leans on residual income, P/B, "
                          "P/E and dividends instead; Yahoo's statement data for lenders is also "
                          "less complete, so check book value against the annual report.", 10))
    if n_years < 3:
        flags.append(Flag("warning", "Short history",
                          f"Only {n_years} year(s) of statements; defaults derived from history are noisy.",
                          15))
    if _latest(c.ebit) <= 0:
        flags.append(Flag("error", "Negative operating income",
                          "Latest EBIT is zero or negative. Value depends entirely on the target-margin "
                          "assumption.", 25))
    if c.book_equity <= 0:
        flags.append(Flag("warning", "Negative book equity",
                          "Accumulated losses or buybacks; invested-capital ratios are distorted.", 10))
    if c.interest_expense.dropna().empty and c.total_debt > 0:
        flags.append(Flag("warning", "Interest expense missing",
                          "Cost of debt uses the default spread instead of a coverage-based rating.", 5))
    if c.financial_currency and c.currency and c.financial_currency != c.currency:
        flags.append(Flag("error", "Currency mismatch",
                          f"Statements are in {c.financial_currency} but the price is in {c.currency}.",
                          30))

    growth = c.revenue.pct_change().dropna()
    if len(growth) >= 2 and growth.std() > 0.25:
        flags.append(Flag("warning", "Volatile revenue",
                          f"Year-on-year revenue growth swings by ±{growth.std():.0%}; "
                          "a single growth rate may not describe the business well.", 10))

    hist_fcff = c.historical_fcff(inp.tax_rate)
    if len(hist_fcff) >= 2 and (hist_fcff < 0).mean() > 0.5:
        flags.append(Flag("warning", "Negative historical free cash flow",
                          "Most past years burned cash; the DCF relies on a turnaround.", 10))

    if c.latest_fy is not None:
        age_days = (datetime.now() - c.latest_fy.to_pydatetime().replace(tzinfo=None)).days
        if age_days > 550:
            flags.append(Flag("warning", "Stale financials",
                              f"Latest annual statement is {age_days // 30} months old.", 10))

    if inp.terminal_growth >= risk_free:
        flags.append(Flag("error", "Terminal growth ≥ risk-free rate",
                          "No company can outgrow the economy forever. Lower terminal growth.", 20))
    if inp.terminal_roic < inp.wacc:
        flags.append(Flag("info", "Terminal ROIC below WACC",
                          "Growth in the terminal period destroys value under these assumptions.", 0))
    if np.isfinite(inp.growth_high) and inp.growth_high > 0.25:
        flags.append(Flag("warning", "Aggressive growth",
                          f"{inp.growth_high:.0%} growth for {inp.high_years} years is rare; "
                          "check it against the company's guidance and industry.", 5))
    return flags
