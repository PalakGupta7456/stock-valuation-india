"""Valuation methods beyond the cash-flow DCF.

Every method returns a ``MethodResult`` with a per-share value, a plausible
range (low/high), and a plain-English reason when it does not apply, so the
dashboard can show all of them side by side on one football-field chart.

Methods
-------
Intrinsic
  * Residual income (Edwards-Bell-Ohlson): book value + PV of returns above
    the cost of equity. The primary model for banks, NBFCs and insurers.
  * Dividend discount (two-stage, payout converging to a sustainable level).
  * Earnings power value (Greenwald): normalised NOPAT / WACC, the value of
    today's business with zero growth.
Relative
  * Peer-median P/E, P/B, EV/EBITDA and EV/Sales.
  * The stock's own historical P/E.
Rules of thumb
  * Graham number and Graham's revised growth formula.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .data import CompanyData, _latest
from .peers import clean_multiple


@dataclass
class MethodResult:
    name: str
    value: float
    low: float
    high: float
    valid: bool
    note: str                     # how the number was built, or why it does not apply
    family: str = ""
    inputs: dict = field(default_factory=dict)
    table: pd.DataFrame | None = None

    def __post_init__(self):
        self.family = self.family or config.METHOD_FAMILY.get(self.name, "")
        if self.valid and not (np.isfinite(self.value) and self.value > 0):
            self.valid = False
            self.note = self.note or "Model produced a zero or negative value."
        if self.valid:
            lo, hi = np.nanmin([self.low, self.value]), np.nanmax([self.high, self.value])
            self.low, self.high = float(max(lo, 0.0)), float(hi)


def na(name: str, why: str) -> MethodResult:
    return MethodResult(name, np.nan, np.nan, np.nan, False, why)


def _ok(x) -> bool:
    return x is not None and np.isfinite(x)


# ---------------------------------------------------------------------------
# Residual income
# ---------------------------------------------------------------------------
def default_roe(c: CompanyData) -> float | None:
    roe = c.roe_hist.tail(3)
    if len(roe):
        return float(np.clip(roe.mean(), -0.3, 0.6))
    ni = _latest(c.net_income, np.nan)
    return float(ni / c.book_equity) if c.book_equity > 0 and np.isfinite(ni) else None


def _ri_value(bv0, roe0, roe_t, ke, g, payout, years=10):
    t = np.arange(1, years + 1)
    roe = roe0 + (roe_t - roe0) * t / years
    bv_open = np.empty(years)
    bv = bv0
    ri = np.empty(years)
    for i in range(years):
        bv_open[i] = bv
        ni = roe[i] * bv
        ri[i] = ni - ke * bv
        bv += ni * (1 - payout)
    dfac = (1 + ke) ** -t
    ri_n1 = (roe_t - ke) * bv
    tv = ri_n1 / (ke - g) if ke - g > 0.005 else np.nan
    value = bv0 + float((ri * dfac).sum()) + tv * dfac[-1]
    table = pd.DataFrame({"Year": t, "Opening book value": bv_open, "ROE": roe,
                          "Net income": roe * bv_open, "Equity charge": ke * bv_open,
                          "Residual income": ri, "Discount factor": dfac, "PV of RI": ri * dfac})
    return value, tv * dfac[-1], table


def residual_income(c: CompanyData, ke: float, g: float, roe0: float | None = None,
                    roe_terminal: float | None = None) -> MethodResult:
    name = "Residual income"
    bv0 = c.bvps
    if not (bv0 > 0):
        return na(name, "Book value is zero or negative.")
    roe0 = default_roe(c) if roe0 is None else roe0
    if roe0 is None:
        return na(name, "Return on equity could not be measured.")
    if roe_terminal is None:
        # Excess returns fade halfway toward the cost of equity, capped at Ke + 6pp.
        roe_terminal = float(min(0.5 * (roe0 + ke), ke + 0.06))
    payout = c.payout
    value, pv_tv, table = _ri_value(bv0, roe0, roe_terminal, ke, g, payout)
    lo = min(_ri_value(bv0, roe0, roe_terminal - 0.03, ke + 0.01, g, payout)[0], value)
    hi = max(_ri_value(bv0, roe0, roe_terminal + 0.03, ke - 0.01, g, payout)[0], value)
    note = (f"Book value ₹{bv0:,.0f}/share plus returns above the {ke:.1%} cost of equity. "
            f"ROE fades from {roe0:.1%} to {roe_terminal:.1%} over 10 years.")
    return MethodResult(name, value, lo, hi, True, note,
                        inputs=dict(bv0=bv0, roe0=roe0, roe_terminal=roe_terminal, ke=ke, g=g,
                                    payout=payout, pv_terminal=pv_tv, book_share=bv0 / value
                                    if value else np.nan),
                        table=table)


# ---------------------------------------------------------------------------
# Dividend discount model
# ---------------------------------------------------------------------------
def _ddm_value(eps0, payout0, g_high, g, ke, roe_t, years=10, high_years=5):
    t = np.arange(1, years + 1)
    fade = (t - high_years) / (years - high_years)
    growth = np.where(t <= high_years, g_high, g_high - (g_high - g) * fade)
    eps = eps0 * np.cumprod(1 + growth)
    payout_t = float(np.clip(1 - g / roe_t, 0.05, 1.0)) if roe_t > g else 1.0
    payout = np.where(t <= high_years, payout0, payout0 + (payout_t - payout0) * fade)
    dps = eps * payout
    dfac = (1 + ke) ** -t
    tv = eps[-1] * (1 + g) * payout_t / (ke - g) if ke - g > 0.005 else np.nan
    value = float((dps * dfac).sum() + tv * dfac[-1])
    table = pd.DataFrame({"Year": t, "EPS growth": growth, "EPS": eps, "Payout": payout,
                          "Dividend": dps, "Discount factor": dfac, "PV of dividend": dps * dfac})
    return value, table, payout_t


def dividend_discount(c: CompanyData, ke: float, g_high: float, g: float,
                      roe_terminal: float) -> MethodResult:
    name = "Dividend discount"
    if c.dividend_per_share <= 0:
        return na(name, "Pays no regular dividend, so a dividend model says little.")
    eps0 = c.eps
    if not (_ok(eps0) and eps0 > 0):
        return na(name, "Earnings are negative; dividends are not covered.")
    payout0 = c.payout
    # Dividends can only grow as fast as retained earnings compound: blend the
    # revenue-growth assumption with sustainable growth = ROE x retention.
    roe0 = default_roe(c)
    sustainable = roe0 * (1 - payout0) if roe0 is not None else g_high
    g_high = float(np.clip(0.5 * g_high + 0.5 * sustainable, -0.05, 0.20))
    value, table, payout_t = _ddm_value(eps0, payout0, g_high, g, ke, roe_terminal)
    lo = _ddm_value(eps0, payout0, g_high, g, ke + 0.01, roe_terminal)[0]
    hi = _ddm_value(eps0, payout0, g_high, g, ke - 0.01, roe_terminal)[0]
    note = (f"Dividends of ₹{c.dividend_per_share:,.1f} today (payout {payout0:.0%}) grow with "
            f"earnings at {g_high:.1%} (half your growth input, half ROE × retention); payout rises to a sustainable {payout_t:.0%} by year 10.")
    return MethodResult(name, value, lo, hi, True, note,
                        inputs=dict(dps=c.dividend_per_share, payout0=payout0, payout_t=payout_t,
                                    g_high=g_high, ke=ke),
                        table=table)


# ---------------------------------------------------------------------------
# Earnings power value
# ---------------------------------------------------------------------------
def earnings_power(c: CompanyData, wacc: float, tax: float) -> MethodResult:
    name = "Earnings power (no growth)"
    margins = (c.ebit / c.revenue).replace([np.inf, -np.inf], np.nan).dropna()
    if margins.empty:
        return na(name, "Operating margin history unavailable.")
    margin = float(margins.mean())
    revenue = _latest(c.revenue)
    nopat = revenue * margin * (1 - tax)
    if nopat <= 0:
        return na(name, "Normalised operating profit is negative.")

    def per_share(w):
        return (nopat / w - c.total_debt + c.cash - c.minority_interest) / c.shares

    note = (f"Average EBIT margin of {margin:.1%} on current revenue, after tax, capitalised at "
            f"WACC {wacc:.1%}. Assumes no growth: the gap to the DCF is what you pay for growth.")
    return MethodResult(name, per_share(wacc), per_share(wacc + 0.01), per_share(max(wacc - 0.01, 0.04)),
                        True, note, inputs=dict(margin=margin, nopat=nopat))


# ---------------------------------------------------------------------------
# Graham
# ---------------------------------------------------------------------------
def graham_number(c: CompanyData) -> MethodResult:
    name = "Graham number"
    eps, bv = c.eps, c.bvps
    if not (_ok(eps) and eps > 0 and bv > 0):
        return na(name, "Needs positive earnings and book value.")
    v = float(np.sqrt(22.5 * eps * bv))
    return MethodResult(name, v, v, v, True,
                        f"√(22.5 × EPS ₹{eps:,.1f} × book ₹{bv:,.0f}): the most a defensive investor "
                        "should pay (P/E 15 × P/B 1.5). Ignores growth, so harsh on quality compounders.")


def graham_formula(c: CompanyData, growth: float,
                   aaa_yield: float = config.AAA_CORPORATE_YIELD) -> MethodResult:
    name = "Graham formula"
    eps = c.eps
    if not (_ok(eps) and eps > 0):
        return na(name, "Needs positive earnings.")
    g = float(np.clip(growth * 100, 0, 15))

    def val(gg):
        return eps * (8.5 + 2 * gg) * 4.4 / (aaa_yield * 100)

    return MethodResult(name, val(g), val(max(g - 3, 0)), val(min(g + 3, 15)), True,
                        f"EPS ₹{eps:,.1f} × (8.5 + 2 × {g:.1f}% growth) × 4.4 / {aaa_yield:.1%} AAA "
                        "bond yield. Graham's 1974 rule of thumb.")


# ---------------------------------------------------------------------------
# Relative valuation
# ---------------------------------------------------------------------------
def _ev_to_share(c: CompanyData, ev: float) -> float:
    return (ev - c.total_debt + c.cash - c.minority_interest) / c.shares


def peer_multiples(c: CompanyData, peers: pd.DataFrame) -> list[MethodResult]:
    out: list[MethodResult] = []
    if peers is None or peers.empty:
        why = ("No peer group found for this industry. Add peer tickers in the sidebar."
               if not c.is_demo else "No peers.")
        return [na(n, why) for n in ("Peer P/E", "Peer EV/EBITDA", "Peer P/B", "Peer EV/Sales")]

    eps, bv = c.eps, c.bvps
    ebitda, rev = _latest(c.ebitda, np.nan), _latest(c.revenue, np.nan)
    specs = [
        ("Peer P/E", "P/E", "pe", eps, lambda m: m * eps, "EPS", "earnings"),
        ("Peer EV/EBITDA", "EV/EBITDA", "ev_ebitda", ebitda, lambda m: _ev_to_share(c, m * ebitda),
         "EBITDA", "EBITDA"),
        ("Peer P/B", "P/B", "pb", bv, lambda m: m * bv, "book value", "book value"),
        ("Peer EV/Sales", "EV/Sales", "ev_sales", rev, lambda m: _ev_to_share(c, m * rev),
         "revenue", "revenue"),
    ]
    for name, col, key, base, fn, label, word in specs:
        if c.is_financial and key in ("ev_ebitda", "ev_sales"):
            out.append(na(name, "Enterprise-value multiples do not work for lenders, whose debt is "
                                "operating capital."))
            continue
        if not (_ok(base) and base > 0):
            out.append(na(name, f"Company's {label} is negative, so the multiple is meaningless."))
            continue
        m = clean_multiple(peers[col], key) if col in peers else pd.Series(dtype=float)
        if len(m) < 2:
            out.append(na(name, f"Fewer than two peers report a usable {col}."))
            continue
        med, p25, p75 = m.median(), m.quantile(0.25), m.quantile(0.75)
        out.append(MethodResult(
            name, fn(med), fn(p25), fn(p75), True,
            f"Peer median {col} of {med:.1f}× applied to the company's {word} "
            f"(peer range {p25:.1f}×–{p75:.1f}×, {len(m)} peers).",
            inputs=dict(multiple=med, p25=p25, p75=p75, n=len(m), base=base)))
    return out


def own_history_pe(c: CompanyData) -> MethodResult:
    name = "Own 5y P/E"
    eps = c.eps
    if not (_ok(eps) and eps > 0):
        return na(name, "Needs positive earnings.")
    pe = clean_multiple(c.historical_pe(), "pe")
    if len(pe) < 2:
        return na(name, "Not enough price and earnings history.")
    med = float(pe.median())
    return MethodResult(name, med * eps, float(pe.min()) * eps, float(pe.max()) * eps, True,
                        f"Median P/E of {med:.1f}× at the last {len(pe)} fiscal year-ends "
                        f"(range {pe.min():.1f}×–{pe.max():.1f}×) on today's EPS ₹{eps:,.1f}. "
                        "Assumes the market's past mood is a fair anchor.",
                        inputs=dict(multiple=med, series=pe))
