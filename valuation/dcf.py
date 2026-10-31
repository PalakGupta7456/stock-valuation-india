"""Two-stage DCF models: Free Cash Flow to Firm (FCFF) and to Equity (FCFE).

Structure (Damodaran-style "story to numbers"):
  * Revenue grows at `growth_high` for `high_years`, then fades linearly to the
    terminal growth rate by the final projection year.
  * Operating margin moves linearly from today's margin to a target margin
    over `margin_years`, then holds.
  * Reinvestment = change in revenue / sales-to-capital ratio.
  * Effective tax rate drifts to the marginal rate during the fade period.
  * Terminal value is consistent with growth: reinvestment rate = g / ROIC.

The core engine is vectorised with NumPy so the same code powers the base
case, scenarios, sensitivity grids and Monte Carlo simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import config


@dataclass
class DCFInputs:
    growth_high: float
    terminal_growth: float
    op_margin_current: float
    op_margin_target: float
    tax_rate: float
    sales_to_capital: float
    terminal_roic: float
    wacc: float
    cost_of_equity: float
    pre_tax_cost_of_debt: float
    debt_ratio: float
    marginal_tax: float = config.MARGINAL_TAX_RATE
    years: int = config.PROJECTION_YEARS
    high_years: int = config.HIGH_GROWTH_YEARS
    margin_years: int = config.MARGIN_CONVERGENCE_YEARS
    mid_year: bool = True

    def with_(self, **kw) -> "DCFInputs":
        return replace(self, **kw)


@dataclass
class Bridge:
    """Balance-sheet items that bridge enterprise value to per-share equity."""
    base_revenue: float
    total_debt: float
    cash: float
    minority_interest: float
    shares: float


@dataclass
class DCFResult:
    method: str
    per_share: float
    equity_value: float
    enterprise_value: float | None
    pv_explicit: float
    pv_terminal: float
    terminal_value: float
    terminal_share: float
    implied_terminal_multiple: float  # TV / terminal-year EBIT (FCFF) or NI (FCFE)
    table: pd.DataFrame

    @property
    def valid(self) -> bool:
        return bool(np.isfinite(self.per_share) and self.per_share > 0)


# ---------------------------------------------------------------------------
# Vectorised core
# ---------------------------------------------------------------------------
def _col(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1, 1)


def project_operations(inp: DCFInputs, base_revenue: float, **arrays) -> dict:
    """Project revenue -> FCFF for one or many parameter sets.

    Any of growth_high, terminal_growth, op_margin_target, sales_to_capital may
    be passed in `arrays` as 1-D arrays (one value per simulation).
    """
    N, H, M = inp.years, inp.high_years, inp.margin_years
    gh = _col(arrays.get("growth_high", inp.growth_high))
    g = _col(arrays.get("terminal_growth", inp.terminal_growth))
    mt = _col(arrays.get("op_margin_target", inp.op_margin_target))
    s2c = _col(arrays.get("sales_to_capital", inp.sales_to_capital))
    m0 = inp.op_margin_current
    t = np.arange(1, N + 1)[None, :].astype(float)

    fade = (t - H) / max(N - H, 1)
    growth = np.where(t <= H, gh, gh - (gh - g) * fade)
    margin = np.where(t <= M, m0 + (mt - m0) * t / M, mt)
    tax = np.where(t <= H, inp.tax_rate, inp.tax_rate + (inp.marginal_tax - inp.tax_rate) * fade)
    tax = np.broadcast_to(tax, growth.shape)

    revenue = base_revenue * np.cumprod(1 + growth, axis=1)
    prev_rev = np.concatenate([np.full((revenue.shape[0], 1), base_revenue), revenue[:, :-1]], axis=1)
    ebit = revenue * margin
    nopat = ebit * (1 - tax)
    reinvestment = (revenue - prev_rev) / s2c
    fcff = nopat - reinvestment
    return dict(t=t, growth=growth, margin=margin, tax=tax, revenue=revenue, ebit=ebit,
                nopat=nopat, reinvestment=reinvestment, fcff=fcff, g=g, mt=mt)


def fcff_per_share(inp: DCFInputs, bridge: Bridge, **arrays) -> dict:
    """Vectorised FCFF valuation. Returns arrays (one per simulation)."""
    p = project_operations(inp, bridge.base_revenue, **arrays)
    wacc = _col(arrays.get("wacc", inp.wacc))
    roic = _col(arrays.get("terminal_roic", inp.terminal_roic))
    g, mt, N = p["g"], p["mt"], inp.years

    timing = p["t"] - 0.5 if inp.mid_year else p["t"]
    dfac = (1 + wacc) ** (-timing)
    pv = p["fcff"] * dfac

    rev_n1 = p["revenue"][:, -1:] * (1 + g)
    ebit_n1 = rev_n1 * mt
    nopat_n1 = ebit_n1 * (1 - inp.marginal_tax)
    fcff_n1 = nopat_n1 * (1 - g / roic)
    spread = wacc - g
    tv = np.where(spread > 0.005, fcff_n1 / np.where(spread > 0.005, spread, 1), np.nan)
    pv_tv = tv * (1 + wacc) ** (-N)

    ev = pv.sum(axis=1, keepdims=True) + pv_tv
    equity = ev - bridge.total_debt + bridge.cash - bridge.minority_interest
    return dict(per_share=(equity / bridge.shares).ravel(), equity=equity.ravel(), ev=ev.ravel(),
                pv_explicit=pv.sum(axis=1), pv_tv=pv_tv.ravel(), tv=tv.ravel(),
                ebit_n1=ebit_n1.ravel(), dfac=dfac, pv=pv, proj=p)


# ---------------------------------------------------------------------------
# Detailed single-case models
# ---------------------------------------------------------------------------
def run_fcff(inp: DCFInputs, bridge: Bridge) -> DCFResult:
    r = fcff_per_share(inp, bridge)
    p = r["proj"]
    table = pd.DataFrame({
        "Year": p["t"][0].astype(int),
        "Revenue": p["revenue"][0],
        "Growth": p["growth"][0],
        "EBIT margin": p["margin"][0],
        "EBIT": p["ebit"][0],
        "Tax rate": p["tax"][0],
        "NOPAT": p["nopat"][0],
        "Reinvestment": p["reinvestment"][0],
        "FCFF": p["fcff"][0],
        "Discount factor": r["dfac"][0],
        "PV of FCFF": r["pv"][0],
    })
    ev = float(r["ev"][0])
    pv_tv = float(r["pv_tv"][0])
    return DCFResult(
        method="FCFF", per_share=float(r["per_share"][0]), equity_value=float(r["equity"][0]),
        enterprise_value=ev, pv_explicit=float(r["pv_explicit"][0]), pv_terminal=pv_tv,
        terminal_value=float(r["tv"][0]), terminal_share=pv_tv / ev if ev else np.nan,
        implied_terminal_multiple=float(r["tv"][0] / r["ebit_n1"][0]) if r["ebit_n1"][0] else np.nan,
        table=table,
    )


def run_fcfe(inp: DCFInputs, bridge: Bridge) -> DCFResult:
    """FCFE = Net income - (1 - debt ratio) x reinvestment, discounted at Ke.

    Debt grows by debt_ratio x reinvestment each year so leverage stays stable;
    interest is charged on opening debt at the pre-tax cost of debt.
    """
    p = project_operations(inp, bridge.base_revenue)
    N, d, kd, ke = inp.years, inp.debt_ratio, inp.pre_tax_cost_of_debt, inp.cost_of_equity
    g = inp.terminal_growth

    ebit, tax, reinv = p["ebit"][0], p["tax"][0], p["reinvestment"][0]
    debt_open = np.empty(N)
    debt = bridge.total_debt
    for i in range(N):
        debt_open[i] = debt
        debt += d * reinv[i]
    interest = kd * debt_open
    net_income = (ebit - interest) * (1 - tax)
    net_borrowing = d * reinv
    equity_reinv = (1 - d) * reinv
    fcfe = net_income - equity_reinv

    timing = p["t"][0] - 0.5 if inp.mid_year else p["t"][0]
    dfac = (1 + ke) ** (-timing)
    pv = fcfe * dfac

    rev_n1 = p["revenue"][0, -1] * (1 + g)
    ebit_n1 = rev_n1 * inp.op_margin_target
    debt_n = debt  # closing debt after year N
    ni_n1 = (ebit_n1 - kd * debt_n) * (1 - inp.marginal_tax)
    reinv_n1 = ebit_n1 * (1 - inp.marginal_tax) * g / inp.terminal_roic
    fcfe_n1 = ni_n1 - (1 - d) * reinv_n1
    tv = fcfe_n1 / (ke - g) if ke - g > 0.005 else np.nan
    pv_tv = tv * (1 + ke) ** (-N)

    equity = pv.sum() + pv_tv + bridge.cash - bridge.minority_interest
    table = pd.DataFrame({
        "Year": p["t"][0].astype(int),
        "Revenue": p["revenue"][0],
        "EBIT": ebit,
        "Interest": interest,
        "Net income": net_income,
        "Reinvestment": reinv,
        "Net borrowing": net_borrowing,
        "FCFE": fcfe,
        "Discount factor": dfac,
        "PV of FCFE": pv,
    })
    total = pv.sum() + pv_tv
    return DCFResult(
        method="FCFE", per_share=float(equity / bridge.shares), equity_value=float(equity),
        enterprise_value=None, pv_explicit=float(pv.sum()), pv_terminal=float(pv_tv),
        terminal_value=float(tv), terminal_share=float(pv_tv / total) if total else np.nan,
        implied_terminal_multiple=float(tv / ni_n1) if ni_n1 else np.nan, table=table,
    )
