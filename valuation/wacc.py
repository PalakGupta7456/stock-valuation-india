"""Cost of capital: regression beta, synthetic rating, CAPM and WACC."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .data import CompanyData, _latest


@dataclass
class BetaResult:
    raw: float | None
    adjusted: float
    r_squared: float | None
    observations: int
    source: str          # "regression", "yahoo", "default"


@dataclass
class WACCResult:
    risk_free: float
    erp: float
    beta: float
    cost_of_equity: float
    interest_coverage: float | None
    rating: str
    default_spread: float
    pre_tax_cost_of_debt: float
    tax_rate: float
    after_tax_cost_of_debt: float
    equity_value: float
    debt_value: float
    weight_equity: float
    weight_debt: float
    wacc: float


def regression_beta(stock: pd.Series, index: pd.Series) -> BetaResult | None:
    """Weekly-return OLS beta vs Nifty 50, Blume-adjusted toward 1."""
    if stock is None or index is None or len(stock) < 30 or len(index) < 30:
        return None
    s = stock.copy()
    i = index.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    i.index = pd.to_datetime(i.index).tz_localize(None).normalize()
    df = pd.concat([s.pct_change(), i.pct_change()], axis=1, join="inner").dropna()
    if len(df) < 52:
        return None
    x, y = df.iloc[:, 1].values, df.iloc[:, 0].values
    var = np.var(x, ddof=1)
    if var == 0:
        return None
    raw = float(np.cov(y, x, ddof=1)[0, 1] / var)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    adjusted = 0.67 * raw + 0.33 * 1.0
    return BetaResult(raw, float(np.clip(adjusted, 0.3, 2.5)), r2, len(df), "regression")


def estimate_beta(c: CompanyData) -> BetaResult:
    reg = regression_beta(c.price_history, c.index_history)
    if reg is not None and (reg.r_squared or 0) >= 0.05:
        return reg
    if c.beta_reported:
        b = float(c.beta_reported)
        return BetaResult(b, float(np.clip(0.67 * b + 0.33, 0.3, 2.5)), None, 0, "yahoo")
    return BetaResult(None, 1.0, None, 0, "default")


def synthetic_rating(coverage: float | None) -> tuple[str, float]:
    if coverage is None or np.isinf(coverage):
        return "AAA", config.SYNTHETIC_RATINGS[0][2]
    for threshold, rating, spread in config.SYNTHETIC_RATINGS:
        if coverage >= threshold:
            return rating, spread
    return "D", 0.20


def compute_wacc(
    c: CompanyData,
    beta: float,
    risk_free: float = config.RISK_FREE_RATE,
    erp: float = config.EQUITY_RISK_PREMIUM,
    tax_rate: float = config.MARGINAL_TAX_RATE,
    spread_override: float | None = None,
) -> WACCResult:
    ke = risk_free + beta * erp

    ebit = _latest(c.ebit)
    interest = _latest(c.interest_expense)
    coverage = (ebit / interest) if interest > 0 else None
    rating, spread = synthetic_rating(coverage)
    if spread_override is not None:
        spread = spread_override
    kd = risk_free + spread
    kd_at = kd * (1 - tax_rate)

    e = c.market_cap
    d = max(c.total_debt, 0.0)
    we = e / (e + d) if (e + d) > 0 else 1.0
    wd = 1 - we
    return WACCResult(
        risk_free=risk_free, erp=erp, beta=beta, cost_of_equity=ke,
        interest_coverage=coverage, rating=rating, default_spread=spread,
        pre_tax_cost_of_debt=kd, tax_rate=tax_rate, after_tax_cost_of_debt=kd_at,
        equity_value=e, debt_value=d, weight_equity=we, weight_debt=wd,
        wacc=we * ke + wd * kd_at,
    )
