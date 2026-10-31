"""Fetch and normalise company data from Yahoo Finance (via yfinance).

All financial series are returned oldest -> newest, in absolute currency
units (INR for NSE/BSE listings). Capex is stored as a positive number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from . import config


@dataclass
class CompanyData:
    ticker: str
    name: str
    sector: str
    industry: str
    currency: str
    financial_currency: str
    price: float
    shares: float
    # Annual statement series (index = fiscal year end, ascending)
    revenue: pd.Series
    ebit: pd.Series
    net_income: pd.Series
    da: pd.Series
    capex: pd.Series
    delta_nwc: pd.Series
    interest_expense: pd.Series
    pretax_income: pd.Series
    tax_provision: pd.Series
    # Latest balance sheet
    total_debt: float
    cash: float
    minority_interest: float
    book_equity: float
    # Market data
    beta_reported: float | None
    price_history: pd.Series
    index_history: pd.Series
    dividend_per_share: float = 0.0
    # Extra fields for relative / equity / dividend models
    ebitda: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    book_equity_hist: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    eps_ttm: float | None = None
    payout_ratio: float | None = None
    analyst_target_mean: float | None = None
    analyst_target_low: float | None = None
    analyst_target_high: float | None = None
    analyst_count: int = 0
    analyst_view: str | None = None
    week52_low: float | None = None
    week52_high: float | None = None
    is_demo: bool = False
    fetched_at: datetime = field(default_factory=datetime.now)
    notes: list[str] = field(default_factory=list)

    # ---- convenience -------------------------------------------------------
    @property
    def market_cap(self) -> float:
        return self.price * self.shares

    @property
    def is_financial(self) -> bool:
        text = f"{self.sector} {self.industry}".lower()
        return self.sector in config.FINANCIAL_SECTORS or any(
            k in text for k in ("bank", "insurance", "nbfc", "credit services", "capital markets")
        )

    @property
    def latest_fy(self) -> pd.Timestamp | None:
        return self.revenue.index[-1] if len(self.revenue) else None

    @property
    def eps(self) -> float:
        """Trailing EPS; falls back to latest annual net income / shares."""
        if self.eps_ttm is not None and np.isfinite(self.eps_ttm):
            return float(self.eps_ttm)
        return _latest(self.net_income, np.nan) / self.shares

    @property
    def bvps(self) -> float:
        return self.book_equity / self.shares

    @property
    def roe_hist(self) -> pd.Series:
        """Net income / average book equity, by fiscal year."""
        be = self.book_equity_hist.dropna()
        if len(be) < 2:
            return (self.net_income / be).dropna() if len(be) else pd.Series(dtype=float)
        avg = (be + be.shift(1)) / 2
        return (self.net_income.reindex(be.index) / avg).replace([np.inf, -np.inf], np.nan).dropna()

    @property
    def payout(self) -> float:
        """Dividend payout ratio, clipped to [0, 1]."""
        if self.payout_ratio is not None and np.isfinite(self.payout_ratio) and self.payout_ratio > 0:
            return float(min(self.payout_ratio, 1.0))
        eps = self.eps
        if eps and eps > 0 and self.dividend_per_share > 0:
            return float(min(self.dividend_per_share / eps, 1.0))
        return 0.0

    def historical_pe(self) -> pd.Series:
        """Price at each fiscal year end / EPS of that year."""
        ph = self.price_history.dropna()
        if ph.empty or self.net_income.dropna().empty:
            return pd.Series(dtype=float)
        ph = ph.copy()
        ph.index = pd.to_datetime(ph.index).tz_localize(None) if ph.index.tz is not None \
            else pd.to_datetime(ph.index)
        out = {}
        for fy, ni in self.net_income.dropna().items():
            if ni <= 0:
                continue
            window = ph[(ph.index >= fy - pd.Timedelta(days=10)) & (ph.index <= fy + pd.Timedelta(days=10))]
            if window.empty:
                continue
            out[fy] = float(window.mean()) / (ni / self.shares)
        return pd.Series(out, dtype=float)

    def historical_fcff(self, tax_rate: float) -> pd.Series:
        nopat = self.ebit * (1 - tax_rate)
        return (nopat + self.da - self.capex - self.delta_nwc).dropna()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalise_ticker(raw: str, suffix: str = config.DEFAULT_SUFFIX) -> str:
    t = raw.strip().upper().replace(" ", "")
    if not t:
        raise ValueError("Please enter a ticker.")
    if t == "DEMO" or t.startswith("^") or "." in t:
        return t
    return f"{t}{suffix}"


def _row(df: pd.DataFrame | None, names: list[str]) -> pd.Series:
    """Return the first matching statement row as an ascending float series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    return pd.Series(dtype=float)


def _latest(s: pd.Series, default: float = 0.0) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else default


def _opt(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean_statement(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    # yfinance often appends an almost-empty oldest column; drop sparse columns
    keep = df.columns[df.notna().mean() > 0.4]
    return df[keep]


# ---------------------------------------------------------------------------
# Main fetch
# ---------------------------------------------------------------------------
def fetch_company(ticker: str) -> CompanyData:
    """Download statements, balance sheet and price history for one ticker."""
    if ticker == "DEMO":
        from .demo import demo_company
        return demo_company()

    import yfinance as yf

    tk = yf.Ticker(ticker)
    try:
        info = tk.info or {}
    except Exception as exc:  # network / rate-limit
        raise RuntimeError(f"Could not reach Yahoo Finance for {ticker}: {exc}") from exc

    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    if not price:
        raise ValueError(
            f"No price found for '{ticker}'. Check the symbol (e.g. RELIANCE.NS, TCS.NS, 500325.BO)."
        )

    inc = _clean_statement(tk.income_stmt)
    cf = _clean_statement(tk.cashflow)
    bs = _clean_statement(tk.balance_sheet)
    if inc is None or inc.empty:
        raise ValueError(f"Yahoo Finance returned no financial statements for {ticker}.")

    notes: list[str] = []

    revenue = _row(inc, ["Total Revenue", "Operating Revenue"])
    ebit = _row(inc, ["EBIT", "Operating Income"])
    if ebit.dropna().empty:
        ebit = _row(inc, ["Pretax Income"]) + _row(inc, ["Interest Expense"]).fillna(0)
        notes.append("EBIT not reported; approximated as pre-tax income + interest.")
    net_income = _row(inc, ["Net Income Common Stockholders", "Net Income"])
    interest = _row(inc, ["Interest Expense", "Interest Expense Non Operating"]).abs()
    pretax = _row(inc, ["Pretax Income"])
    tax = _row(inc, ["Tax Provision"])

    da = _row(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion"])
    if da.dropna().empty:
        da = _row(inc, ["Reconciled Depreciation"])
    capex = _row(cf, ["Capital Expenditure"]).abs()
    # yfinance reports the *cash-flow effect*; an increase in NWC is negative.
    delta_nwc = -_row(cf, ["Change In Working Capital"])

    total_debt = _latest(_row(bs, ["Total Debt"]))
    cash = _latest(_row(bs, ["Cash Cash Equivalents And Short Term Investments",
                             "Cash And Cash Equivalents"]))
    minority = _latest(_row(bs, ["Minority Interest"]))
    book_equity = _latest(_row(bs, ["Stockholders Equity", "Common Stock Equity"]))
    shares = info.get("sharesOutstanding") or _latest(_row(bs, ["Ordinary Shares Number",
                                                               "Share Issued"]))
    if not shares:
        raise ValueError(f"Shares outstanding unavailable for {ticker}.")

    ebitda = _row(inc, ["EBITDA", "Normalized EBITDA"])
    if ebitda.dropna().empty:
        ebitda = ebit + da.reindex(ebit.index).fillna(0)
    book_hist = _row(bs, ["Stockholders Equity", "Common Stock Equity"])

    idx = revenue.index
    align = lambda s: s.reindex(idx) if len(idx) else s  # noqa: E731

    hist = tk.history(period="5y", interval="1wk", auto_adjust=True)
    try:
        idx_hist = yf.Ticker(config.MARKET_INDEX).history(period="5y", interval="1wk",
                                                          auto_adjust=True)
        index_close = idx_hist["Close"]
    except Exception:
        index_close = pd.Series(dtype=float)
        notes.append("Nifty 50 history unavailable; regression beta skipped.")

    return CompanyData(
        ticker=ticker,
        name=info.get("longName") or info.get("shortName") or ticker,
        sector=info.get("sector") or "Unknown",
        industry=info.get("industry") or "Unknown",
        currency=info.get("currency") or "INR",
        financial_currency=info.get("financialCurrency") or info.get("currency") or "INR",
        price=float(price),
        shares=float(shares),
        revenue=revenue,
        ebit=align(ebit),
        net_income=align(net_income),
        da=align(da),
        capex=align(capex),
        delta_nwc=align(delta_nwc),
        interest_expense=align(interest),
        pretax_income=align(pretax),
        tax_provision=align(tax),
        total_debt=total_debt,
        cash=cash,
        minority_interest=minority,
        book_equity=book_equity,
        beta_reported=info.get("beta"),
        price_history=hist["Close"] if "Close" in hist else pd.Series(dtype=float),
        index_history=index_close,
        dividend_per_share=float(info.get("dividendRate") or info.get("trailingAnnualDividendRate")
                                 or 0.0),
        ebitda=align(ebitda),
        book_equity_hist=book_hist,
        eps_ttm=_opt(info.get("trailingEps")),
        payout_ratio=_opt(info.get("payoutRatio")),
        analyst_target_mean=_opt(info.get("targetMeanPrice")),
        analyst_target_low=_opt(info.get("targetLowPrice")),
        analyst_target_high=_opt(info.get("targetHighPrice")),
        analyst_count=int(info.get("numberOfAnalystOpinions") or 0),
        analyst_view=info.get("recommendationKey"),
        week52_low=_opt(info.get("fiftyTwoWeekLow")),
        week52_high=_opt(info.get("fiftyTwoWeekHigh")),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Historical-derived default assumptions
# ---------------------------------------------------------------------------
def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def derive_defaults(c: CompanyData) -> dict:
    """Turn the historical record into sensible starting assumptions."""
    rev = c.revenue.dropna()
    yrs = max(len(rev) - 1, 1)
    if len(rev) >= 2 and rev.iloc[0] > 0 and rev.iloc[-1] > 0:
        cagr = (rev.iloc[-1] / rev.iloc[0]) ** (1 / yrs) - 1
    else:
        cagr = 0.08

    margins = (c.ebit / c.revenue).replace([np.inf, -np.inf], np.nan).dropna()
    cur_margin = float(margins.iloc[-1]) if len(margins) else 0.12
    avg_margin = float(margins.mean()) if len(margins) else cur_margin

    eff_tax = (c.tax_provision / c.pretax_income).replace([np.inf, -np.inf], np.nan)
    eff_tax = eff_tax[(eff_tax > 0) & (eff_tax < 0.6)].dropna()
    tax = float(eff_tax.mean()) if len(eff_tax) else config.MARGINAL_TAX_RATE

    invested_capital = c.book_equity + c.total_debt - c.cash
    if invested_capital > 0 and len(rev):
        s2c = rev.iloc[-1] / invested_capital
    else:
        s2c = 1.5

    nopat = _latest(c.ebit) * (1 - tax)
    roic = nopat / invested_capital if invested_capital > 0 else 0.15

    return {
        "growth_high": _clip(cagr, -0.05, 0.30),
        "op_margin_current": _clip(cur_margin, -0.5, 0.8),
        "op_margin_target": _clip(0.5 * cur_margin + 0.5 * avg_margin, 0.0, 0.8),
        "tax_rate": _clip(tax, 0.15, 0.35),
        "sales_to_capital": _clip(s2c, 0.4, 5.0),
        "current_roic": roic,
        "hist_revenue_cagr": cagr,
    }
