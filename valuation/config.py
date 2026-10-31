"""Market-level defaults for valuing Indian equities.

Every value here is only a *starting point*: the dashboard lets the user
override all of them. Update these when the market moves materially.
"""

# --- Rates (decimal) ---------------------------------------------------------
# India 10Y G-sec traded roughly 7.0-7.4% in Oct 2026 (3-year high of ~7.37% on 12 Oct 2026).
RISK_FREE_RATE = 0.072
# Total equity risk premium for India (mature-market ERP + country risk premium).
# Damodaran-style estimates for 2026 cluster around 6.9-7.3%.
EQUITY_RISK_PREMIUM = 0.070
# Corporate tax rate under the new regime (Sec 115BAA incl. surcharge & cess).
MARGINAL_TAX_RATE = 0.2517
# Long-run nominal growth cap. Must stay below the risk-free rate.
DEFAULT_TERMINAL_GROWTH = 0.050

# --- Benchmarks --------------------------------------------------------------
MARKET_INDEX = "^NSEI"          # Nifty 50, used for regression beta
DEFAULT_SUFFIX = ".NS"          # NSE; ".BO" for BSE

# --- Model structure ---------------------------------------------------------
PROJECTION_YEARS = 10
HIGH_GROWTH_YEARS = 5
MARGIN_CONVERGENCE_YEARS = 5

# --- Recommendation bands (upside = fair value / price - 1) ------------------
RATING_BANDS = [
    (0.30, "Strong Buy"),
    (0.15, "Buy"),
    (-0.10, "Hold"),
    (-0.25, "Sell"),
    (-1.00, "Strong Sell"),
]

# Sectors where an operating-cash-flow DCF is not a reliable primary method.
FINANCIAL_SECTORS = {"Financial Services", "Financial", "Banks", "Insurance"}

# Synthetic credit rating from interest coverage (large non-financial firms).
# (min coverage, rating, default spread) — approximates Damodaran's table.
SYNTHETIC_RATINGS = [
    (8.50, "AAA", 0.0059),
    (6.50, "AA", 0.0070),
    (5.50, "A+", 0.0092),
    (4.25, "A", 0.0107),
    (3.00, "A-", 0.0121),
    (2.50, "BBB", 0.0147),
    (2.25, "BB+", 0.0183),
    (2.00, "BB", 0.0220),
    (1.75, "B+", 0.0275),
    (1.50, "B", 0.0342),
    (1.25, "B-", 0.0524),
    (0.80, "CCC", 0.0880),
    (0.65, "CC", 0.1183),
    (0.20, "C", 0.1600),
    (-1e9, "D", 0.2000),
]

CRORE = 1e7

# --- Graham formula ------------------------------------------------------------
# Graham's revised formula scales by (4.4 / current AAA corporate bond yield).
# Indian AAA PSU / corporate bonds yielded roughly 7.3-7.8% in late 2026.
AAA_CORPORATE_YIELD = 0.076

# --- Method weights for the blended fair value (renormalised over valid models) --
# Users can change every weight in the sidebar. Graham is shown for reference.
METHOD_WEIGHTS = {
    "DCF (FCFF)": 25, "DCF (FCFE)": 10, "Residual income": 15,
    "Peer P/E": 12, "Peer EV/EBITDA": 12, "Peer P/B": 3, "Peer EV/Sales": 3,
    "Own 5y P/E": 5, "Dividend discount": 5, "Earnings power (no growth)": 10,
    "Graham number": 0, "Graham formula": 0,
}
# Banks, NBFCs and insurers: debt is raw material, so enterprise-value and
# cash-flow models are switched off and equity / book-based models lead.
FINANCIAL_METHOD_WEIGHTS = {
    "DCF (FCFF)": 0, "DCF (FCFE)": 0, "Residual income": 35,
    "Peer P/E": 20, "Peer EV/EBITDA": 0, "Peer P/B": 20, "Peer EV/Sales": 0,
    "Own 5y P/E": 10, "Dividend discount": 15, "Earnings power (no growth)": 0,
    "Graham number": 0, "Graham formula": 0,
}
METHOD_FAMILY = {
    "DCF (FCFF)": "Intrinsic", "DCF (FCFE)": "Intrinsic", "Residual income": "Intrinsic",
    "Dividend discount": "Intrinsic", "Earnings power (no growth)": "Intrinsic",
    "Peer P/E": "Relative", "Peer EV/EBITDA": "Relative", "Peer P/B": "Relative",
    "Peer EV/Sales": "Relative", "Own 5y P/E": "Relative",
    "Graham number": "Rule of thumb", "Graham formula": "Rule of thumb",
}
# Multiples outside these bounds are treated as noise when taking peer medians.
MULTIPLE_BOUNDS = {"pe": (0.5, 150), "pb": (0.1, 40), "ev_ebitda": (0.5, 80), "ev_sales": (0.05, 40)}
