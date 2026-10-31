# Methodology

This page documents every model in Intrinsic, the assumptions behind it, and how the
results are combined. Code references point to the `valuation/` package.

## 1. Data

| Item | Source | Notes |
|---|---|---|
| Price, shares, EPS, book value/share, dividend, analyst targets | Yahoo Finance quote summary | Live, cached 1 hour in the app |
| Income statement, balance sheet, cash flow | Yahoo Finance annual statements | Usually 4 fiscal years for Indian companies |
| Weekly prices (5 years) for the stock and Nifty 50 (`^NSEI`) | Yahoo Finance | For regression beta and historical P/E |
| Peer multiples | Yahoo Finance, per peer | Curated map keyed on Yahoo's industry label, topped up with large same-sector companies when fewer than 4 (`peers.py`) |
| Company list for search | NSE `EQUITY_L.csv`, BSE active scrips | Refreshed weekly by GitHub Actions |

**Fallbacks** (`data.py`): numeric BSE scrip codes are translated to Yahoo's `SYMBOL.BO` form; statements
reported in another currency (e.g. Infosys, which files in USD) are converted at the spot rate; Yahoo's
trailing EPS is replaced by net income ÷ shares when the two disagree by more than 3×; a symbol with no price on NSE is retried on BSE (and vice versa); if annual
statements are missing, the trailing-twelve-month snapshot (revenue, operating margin, net income, EBITDA,
debt, cash, book value) is used and flagged; shares outstanding fall back to implied shares, then market
cap ÷ price. Every fallback adds a note and, where it matters, a confidence penalty.

## 2. Cost of capital (`wacc.py`)

```
Cost of equity  Ke   = Rf + β × ERP
Cost of debt    Kd   = Rf + default spread(synthetic rating from EBIT / interest)
WACC                 = E/(D+E) × Ke + D/(D+E) × Kd × (1 − t_marginal)
```

- **Rf** = India 10-year G-sec yield (7.2% default, editable).
- **ERP** = India total equity risk premium (7.0% default): mature-market premium plus country risk.
- **β** = OLS slope of weekly stock returns on Nifty 50 returns over up to 5 years, Blume-adjusted
  (`0.67 × raw + 0.33`). Falls back to Yahoo's beta if R² < 0.05, then to 1.0.
- **Synthetic rating** maps interest coverage to a default spread (Damodaran's large-firm table).
- **Marginal tax** = 25.17% (Sec 115BAA incl. surcharge and cess).

## 3. Intrinsic models

### 3.1 DCF: free cash flow to the firm (`dcf.py`)

Ten-year, two-stage projection in the "story to numbers" style:

- Revenue grows at `g_high` for years 1–5, then fades linearly to terminal growth `g` by year 10.
- EBIT margin moves linearly from today's margin to a target margin over 5 years.
- Reinvestment = Δrevenue ÷ sales-to-capital ratio.
- Tax rate drifts from the effective to the marginal rate during the fade.

```
FCFF_t = EBIT_t × (1 − tax_t) − Reinvestment_t
TV     = NOPAT_{11} × (1 − g / ROIC_terminal) / (WACC − g)        # growth-consistent terminal value
Equity = Σ PV(FCFF) + PV(TV) − debt + cash − minority interest
```

Mid-year discounting by default. Terminal ROIC defaults to halfway between today's ROIC and WACC,
bounded to [WACC, WACC + 8pp], so excess returns fade but do not vanish.

### 3.2 DCF: free cash flow to equity

Same operating projection; debt grows by `debt ratio × reinvestment` so leverage stays constant;
interest is charged on opening debt.

```
FCFE_t = Net income_t − (1 − debt ratio) × Reinvestment_t      discounted at Ke
```

### 3.3 Residual income (`methods.residual_income`)

```
Value = BV_0 + Σ_{t=1..10} (ROE_t − Ke) × BV_{t−1} / (1 + Ke)^t + TV / (1 + Ke)^10
BV_t  = BV_{t−1} + NI_t × (1 − payout)
TV    = (ROE_T − Ke) × BV_10 / (Ke − g)
```

ROE starts at the 3-year average and fades linearly to `ROE_T = min(½(ROE_0 + Ke), Ke + 6pp)`.
It is the lead model for banks, NBFCs and insurers because it needs no free-cash-flow figure, and
it is anchored on book value, which for lenders is close to marked-to-market.

### 3.4 Dividend discount (`methods.dividend_discount`)

EPS grows with the revenue growth assumption for 5 years, then fades to `g`. Payout moves from today's
level to the sustainable payout `1 − g / ROE_T`, so the terminal value is consistent with growth:

```
Value = Σ_{t=1..10} EPS_t × payout_t / (1 + Ke)^t + EPS_10 (1 + g) (1 − g/ROE_T) / (Ke − g) / (1 + Ke)^10
```

Not applied to companies that pay no dividend.

### 3.5 Earnings power value (`methods.earnings_power`)

Greenwald's no-growth value: average historical EBIT margin × current revenue × (1 − tax), capitalised
at WACC, then bridged to equity. Comparing it with the market price shows how much of the price is a
bet on growth.

## 4. Relative models (`methods.peer_multiples`, `methods.own_history_pe`)

| Method | Value per share |
|---|---|
| Peer P/E | median peer P/E × trailing EPS |
| Peer P/B | median peer P/B × book value per share |
| Peer EV/EBITDA | (median EV/EBITDA × EBITDA − debt + cash − minority) ÷ shares |
| Peer EV/Sales | (median EV/Sales × revenue − debt + cash − minority) ÷ shares |
| Own 5y P/E | median of the stock's P/E at each fiscal year-end × trailing EPS |

Peer multiples outside sane bounds (e.g. P/E ≤ 0.5 or > 150) are dropped before taking medians; at
least two valid peers are required. The low–high range uses the peer 25th–75th percentiles.
EV multiples are switched off for financial companies.

## 5. Rules of thumb

- **Graham number** = √(22.5 × EPS × BVPS), i.e. P/E 15 × P/B 1.5.
- **Graham formula** = EPS × (8.5 + 2g) × 4.4 / Y, with g in % (capped at 15) and Y the Indian AAA
  corporate bond yield (7.6% default).

Both are shown for context with zero weight by default; they ignore the cost of capital and are harsh
on high-ROE compounders.

## 6. Blending and the verdict (`recommend.py`)

1. Each method that produces a positive value gets its weight (editable in the app); weights are
   renormalised over the methods that apply. Defaults split roughly **50/50 between intrinsic and relative**
   models: FCFF 20, residual income 15, FCFE 10, DDM 5 · peer P/E 15, peer EV/EBITDA 15, own P/E 12,
   peer P/B 5, peer EV/Sales 3. Earnings power and the Graham rules are shown as reference points
   (weight 0). For lenders: residual income 30, P/E 20, P/B 20, DDM 15, own P/E 15.
2. **Fair value** = weighted mean. **Range** = weighted 25th–75th percentile across methods.
3. **Upside** = fair value ÷ price − 1, mapped to a rating:

| Upside | ≥ +30% | +15% to +30% | −10% to +15% | −25% to −10% | < −25% |
|---|---|---|---|---|---|
| Rating | Strong Buy | Buy | Hold | Sell | Strong Sell |

4. **Confidence** starts at 100 and loses points for data-quality flags (financial company, short or stale
   history, negative EBIT or book value, volatile revenue, cash burn, currency mismatch, aggressive
   growth) and for model disagreement: fewer than 3 weighted methods (−15), cross-method dispersion
   above 35% / 60% (−12 / −25), intrinsic and relative blends more than 1.5× / 2× apart (−10 / −20),
   the blend's direction contradicted by most methods (−10), terminal value above 80% of EV (−8), and a
   coin-flip Monte Carlo (−5).
5. **A call is only as strong as the evidence.** With low confidence (below 45), Strong Buy/Sell drop one
   notch and Buy/Sell become Hold. Consensus ("6 of 9 methods above price") counts only methods that carry
   weight in the blend.

**Why intrinsic and relative values often disagree for Indian stocks.** With a 10-year G-sec near 7% and
an equity risk premium near 7%, the cost of equity for an average-risk Indian company is about 14%. At that
discount rate, a company growing 10–12% is worth roughly 12–18× earnings, while Indian large caps have
traded at 20–40×. Cash-flow models therefore tend to sit below the market and peer multiples near it. The
app shows both sub-blends instead of hiding the gap, and lowers confidence when they are far apart.

## 7. Stress tests (`analysis.py`)

- **Monte Carlo** (5,000 runs, vectorised): revenue growth ~ N(g_high, 0.3|g_high| + 2pp), target
  margin ~ N(m, 0.15m + 1pp), WACC ~ N(WACC, 1pp), terminal growth ~ U(g ± 1pp) capped below WACC,
  sales-to-capital × lognormal(0, 0.2).
- **Sensitivity grid**: WACC × terminal growth, 7 × 7.
- **Tornado**: value swing when each driver is flexed by a plausible amount.
- **Scenarios**: bear / base / bull shifts to growth, margin, WACC and terminal growth.
- **Reverse DCF**: bisection on `g_high` until the FCFF value equals the market price.

## 8. Known limitations

- Yahoo Finance gives only ~4 years of statements and is thinner for small caps and lenders.
- Peer groups are curated per industry; unusual companies need a hand-picked peer list.
- No sum-of-the-parts treatment for conglomerates or holding companies.
- The output is a model, not advice. Verify inputs against the annual report.
