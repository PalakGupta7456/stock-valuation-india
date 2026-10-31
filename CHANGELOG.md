# Changelog

## v1.0.0 — 1 Nov 2026

First public release of **Intrinsic**, a multi-method valuation dashboard for NSE and BSE stocks.

### What's in it
- **12 valuation methods**: FCFF and FCFE DCF, residual income, dividend discount, earnings power, peer P/E, EV/EBITDA, P/B and EV/Sales, own 5-year P/E, Graham number and formula
- **Any listed company**: search 5,200+ NSE/BSE companies by name, symbol or BSE scrip code, with live Yahoo Finance data and shareable `?ticker=` links
- **One verdict**: weighted fair value and range, method consensus, confidence score (0–95) and a Buy / Hold / Sell call that turns to Hold when confidence is low
- **Sector-aware**: banks, NBFCs and insurers switch to residual income, P/B, P/E and dividends; diversified groups are flagged
- **Indian cost of capital**: G-sec, India ERP, beta vs Nifty 50, synthetic credit rating, Sec 115BAA tax
- **Stress tests**: Monte Carlo, WACC × growth heatmap, tornado, scenarios, reverse DCF
- **Excel export** of the full model

### Quality
- 45 unit tests with closed-form checks; lint and app smoke test on Python 3.11–3.13
- Weekly live check that values 32 real stocks ([latest report](https://github.com/PalakGupta7456/stock-valuation-india/blob/main/reports/live_check.md))

Educational project, not investment advice.
