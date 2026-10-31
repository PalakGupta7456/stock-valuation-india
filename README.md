# Intrinsic — what is an Indian stock actually worth?

[![CI](https://github.com/OWNER/stock-valuation-india/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/stock-valuation-india/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-0E5160)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-D98E1F)
![License: MIT](https://img.shields.io/badge/license-MIT-5D6B70)

Type an NSE or BSE ticker. **Intrinsic** values the company **twelve different ways**, blends them into one fair value with sector-aware weights, compares it with the market price and gives a **Buy / Hold / Sell** call, with a confidence score that is honest about how far the numbers can be trusted.

![Overview](docs/screenshots/overview.png)

---

## What it does

| | Method | What it answers |
|---|---|---|
| **Intrinsic** | DCF: free cash flow to the firm (FCFF) | What are the operating cash flows worth at the company's WACC? |
| | DCF: free cash flow to equity (FCFE) | What is left for shareholders after debt, discounted at the cost of equity? |
| | Residual income (Edwards-Bell-Ohlson) | Book value plus the present value of returns above the cost of equity. The main model for banks and NBFCs. |
| | Dividend discount (two-stage) | What are the future dividends worth, with payout rising to a sustainable level? |
| | Earnings power value (Greenwald) | What is today's business worth with **zero** growth? The gap to the DCF is what you pay for growth. |
| **Relative** | Peer-median P/E, EV/EBITDA, P/B, EV/Sales | What would the stock trade at on its peers' multiples? |
| | Own 5-year P/E | What would it trade at on its own historical multiple? |
| **Rule of thumb** | Graham number, Graham formula | Benjamin Graham's defensive-investor ceilings (shown for reference, weight 0 by default). |

On top of the methods:

- **Blended fair value and range.** A weighted average of every method that applies, plus the weighted 25th–75th percentile across methods.
- **Sector awareness.** For banks, NBFCs and insurers, cash-flow DCFs and EV multiples are switched off and the blend leans on residual income, P/B, P/E and dividends.
- **Consensus.** How many methods put the value above the market price.
- **Confidence score (0–100).** It starts at 100 and drops for data problems (short history, negative earnings, stale or missing statements, currency mismatch) and for disagreement between methods, especially when intrinsic and market-relative methods are far apart.
- **Stress tests.** 5,000-run Monte Carlo, WACC × terminal-growth heatmap, tornado chart of the drivers, bear/base/bull scenarios and a **reverse DCF** (what growth is the price assuming?).
- **Indian cost of capital.** 10-year G-sec risk-free rate, India equity risk premium, beta regressed on 5 years of weekly returns against the Nifty 50 (Blume-adjusted), synthetic credit rating from interest coverage, 25.17% marginal tax under Sec 115BAA.
- **Every assumption is editable** in the sidebar: growth, margins, terminal growth, cost of capital, peer list and the weight of each method.
- **Excel export** of the full model (summary, all methods, FCFF/FCFE, residual income, DDM, peers, scenarios, sensitivity, data checks).
- **Works offline** with a synthetic `DEMO` company, so it can be tried, tested and demoed without internet.

## Screenshots

| All twelve methods on one chart | Peer comparison |
|---|---|
| ![All methods](docs/screenshots/methods.png) | ![Relative](docs/screenshots/relative.png) |

| Residual income, dividends and no-growth value | Stress tests |
|---|---|
| ![Residual income](docs/screenshots/residual.png) | ![Stress tests](docs/screenshots/stress.png) |

## Quick start

```bash
git clone https://github.com/OWNER/stock-valuation-india.git
cd stock-valuation-india
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, type a ticker (`TCS`, `HDFCBANK`, `RELIANCE`, `ASIANPAINT`, a BSE code like `500325` with the BSE toggle) or `DEMO`.

### Use it from Python

```python
from valuation import fetch_company, value_company, Assumptions
from valuation.peers import suggest_peers, fetch_peer_multiples

c = fetch_company("TCS.NS")
peers = fetch_peer_multiples(suggest_peers(c.ticker, c.industry))
rep = value_company(c, Assumptions(growth_high=0.09), peers)

print(rep.verdict.rating, rep.verdict.fair_value, rep.verdict.upside)
print(rep.method_table[["Method", "Value", "Upside", "Weight"]])
```

### Deploy for free

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), choose the repo, branch `main`, file `app.py`.
3. Add the public URL to the top of this README.

## How the verdict is built

```
fair value  = Σ weight_i × value_i     over methods that produce a positive value
upside      = fair value / price − 1
rating      = Strong Buy ≥ +30% | Buy ≥ +15% | Hold ≥ −10% | Sell ≥ −25% | Strong Sell
```

Default weights (non-financial): FCFF 25, residual income 15, peer P/E 12, peer EV/EBITDA 12, FCFE 10, earnings power 10, own P/E 5, DDM 5, peer P/B 3, peer EV/Sales 3. Financials: residual income 35, P/E 20, P/B 20, DDM 15, own P/E 10. Weights are renormalised over the methods that apply. A Strong Buy/Sell with low confidence is pulled one notch toward Hold; a Buy/Sell with very low confidence becomes Hold.

## Project structure

```
app.py                  Streamlit dashboard
valuation/
  config.py             Market defaults (G-sec, ERP, tax, rating bands, method weights)
  data.py               Yahoo Finance download and cleaning
  peers.py              Curated Indian peer groups and peer multiples
  wacc.py               Beta regression, synthetic rating, CAPM, WACC
  dcf.py                Vectorised FCFF / FCFE engine
  methods.py            Residual income, DDM, EPV, Graham, peer and historical multiples
  analysis.py           Sensitivity, tornado, scenarios, Monte Carlo, reverse DCF
  quality.py            Data-quality checks that reduce confidence
  recommend.py          Blend, consensus, confidence and rating
  export.py             Excel workbook export
  demo.py               Synthetic company for offline use and tests
tests/                  39 unit tests with closed-form checks
```

## Limitations

- **Data comes from Yahoo Finance.** Indian coverage is good for large and mid caps but thinner for small caps and financials, and it only gives four years of annual statements. Always check key numbers against the annual report.
- **Peers are curated by Yahoo industry label.** Unusual businesses may get poor peers; edit the list in the sidebar.
- **Market defaults** (G-sec yield, ERP, AAA yield) are set in `valuation/config.py` as of late 2026 and should be refreshed when rates move.
- **This is an educational tool, not investment advice.** It is not a SEBI-registered research analyst or adviser.

## Roadmap

- [ ] Screener.in / NSE filings as a second data source for 10-year history and better coverage of banks
- [ ] Live G-sec yield and India ERP instead of config values
- [ ] Sum-of-the-parts valuation for conglomerates (Reliance, L&T, Bajaj Finserv)
- [ ] Bank-specific checks (NIM, GNPA, CASA, capital adequacy)
- [ ] Watchlist mode: value 20 tickers at once and rank by upside × confidence
- [ ] Backtest: how did past "Buy" calls perform over 12 months?
- [ ] PDF one-pager export

## License

MIT, see [LICENSE](LICENSE).
