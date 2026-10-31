"""Intrinsic: multi-method valuation dashboard for NSE/BSE stocks.

Run:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from valuation import Assumptions, config, fetch_company, normalise_ticker, value_company
from valuation.data import derive_defaults, search_tickers
from valuation.export import to_excel
from valuation.peers import demo_peers, fetch_peer_multiples, suggest_peers
from valuation.universe import load_universe
from valuation.wacc import estimate_beta

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK, TEAL, MIST, PAPER = "#18262B", "#0E5160", "#C9D3CD", "#F2F4F0"
INDIGO = "#4F5BA6"           # relative (market-based) methods
SAFFRON = "#D98E1F"          # reserved for the market price, everywhere
GAIN, LOSS, HOLD = "#2F7D57", "#B23B3B", "#5D6B70"
FAMILY_COLOR = {"Intrinsic": TEAL, "Relative": INDIGO, "Rule of thumb": HOLD}
RATING_COLOR = {"Strong Buy": GAIN, "Buy": GAIN, "Hold": HOLD, "Sell": LOSS, "Strong Sell": LOSS}

st.set_page_config(page_title="Intrinsic — Indian stock valuation", page_icon="₹", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Instrument+Serif&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, label {{
  font-family: 'Instrument Sans', system-ui, sans-serif;
}}
.block-container {{ padding-top: 3.2rem; max-width: 1320px; }}
h1, h2, h3 {{ font-family: 'Instrument Serif', Georgia, serif !important; font-weight: 400 !important;
  letter-spacing: -0.01em; color: {INK}; }}
.brand {{ font-family: 'Instrument Serif', serif; font-size: 1.9rem; color: {TEAL}; line-height: 1; }}
.brand small {{ font-family: 'Instrument Sans'; font-size: .8rem; color: {HOLD}; display:block; margin-top:.35rem; }}

.stMarkdown p.company {{ font-family:'Instrument Serif', serif !important; font-size: 2.6rem !important;
  line-height:1.05; color:{INK}; margin:0; }}
.meta {{ color:{HOLD}; font-size:.92rem; margin-top:.35rem; }}

.verdict {{ border-left: 6px solid var(--vc); background: #fff; padding: 1.4rem 1.6rem; border-radius: 4px;
  margin: 1.2rem 0 .6rem; }}
.verdict .rating {{ color: var(--vc); font-weight: 600; font-size: 1.05rem; }}
.verdict .line {{ font-family:'Instrument Serif', serif; font-size: 2.05rem; line-height: 1.2; color:{INK};
  margin:.3rem 0 .5rem; max-width: 52ch; }}
.verdict .sub {{ color:{HOLD}; font-size:.95rem; max-width: 85ch; }}
.price {{ color:{SAFFRON}; font-weight:600; }}
.pill {{ display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.78rem; font-weight:600;
  color:#fff; background: var(--pc); margin-right:.3rem; vertical-align: middle; }}

.kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 0; margin: .8rem 0 1rem;
  border-top:1px solid {MIST}; border-bottom:1px solid {MIST}; }}
.kpi {{ padding: .9rem 1rem; border-right:1px solid {MIST}; }}
.kpi:last-child {{ border-right: none; }}
.kpi .v {{ font-family:'Instrument Serif', serif; font-size: 1.75rem; color:{INK}; line-height:1.1; }}
.kpi .k {{ color:{HOLD}; font-size:.82rem; }}
.kpi .n {{ color:{HOLD}; font-size:.78rem; margin-top:.15rem; }}

.mcard {{ background:#fff; border-radius:4px; padding: .9rem 1rem; border-top: 3px solid var(--mc); height: 100%; }}
.mcard .t {{ font-size:.82rem; color:{HOLD}; }}
.mcard .v {{ font-family:'Instrument Serif', serif; font-size:1.6rem; color:{INK}; }}
.mcard .u {{ font-size:.85rem; font-weight:600; }}
.mcard .d {{ font-size:.82rem; color:{HOLD}; margin-top:.3rem; }}

.flag {{ padding:.75rem 1rem; border-radius:4px; margin-bottom:.5rem; background:#fff; border-left:4px solid var(--fc); }}
.flag b {{ color: {INK}; }}
.flag div {{ color:{HOLD}; font-size:.92rem; }}
.note {{ color:{HOLD}; font-size:.85rem; }}
.stTabs [data-baseweb="tab"] {{ font-size: .95rem; }}
@media (max-width: 640px) {{ .stMarkdown p.company {{ font-size: 2rem !important; }} .verdict .line {{ font-size: 1.5rem; }} }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fin(x) -> bool:
    return x is not None and np.isfinite(x)


def inr(x: float | None, dp: int = 0) -> str:
    return f"₹{x:,.{dp}f}" if fin(x) else "—"


def cr(x: float) -> str:
    return f"₹{x / config.CRORE:,.0f} Cr" if fin(x) else "—"


def pct(x: float | None, dp: int = 1, sign: bool = False) -> str:
    if not fin(x):
        return "—"
    return f"{x:+.{dp}%}" if sign else f"{x:.{dp}%}"


def mult(x: float | None) -> str:
    return f"{x:.1f}×" if fin(x) else "—"


def updown_color(u: float) -> str:
    return GAIN if fin(u) and u > 0 else LOSS if fin(u) else HOLD


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Instrument Sans, sans-serif", color=INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_family="Instrument Sans"),
    )
    fig.update_xaxes(gridcolor=MIST, zerolinecolor=MIST)
    fig.update_yaxes(gridcolor=MIST, zerolinecolor=MIST)
    return fig


def price_line(fig: go.Figure, price: float, horizontal: bool = False, pos: str = "top left") -> None:
    kw = dict(line_color=SAFFRON, line_width=2.5, annotation_text=f"Market price {inr(price)}",
              annotation_font_color=SAFFRON)
    if horizontal:
        fig.add_hline(y=price, annotation_position=pos, **kw)
    else:
        fig.add_vline(x=price, annotation_position=pos, **kw)


def money_table(df: pd.DataFrame, pct_cols: list[str], ratio_cols: list[str],
                per_share: bool = False) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "Year":
            out[col] = out[col].map(lambda y: f"Y{y}")
        elif col in pct_cols:
            out[col] = out[col].map(pct)
        elif col in ratio_cols:
            out[col] = out[col].map(lambda x: f"{x:.3f}")
        elif per_share:
            out[col] = out[col].map(lambda x: f"{x:,.1f}")
        else:
            out[col] = out[col].map(lambda x: f"{x / config.CRORE:,.0f}")
    return out.set_index("Year").T


# ---------------------------------------------------------------------------
# Data loading (cached for an hour)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load(ticker: str):
    return fetch_company(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def load_peers(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_peer_multiples(list(tickers))


@st.cache_data(ttl=86400, show_spinner=False)
def universe() -> pd.DataFrame:
    return load_universe()


@st.cache_data(ttl=3600, show_spinner=False)
def name_search(q: str) -> list[dict]:
    return search_tickers(q)


def resolve(raw: str, suffix: str) -> tuple[str | None, list[dict]]:
    """Turn whatever the user typed (symbol, BSE code or company name) into a Yahoo ticker."""
    text = raw.strip()
    if not text:
        return None, []
    compact = text.upper().replace(" ", "")
    if compact == "DEMO" or "." in compact or compact.isdigit():
        return normalise_ticker(compact, ".BO" if compact.isdigit() else suffix), []
    u = universe()
    if len(u):
        exact = u[u.symbol.str.replace(r"\.(NS|BO)$", "", regex=True) == compact]
        if len(exact):
            return exact.symbol.iloc[0], []
        by_name = u[u.name.str.lower().str.contains(text.lower(), regex=False)]
        if len(by_name):
            hits = [{"symbol": r.symbol, "name": r.name} for r in by_name.head(8).itertuples()]
            return hits[0]["symbol"], hits
    hits = name_search(text)
    for h in hits:                                   # typed an exact symbol Yahoo knows
        if h["symbol"].split(".")[0] == compact:
            return h["symbol"], []
    if hits:
        return hits[0]["symbol"], hits
    return normalise_ticker(compact, suffix), []


def set_ticker(t: str) -> None:
    st.session_state.ticker = t
    st.query_params["ticker"] = t.removesuffix(".NS")


# Shareable links: /?ticker=TCS or /?ticker=500325.BO
if "qp_done" not in st.session_state:
    st.session_state.qp_done = True
    qp = st.query_params.get("ticker")
    if qp:
        st.session_state.ticker = normalise_ticker(qp)

# ---------------------------------------------------------------------------
# Sidebar: ticker
# ---------------------------------------------------------------------------
U = universe()
with st.sidebar:
    st.markdown('<div class="brand">Intrinsic<small>12 valuation methods for NSE &amp; BSE stocks</small></div>',
                unsafe_allow_html=True)
    st.write("")
    with st.form("ticker_form", border=False):
        picked = None
        if len(U):
            picked = st.selectbox(f"Search {len(U):,} listed companies", U.label.tolist(), index=None,
                                  placeholder="Start typing a name or symbol…")
        raw = st.text_input("…or type any ticker, BSE code or company name" if len(U) else
                            "Ticker, BSE code or company name",
                            placeholder="e.g. TCS, 500325, asian paints")
        exch = st.radio("Default exchange for typed symbols", ["NSE", "BSE"], horizontal=True)
        go_btn = st.form_submit_button("Value this stock", type="primary", width="stretch")
    if go_btn:
        if raw.strip():
            with st.spinner("Finding the company…"):
                t, hits = resolve(raw, ".NS" if exch == "NSE" else ".BO")
            st.session_state.matches = hits
            if t:
                set_ticker(t)
        elif picked:
            set_ticker(U.loc[U.label == picked, "symbol"].iloc[0])
            st.session_state.matches = []
    hits = st.session_state.get("matches") or []
    if len(hits) > 1:
        st.caption("Other matches:")
        for h in hits[1:6]:
            if st.button(f"{h['name'][:34]} · {h['symbol']}", key=f"m_{h['symbol']}", width="stretch"):
                set_ticker(h["symbol"])
                st.session_state.matches = []
                st.rerun()
    st.caption("Works for any NSE or BSE listing on Yahoo Finance. Type DEMO for an offline sample.")

ticker = st.session_state.get("ticker")

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if not ticker:
    st.markdown('<p class="company">What is an Indian stock actually worth?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="meta" style="max-width:72ch">Enter an NSE or BSE ticker. Intrinsic values the company '
        'twelve ways (discounted cash flow, residual income, dividend discount, earnings power, peer '
        'multiples, its own history and Graham\'s rules), blends them with sector-aware weights, '
        'stress-tests the answer and tells you whether the market price looks cheap or expensive.</p>',
        unsafe_allow_html=True)
    st.write("")
    cols = st.columns(6)
    for col, t in zip(cols, ["TCS", "HDFCBANK", "RELIANCE", "ASIANPAINT", "ITC", "DEMO"], strict=False):
        if col.button(t, width="stretch"):
            set_ticker(normalise_ticker(t))
            st.rerun()
    st.write("")
    a, b, c_ = st.columns(3)
    a.markdown(f'<div class="mcard" style="--mc:{TEAL}"><div class="t">Intrinsic</div>'
               '<div class="d">What the cash flows, book value and dividends are worth at India\'s cost '
               'of capital: FCFF and FCFE DCF, residual income, DDM, earnings power.</div></div>',
               unsafe_allow_html=True)
    b.markdown(f'<div class="mcard" style="--mc:{INDIGO}"><div class="t">Relative</div>'
               '<div class="d">What the market pays for similar businesses: peer-median P/E, EV/EBITDA, '
               'P/B and EV/Sales, plus the stock\'s own historical P/E.</div></div>',
               unsafe_allow_html=True)
    c_.markdown(f'<div class="mcard" style="--mc:{SAFFRON}"><div class="t">Verdict</div>'
                '<div class="d">A weighted fair value, a range, how many methods agree, a confidence '
                'score that drops when data is thin or models disagree, and a Buy / Hold / Sell call.'
                '</div></div>', unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# Load company
# ---------------------------------------------------------------------------
try:
    with st.spinner(f"Fetching {ticker} from Yahoo Finance…"):
        company = load(ticker)
except Exception as exc:  # noqa: BLE001
    st.error(f"Couldn't value {ticker}. {exc}")
    st.info("Try the company name instead of the symbol (e.g. “asian paints”), a BSE code such as 500325, "
            "or DEMO to see the app working offline. Yahoo Finance occasionally rate-limits; waiting a minute "
            "and retrying usually helps.")
    st.stop()

d = derive_defaults(company)
beta_auto = estimate_beta(company).adjusted
K = ticker  # widget keys are per-ticker so defaults reset when the ticker changes
default_weights = config.FINANCIAL_METHOD_WEIGHTS if company.is_financial else config.METHOD_WEIGHTS
suffix = ".BO" if ticker.endswith(".BO") else ".NS"

with st.sidebar:
    st.divider()
    st.markdown("**Assumptions**")
    st.caption("Pre-filled from the company's history and current Indian market data. Change any of them.")
    with st.expander("Cost of capital", expanded=False):
        rf = st.number_input("Risk-free rate (10Y G-sec), %", 3.0, 12.0, config.RISK_FREE_RATE * 100, 0.05,
                             key=f"{K}rf") / 100
        erp = st.number_input("Equity risk premium, %", 3.0, 12.0, config.EQUITY_RISK_PREMIUM * 100, 0.1,
                              key=f"{K}erp") / 100
        beta = st.number_input("Beta", 0.3, 2.5, round(float(beta_auto), 2), 0.05, key=f"{K}beta")
    with st.expander("Growth and profitability", expanded=True):
        gh = st.slider("Revenue growth, years 1–5 (%)", -10.0, 40.0, round(d["growth_high"] * 100, 1), 0.5,
                       key=f"{K}gh") / 100
        mt = st.slider("Target EBIT margin (%)", 0.0, 60.0,
                       round(max(d["op_margin_target"], 0) * 100, 1), 0.5, key=f"{K}mt") / 100
        tg = st.slider("Terminal growth (%)", 2.0, 7.0, config.DEFAULT_TERMINAL_GROWTH * 100, 0.25,
                       key=f"{K}tg") / 100
        s2c = st.slider("Sales-to-capital ratio", 0.3, 5.0, round(d["sales_to_capital"], 2), 0.05,
                        key=f"{K}s2c", help="₹ of extra revenue generated per ₹1 reinvested.")
        tax = st.slider("Effective tax rate (%)", 15.0, 35.0, round(d["tax_rate"] * 100, 1), 0.5,
                        key=f"{K}tax") / 100
        roic_auto = st.checkbox("Auto terminal ROIC / ROE", True, key=f"{K}roica",
                                help="Excess returns fade halfway toward the cost of capital: some moat survives.")
        roic = None if roic_auto else st.slider("Terminal ROIC (%)", 5.0, 40.0, 15.0, 0.5,
                                                key=f"{K}roic") / 100
        roe_t = None if roic_auto else st.slider("Terminal ROE, residual income (%)", 5.0, 40.0, 16.0, 0.5,
                                                 key=f"{K}roe") / 100
    with st.expander("Peer group", expanded=False):
        if company.is_demo:
            st.caption("Demo uses five synthetic peers.")
            peer_text = ""
        else:
            sugg = [p.split(".")[0] for p in suggest_peers(ticker, company.industry)]
            peer_text = st.text_area("Peer tickers (comma-separated)", ", ".join(sugg), key=f"{K}peers",
                                     help=f"Suggested from Yahoo industry: {company.industry}.")
            if not sugg:
                st.caption(f"No curated peers for “{company.industry}”. Add 3–6 comparable companies.")
    with st.expander("Method weights", expanded=False):
        st.caption("Relative weight of each method in the blended fair value. Models that cannot run "
                   "for this company are dropped and the rest re-weighted."
                   + (" Financial-company defaults applied." if company.is_financial else ""))
        weights = {m: st.slider(m, 0, 50, int(w0), 1, key=f"{K}w_{m}") for m, w0 in default_weights.items()}
    with st.expander("Model settings"):
        mos = st.slider("Margin of safety (%)", 0, 40, 15, 1, key=f"{K}mos") / 100
        mid = st.checkbox("Mid-year discounting", True, key=f"{K}mid")
        runs = st.select_slider("Monte Carlo runs", [1000, 2500, 5000, 10000], 5000, key=f"{K}runs")

# --- peers --------------------------------------------------------------------
if company.is_demo:
    peers = demo_peers()
else:
    peer_list = tuple(dict.fromkeys(normalise_ticker(p, suffix) for p in peer_text.replace("\n", ",").split(",")
                                    if p.strip() and normalise_ticker(p, suffix) != ticker))
    with st.spinner("Fetching peer multiples…"):
        peers = load_peers(peer_list) if peer_list else pd.DataFrame()

assumptions = Assumptions(risk_free=rf, erp=erp, beta=beta, terminal_growth=tg, growth_high=gh,
                          op_margin_target=mt, tax_rate=tax, sales_to_capital=s2c, terminal_roic=roic,
                          roe_terminal=roe_t, method_weights=weights, margin_of_safety=mos, mid_year=mid,
                          mc_runs=runs)
rep = value_company(company, assumptions, peers)
v, c, w, inp = rep.verdict, rep.company, rep.wacc, rep.inputs
vc = RATING_COLOR.get(v.rating, HOLD)
mtab = rep.method_table


def family_blend(fam: str) -> float:
    sub = mtab[(mtab.Family == fam) & mtab.Applies]
    wt = sub.Weight.where(sub.Weight > 0)
    if wt.notna().any():
        return float((sub.Value * wt).sum() / wt.sum())
    return float(sub.Value.median()) if len(sub) else np.nan


intrinsic_v, relative_v = family_blend("Intrinsic"), family_blend("Relative")

# ---------------------------------------------------------------------------
# Header + verdict
# ---------------------------------------------------------------------------
fy = c.latest_fy.strftime("FY%y") if c.latest_fy is not None else "n/a"
st.markdown(f'<p class="company">{c.name}</p>'
            f'<p class="meta">{c.ticker} &nbsp;|&nbsp; {c.sector}, {c.industry} &nbsp;|&nbsp; '
            f'Market cap {cr(c.market_cap)} &nbsp;|&nbsp; P/E {mult(c.price / c.eps if c.eps > 0 else np.nan)}'
            f' &nbsp;|&nbsp; Financials to {fy}</p>', unsafe_allow_html=True)

if fin(v.upside):
    direction = "below" if v.upside > 0 else "above"
    line = (f"Trading at <span class='price'>{inr(c.price)}</span>, {abs(v.upside):.0%} {direction} "
            f"a blended fair value of {inr(v.fair_value)}.")
else:
    line = f"Trading at <span class='price'>{inr(c.price)}</span>. The models could not produce a positive value."

gap_txt = ""
if fin(intrinsic_v) and fin(relative_v):
    gap_txt = (f"Intrinsic methods say {inr(intrinsic_v)}; market-relative methods say {inr(relative_v)}. ")
igr = (f"The price implies {pct(rep.implied_growth)} annual revenue growth for five years, versus "
       f"{pct(d['hist_revenue_cagr'])} historically. " if rep.implied_growth is not None else "")
st.markdown(f"""
<div class="verdict" style="--vc:{vc}">
  <div class="rating"><span class="pill" style="--pc:{vc}">{v.rating}</span> {v.valuation_label} &nbsp;·&nbsp;
   {v.confidence_label} confidence ({v.confidence}/100) &nbsp;·&nbsp; {v.methods_above_price} of
   {v.methods_valid} methods above market price</div>
  <div class="line">{line}</div>
  <div class="sub">{gap_txt}{igr}Fair-value range {inr(v.range_low)}–{inr(v.range_high)}. Buy below
   {inr(v.buy_below)} for a {mos:.0%} margin of safety.</div>
</div>""", unsafe_allow_html=True)

an = (f"{c.analyst_count} analysts" + (f", consensus “{c.analyst_view.replace('_', ' ')}”" if c.analyst_view else "")
      if c.analyst_count else "No analyst coverage on Yahoo")
kpis = [
    ("Blended fair value", inr(v.fair_value), f"{len([x for x in v.weights.values() if x > 0])} methods weighted"),
    ("Upside", pct(v.upside, 0, sign=True), "Fair value vs market price"),
    ("Intrinsic", inr(intrinsic_v), "Cash flow, book & dividend models"),
    ("Relative", inr(relative_v), "Peer & historical multiples"),
    ("Chance undervalued", pct(rep.monte_carlo.prob_undervalued, 0), "DCF Monte Carlo"),
    ("Analyst target", inr(c.analyst_target_mean), an),
]
st.markdown('<div class="kpis">' + "".join(
    f'<div class="kpi"><div class="k">{k}</div><div class="v">{val}</div><div class="n">{n}</div></div>'
    for k, val, n in kpis) + "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs(["All methods", "DCF", "Relative", "Residual income & dividends", "Cost of capital",
                "Stress tests", "Data checks", "History"])

# --- All methods (football field) -------------------------------------------
with tabs[0]:
    order = mtab.iloc[::-1]
    labels = [f"{r.Method}  ·  {r.Weight:.0%}" if r.Applies else f"{r.Method}  ·  n/a" for r in order.itertuples()]
    fig = go.Figure()
    if fin(v.range_low) and fin(v.range_high):
        fig.add_vrect(x0=v.range_low, x1=v.range_high, fillcolor=TEAL, opacity=0.07, line_width=0)
    for lab, r in zip(labels, order.itertuples(), strict=False):
        col = FAMILY_COLOR.get(r.Family, HOLD)
        if not r.Applies:
            continue
        fig.add_trace(go.Bar(y=[lab], x=[r.High - r.Low], base=[r.Low], orientation="h", marker_color=col,
                             opacity=0.28, showlegend=False,
                             hovertemplate=f"{r.Method}<br>Range {inr(r.Low)} – {inr(r.High)}<extra></extra>"))
        fig.add_trace(go.Scatter(y=[lab], x=[r.Value], mode="markers", showlegend=False,
                                 marker=dict(color=col, size=12 if r.Weight > 0 else 9,
                                             symbol="circle" if r.Weight > 0 else "circle-open",
                                             line=dict(width=2, color=col)),
                                 hovertemplate=f"{r.Method}: {inr(r.Value)} ({pct(r.Upside, 0, True)})<extra></extra>"))
    # keep n/a rows on the axis
    fig.add_trace(go.Scatter(y=labels, x=[None] * len(labels), mode="markers", showlegend=False, hoverinfo="skip"))
    for fam, col in FAMILY_COLOR.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color=col, size=10), name=fam))
    if fin(v.fair_value):
        fig.add_vline(x=v.fair_value, line_color=INK, line_dash="dot",
                      annotation_text=f"Fair value {inr(v.fair_value)}",
                      annotation_position="bottom left" if v.fair_value < c.price else "bottom right")
    if fin(c.analyst_target_mean):
        fig.add_vline(x=c.analyst_target_mean, line_color=HOLD, line_dash="dash", line_width=1,
                      annotation_text=f"Analysts {inr(c.analyst_target_mean)}",
                      annotation_position="top right" if c.analyst_target_mean > c.price else "top left",
                      annotation_font_color=HOLD)
    price_line(fig, c.price)
    fig.update_layout(barmode="overlay", xaxis_title="Value per share (₹)")
    fig.update_yaxes(categoryorder="array", categoryarray=labels)
    st.plotly_chart(style_fig(fig, 520), width="stretch")
    st.markdown('<p class="note">Each row is one valuation method: the dot is its central value, the bar its '
                'plausible range, the percentage its weight in the blend. Hollow dots are shown for reference '
                'only. The shaded band is the middle 50% of the weighted estimates. When the saffron market-price '
                'line sits left of most dots, the stock looks cheap; right of them, expensive.</p>',
                unsafe_allow_html=True)

    st.subheader("Method by method")
    for fam in ["Intrinsic", "Relative", "Rule of thumb"]:
        sub = [m for m in rep.methods if m.family == fam]
        cols = st.columns(len(sub) if len(sub) <= 5 else 5)
        for i, m in enumerate(sub):
            u = m.value / c.price - 1 if m.valid else np.nan
            wt = v.weights.get(m.name, 0)
            body = (f'<div class="v">{inr(m.value)}</div><div class="u" style="color:{updown_color(u)}">'
                    f'{pct(u, 0, True)} vs price · weight {wt:.0%}</div>' if m.valid else
                    '<div class="v" style="color:#9AA5A8">n/a</div>')
            cols[i % len(cols)].markdown(
                f'<div class="mcard" style="--mc:{FAMILY_COLOR[fam]}"><div class="t">{m.name}</div>{body}'
                f'<div class="d">{m.note}</div></div>', unsafe_allow_html=True)
        st.write("")

# --- DCF -----------------------------------------------------------------------
with tabs[1]:
    st.subheader(f"Free cash flow to the firm · {inr(rep.fcff.per_share)} per share")
    t = rep.fcff.table
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Y{y}" for y in t["Year"]], y=t["FCFF"] / config.CRORE, name="FCFF",
                         marker_color=TEAL))
    fig.add_trace(go.Bar(x=[f"Y{y}" for y in t["Year"]], y=t["PV of FCFF"] / config.CRORE,
                         name="Present value", marker_color=MIST))
    fig.update_layout(barmode="group", yaxis_title="₹ Cr")
    left, right = st.columns([3, 2])
    left.plotly_chart(style_fig(fig, 340), width="stretch")
    b = rep.bridge
    wf = go.Figure(go.Waterfall(
        orientation="v", measure=["relative", "relative", "total", "relative", "relative", "relative", "total"],
        x=["Yrs 1–10", "Terminal", "EV", "Debt", "Cash", "Minority", "Equity"],
        y=np.array([rep.fcff.pv_explicit, rep.fcff.pv_terminal, 0, -b.total_debt, b.cash,
                    -b.minority_interest, 0]) / config.CRORE,
        increasing_marker_color=TEAL, decreasing_marker_color=LOSS, totals_marker_color=INK,
        connector_line_color=MIST))
    wf.update_layout(yaxis_title="₹ Cr", showlegend=False)
    right.plotly_chart(style_fig(wf, 340), width="stretch")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Terminal value share of EV", pct(rep.fcff.terminal_share, 0))
    m2.metric("Implied terminal EV / EBIT", mult(rep.fcff.implied_terminal_multiple))
    m3.metric("Terminal ROIC", pct(inp.terminal_roic))
    m4.metric("WACC", pct(w.wacc, 2))
    with st.expander("FCFF projection table (₹ Cr)"):
        st.caption("FCFF = EBIT × (1 − tax) − reinvestment, reinvestment = Δrevenue ÷ sales-to-capital.")
        st.dataframe(money_table(t, ["Growth", "EBIT margin", "Tax rate"], ["Discount factor"]), width="stretch")

    st.subheader(f"Free cash flow to equity · {inr(rep.fcfe.per_share)} per share")
    t = rep.fcfe.table
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Y{y}" for y in t["Year"]], y=t["Net income"] / config.CRORE, name="Net income",
                         marker_color=MIST))
    fig.add_trace(go.Bar(x=[f"Y{y}" for y in t["Year"]], y=t["FCFE"] / config.CRORE, name="FCFE",
                         marker_color=TEAL))
    fig.update_layout(barmode="group", yaxis_title="₹ Cr")
    st.plotly_chart(style_fig(fig, 300), width="stretch")
    m1, m2, m3 = st.columns(3)
    m1.metric("Equity value", cr(rep.fcfe.equity_value))
    m2.metric("Terminal value share", pct(rep.fcfe.terminal_share, 0))
    m3.metric("Implied terminal P/E", mult(rep.fcfe.implied_terminal_multiple))
    with st.expander("FCFE projection table (₹ Cr)"):
        st.caption("FCFE = net income − (1 − debt ratio) × reinvestment; discounted at the cost of equity.")
        st.dataframe(money_table(t, [], ["Discount factor"]), width="stretch")

# --- Relative ------------------------------------------------------------------
with tabs[2]:
    own = {
        "P/E": c.price / c.eps if c.eps > 0 else np.nan,
        "P/B": c.price / c.bvps if c.bvps > 0 else np.nan,
        "EV/EBITDA": (c.market_cap + c.total_debt - c.cash) / c.ebitda.dropna().iloc[-1]
        if len(c.ebitda.dropna()) and c.ebitda.dropna().iloc[-1] > 0 else np.nan,
        "EV/Sales": (c.market_cap + c.total_debt - c.cash) / c.revenue.dropna().iloc[-1],
    }
    if rep.peers.empty:
        st.info("No peer data. Add peer tickers in the sidebar under “Peer group”.")
    else:
        cols = st.columns(4)
        for i, col in enumerate(["P/E", "EV/EBITDA", "P/B", "EV/Sales"]):
            pdf = rep.peers.dropna(subset=[col])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=pdf[col], y=[n[:22] for n in pdf["Name"]], orientation="h",
                                 marker_color=INDIGO, opacity=0.55, name="Peers",
                                 hovertemplate="%{y}: %{x:.1f}×<extra></extra>"))
            if fin(own[col]):
                fig.add_trace(go.Bar(x=[own[col]], y=[c.name[:22]], orientation="h", marker_color=INK,
                                     name=c.ticker, hovertemplate="%{y}: %{x:.1f}×<extra></extra>"))
            if len(pdf):
                fig.add_vline(x=pdf[col].median(), line_dash="dot", line_color=INDIGO,
                              annotation_text=f"median {pdf[col].median():.1f}×", annotation_position="top")
            fig.update_layout(title=dict(text=col, font=dict(size=15)), showlegend=False,
                              yaxis=dict(categoryorder="total ascending"))
            cols[i].plotly_chart(style_fig(fig, 300), width="stretch")
        st.caption("Dark bar is the company at today's price; blue bars are peers. Yahoo multiples are trailing "
                   "twelve months; outliers (negative or extreme values) are excluded from medians.")
        show = rep.peers.copy()
        own_row = {"Ticker": c.ticker, "Name": f"▶ {c.name}", "Market cap": c.market_cap, **own,
                   "ROE": c.roe_hist.iloc[-1] if len(c.roe_hist) else np.nan,
                   "Revenue growth": c.revenue.pct_change().dropna().iloc[-1] if len(c.revenue.dropna()) > 1 else np.nan}
        show = pd.concat([pd.DataFrame([own_row]), show], ignore_index=True)
        show["Market cap"] = show["Market cap"].map(cr)
        for col in ["P/E", "P/B", "EV/EBITDA", "EV/Sales"]:
            show[col] = show[col].map(mult)
        for col in ["ROE", "Revenue growth"]:
            show[col] = show[col].map(pct)
        st.dataframe(show, hide_index=True, width="stretch")

    st.subheader("What each multiple implies")
    rel = mtab[mtab.Family == "Relative"].copy()
    rel_show = pd.DataFrame({"Method": rel.Method, "Value / share": rel.Value.map(inr),
                             "Range": [f"{inr(a)} – {inr(b_)}" if ok else "—"
                                       for a, b_, ok in zip(rel.Low, rel.High, rel.Applies, strict=False)],
                             "vs price": rel.Upside.map(lambda x: pct(x, 0, True)), "How": rel.Note})
    st.dataframe(rel_show, hide_index=True, width="stretch")
    pe_hist = c.historical_pe()
    if len(pe_hist) >= 2:
        fig = go.Figure(go.Scatter(x=[f"FY{x:%y}" for x in pe_hist.index], y=pe_hist.values, mode="lines+markers",
                                   line=dict(color=INDIGO, width=2.5), name="P/E at year end"))
        if fin(own["P/E"]):
            fig.add_hline(y=own["P/E"], line_color=SAFFRON, annotation_text=f"Today {own['P/E']:.1f}×",
                          annotation_font_color=SAFFRON)
        fig.update_layout(title=dict(text="Own P/E history", font=dict(size=15)), yaxis_title="P/E (×)")
        st.plotly_chart(style_fig(fig, 280), width="stretch")

# --- Residual income & dividends -------------------------------------------
with tabs[3]:
    ri = rep.method("Residual income")
    left, right = st.columns(2)
    with left:
        st.subheader(f"Residual income · {inr(ri.value) if ri.valid else 'n/a'}")
        if ri.valid:
            i_ = ri.inputs
            pv_ri = ri.table["PV of RI"].sum()
            wf = go.Figure(go.Waterfall(
                orientation="v", measure=["relative", "relative", "relative", "total"],
                x=["Book value", "Excess returns<br>yrs 1–10", "Terminal", "Value"],
                y=[i_["bv0"], pv_ri, i_["pv_terminal"], 0],
                increasing_marker_color=TEAL, decreasing_marker_color=LOSS, totals_marker_color=INK,
                connector_line_color=MIST, text=[inr(i_["bv0"]), inr(pv_ri), inr(i_["pv_terminal"]), inr(ri.value)],
                textposition="outside"))
            wf.update_layout(yaxis_title="₹ per share", showlegend=False)
            price_line(wf, c.price, horizontal=True)
            st.plotly_chart(style_fig(wf, 360), width="stretch")
            st.markdown(f'<p class="note">{ri.note} Value = book value + present value of (ROE − cost of equity) '
                        '× book value. It is the main model for banks and NBFCs because it never needs a '
                        'free-cash-flow figure.</p>', unsafe_allow_html=True)
            roe = c.roe_hist
            if len(roe):
                st.caption("Historical ROE: " + ", ".join(f"FY{k:%y} {pct(x)}" for k, x in roe.items()))
            with st.expander("Residual income table (₹ per share)"):
                st.dataframe(money_table(ri.table, ["ROE"], ["Discount factor"], per_share=True), width="stretch")
        else:
            st.info(ri.note)
    with right:
        ddm = rep.method("Dividend discount")
        st.subheader(f"Dividend discount · {inr(ddm.value) if ddm.valid else 'n/a'}")
        if ddm.valid:
            t = ddm.table
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"Y{y}" for y in t["Year"]], y=t["Dividend"], name="Dividend / share",
                                 marker_color=TEAL))
            fig.add_trace(go.Scatter(x=[f"Y{y}" for y in t["Year"]], y=t["Payout"], name="Payout ratio",
                                     yaxis="y2", mode="lines+markers", line=dict(color=INK)))
            fig.update_layout(yaxis=dict(title="₹ per share"),
                              yaxis2=dict(title="Payout", overlaying="y", side="right", tickformat=".0%",
                                          showgrid=False, range=[0, 1.05]))
            st.plotly_chart(style_fig(fig, 360), width="stretch")
            st.markdown(f'<p class="note">{ddm.note}</p>', unsafe_allow_html=True)
        else:
            st.info(ddm.note)

    st.subheader("No-growth and rule-of-thumb checks")
    cols = st.columns(3)
    for col, name in zip(cols, ["Earnings power (no growth)", "Graham number", "Graham formula"], strict=False):
        m = rep.method(name)
        u = m.value / c.price - 1 if m.valid else np.nan
        col.markdown(f'<div class="mcard" style="--mc:{FAMILY_COLOR[m.family]}"><div class="t">{name}</div>'
                     f'<div class="v">{inr(m.value) if m.valid else "n/a"}</div>'
                     f'<div class="u" style="color:{updown_color(u)}">{pct(u, 0, True) if m.valid else ""}</div>'
                     f'<div class="d">{m.note}</div></div>', unsafe_allow_html=True)
    epv = rep.method("Earnings power (no growth)")
    if epv.valid and rep.fcff.valid:
        st.write("")
        growth_share = 1 - epv.value / c.price
        st.markdown(f"**Growth priced in:** the business as it stands, with no growth, is worth about "
                    f"{inr(epv.value)}. That means {pct(max(growth_share, 0), 0)} of today's price is a bet on "
                    "future growth." if growth_share > 0 else
                    f"**No growth needed:** today's earnings alone, with zero growth, justify {inr(epv.value)} — "
                    "above the market price.")

# --- Cost of capital --------------------------------------------------------
with tabs[4]:
    left, right = st.columns([2, 3])
    with left:
        rows = [
            ("Risk-free rate (10Y G-sec)", pct(w.risk_free, 2)),
            ("Equity risk premium (India)", pct(w.erp, 2)),
            ("Beta (Blume-adjusted)", f"{w.beta:.2f}"),
            ("Cost of equity", pct(w.cost_of_equity, 2)),
            ("Interest coverage", f"{w.interest_coverage:.1f}×" if w.interest_coverage else "No interest"),
            ("Synthetic rating", w.rating),
            ("Pre-tax cost of debt", pct(w.pre_tax_cost_of_debt, 2)),
            ("After-tax cost of debt", pct(w.after_tax_cost_of_debt, 2)),
            ("Equity weight (market)", pct(w.weight_equity)),
            ("Debt weight", pct(w.weight_debt)),
            ("WACC", pct(w.wacc, 2)),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Component", "Value"]), hide_index=True,
                     width="stretch", height=420)
    with right:
        bt = rep.beta
        s = c.price_history.copy()
        ix = c.index_history.copy()
        if len(s) > 30 and len(ix) > 30:
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            ix.index = pd.to_datetime(ix.index).tz_localize(None).normalize()
            r = pd.concat([s.pct_change(), ix.pct_change()], axis=1, join="inner").dropna()
            r.columns = ["stock", "nifty"]
            fig = go.Figure(go.Scatter(x=r["nifty"], y=r["stock"], mode="markers",
                                       marker=dict(color=TEAL, opacity=0.45, size=7), name="Weekly returns"))
            if bt.raw is not None and bt.source == "regression":
                xs = np.linspace(r["nifty"].min(), r["nifty"].max(), 20)
                a = r["stock"].mean() - bt.raw * r["nifty"].mean()
                fig.add_trace(go.Scatter(x=xs, y=a + bt.raw * xs, mode="lines", line=dict(color=INK),
                                         name=f"Raw beta {bt.raw:.2f}"))
            fig.update_layout(xaxis_title="Nifty 50 weekly return", yaxis_title=f"{c.ticker} weekly return",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            st.plotly_chart(style_fig(fig, 380), width="stretch")
        src = {"regression": f"Regression on {bt.observations} weekly returns vs Nifty 50"
                             f"{f', R² {bt.r_squared:.2f}' if bt.r_squared else ''}",
               "yahoo": "Yahoo Finance reported beta (regression too weak)", "default": "Default of 1.0"}
        st.caption(f"Beta source: {src[bt.source]}. Raw beta is pulled one-third toward 1 (Blume) "
                   "because betas mean-revert over time.")

# --- Stress tests -------------------------------------------------------------
with tabs[5]:
    mc = rep.monte_carlo
    st.subheader("Monte Carlo (FCFF DCF)")
    vals = mc.values
    if len(vals):
        clip = vals[(vals > np.percentile(vals, 0.5)) & (vals < np.percentile(vals, 99.5))]
        fig = go.Figure(go.Histogram(x=clip, nbinsx=70, marker_color=TEAL, opacity=0.75))
        for q, lab in [(mc.p10, "P10"), (mc.p50, "Median"), (mc.p90, "P90")]:
            fig.add_vline(x=q, line_color=INK, line_dash="dot", annotation_text=f"{lab} {inr(q)}",
                          annotation_position="top right")
        price_line(fig, c.price, pos="bottom right" if c.price > mc.p50 else "bottom left")
        fig.update_layout(xaxis_title="FCFF value per share (₹)", yaxis_title="Simulations", showlegend=False)
        st.plotly_chart(style_fig(fig, 360), width="stretch")
        st.markdown(
            f"In **{mc.prob_undervalued:.0%}** of {mc.n_valid:,} simulations the stock is worth more than "
            f"{inr(c.price)}. Each run draws revenue growth, target margin, WACC, terminal growth and "
            "sales-to-capital from distributions centred on your assumptions.")
    else:
        st.warning("No valid simulations: check that terminal growth is well below WACC.")

    left, right = st.columns(2)
    with left:
        st.subheader("WACC × terminal growth")
        sens = rep.sensitivity
        rel_ = sens.values / c.price - 1
        lim = max(0.3, np.nanmax(np.abs(rel_))) if np.isfinite(rel_).any() else 0.3
        fig = go.Figure(go.Heatmap(
            z=rel_, x=[pct(g, 2) for g in sens.columns], y=[pct(x, 2) for x in sens.index],
            text=[[inr(val) for val in row] for row in sens.values], texttemplate="%{text}",
            colorscale=[[0, LOSS], [0.5, "#F7F7F2"], [1, GAIN]], zmin=-lim, zmax=lim,
            colorbar=dict(title="vs price", tickformat="+.0%"),
            hovertemplate="WACC %{y}, g %{x}<br>Value %{text}<extra></extra>"))
        fig.update_layout(xaxis_title="Terminal growth", yaxis_title="WACC")
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category", autorange="reversed")
        st.plotly_chart(style_fig(fig, 420), width="stretch")
    with right:
        st.subheader("Which assumption matters most")
        tor = rep.tornado
        fig = go.Figure()
        fig.add_trace(go.Bar(y=tor["Driver"], x=tor["Low"] - tor["Base"], base=tor["Base"], orientation="h",
                             marker_color=LOSS, name="Value falls"))
        fig.add_trace(go.Bar(y=tor["Driver"], x=tor["High"] - tor["Base"], base=tor["Base"], orientation="h",
                             marker_color=GAIN, name="Value rises"))
        fig.update_layout(barmode="overlay", xaxis_title="FCFF value per share (₹)")
        price_line(fig, c.price)
        st.plotly_chart(style_fig(fig, 420), width="stretch")

    st.subheader("Scenarios")
    show = rep.scenarios.copy()
    for col in ["Revenue growth", "Target margin", "WACC", "Terminal g"]:
        show[col] = show[col].map(pct)
    for col in ["FCFF value", "FCFE value"]:
        show[col] = show[col].map(inr)
    st.dataframe(show, hide_index=True, width="stretch")

# --- Data checks --------------------------------------------------------------
with tabs[6]:
    colors = {"error": LOSS, "warning": SAFFRON, "info": TEAL}
    st.markdown(f"**Confidence {v.confidence}/100 ({v.confidence_label}).** It starts at 100 and drops for "
                "each data problem below and whenever the methods disagree with each other.")
    if not rep.flags:
        st.success("No data-quality issues found.")
    for f in rep.flags:
        st.markdown(f'<div class="flag" style="--fc:{colors[f.level]}"><b>{f.title}</b>'
                    f'{f" (−{f.penalty} confidence)" if f.penalty else ""}<div>{f.detail}</div></div>',
                    unsafe_allow_html=True)
    if v.reasons:
        st.markdown("**Model checks**")
        for r_ in v.reasons:
            st.markdown(f"- {r_}")
    for n in c.notes:
        st.caption(n)

# --- History ----------------------------------------------------------------
with tabs[7]:
    hist = pd.DataFrame({
        "Revenue": c.revenue, "EBITDA": c.ebitda, "EBIT": c.ebit, "Net income": c.net_income,
        "D&A": c.da, "Capex": c.capex, "Δ Working capital": c.delta_nwc,
        "FCFF (historical)": c.historical_fcff(inp.tax_rate),
    })
    hist.index = [f"FY{d_.strftime('%y')}" for d_ in hist.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hist.index, y=hist["Revenue"] / config.CRORE, name="Revenue", marker_color=MIST))
    fig.add_trace(go.Bar(x=hist.index, y=hist["EBIT"] / config.CRORE, name="EBIT", marker_color=TEAL))
    fig.add_trace(go.Bar(x=hist.index, y=hist["FCFF (historical)"] / config.CRORE, name="FCFF",
                         marker_color=INK))
    fig.update_layout(barmode="group", yaxis_title="₹ Cr")
    left, right = st.columns([3, 2])
    left.plotly_chart(style_fig(fig, 340), width="stretch")
    ph = c.price_history.dropna()
    if len(ph):
        pfig = go.Figure(go.Scatter(x=ph.index, y=ph.values, mode="lines", line=dict(color=INK, width=1.6),
                                    name="Price"))
        if fin(v.fair_value):
            pfig.add_hline(y=v.fair_value, line_color=TEAL, line_dash="dot",
                           annotation_text=f"Fair value {inr(v.fair_value)}", annotation_font_color=TEAL)
        pfig.update_layout(yaxis_title="₹", showlegend=False, title=dict(text="5-year price", font=dict(size=15)))
        right.plotly_chart(style_fig(pfig, 340), width="stretch")
    table = (hist / config.CRORE).T.map(lambda x: f"{x:,.0f}" if np.isfinite(x) else "—")
    table.loc["EBIT margin"] = (c.ebit / c.revenue).map(pct).values
    st.dataframe(table, width="stretch")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
f1, f2 = st.columns([3, 1])
f1.markdown(
    f'<p class="note">Data from Yahoo Finance, fetched {c.fetched_at:%d %b %Y %H:%M}. '
    'Educational tool, not investment advice. Intrinsic is not a SEBI-registered adviser; check figures '
    'against the annual report before acting.</p>', unsafe_allow_html=True)
f2.download_button("Download model (Excel)", to_excel(rep), file_name=f"{c.ticker}_valuation.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   width="stretch")
