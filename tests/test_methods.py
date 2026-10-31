"""Tests for the non-DCF methods and the multi-method verdict.

Each model is checked against a case with a closed-form answer.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from valuation import Assumptions, fetch_company, value_company
from valuation.analysis import MonteCarloResult
from valuation.methods import (
    MethodResult,
    _ddm_value,
    _ri_value,
    earnings_power,
    graham_formula,
    graham_number,
    own_history_pe,
    peer_multiples,
    residual_income,
)
from valuation.peers import PEER_COLUMNS, demo_peers, suggest_peers
from valuation.recommend import build_verdict

DEMO = fetch_company("DEMO")
NO_MC = MonteCarloResult(np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan, 0)


# --- residual income ---------------------------------------------------------
def test_ri_equals_book_when_roe_equals_cost_of_equity():
    value, _, _ = _ri_value(bv0=100, roe0=0.12, roe_t=0.12, ke=0.12, g=0.04, payout=0.4)
    assert value == pytest.approx(100)


def test_ri_perpetuity_with_full_payout():
    # Full payout -> book value constant; RI = (ROE - Ke) * BV forever, no growth.
    value, _, _ = _ri_value(bv0=100, roe0=0.20, roe_t=0.20, ke=0.12, g=0.0, payout=1.0)
    assert value == pytest.approx(100 + (0.20 - 0.12) * 100 / 0.12)


def test_ri_below_book_when_roe_below_ke():
    value, _, _ = _ri_value(bv0=100, roe0=0.08, roe_t=0.08, ke=0.12, g=0.03, payout=0.5)
    assert value < 100


def test_ri_rejects_negative_book():
    c = replace(DEMO, book_equity=-1.0)
    assert not residual_income(c, 0.13, 0.05).valid


# --- dividend discount --------------------------------------------------------
def test_ddm_collapses_to_gordon_growth():
    ke, g, roe = 0.12, 0.05, 0.20
    payout = 1 - g / roe          # already at the sustainable payout
    value, _, _ = _ddm_value(eps0=10, payout0=payout, g_high=g, g=g, ke=ke, roe_t=roe)
    d1 = 10 * (1 + g) * payout
    assert value == pytest.approx(d1 / (ke - g))


def test_ddm_not_applicable_without_dividend():
    c = replace(DEMO, dividend_per_share=0.0)
    m = value_company(c, Assumptions(mc_runs=200)).method("Dividend discount")
    assert not m.valid and "dividend" in m.note.lower()


# --- earnings power / Graham ---------------------------------------------------
def test_epv_matches_formula():
    m = earnings_power(DEMO, wacc=0.12, tax=0.25)
    margin = float((DEMO.ebit / DEMO.revenue).mean())
    nopat = DEMO.revenue.iloc[-1] * margin * 0.75
    expected = (nopat / 0.12 - DEMO.total_debt + DEMO.cash - DEMO.minority_interest) / DEMO.shares
    assert m.value == pytest.approx(expected)
    assert m.low < m.value < m.high


def test_graham_number():
    m = graham_number(DEMO)
    assert m.value == pytest.approx(np.sqrt(22.5 * DEMO.eps * DEMO.bvps))


def test_graham_formula_caps_growth_at_15pct():
    a = graham_formula(DEMO, 0.15).value
    b = graham_formula(DEMO, 0.40).value
    assert a == pytest.approx(b)


def test_graham_needs_positive_eps():
    c = replace(DEMO, eps_ttm=-5.0)
    assert not graham_number(c).valid and not graham_formula(c, 0.1).valid


# --- relative ----------------------------------------------------------------
def test_peer_pe_uses_median():
    peers = demo_peers()
    pe = next(m for m in peer_multiples(DEMO, peers) if m.name == "Peer P/E")
    assert pe.value == pytest.approx(peers["P/E"].median() * DEMO.eps)


def test_peer_outliers_excluded():
    peers = demo_peers()
    peers.loc[0, "P/E"] = 900.0           # absurd multiple
    peers.loc[1, "P/E"] = -12.0           # loss-maker
    pe = next(m for m in peer_multiples(DEMO, peers) if m.name == "Peer P/E")
    assert pe.inputs["n"] == 3


def test_ev_multiples_off_for_financials():
    bank = replace(DEMO, sector="Financial Services", industry="Banks - Regional")
    res = {m.name: m for m in peer_multiples(bank, demo_peers())}
    assert not res["Peer EV/EBITDA"].valid and not res["Peer EV/Sales"].valid
    assert res["Peer P/B"].valid


def test_too_few_peers_is_not_applicable():
    one = demo_peers().head(1)
    assert all(not m.valid for m in peer_multiples(DEMO, one))
    assert all(not m.valid for m in peer_multiples(DEMO, pd.DataFrame(columns=PEER_COLUMNS)))


def test_own_history_pe():
    m = own_history_pe(DEMO)
    assert m.valid
    assert m.value == pytest.approx(DEMO.historical_pe().median() * DEMO.eps)


def test_peer_suggestions_exclude_self():
    peers = suggest_peers("TCS.NS", "Information Technology Services")
    assert "TCS.NS" not in peers and "INFY.NS" in peers
    assert suggest_peers("500325.BO", "Oil & Gas Refining & Marketing")[0].endswith(".NS")
    assert suggest_peers("XYZ.NS", "Some Unknown Industry") == []


def test_thin_industry_topped_up_from_sector():
    peers = suggest_peers("COALINDIA.NS", "Thermal Coal", "Energy")
    assert len(peers) >= 4 and "COALINDIA.NS" not in peers
    assert "NMDC.NS" in peers                       # industry peer kept first
    assert len(suggest_peers("NEWCO.NS", "Unlisted Industry", "Real Estate")) >= 4


def test_bse_codes_normalise_to_bo():
    from valuation import normalise_ticker
    assert normalise_ticker("500325") == "500325.BO"


# --- blended verdict ----------------------------------------------------------
def _m(name, value, valid=True):
    return MethodResult(name, value, value * 0.9, value * 1.1, valid, "")


def test_blend_renormalises_over_valid_methods():
    methods = [_m("DCF (FCFF)", 120), _m("Peer P/E", 80), _m("Residual income", 1, valid=False)]
    v = build_verdict(100, methods, {"DCF (FCFF)": 30, "Peer P/E": 10, "Residual income": 60}, NO_MC, [])
    assert v.weights == pytest.approx({"DCF (FCFF)": 0.75, "Peer P/E": 0.25})
    assert v.fair_value == pytest.approx(0.75 * 120 + 0.25 * 80)
    assert v.methods_above_price == 1 and v.methods_valid == 2


def test_disagreement_lowers_confidence():
    agree = [_m("DCF (FCFF)", 130), _m("Residual income", 125), _m("Peer P/E", 128), _m("Peer P/B", 132)]
    split = [_m("DCF (FCFF)", 60), _m("Residual income", 55), _m("Peer P/E", 200), _m("Peer P/B", 210)]
    w = {"DCF (FCFF)": 1, "Residual income": 1, "Peer P/E": 1, "Peer P/B": 1}
    assert build_verdict(100, agree, w, NO_MC, []).confidence > build_verdict(100, split, w, NO_MC, []).confidence


def test_zero_weight_methods_fall_back_to_equal_blend():
    methods = [_m("Peer P/E", 90), _m("Peer P/B", 110)]
    v = build_verdict(100, methods, {"DCF (FCFF)": 100}, NO_MC, [])
    assert v.fair_value == pytest.approx(100)


def test_full_pipeline_runs_twelve_methods():
    rep = value_company(DEMO, Assumptions(mc_runs=300), demo_peers())
    assert len(rep.methods) == 12
    assert rep.method_table["Applies"].sum() == 12
    assert np.isfinite(rep.verdict.fair_value)
    assert rep.verdict.range_low <= rep.verdict.fair_value <= rep.verdict.range_high


def test_financial_company_switches_to_equity_models():
    bank = replace(DEMO, sector="Financial Services", industry="Banks - Regional")
    rep = value_company(bank, Assumptions(mc_runs=300), demo_peers())
    w = rep.verdict.weights
    assert w.get("DCF (FCFF)", 0) == 0 and w.get("Peer EV/EBITDA", 0) == 0
    assert w["Residual income"] >= 0.25


def test_low_confidence_gives_no_directional_call():
    from valuation.quality import Flag
    methods = [_m("DCF (FCFF)", 300), _m("Peer P/E", 280), _m("Residual income", 320)]
    w = {"DCF (FCFF)": 1, "Peer P/E": 1, "Residual income": 1}
    bad_data = [Flag("error", "x", "", 60)]
    v = build_verdict(100, methods, w, NO_MC, bad_data)
    assert v.raw_rating == "Strong Buy" and v.rating == "Hold"


def test_strong_call_needs_families_to_agree():
    methods = [_m("DCF (FCFF)", 40), _m("Residual income", 45), _m("Peer P/E", 120), _m("Peer P/B", 125)]
    w = {"DCF (FCFF)": 1, "Residual income": 1, "Peer P/E": 0.2, "Peer P/B": 0.2}
    v = build_verdict(200, methods, w, NO_MC, [])
    assert v.raw_rating == "Strong Sell" and v.rating in ("Sell", "Hold")
