import numpy as np
import pytest

from valuation import Assumptions, fetch_company, normalise_ticker, value_company
from valuation.analysis import implied_growth, sensitivity_grid
from valuation.dcf import Bridge, DCFInputs, fcff_per_share, run_fcfe, run_fcff
from valuation.recommend import rate
from valuation.wacc import synthetic_rating


def flat_inputs(**kw):
    """No growth, constant margin, tax already at marginal -> pure perpetuity."""
    base = dict(growth_high=0.0, terminal_growth=0.0, op_margin_current=0.2,
                op_margin_target=0.2, tax_rate=0.25, marginal_tax=0.25,
                sales_to_capital=2.0, terminal_roic=0.15, wacc=0.10,
                cost_of_equity=0.12, pre_tax_cost_of_debt=0.08, debt_ratio=0.0,
                mid_year=False)
    base.update(kw)
    return DCFInputs(**base)


BRIDGE = Bridge(base_revenue=1000.0, total_debt=200.0, cash=50.0, minority_interest=0.0,
                shares=10.0)


def test_zero_growth_fcff_equals_perpetuity():
    r = run_fcff(flat_inputs(), BRIDGE)
    nopat = 1000 * 0.2 * 0.75
    assert r.enterprise_value == pytest.approx(nopat / 0.10, rel=1e-9)
    assert r.per_share == pytest.approx((nopat / 0.10 - 200 + 50) / 10, rel=1e-9)


def test_unlevered_fcfe_matches_fcff_when_no_debt():
    b = Bridge(1000.0, 0.0, 0.0, 0.0, 10.0)
    inp = flat_inputs(growth_high=0.08, terminal_growth=0.04, cost_of_equity=0.10)
    assert run_fcfe(inp, b).per_share == pytest.approx(run_fcff(inp, b).per_share, rel=1e-9)


def test_vectorised_engine_matches_single_case():
    inp = flat_inputs(growth_high=0.12, terminal_growth=0.05, op_margin_current=0.15,
                      op_margin_target=0.2, tax_rate=0.22)
    single = run_fcff(inp, BRIDGE).per_share
    vec = fcff_per_share(inp, BRIDGE, growth_high=np.array([0.10, 0.12, 0.14]))["per_share"]
    assert vec[1] == pytest.approx(single)
    assert vec[0] < vec[1] < vec[2]


def test_higher_wacc_lowers_value():
    inp = flat_inputs(growth_high=0.1, terminal_growth=0.05)
    grid = sensitivity_grid(inp, BRIDGE)
    assert (np.diff(grid.values, axis=0) < 0).all()   # down the rows: higher WACC
    assert (np.diff(grid.values, axis=1) > 0).all()   # across: higher g (ROIC > WACC)


def test_terminal_growth_above_wacc_is_nan():
    r = run_fcff(flat_inputs(terminal_growth=0.12), BRIDGE)
    assert not r.valid


def test_reverse_dcf_recovers_price():
    inp = flat_inputs(growth_high=0.1, terminal_growth=0.04)
    price = run_fcff(inp.with_(growth_high=0.15), BRIDGE).per_share
    g = implied_growth(inp, BRIDGE, price)
    assert g == pytest.approx(0.15, abs=1e-4)


@pytest.mark.parametrize("cov,expected", [(20, "AAA"), (5, "A"), (2.1, "BB"), (0.1, "D")])
def test_synthetic_rating(cov, expected):
    assert synthetic_rating(cov)[0] == expected


@pytest.mark.parametrize("upside,label", [(0.5, "Strong Buy"), (0.2, "Buy"), (0.0, "Hold"),
                                          (-0.15, "Sell"), (-0.5, "Strong Sell")])
def test_rating_bands(upside, label):
    assert rate(upside) == label


def test_ticker_normalisation():
    assert normalise_ticker(" reliance ") == "RELIANCE.NS"
    assert normalise_ticker("500325", ".BO") == "500325.BO"
    assert normalise_ticker("TCS.NS") == "TCS.NS"


def test_demo_pipeline_end_to_end():
    rep = value_company(fetch_company("DEMO"), Assumptions(mc_runs=500))
    assert rep.fcff.valid and rep.fcfe.valid
    assert 0.08 < rep.wacc.wacc < 0.18
    assert rep.verdict.rating in {"Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"}
    assert 0 <= rep.verdict.confidence <= 100
    assert rep.monte_carlo.p10 < rep.monte_carlo.p50 < rep.monte_carlo.p90


def test_user_overrides_flow_through():
    c = fetch_company("DEMO")
    low = value_company(c, Assumptions(growth_high=0.05, mc_runs=200)).fcff.per_share
    high = value_company(c, Assumptions(growth_high=0.20, mc_runs=200)).fcff.per_share
    assert high > low
