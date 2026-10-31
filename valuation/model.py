"""One-call pipeline: company data + assumptions -> full valuation report."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .analysis import MonteCarloResult, driver_tornado, implied_growth, monte_carlo, run_scenarios, sensitivity_grid
from .data import CompanyData, derive_defaults
from .dcf import Bridge, DCFInputs, DCFResult, run_fcfe, run_fcff
from .methods import (
    MethodResult,
    dividend_discount,
    earnings_power,
    graham_formula,
    graham_number,
    own_history_pe,
    peer_multiples,
    residual_income,
)
from .quality import Flag, run_checks
from .recommend import Verdict, build_verdict, method_table
from .wacc import BetaResult, WACCResult, compute_wacc, estimate_beta


@dataclass
class Assumptions:
    """Everything a user can override. None = use the auto-derived default."""
    risk_free: float = config.RISK_FREE_RATE
    erp: float = config.EQUITY_RISK_PREMIUM
    beta: float | None = None
    terminal_growth: float = config.DEFAULT_TERMINAL_GROWTH
    growth_high: float | None = None
    op_margin_target: float | None = None
    tax_rate: float | None = None
    sales_to_capital: float | None = None
    terminal_roic: float | None = None
    roe_terminal: float | None = None
    method_weights: dict | None = None   # None = sector-appropriate defaults
    margin_of_safety: float = 0.15
    mid_year: bool = True
    mc_runs: int = 5000


@dataclass
class ValuationReport:
    company: CompanyData
    defaults: dict
    beta: BetaResult
    wacc: WACCResult
    inputs: DCFInputs
    bridge: Bridge
    fcff: DCFResult
    fcfe: DCFResult
    scenarios: pd.DataFrame
    sensitivity: pd.DataFrame
    tornado: pd.DataFrame
    monte_carlo: MonteCarloResult
    implied_growth: float | None
    flags: list[Flag]
    methods: list[MethodResult]
    method_table: pd.DataFrame
    peers: pd.DataFrame
    verdict: Verdict

    def method(self, name: str) -> MethodResult:
        return next(m for m in self.methods if m.name == name)


def value_company(c: CompanyData, a: Assumptions | None = None,
                  peers: pd.DataFrame | None = None) -> ValuationReport:
    a = a or Assumptions()
    d = derive_defaults(c)

    beta_res = estimate_beta(c)
    beta = a.beta if a.beta is not None else beta_res.adjusted
    tax = a.tax_rate if a.tax_rate is not None else d["tax_rate"]
    w = compute_wacc(c, beta, a.risk_free, a.erp, config.MARGINAL_TAX_RATE)

    terminal_roic = a.terminal_roic
    if terminal_roic is None:
        # Some competitive advantage persists, but excess returns fade:
        # halfway between today's ROIC and WACC, bounded to [WACC, WACC + 8pp].
        terminal_roic = float(np.clip(0.5 * (d["current_roic"] + w.wacc), w.wacc, w.wacc + 0.08))

    inp = DCFInputs(
        growth_high=a.growth_high if a.growth_high is not None else d["growth_high"],
        terminal_growth=a.terminal_growth,
        op_margin_current=d["op_margin_current"],
        op_margin_target=a.op_margin_target if a.op_margin_target is not None else d["op_margin_target"],
        tax_rate=tax,
        sales_to_capital=a.sales_to_capital if a.sales_to_capital is not None else d["sales_to_capital"],
        terminal_roic=terminal_roic,
        wacc=w.wacc,
        cost_of_equity=w.cost_of_equity,
        pre_tax_cost_of_debt=w.pre_tax_cost_of_debt,
        debt_ratio=w.weight_debt,
        mid_year=a.mid_year,
    )
    bridge = Bridge(
        base_revenue=float(c.revenue.dropna().iloc[-1]),
        total_debt=c.total_debt, cash=c.cash,
        minority_interest=c.minority_interest, shares=c.shares,
    )

    fcff = run_fcff(inp, bridge)
    fcfe = run_fcfe(inp, bridge)
    mc = monte_carlo(inp, bridge, c.price, n=a.mc_runs)
    scenarios = run_scenarios(inp, bridge)
    sc = scenarios.set_index("Scenario")
    flags = run_checks(c, inp, a.risk_free)

    # --- every valuation method ---------------------------------------------
    ke, g = w.cost_of_equity, a.terminal_growth
    dcf_note = ("" if not c.is_financial else
                " Not weighted for financial companies: debt is their raw material.")
    methods = [
        MethodResult("DCF (FCFF)", fcff.per_share, mc.p10, mc.p90, fcff.valid,
                     f"10-year free cash flow to the firm at WACC {w.wacc:.1%}; range is the "
                     f"Monte Carlo P10–P90.{dcf_note}" if fcff.valid else
                     "Free cash flow to the firm gave no positive value under these assumptions.",
                     table=fcff.table),
        MethodResult("DCF (FCFE)", fcfe.per_share, sc.loc["Bear", "FCFE value"],
                     sc.loc["Bull", "FCFE value"], fcfe.valid,
                     f"10-year free cash flow to equity at cost of equity {ke:.1%}; range is "
                     f"bear–bull scenarios.{dcf_note}" if fcfe.valid else
                     "Free cash flow to equity gave no positive value under these assumptions.",
                     table=fcfe.table),
    ]
    ri = residual_income(c, ke, g, roe_terminal=a.roe_terminal)
    roe_t = ri.inputs.get("roe_terminal", ke + 0.03) if ri.valid else ke + 0.03
    methods += [
        ri,
        dividend_discount(c, ke, inp.growth_high, g, max(roe_t, g + 0.02)),
        earnings_power(c, w.wacc, inp.tax_rate),
        *peer_multiples(c, peers),
        own_history_pe(c),
        graham_number(c),
        graham_formula(c, inp.growth_high),
    ]
    raw_w = a.method_weights or (config.FINANCIAL_METHOD_WEIGHTS if c.is_financial
                                 else config.METHOD_WEIGHTS)
    verdict = build_verdict(c.price, methods, raw_w, mc, flags, a.margin_of_safety,
                            dcf_terminal_share=fcff.terminal_share if not c.is_financial else np.nan)

    return ValuationReport(
        company=c, defaults=d, beta=beta_res, wacc=w, inputs=inp, bridge=bridge,
        fcff=fcff, fcfe=fcfe, scenarios=scenarios,
        sensitivity=sensitivity_grid(inp, bridge), tornado=driver_tornado(inp, bridge),
        monte_carlo=mc, implied_growth=implied_growth(inp, bridge, c.price),
        flags=flags, methods=methods, method_table=method_table(methods, verdict.weights, c.price),
        peers=peers if peers is not None else pd.DataFrame(), verdict=verdict,
    )
