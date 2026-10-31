"""Robustness checks around the base DCF: sensitivity, scenarios, Monte Carlo
simulation and reverse DCF (what growth is the market pricing in?)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dcf import Bridge, DCFInputs, fcff_per_share, run_fcfe, run_fcff


# ---------------------------------------------------------------------------
# Sensitivity grid
# ---------------------------------------------------------------------------
def sensitivity_grid(inp: DCFInputs, bridge: Bridge, wacc_step=0.005, g_step=0.005,
                     n=7) -> pd.DataFrame:
    """Per-share FCFF value across WACC (rows) x terminal growth (columns)."""
    half = n // 2
    waccs = inp.wacc + wacc_step * np.arange(-half, half + 1)
    gs = inp.terminal_growth + g_step * np.arange(-half, half + 1)
    W, G = np.meshgrid(waccs, gs, indexing="ij")
    vals = fcff_per_share(inp, bridge, wacc=W.ravel(), terminal_growth=G.ravel())["per_share"]
    return pd.DataFrame(vals.reshape(W.shape), index=np.round(waccs, 4), columns=np.round(gs, 4))


def driver_tornado(inp: DCFInputs, bridge: Bridge) -> pd.DataFrame:
    """How much each driver moves value when flexed by a plausible amount.

    Low/High are the lower and higher resulting values, whichever direction of the flex causes them.
    """
    base = fcff_per_share(inp, bridge)["per_share"][0]
    flexes = {
        "Revenue growth (±3pp)": ("growth_high", 0.03),
        "Target EBIT margin (±2pp)": ("op_margin_target", 0.02),
        "WACC (±1pp)": ("wacc", 0.01),
        "Terminal growth (±0.5pp)": ("terminal_growth", 0.005),
        "Sales-to-capital (±25%)": ("sales_to_capital", None),
        "Terminal ROIC (±3pp)": ("terminal_roic", 0.03),
    }
    rows = []
    for label, (key, step) in flexes.items():
        cur = getattr(inp, key)
        lo_v, hi_v = (cur * 0.75, cur * 1.25) if step is None else (cur - step, cur + step)
        lo = fcff_per_share(inp, bridge, **{key: lo_v})["per_share"][0]
        hi = fcff_per_share(inp, bridge, **{key: hi_v})["per_share"][0]
        rows.append({"Driver": label, "Low": min(lo, hi), "High": max(lo, hi), "Swing": abs(hi - lo)})
    df = pd.DataFrame(rows).sort_values("Swing")
    df["Base"] = base
    return df


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
SCENARIOS = {
    "Bear": dict(growth=-0.04, margin=-0.03, wacc=+0.010, g=-0.005),
    "Base": dict(growth=0.0, margin=0.0, wacc=0.0, g=0.0),
    "Bull": dict(growth=+0.04, margin=+0.02, wacc=-0.005, g=+0.005),
}


def run_scenarios(inp: DCFInputs, bridge: Bridge) -> pd.DataFrame:
    rows = []
    for name, s in SCENARIOS.items():
        sc = inp.with_(
            growth_high=inp.growth_high + s["growth"],
            op_margin_target=max(inp.op_margin_target + s["margin"], 0.0),
            wacc=inp.wacc + s["wacc"],
            cost_of_equity=inp.cost_of_equity + s["wacc"],
            terminal_growth=inp.terminal_growth + s["g"],
        )
        rows.append({
            "Scenario": name,
            "Revenue growth": sc.growth_high,
            "Target margin": sc.op_margin_target,
            "WACC": sc.wacc,
            "Terminal g": sc.terminal_growth,
            "FCFF value": run_fcff(sc, bridge).per_share,
            "FCFE value": run_fcfe(sc, bridge).per_share,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
@dataclass
class MonteCarloResult:
    values: np.ndarray
    p10: float
    p50: float
    p90: float
    mean: float
    prob_undervalued: float   # P(value > price)
    n_valid: int


def monte_carlo(inp: DCFInputs, bridge: Bridge, price: float, n: int = 5000,
                seed: int = 42) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    growth = rng.normal(inp.growth_high, 0.3 * abs(inp.growth_high) + 0.02, n)
    margin = np.clip(rng.normal(inp.op_margin_target, 0.15 * abs(inp.op_margin_target) + 0.01, n),
                     0.0, 0.9)
    wacc = np.clip(rng.normal(inp.wacc, 0.01, n), 0.05, 0.25)
    g = rng.uniform(inp.terminal_growth - 0.01, inp.terminal_growth + 0.01, n)
    g = np.minimum(g, wacc - 0.01)
    s2c = inp.sales_to_capital * rng.lognormal(0.0, 0.2, n)

    vals = fcff_per_share(inp, bridge, growth_high=growth, op_margin_target=margin, wacc=wacc,
                          terminal_growth=g, sales_to_capital=s2c)["per_share"]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return MonteCarloResult(vals, np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    return MonteCarloResult(
        values=vals,
        p10=float(np.percentile(vals, 10)), p50=float(np.percentile(vals, 50)),
        p90=float(np.percentile(vals, 90)), mean=float(vals.mean()),
        prob_undervalued=float((vals > price).mean()), n_valid=len(vals),
    )


# ---------------------------------------------------------------------------
# Reverse DCF
# ---------------------------------------------------------------------------
def implied_growth(inp: DCFInputs, bridge: Bridge, price: float,
                   lo: float = -0.20, hi: float = 0.60) -> float | None:
    """Solve for the high-growth rate that makes FCFF value equal the price."""
    f = lambda x: fcff_per_share(inp, bridge, growth_high=x)["per_share"][0] - price  # noqa: E731
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-6 * max(price, 1):
            return mid
        if flo * fm < 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)
