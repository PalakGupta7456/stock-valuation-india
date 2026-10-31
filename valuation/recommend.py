"""Blend every valuation method into one fair value, verdict and confidence score."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .analysis import MonteCarloResult
from .methods import MethodResult
from .quality import Flag


@dataclass
class Verdict:
    fair_value: float
    price: float
    upside: float
    rating: str
    raw_rating: str
    valuation_label: str     # Undervalued / Fairly valued / Overvalued
    confidence: int
    confidence_label: str
    buy_below: float
    weights: dict            # normalised weight actually used per method
    reasons: list[str]
    range_low: float         # weighted 25th / 75th percentile across methods
    range_high: float
    methods_above_price: int
    methods_valid: int
    dispersion: float        # weighted coefficient of variation across methods

    @property
    def consensus(self) -> float:
        return self.methods_above_price / self.methods_valid if self.methods_valid else np.nan


_ORDER = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]


def rate(upside: float) -> str:
    for threshold, label in config.RATING_BANDS:
        if upside >= threshold:
            return label
    return "Strong Sell"


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), cum, v))


def method_table(methods: list[MethodResult], weights: dict, price: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "Method": m.name, "Family": m.family, "Value": m.value if m.valid else np.nan,
        "Low": m.low if m.valid else np.nan, "High": m.high if m.valid else np.nan,
        "Upside": (m.value / price - 1) if m.valid else np.nan,
        "Weight": weights.get(m.name, 0.0), "Applies": m.valid, "Note": m.note,
    } for m in methods])


def build_verdict(price: float, methods: list[MethodResult], raw_weights: dict,
                  mc: MonteCarloResult, flags: list[Flag], margin_of_safety: float = 0.15,
                  dcf_terminal_share: float = np.nan) -> Verdict:
    reasons: list[str] = []
    valid = [m for m in methods if m.valid]
    w = {m.name: float(raw_weights.get(m.name, 0)) for m in valid}
    total = sum(w.values())
    if total <= 0 and valid:   # every weighted model failed: fall back to an equal blend
        w = {m.name: 1.0 for m in valid if m.family != "Rule of thumb"} or {m.name: 1.0 for m in valid}
        total = sum(w.values())
        reasons.append("None of the preferred models applied, so the available ones are equally weighted.")
    weights = {k: v / total for k, v in w.items()} if total else {}

    used = [m for m in valid if weights.get(m.name, 0) > 0]
    if used:
        vals = np.array([m.value for m in used])
        wts = np.array([weights[m.name] for m in used])
        fair = float((vals * wts).sum())
        lo = _weighted_quantile(vals, wts, 0.25)
        hi = _weighted_quantile(vals, wts, 0.75)
        disp = float(np.sqrt((wts * (vals - fair) ** 2).sum()) / fair) if fair > 0 else np.nan
    else:
        fair = lo = hi = disp = float("nan")

    upside = fair / price - 1 if np.isfinite(fair) and fair > 0 else float("nan")
    raw = rate(upside) if np.isfinite(upside) else "No rating"
    above = sum(1 for m in valid if m.family != "Rule of thumb" and m.value > price)
    n_valid = sum(1 for m in valid if m.family != "Rule of thumb")

    # --- confidence -------------------------------------------------------
    score = 100 - sum(f.penalty for f in flags)
    if len(used) < 3:
        score -= 15
        reasons.append(f"Only {len(used)} weighted method(s) produced a value.")
    if np.isfinite(disp):
        if disp > 0.60:
            score -= 25
            reasons.append(f"Methods disagree sharply (spread ±{disp:.0%} around the blend).")
        elif disp > 0.35:
            score -= 12
            reasons.append(f"Methods disagree noticeably (spread ±{disp:.0%} around the blend).")
    fam = {}
    for f_name in ("Intrinsic", "Relative"):
        ms = [m for m in used if m.family == f_name]
        if ms:
            fw = np.array([weights[m.name] for m in ms])
            fam[f_name] = float((np.array([m.value for m in ms]) * fw).sum() / fw.sum())
    if len(fam) == 2:
        ratio = max(fam.values()) / min(fam.values())
        if ratio > 2.0:
            score -= 20
            reasons.append(f"Intrinsic methods (₹{fam['Intrinsic']:,.0f}) and market multiples "
                           f"(₹{fam['Relative']:,.0f}) differ by {ratio:.1f}×: either peers are richly "
                           "priced or the cash-flow assumptions are too conservative.")
        elif ratio > 1.5:
            score -= 10
            reasons.append(f"Intrinsic methods (₹{fam['Intrinsic']:,.0f}) and market multiples "
                           f"(₹{fam['Relative']:,.0f}) differ by {ratio:.1f}×.")
    if np.isfinite(upside) and n_valid:
        agree = above / n_valid if upside > 0 else 1 - above / n_valid
        if agree < 0.4:
            score -= 10
            reasons.append(f"The blend says {'cheap' if upside > 0 else 'expensive'}, but most "
                           "individual methods disagree.")
    if np.isfinite(dcf_terminal_share) and dcf_terminal_share > 0.80:
        score -= 8
        reasons.append(f"DCF terminal value is {dcf_terminal_share:.0%} of enterprise value; the result "
                       "hinges on long-run assumptions.")
    if np.isfinite(mc.prob_undervalued) and 0.4 <= mc.prob_undervalued <= 0.6:
        score -= 5
        reasons.append("DCF Monte Carlo is close to a coin flip on under- vs over-valuation.")
    score = int(np.clip(score, 0, 100))
    conf_label = "High" if score >= 70 else "Medium" if score >= 45 else "Low"

    # --- conviction adjustment: extreme calls need at least medium confidence
    rating = raw
    if raw in ("Strong Buy", "Strong Sell") and conf_label == "Low":
        rating = _ORDER[_ORDER.index(raw) + (-1 if raw == "Strong Buy" else 1)]
        reasons.append(f"Downgraded from {raw} because confidence is low.")
    if raw in ("Buy", "Sell") and score < 25:
        rating = "Hold"
        reasons.append(f"Downgraded from {raw} to Hold because confidence is very low.")

    if not np.isfinite(upside):
        label = "Not determinable"
    elif upside >= 0.15:
        label = "Undervalued"
    elif upside <= -0.10:
        label = "Overvalued"
    else:
        label = "Fairly valued"

    return Verdict(fair_value=fair, price=price, upside=upside, rating=rating, raw_rating=raw,
                   valuation_label=label, confidence=score, confidence_label=conf_label,
                   buy_below=fair * (1 - margin_of_safety) if np.isfinite(fair) else float("nan"),
                   weights=weights, reasons=reasons, range_low=lo, range_high=hi,
                   methods_above_price=above, methods_valid=n_valid, dispersion=disp)
