"""Export a valuation report to a multi-sheet Excel workbook."""
from __future__ import annotations

import io

import pandas as pd

from . import config
from .model import ValuationReport


def to_excel(rep: ValuationReport) -> bytes:
    c, v, w, i = rep.company, rep.verdict, rep.wacc, rep.inputs
    summary = pd.DataFrame([
        ("Company", c.name), ("Ticker", c.ticker), ("Sector", c.sector),
        ("Market price (₹)", c.price),
        ("FCFF value per share (₹)", rep.fcff.per_share),
        ("FCFE value per share (₹)", rep.fcfe.per_share),
        ("Blended fair value (₹)", v.fair_value),
        ("Fair value range, P25–P75 across methods (₹)", f"{v.range_low:,.0f} – {v.range_high:,.0f}"),
        ("Methods above market price", f"{v.methods_above_price} of {v.methods_valid}"),
        ("Analyst mean target (₹)", c.analyst_target_mean),
        ("Upside / (downside)", v.upside),
        ("Rating", v.rating), ("Valuation", v.valuation_label),
        ("Confidence (0-100)", v.confidence),
        ("Buy-below price (₹)", v.buy_below),
        ("Market-implied 5y growth", rep.implied_growth),
        ("Monte Carlo P(value > price)", rep.monte_carlo.prob_undervalued),
        ("Generated", c.fetched_at.strftime("%Y-%m-%d %H:%M")),
    ], columns=["Item", "Value"])

    assumptions = pd.DataFrame([
        ("Risk-free rate", w.risk_free), ("Equity risk premium", w.erp), ("Beta", w.beta),
        ("Cost of equity", w.cost_of_equity), ("Synthetic rating", w.rating),
        ("Pre-tax cost of debt", w.pre_tax_cost_of_debt), ("Weight of debt", w.weight_debt),
        ("WACC", w.wacc), ("Revenue growth (yrs 1-5)", i.growth_high),
        ("Terminal growth", i.terminal_growth), ("Current EBIT margin", i.op_margin_current),
        ("Target EBIT margin", i.op_margin_target), ("Effective tax rate", i.tax_rate),
        ("Marginal tax rate", i.marginal_tax), ("Sales-to-capital", i.sales_to_capital),
        ("Terminal ROIC", i.terminal_roic), ("Mid-year convention", i.mid_year),
    ], columns=["Assumption", "Value"])

    def crore(df: pd.DataFrame, money_cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in money_cols:
            out[col] = out[col] / config.CRORE
        return out

    fcff = crore(rep.fcff.table, ["Revenue", "EBIT", "NOPAT", "Reinvestment", "FCFF", "PV of FCFF"])
    fcfe = crore(rep.fcfe.table, ["Revenue", "EBIT", "Interest", "Net income", "Reinvestment",
                                  "Net borrowing", "FCFE", "PV of FCFE"])
    sens = rep.sensitivity.copy()
    sens.index = [f"WACC {x:.2%}" for x in sens.index]
    sens.columns = [f"g {x:.2%}" for x in sens.columns]
    flags = pd.DataFrame([(f.level, f.title, f.detail) for f in rep.flags],
                         columns=["Level", "Check", "Detail"])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        assumptions.to_excel(xw, sheet_name="Assumptions", index=False)
        fcff.to_excel(xw, sheet_name="FCFF (₹ Cr)", index=False)
        fcfe.to_excel(xw, sheet_name="FCFE (₹ Cr)", index=False)
        methods = rep.method_table.copy()
        methods.to_excel(xw, sheet_name="All methods", index=False)
        ri = rep.method("Residual income")
        if ri.table is not None:
            ri.table.to_excel(xw, sheet_name="Residual income (₹ per sh)", index=False)
        ddm = rep.method("Dividend discount")
        if ddm.table is not None:
            ddm.table.to_excel(xw, sheet_name="DDM (₹ per share)", index=False)
        if not rep.peers.empty:
            rep.peers.to_excel(xw, sheet_name="Peers", index=False)
        rep.scenarios.to_excel(xw, sheet_name="Scenarios", index=False)
        sens.to_excel(xw, sheet_name="Sensitivity (₹ per share)")
        flags.to_excel(xw, sheet_name="Data checks", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(cell.value or "")) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 60)
    return buf.getvalue()
