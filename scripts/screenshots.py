"""Capture README screenshots of the running app with real stocks.

Used by .github/workflows/screenshots.yml:
    streamlit run app.py --server.port 8501 &
    python scripts/screenshots.py TCS HDFCBANK
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
URL = "http://localhost:8501"
TABS = {"All methods": "methods", "DCF": "dcf", "Relative": "relative",
        "Residual income & dividends": "residual", "Stress tests": "stress", "Data checks": "checks"}


def wait_loaded(page) -> None:
    page.get_by_role("tab", name="All methods", exact=True).wait_for(timeout=180_000)
    page.wait_for_timeout(4000)


def shoot(page, ticker: str, prefix: str, full: bool) -> None:
    page.goto(f"{URL}/?ticker={ticker}", wait_until="networkidle")
    wait_loaded(page)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / f"{prefix}overview.png"))
    if not full:
        return
    page.set_viewport_size({"width": 1440, "height": 2600})
    page.wait_for_timeout(2000)
    for tab, name in TABS.items():
        page.get_by_role("tab", name=tab, exact=True).click()
        page.wait_for_timeout(3500)
        box = page.get_by_role("tablist").first.bounding_box()
        page.screenshot(path=str(OUT / f"{prefix}{name}.png"),
                        clip={"x": 300, "y": box["y"] - 10, "width": 1140,
                              "height": min(2590 - box["y"], 1700)})


def main(tickers: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(4000)
        page.screenshot(path=str(OUT / "landing.png"))
        for i, t in enumerate(tickers):
            # first ticker gets the full tab set; others get an overview shot each
            shoot(page, t, "" if i == 0 else f"{t.lower()}_", full=(i == 0))
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1:] or ["TCS", "HDFCBANK"])
