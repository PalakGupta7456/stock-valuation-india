"""Render docs/banner.png, the README header image.

Needs the Instrument Serif / Instrument Sans woff2 files (from @fontsource on npm)
and Playwright:  python scripts/make_banner.py <fonts_dir>
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROWS = [("DCF · FCFF", .30, .52, .40, "#0E5160"), ("DCF · FCFE", .32, .50, .41, "#0E5160"),
        ("Residual income", .28, .49, .38, "#0E5160"), ("Dividend discount", .34, .44, .39, "#0E5160"),
        ("Earnings power", .22, .27, .245, "#0E5160"), ("Peer P/E", .58, .80, .68, "#4F5BA6"),
        ("Peer EV/EBITDA", .56, .76, .66, "#4F5BA6"), ("Peer P/B", .44, .62, .52, "#4F5BA6"),
        ("Own 5y P/E", .66, .92, .82, "#4F5BA6")]


def build(fonts: Path) -> str:
    def b64(name: str) -> str:
        return base64.b64encode(next(fonts.rglob(name)).read_bytes()).decode()

    serif, sans, sans6 = (b64("instrument-serif-latin-400-normal.woff2"),
                          b64("instrument-sans-latin-400-normal.woff2"),
                          b64("instrument-sans-latin-600-normal.woff2"))
    W, x0, step = 330, 124, 31
    h = len(ROWS) * step
    svg = [f'<rect x="{x0 + .38 * W}" y="8" width="{.26 * W}" height="{h + 2}" fill="#0E5160" opacity=".06"/>']
    for i, (lab, lo, hi, v, c) in enumerate(ROWS):
        y = 24 + i * step
        svg += [f'<text x="{x0 - 12}" y="{y + 4}" text-anchor="end" class="lab">{lab}</text>',
                f'<line x1="{x0}" x2="{x0 + W}" y1="{y}" y2="{y}" stroke="#C9D3CD"/>',
                f'<rect x="{x0 + lo * W}" y="{y - 9}" width="{(hi - lo) * W}" height="18" rx="3" fill="{c}" opacity=".22"/>',
                f'<circle cx="{x0 + v * W}" cy="{y}" r="6.5" fill="{c}"/>']
    px, fv = x0 + .55 * W, x0 + .47 * W
    svg += [f'<line x1="{fv}" x2="{fv}" y1="8" y2="{h + 14}" stroke="#18262B" stroke-dasharray="3 4" stroke-width="1.5"/>',
            f'<line x1="{px}" x2="{px}" y1="4" y2="{h + 14}" stroke="#D98E1F" stroke-width="3"/>',
            f'<text x="{px + 7}" y="{h + 13}" class="price">Market price</text>',
            f'<text x="{fv - 7}" y="{h + 13}" text-anchor="end" class="fv">Fair value</text>']
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
@font-face{{font-family:IS;src:url(data:font/woff2;base64,{serif})}}
@font-face{{font-family:IN;src:url(data:font/woff2;base64,{sans});font-weight:400}}
@font-face{{font-family:IN;src:url(data:font/woff2;base64,{sans6});font-weight:600}}
html,body{{margin:0}} body{{width:1280px;height:440px;background:#F2F4F0;font-family:IN;color:#18262B;position:relative;overflow:hidden}}
.l{{position:absolute;left:64px;top:54px;width:600px}}
.brand{{font-family:IS;color:#0E5160;font-size:30px}}
.brand span{{font-family:IN;font-size:13px;color:#5D6B70;margin-left:10px;letter-spacing:.06em;text-transform:uppercase}}
h1{{font-family:IS;font-weight:400;font-size:60px;line-height:1.0;margin:20px 0 18px;letter-spacing:-.01em}}
h1 em{{color:#D98E1F;font-style:normal}}
p{{font-size:18px;color:#5D6B70;margin:0 0 24px;line-height:1.45;max-width:540px}}
.chips span{{display:inline-block;border:1px solid #C9D3CD;background:#fff;border-radius:999px;padding:5px 12px;margin:0 6px 8px 0;font-size:14px;font-weight:600;color:#0E5160}}
.r{{position:absolute;right:56px;top:62px;background:#fff;border-radius:8px;padding:16px 20px 10px;box-shadow:0 1px 0 #C9D3CD}}
.lab{{font:12.5px IN;fill:#5D6B70}} .price{{font:600 12.5px IN;fill:#D98E1F}} .fv{{font:12.5px IN;fill:#18262B}}
.cap{{font-size:11.5px;color:#5D6B70;margin-top:2px}}
.bar{{position:absolute;left:0;right:0;bottom:0;height:6px;background:linear-gradient(90deg,#0E5160 0 60%,#4F5BA6 60% 88%,#D98E1F 88%)}}
</style></head><body>
<div class="l"><div class="brand">Intrinsic<span>NSE &amp; BSE valuation</span></div>
<h1>What is an Indian stock<br><em>actually</em> worth?</h1>
<p>Type any ticker. Twelve valuation methods, live market data, one fair value and a Buy / Hold / Sell call you can audit.</p>
<div class="chips"><span>DCF</span><span>Residual income</span><span>Dividend discount</span><span>Peer multiples</span><span>Monte Carlo</span><span>Reverse DCF</span></div></div>
<div class="r"><svg width="{x0 + W + 14}" height="{h + 22}">{"".join(svg)}</svg><div class="cap">Illustrative · every method on one chart</div></div>
<div class="bar"></div></body></html>'''


def main(fonts_dir: str) -> None:
    from playwright.sync_api import sync_playwright

    html = ROOT / "docs" / "_banner.html"
    html.write_text(build(Path(fonts_dir)), encoding="utf-8")
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        p = b.new_page(viewport={"width": 1280, "height": 440}, device_scale_factor=2)
        p.goto(html.as_uri())
        p.wait_for_timeout(800)
        p.screenshot(path=str(ROOT / "docs" / "banner.png"))
        b.close()
    html.unlink()


if __name__ == "__main__":
    main(sys.argv[1])
