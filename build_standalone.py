"""
Turn the published dashboard into one self-contained .html file.

The hosted version leans on two things the page does not carry itself: the design
tokens (fonts, colours, spacing) and a `dash` object that serves the data. This
script supplies both - tokens as plain CSS, data embedded as JSON - so the result
opens from disk with no server, no network and no dependencies.

Output: dist/player-value-analyzer.html
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

SRC = Path("dashboard_v2/index.html")
DATA_DIR = Path("dashboard_v2")
OUT = Path("dist/player-value-analyzer.html")

DATASETS = ["players", "accuracy", "examples", "worked_example"]
# the page holds eleven seasons; a file people email around does not need all of them
KEEP_SEASONS = 6

TOKENS = """
:root {
  color-scheme: light;
  --font-anthropic-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-anthropic-serif: ui-serif, Georgia, Cambria, "Times New Roman", serif;

  --color-bg: #faf9f7;
  --color-panel: #ffffff;
  --color-fg: #1f1e1c;
  --color-fg-muted: #6b6760;
  --color-border-line: #e4e0d9;
  --color-ok: #117a56;
  --color-warn: #a86613;
  --color-bad: #b4332b;

  --cds-radius: 10px;
  --cds-pad-xs: 4px;  --cds-pad-sm: 8px;  --cds-pad-md: 12px;
  --cds-pad-lg: 18px; --cds-pad-xl: 26px;
  --cds-gap-xs: 4px;  --cds-gap-sm: 8px;  --cds-gap-md: 14px;
  --cds-gap-lg: 22px; --cds-gap-xl: 34px;

  --cds-font-size-caption: 0.8125rem;
  --cds-font-size-body: 0.9375rem;
  --cds-font-size-heading: 1.125rem;
  --cds-font-size-title: 1.75rem;
  --cds-font-weight-medium: 560;

  --cds-chart-grid: #ebe7e0;
  --cds-chart-axis: #c9c3b8;
  --cds-chart-reference: #a8a195;
  --cds-chart-muted: #b9b2a6;

  --cds-surface-popover: #ffffff;
  --cds-border: #ddd8cf;
  --cds-shadow-popover: 0 8px 24px rgba(31, 30, 28, .14);
  --cds-dur-slow: .25s;
  --cds-ease-out: cubic-bezier(.2, .8, .3, 1);
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --color-bg: #191817;
    --color-panel: #221f1e;
    --color-fg: #f2efe9;
    --color-fg-muted: #a39d94;
    --color-border-line: #38332f;
    --color-ok: #4ec08f;
    --color-warn: #d99a3f;
    --color-bad: #e8695e;
    --cds-chart-grid: #302c29;
    --cds-chart-axis: #4a443f;
    --cds-chart-reference: #6a635b;
    --cds-chart-muted: #595249;
    --cds-surface-popover: #262321;
    --cds-border: #413b36;
    --cds-shadow-popover: 0 8px 24px rgba(0, 0, 0, .5);
  }
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --color-bg: #191817;
  --color-panel: #221f1e;
  --color-fg: #f2efe9;
  --color-fg-muted: #a39d94;
  --color-border-line: #38332f;
  --color-ok: #4ec08f;
  --color-warn: #d99a3f;
  --color-bad: #e8695e;
  --cds-chart-grid: #302c29;
  --cds-chart-axis: #4a443f;
  --cds-chart-reference: #6a635b;
  --cds-chart-muted: #595249;
  --cds-surface-popover: #262321;
  --cds-border: #413b36;
  --cds-shadow-popover: 0 8px 24px rgba(0, 0, 0, .5);
}

* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--color-bg);
  color: var(--color-fg);
  font-family: var(--font-anthropic-sans);
  font-size: var(--cds-font-size-body);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
#dash-root { max-width: 1100px; margin-inline: auto; padding: 28px 16px 64px; }
img { max-width: 100%; }
[hidden] { display: none !important; }
"""

UNPACK = """
(() => {
  const packed = JSON.parse(document.getElementById('dash-data').textContent);
  const out = {};
  for (const [name, pack] of Object.entries(packed)) {
    out[name] = pack.rows.map((row) => {
      const obj = {};
      pack.columns.forEach((c, i) => { obj[c] = row[i]; });
      return obj;
    });
  }
  window.__DASH_DATA__ = out;
})();
"""

# the slice of the hosted `dash` API this page actually uses
SHIM = """
(() => {
  const DATA = window.__DASH_DATA__ || {};
  const subs = [];
  let params = {};
  const notify = () => subs.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
  window.dash = {
    data: (id) => DATA[id]
      ? { status: 'ok', data: DATA[id], columns: Object.keys(DATA[id][0] || {}), meta: [], refreshing: false }
      : { status: 'missing', data: [], columns: [], meta: [], refreshing: false },
    onData: (fn) => { subs.push(fn); fn(); },
    params: () => ({ ...params }),
    setParams: (next) => { params = { ...params, ...next }; notify(); },
    resetParams: () => { params = {}; notify(); },
    setLink: () => {},
    refresh: () => {},
    calc: () => {},
    loader: () => {},
    colors: ['#3f7fd4', '#7a5cc6', '#2f9e74', '#c9972f', '#cf5a5a', '#4aa3b8', '#9b6a3f', '#8a8f98'],
  };
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(notify, 150);
  });
})();
"""


def load_dataset(name: str) -> dict:
    """Columns once, rows as plain arrays - repeating 30 key names 14,000 times triples the file."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    if name == "players" and "season" in df.columns:
        seasons = sorted(df["season"].unique())[-KEEP_SEASONS:]
        df = df[df["season"].isin(seasons)]
    # NaN is not valid JSON, and it survives to_numpy() even after a where() on the frame
    rows = [[None if isinstance(v, float) and math.isnan(v) else v for v in row]
            for row in df.to_numpy().tolist()]
    return {"columns": list(df.columns), "rows": rows}


def main() -> None:
    page = SRC.read_text(encoding="utf-8")
    style = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    body = page[page.index("</style>") + len("</style>"):page.index("<script>")].strip()

    data = {name: load_dataset(name) for name in DATASETS}
    rows = sum(len(v["rows"]) for v in data.values())
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>What a Footballer Is Worth</title>
<style>{TOKENS}
{style}</style>
</head>
<body>
<div id="dash-root">
{body}
</div>
<script id="dash-data" type="application/json">{payload}</script>
<script>{UNPACK}</script>
<script>{SHIM}</script>
<script>{script}</script>
</body>
</html>
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    size = OUT.stat().st_size / 1_048_576
    print(f"wrote {OUT}  ({size:.1f} MB, {rows:,} rows embedded)")
    for name, v in data.items():
        print(f"  {name:<16}{len(v['rows']):>7,} rows")


if __name__ == "__main__":
    main()
