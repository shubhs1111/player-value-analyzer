"""
Turn the current "Scouting Register" dashboard into one self-contained .html file.

Unlike the earlier draft (build_standalone.py), this page is already fully
self-styled (the `.sr` wrapper carries its own dark theme, fonts and colours
as plain CSS variables) - it does not lean on the Appifact design-system
tokens at all. So this script only needs to supply the one thing the page
really depends on: a `dash` object that serves the data. Data is embedded as
JSON so the result opens from disk with no server, no network and no
dependencies.

Output: dist/scouting-register.html
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
SRC = ROOT / "dashboard_v2" / "index_new.html"
DATA_DIR = ROOT / "dashboard_v2"
OUT = ROOT / "dist" / "scouting-register.html"

DATASETS = ["players", "accuracy", "examples", "worked_example"]
# the page holds many seasons; a file people email around does not need all of them
KEEP_SEASONS = 6

PAGE_SHELL_CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; }
body { background: #0e1012; }
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
<style>{PAGE_SHELL_CSS}
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
