#!/usr/bin/env python3
"""Run build_aapl_mockup.py with monkey-patches to capture each Plotly figure
and section HTML separately, then dump to a single ticker_data/<TICKER>.json
consumed by ticker.html shell.

This is the pragmatic bridge that keeps build_aapl_mockup.py as the single
source of truth for data + chart construction, while the shell template
becomes a deploy-once asset.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = ROOT / "ticker_data"
OUT_DIR.mkdir(exist_ok=True)


def patched_build(ticker: str, cik: str) -> dict:
    """Load build_aapl_mockup as a module, monkey-patch fig.to_html to capture
    figures, monkey-patch the output writer to capture HTML, run main().
    """
    # Read source and rewrite the AAPL-specific identifiers
    src = (HERE / "build_aapl_mockup.py").read_text(encoding="utf-8")
    src = src.replace('CIK = "0000320193"', f'CIK = "{cik}"')
    src = src.replace('TICKER = "AAPL"', f'TICKER = "{ticker.upper()}"')
    temp_html_path = ROOT / f"_{ticker.lower()}_temp.html"
    src = src.replace(
        'OUT_FILE = Path(__file__).resolve().parents[1] / "aapl_mockup.html"',
        f'OUT_FILE = Path(r"{temp_html_path}")',
    )
    # Map AAPL-specific chart IDs to the generic IDs the shell expects.
    # The base mockup uses aapl-X-chart; the shell uses chart-X. Also rename
    # the bare "aapl-chart" (main financial chart) to "chart-financial".
    chart_id_map = {
        "aapl-chart":                "chart-financial",
        "aapl-forecast-chart":       "chart-forecast",
        "aapl-yoy-chart":            "chart-yoy",
        "aapl-expenses-chart":       "chart-expenses",
        "aapl-segments-prod-chart":  "chart-segments-prod",
        "aapl-segments-geo-chart":   "chart-segments-geo",
        "aapl-fwd-est-chart":        "chart-fwd-est",
        "aapl-balance-chart":        "chart-balance",
        "aapl-epsdiv-chart":         "chart-epsdiv",
        "aapl-valuation-chart":      "chart-valuation",
        "aapl-impliedgrowth-chart":  "chart-impliedgrowth",
    }
    # Order by length descending so longer matches replace first
    for old, new in sorted(chart_id_map.items(), key=lambda kv: -len(kv[0])):
        src = src.replace(old, new)
    src = src.replace('"Apple Inc."', f'"{ticker.upper()}"')
    # (Segments are now gated on TICKER=="AAPL" inside build_aapl_mockup.py, so
    # no per-ticker source rewrite is needed to suppress AAPL's hardcoded data.)

    tmp = pathlib.Path(f"/tmp/build_{ticker.lower()}_json.py")
    tmp.write_text(src, encoding="utf-8")

    # Monkey-patch BaseFigure.to_html (where to_html is actually defined)
    import plotly.basedatatypes as bd
    captures = {"figures": {}, "div_order": []}
    orig = bd.BaseFigure.to_html

    def patched_to_html(self, *args, **kwargs):
        div_id = kwargs.get("div_id") or "chart-unknown"
        try:
            captures["figures"][div_id] = self.to_dict()
            captures["div_order"].append(div_id)
        except Exception as e:
            print(f"[capture] failed for {div_id}: {e}")
        return orig(self, *args, **kwargs)

    bd.BaseFigure.to_html = patched_to_html

    # Use runpy so the module runs as __main__ (which is what triggers main())
    import runpy
    try:
        runpy.run_path(str(tmp), run_name="__main__")
    except SystemExit:
        pass  # main() calls sys.exit(0) which raises SystemExit
    finally:
        bd.BaseFigure.to_html = orig

    # Read the rendered HTML before deleting the temp file, so the caller
    # can extract section HTML strings without depending on a legacy
    # per-ticker file in ROOT.
    try:
        captures["rendered_html"] = temp_html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        captures["rendered_html"] = ""
    try:
        temp_html_path.unlink(missing_ok=True)
    except Exception:
        pass

    return captures


def extract_sections(html: str) -> dict:
    """Pull section HTML strings out of the rendered page so the shell can
    populate placeholders without re-rendering.
    """
    out = {}

    # Each <div class="data-table"><h2>TITLE</h2>...</div> is a section.
    # We extract the *inner* content (after the title strip) up to the matching
    # </div>. The mockup nests divs, so use a balanced-bracket extraction.
    def section_after_h2(title: str) -> str | None:
        # Find the section block by header
        pat = re.compile(
            r'<div class="data-table[^"]*"[^>]*><h2[^>]*>' +
            re.escape(title) + r'</h2>(.*?)</div>\s*(?=<div class="data-table"|$)',
            re.DOTALL,
        )
        m = pat.search(html)
        return m.group(1) if m else None

    # Pull out the body of each section
    titles = {
        "headlines": "Latest Headlines",
        "financial_body": "Revenue, Net Income, EPS",
        "forecast_body": "Forecasts",
        "yoy_body": "Y/Y % Change",
        "expenses_body": "Expenses",
        "segments_body": "Segment Performance",
        "fwdest_body": "Forward Estimate History",
        "balance_body": "Balance Sheet",
        "epsdiv_body": "EPS &amp; Dividend",
        "valuation_body": "Valuation",
        "impliedgrowth_body": "Implied Growth",
    }
    for k, t in titles.items():
        s = section_after_h2(t)
        if s is not None:
            out[k] = s.strip()
    return out


def split_section_body(body: str) -> dict:
    """Inside a section body extract the legend strip, the chart div script
    output, the table HTML(s), and the trailing footnote so the shell can put
    each into its own slot. The chart div is replaced by the shell's empty
    container, so we just need to drop the embedded Plotly script (it'll be
    re-run from JSON)."""
    out = {}

    # Footnote
    m = re.search(r'<p class="m-footnote"[^>]*>(.*?)</p>', body, re.DOTALL)
    if m:
        out["footnote"] = m.group(1).strip()

    # Legend strip
    m = re.search(r'(<div class="(?:seg|yoy)-strip[^>]*>.*?</div>)', body, re.DOTALL)
    if m:
        out["legend"] = m.group(1)

    # All tables — we keep raw HTML strings
    tables = re.findall(r'<table class="ft[^"]*"[^>]*>.*?</table>', body, re.DOTALL)
    # Also expand-row buttons (collapsible "Show all N years")
    expand = re.search(r'<div class="expand-row">.*?</div>', body, re.DOTALL)
    if tables:
        out["tables"] = tables
        if expand:
            out["expand"] = expand.group(0)
    return out


def build_metrics_from_html(html: str) -> list[dict]:
    """Extract the 8 .m metric cells from the rendered HTML."""
    out = []
    for m in re.finditer(
        r'<div class="m"><div class="m-lbl">(.*?)</div>'
        r'<div class="m-val">(.*?)</div></div>',
        html, re.DOTALL,
    ):
        label = m.group(1).strip()
        val_raw = m.group(2).strip()
        # Pull off optional <span class="m-sub">...</span>
        sub_m = re.search(r'<span class="m-sub">(.*?)</span>', val_raw)
        if sub_m:
            sub = sub_m.group(1)
            value = val_raw.replace(sub_m.group(0), "").strip()
        else:
            sub = ""
            value = val_raw
        out.append({"label": label, "value": value, "sub": sub})
    return out


def build_headlines_from_html(html: str) -> list[dict]:
    out = []
    for m in re.finditer(
        r'<a class="hl" href="([^"]+)"[^>]*>'
        r'<div class="hl-title">(.*?)</div>'
        r'<div class="hl-meta">(.*?)</div></a>',
        html, re.DOTALL,
    ):
        url, title, meta = m.group(1), m.group(2).strip(), m.group(3).strip()
        # meta is "Source • 3h ago"
        parts = meta.rsplit(" • ", 1)
        src = parts[0] if parts else meta
        time_ago = parts[1] if len(parts) > 1 else ""
        out.append({"url": url, "title": title, "src": src, "time_ago": time_ago})
    return out


def build_data(ticker: str, cik: str, name: str = "") -> dict:
    captures = patched_build(ticker, cik)

    html = captures.get("rendered_html") or ""
    if not html:
        # Fallback: read AAPL's persisted mockup (only AAPL writes one).
        html_path = ROOT / "aapl_mockup.html"
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8")
        else:
            raise SystemExit("no rendered HTML found")

    sections = extract_sections(html)
    metrics = build_metrics_from_html(html)
    headlines = build_headlines_from_html(html)

    # Build the html map the shell expects
    html_map: dict[str, str] = {}
    if "headlines" in sections:
        # headlines already rendered as <div class="hl-list">...</div>
        # we pass through structured array above; keep flag only
        pass

    # Financial body — extract just the inner table HTML
    fin = split_section_body(sections.get("financial_body", ""))
    if fin.get("tables"):
        html_map["financial"] = (
            '<div style="overflow-x:auto">' + fin["tables"][0] + "</div>"
            + (fin.get("expand", ""))
        )

    fc = split_section_body(sections.get("forecast_body", ""))
    if fc.get("tables"):
        html_map["forecast"] = '<div style="overflow-x:auto">' + fc["tables"][0] + "</div>"
    if fc.get("footnote"):
        html_map["forecast_footnote"] = fc["footnote"]

    yoy = split_section_body(sections.get("yoy_body", ""))
    if yoy.get("legend"):
        html_map["yoy_legend"] = yoy["legend"]
    if yoy.get("tables"):
        html_map["yoy_table"] = '<div style="overflow-x:auto">' + yoy["tables"][0] + "</div>"

    exp = split_section_body(sections.get("expenses_body", ""))
    if exp.get("tables"):
        html_map["expenses_abs"] = '<div style="overflow-x:auto">' + exp["tables"][0] + "</div>"
        if len(exp["tables"]) > 1:
            html_map["expenses_chg"] = '<div style="overflow-x:auto">' + exp["tables"][1] + "</div>"

    # Segments — there are 2 legends and 2 tables
    seg_body = sections.get("segments_body", "")
    seg_legends = re.findall(r'<div class="seg-strip[^>]*>.*?</div>', seg_body, re.DOTALL)
    seg_tables  = re.findall(r'<table class="ft[^"]*"[^>]*>.*?</table>', seg_body, re.DOTALL)
    if len(seg_legends) >= 1: html_map["prod_legend"] = seg_legends[0]
    if len(seg_legends) >= 2: html_map["geo_legend"]  = seg_legends[1]
    if len(seg_tables) >= 1:  html_map["prod_table"] = '<div style="overflow-x:auto">' + seg_tables[0] + "</div>"
    if len(seg_tables) >= 2:  html_map["geo_table"]  = '<div style="overflow-x:auto">' + seg_tables[1] + "</div>"
    seg_footnote = re.search(r'<p class="m-footnote">(.*?)</p>', seg_body, re.DOTALL)
    if seg_footnote: html_map["segments_footnote"] = seg_footnote.group(1).strip()

    fwd = split_section_body(sections.get("fwdest_body", ""))
    if fwd.get("legend"): html_map["fwdest_legend"] = fwd["legend"]
    if fwd.get("tables"): html_map["fwdest_table"] = '<div style="overflow-x:auto">' + fwd["tables"][0] + "</div>"

    bs = split_section_body(sections.get("balance_body", ""))
    if bs.get("legend"): html_map["balance_legend"] = bs["legend"]
    if bs.get("tables"): html_map["balance_table"] = '<div style="overflow-x:auto">' + bs["tables"][0] + "</div>"

    epsdiv = split_section_body(sections.get("epsdiv_body", ""))
    if epsdiv.get("legend"): html_map["epsdiv_legend"] = epsdiv["legend"]
    if epsdiv.get("footnote"): html_map["epsdiv_footnote"] = epsdiv["footnote"]

    val = split_section_body(sections.get("valuation_body", ""))
    if val.get("legend"): html_map["valuation_legend"] = val["legend"]
    if val.get("tables"):
        html_map["valuation_assumptions"] = '<div style="overflow-x:auto">' + val["tables"][0] + "</div>"
        if len(val["tables"]) > 1:
            html_map["valuation_table"] = '<div style="overflow-x:auto">' + val["tables"][1] + "</div>"

    ig = split_section_body(sections.get("impliedgrowth_body", ""))
    if ig.get("legend"): html_map["impliedgrowth_legend"] = ig["legend"]
    if ig.get("tables"): html_map["impliedgrowth_table"] = '<div style="overflow-x:auto">' + ig["tables"][0] + "</div>"

    # Financial range button metadata — pull from JS in the rendered HTML
    fmeta = {}
    for key, var in [
        ("n", r"var N\s*=\s*(\d+)"),
        ("rev_labels_full",  r"var REV_FULL\s*=\s*(\[[^\]]+\])"),
        ("rev_labels_short", r"var REV_SHORT\s*=\s*(\[[^\]]+\])"),
        ("eps_labels_full",  r"var EPS_FULL\s*=\s*(\[[^\]]+\])"),
        ("eps_labels_short", r"var EPS_SHORT\s*=\s*(\[[^\]]+\])"),
    ]:
        m = re.search(var, html)
        if m:
            v = m.group(1)
            fmeta[key] = int(v) if key == "n" else json.loads(v)

    # Metrics footnote
    m = re.search(r'<p class="m-footnote">(.*?)</p>', html, re.DOTALL)
    metrics_footnote = m.group(1).strip() if m else ""

    return {
        "ticker": ticker.upper(),
        "name": name or ticker.upper(),
        "metrics": metrics,
        "metrics_footnote": metrics_footnote,
        "headlines": headlines,
        "html": html_map,
        "figures": captures["figures"],
        "financial_meta": fmeta,
    }


# ------------------------------------------------------------ cik lookup --
def cik_for(ticker: str) -> str | None:
    import requests
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "Nick Daly ndaly111@gmail.com"},
        timeout=30,
    )
    r.raise_for_status()
    for entry in r.json().values():
        if entry.get("ticker", "").upper() == ticker.upper():
            return f"{int(entry['cik_str']):010d}"
    return None


def load_cik_map() -> dict:
    """Download SEC's company_tickers.json once and return {TICKER: cik}.

    cik_for() previously re-downloaded this ~1MB file for every ticker; doing it
    once removes ~N redundant SEC round-trips and avoids rate-limiting when the
    builds run in parallel.
    """
    import requests
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "Nick Daly ndaly111@gmail.com"},
        timeout=30,
    )
    r.raise_for_status()
    out = {}
    for entry in r.json().values():
        t = str(entry.get("ticker", "")).upper()
        if t:
            out[t] = f"{int(entry['cik_str']):010d}"
    return out


def _build_one(args):
    """Build + write one ticker's JSON. Runs in a worker process; returns a
    (ticker, size_or_None, error_or_None) tuple. Self-contained error handling
    so one bad ticker can never abort the batch."""
    tk, cik = args
    try:
        if not cik:
            return (tk, None, "no CIK")
        d = build_data(tk, cik)
        out_path = OUT_DIR / f"{tk.upper()}.json"
        out_path.write_text(json.dumps(d, separators=(",", ":"), default=str), encoding="utf-8")
        return (tk, out_path.stat().st_size, None)
    except Exception as e:  # noqa: BLE001 - isolate per-ticker failures
        import traceback
        traceback.print_exc()
        return (tk, None, str(e))


def main():
    tickers = sys.argv[1:] or ["AAPL"]

    # Resolve CIKs once (single SEC download) instead of per ticker.
    try:
        cik_map = load_cik_map()
    except Exception as e:  # noqa: BLE001
        print(f"[gen_ticker_json] CIK map load failed ({e}); per-ticker fallback")
        cik_map = None
    work = [(tk, (cik_map.get(tk.upper()) if cik_map is not None else cik_for(tk)))
            for tk in tickers]

    def run_sequential():
        n = 0
        for w in work:
            tk, size, err = _build_one(w)
            if err:
                print(f"[{tk}] FAILED: {err}")
            else:
                n += 1
                print(f"[{tk}] wrote {size:,} bytes")
        return n

    # Each ticker build is independent and network-bound, so fan out across
    # processes (the build monkey-patches a global Plotly method, so processes —
    # not threads — keep them isolated). GEN_WORKERS=1 forces sequential.
    workers = int(os.environ.get("GEN_WORKERS", "4") or "4")
    ok = 0
    if workers > 1 and len(work) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_build_one, w) for w in work]
                for fut in as_completed(futs):
                    tk, size, err = fut.result()
                    if err:
                        print(f"[{tk}] FAILED: {err}")
                    else:
                        ok += 1
                        print(f"[{tk}] wrote {size:,} bytes")
        except Exception as e:  # noqa: BLE001 - never leave pages unbuilt
            print(f"[gen_ticker_json] parallel run failed ({e}); retrying sequentially")
            ok = run_sequential()
    else:
        ok = run_sequential()

    print(f"[gen_ticker_json] {ok}/{len(work)} tickers built "
          f"({workers} workers)")


if __name__ == "__main__":
    main()
