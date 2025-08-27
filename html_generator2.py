#!/usr/bin/env python3
# html_generator2.py — layout-safe: single-path table reader + sequential render + dashboard function
# --------------------------------------------------------------------------------------------------
from jinja2 import Environment, FileSystemLoader
import os, sqlite3, pandas as pd, yfinance as yf, re

DB_PATH = "Stock Data.db"
env = Environment(loader=FileSystemLoader("templates"))

# ─────────────────────────────────────────────────────────────────────────────
# Small, safe utilities (no template/CSS overrides)
# ─────────────────────────────────────────────────────────────────────────────
def get_file_or_placeholder(path: str, ph: str = "No data available") -> str:
    try:
        return open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return ph

def get_company_short_name(tk: str, cur) -> str:
    cur.execute("SELECT short_name FROM Tickers_Info WHERE ticker=?", (tk,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0]
    name = (yf.Ticker(tk).info or {}).get("shortName", "").strip() or tk
    cur.execute("UPDATE Tickers_Info SET short_name=? WHERE ticker=?", (name, tk))
    cur.connection.commit()
    return name

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-PATH table reader (matches generate_segment_tables.py)
# charts/{TICKER}_segments.html
# ─────────────────────────────────────────────────────────────────────────────
def get_segment_table_html(ticker: str, charts_dir_fs: str) -> str:
    path = f"{charts_dir_fs}/{ticker}_segments.html"
    try:
        return open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return f"No segment data available for {ticker}."

# ─────────────────────────────────────────────────────────────────────────────
# Carousels: gather all PNGs under charts/{TICKER}/ and render a single row
# ─────────────────────────────────────────────────────────────────────────────
_slug_pat = re.compile(r'^(?P<tkr>[A-Za-z0-9]+)_(?P<slug>[a-z0-9-]+)_.+\.png$', re.IGNORECASE)

def _collect_pngs(ticker: str, charts_dir_fs: str):
    seg_dir = os.path.join(charts_dir_fs, ticker)
    if not os.path.isdir(seg_dir):
        return []
    pngs = [f for f in sorted(os.listdir(seg_dir)) if f.lower().endswith(".png")]
    # keep only this ticker’s files (defensive if directory has leftovers)
    out = []
    for f in pngs:
        m = _slug_pat.match(f)
        if m and m.group("tkr").upper() == ticker.upper():
            out.append(f)
    return out or pngs  # fall back to all PNGs if slug pattern doesn’t match

def build_single_carousel_html(ticker: str, charts_dir_fs: str, charts_dir_web: str) -> str:
    files = _collect_pngs(ticker, charts_dir_fs)
    if not files:
        return ""
    items = [
        f'<div class="carousel-item"><img class="chart-img" src="{charts_dir_web}/{ticker}/{fn}" alt="{fn}"></div>'
        for fn in files
    ]
    # Use your page’s existing styling: .chart-img, .chart-block etc.
    # Only add a narrow wrapper class name to avoid clashing with your CSS:
    return '<div class="carousel-container">\n' + "\n".join(items) + "\n</div>"

# ─────────────────────────────────────────────────────────────────────────────
# Build the Segment Performance block: [carousel] then [table]
# (Single combined table snippet)
# ─────────────────────────────────────────────────────────────────────────────
def build_segment_performance_block(ticker: str, charts_dir_fs: str, charts_dir_web: str) -> str:
    table_html = get_segment_table_html(ticker, charts_dir_fs)
    # If no table exists, we still show the images so there’s at least content
    carousel = build_single_carousel_html(ticker, charts_dir_fs, charts_dir_web)

    if "No segment data available" in table_html:
        if not carousel:
            return ""  # truly nothing to show
        return (
            '<div class="chart-block">\n'
            f'<h2>Segment Performance</h2>\n'
            f"{carousel}\n"
            '<div class="table-wrap"><p>No segment table found for this ticker.</p></div>\n'
            "</div>"
        )

    # table exists → show carousel above table
    # Do not inject extra CSS; leave your theme to style .chart-block/.table-wrap
    # If your snippet already includes <style>, it will render as-is.
    has_table_wrap = ('class="seg-table"' in table_html) or ('class="table-wrap"' in table_html)
    safe_table = table_html if has_table_wrap else f'<div class="table-wrap">{table_html}</div>'

    return (
        '<div class="chart-block">\n'
        f'<h2>Segment Performance</h2>\n'
        f"{(carousel + '\n') if carousel else ''}"
        f"{safe_table}\n"
        "</div>"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard summary (present so main_remote.py imports succeed)
# ─────────────────────────────────────────────────────────────────────────────
def generate_dashboard_table(raw_rows):
    """
    Minimal dashboard builder preserved for compatibility with main_remote.py.
    If you already build a dashboard elsewhere, you can pass it straight through.
    """
    base_cols = [
        "Ticker", "Share Price",
        "Nick's TTM Value", "Nick's Forward Value",
        "Finviz TTM Value", "Finviz Forward Value"
    ]
    df = pd.DataFrame(raw_rows, columns=base_cols) if raw_rows else pd.DataFrame(columns=base_cols)

    # Up to you: look up any live stats before rendering. We leave it empty-safe.
    # We still return a valid HTML block + an empty stats dict so callers won’t crash.
    if df.empty:
        html = (
            '<div class="table-wrap">'
            '<table border="1" class="dataframe"><thead><tr>'
            + "".join(f"<th>{h}</th>" for h in base_cols) +
            "</tr></thead><tbody></tbody></table></div>"
        )
        return html, {}

    # Example formatting parity with your older table
    sp_num = pd.to_numeric(df["Share Price"], errors="coerce")
    df["Share Price_num"]  = sp_num
    df["Share Price_disp"] = sp_num.map(lambda x: f"{x:.2f}" if pd.notnull(x) else "–")

    pct_cols = base_cols[2:]
    for col in pct_cols:
        num = pd.to_numeric(df[col].astype(str).str.rstrip('%'), errors="coerce")
        df[col + "_num"]  = num
        df[col + "_disp"] = num.map(lambda x: f"{x:.1f}" if pd.notnull(x) else "–")

    body = []
    for _, r in df.iterrows():
        cells = [
            f"<td>{r['Ticker']}</td>",
            f'<td data-order="{r["Share Price_num"] if pd.notnull(r["Share Price_num"]) else -999}">{r["Share Price_disp"]}</td>'
        ]
        for col in pct_cols:
            num, disp = r[col + "_num"], r[col + "_disp"]
            cells.append(f'<td class="pct" data-order="{num if pd.notnull(num) else -999}">{disp}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")

    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in base_cols) + "</tr></thead>"
    table  = '<table id="sortable-table" style="width:100%">' + thead + "<tbody>" + "".join(body) + "</tbody></table>"
    return '<div class="table-wrap">' + table + "</div>", {}

# ─────────────────────────────────────────────────────────────────────────────
# Page generation: preserve your templates, fill placeholders only
# ─────────────────────────────────────────────────────────────────────────────
def prepare_and_generate_ticker_pages(tickers, charts_dir_fs="charts"):
    charts_dir_web = "../" + charts_dir_fs
    os.makedirs("pages", exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for t in tickers:
            seg_block = build_segment_performance_block(t, charts_dir_fs, charts_dir_web)

            d = {
                "ticker":                        t,
                "company_name":                  get_company_short_name(t, cur),

                # Existing fragments you already embed
                "ticker_info":                   get_file_or_placeholder(f"{charts_dir_fs}/{t}_ticker_info.html"),
                "financial_table":               get_file_or_placeholder(f"{charts_dir_fs}/{t}_rev_net_table.html"),
                "yoy_growth_table_html":         get_file_or_placeholder(f"{charts_dir_fs}/{t}_yoy_growth_tbl.html"),
                "balance_sheet_table_html":      get_file_or_placeholder(f"{charts_dir_fs}/{t}_balance_sheet_table.html"),
                "valuation_info_table":          get_file_or_placeholder(f"{charts_dir_fs}/{t}_valuation_info.html"),
                "valuation_data_table":          get_file_or_placeholder(f"{charts_dir_fs}/{t}_valuation_table.html"),
                "expense_abs_html":              get_file_or_placeholder(f"{charts_dir_fs}/{t}_expense_absolute.html"),
                "expense_yoy_html":              get_file_or_placeholder(f"{charts_dir_fs}/{t}_yoy_expense_change.html"),
                "unmapped_expense_html":         get_file_or_placeholder(f"{charts_dir_fs}/{t}_unmapped_fields.html", "No unmapped expenses."),
                "implied_growth_table_html":     get_file_or_placeholder(f"{charts_dir_fs}/{t}_implied_growth_summary.html", "No implied growth data available."),

                # Provide both keys so your template can use either
                "segment_sections_html":         seg_block,
                "segment_table_html":            seg_block,

                # Paths for chart images (your template already references these)
                "revenue_net_income_chart_path": f"{charts_dir_web}/{t}_revenue_net_income_chart.png",
                "eps_chart_path":                f"{charts_dir_web}/{t}_eps_chart.png",
                "forecast_rev_net_chart_path":   f"{charts_dir_web}/{t}_Revenue_Net_Income_Forecast.png",
                "forecast_eps_chart_path":       f"{charts_dir_web}/{t}_EPS_Forecast.png",
                "revenue_yoy_change_chart_path": f"{charts_dir_web}/{t}_revenue_yoy_change.png",
                "eps_yoy_change_chart_path":     f"{charts_dir_web}/{t}_eps_yoy_change.png",
                "expense_chart_path":            f"{charts_dir_web}/{t}_rev_expense_chart.png",
                "expense_percent_chart_path":    f"{charts_dir_web}/{t}_expense_percent_chart.png",
                "balance_sheet_chart_path":      f"{charts_dir_web}/{t}_balance_sheet_chart.png",
                "valuation_chart":               f"{charts_dir_web}/{t}_valuation_chart.png",
                "eps_dividend_chart_path":       f"{charts_dir_web}/{t}_eps_dividend_forecast.png",
                "implied_growth_chart_path":     f"{charts_dir_web}/{t}_implied_growth_plot.png",
            }

            tpl = env.get_template("ticker_template.html")
            rendered = tpl.render(ticker_data=d)
            with open(f"pages/{t}_page.html", "w", encoding="utf-8") as f:
                f.write(rendered)

def create_home_page(*_args, **_kwargs):
    """No-op: your home page is already produced elsewhere in your pipeline."""
    pass

def html_generator2(tickers, financial_data, full_dashboard_html, avg_values, spy_qqq_growth_html=""):
    """
    Minimal entrypoint used by main_remote.py; just render ticker pages.
    """
    prepare_and_generate_ticker_pages(tickers)
