#!/usr/bin/env python3
# html_generator2.py — sequential Segment Performance (NO tabs), single-path table, dashboard helper
# --------------------------------------------------------------------------------------------------
from jinja2 import Environment, FileSystemLoader
import os, sqlite3, pandas as pd, yfinance as yf, re

DB_PATH = "Stock Data.db"
env = Environment(loader=FileSystemLoader("templates"))

# ─────────────────────────────────────────────────────────────────────────────
# Safe utilities (no template/CSS overrides)
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
# SINGLE-PATH table reader (matches normalized canonical path)
# charts/{TICKER}/{TICKER}_segments_table.html
# ─────────────────────────────────────────────────────────────────────────────
def get_segment_table_html(ticker: str, charts_dir_fs: str) -> str:
    path = f"{charts_dir_fs}/{ticker}/{ticker}_segments_table.html"
    try:
        return open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return f"No segment data available for {ticker}."

# ─────────────────────────────────────────────────────────────────────────────
# Carousel: gather all PNGs under charts/{TICKER}/ (issuer-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
_slug_pat = re.compile(r'^(?P<tkr>[A-Za-z0-9]+)_(?P<slug>[a-z0-9-]+)_.+\.png$', re.IGNORECASE)

def _collect_pngs(ticker: str, charts_dir_fs: str):
    seg_dir = os.path.join(charts_dir_fs, ticker)
    if not os.path.isdir(seg_dir):
        return []
    pngs = [f for f in sorted(os.listdir(seg_dir)) if f.lower().endswith(".png")]
    out = []
    for f in pngs:
        m = _slug_pat.match(f)
        if m and m.group("tkr").upper() == ticker.upper():
            out.append(f)
    return out or pngs

def build_single_carousel_html(ticker: str, charts_dir_fs: str, charts_dir_web: str) -> str:
    files = _collect_pngs(ticker, charts_dir_fs)
    if not files:
        return ""
    items = [
        f'<div class="carousel-item"><img class="chart-img" src="{charts_dir_web}/{ticker}/{fn}" alt="{fn}"></div>'
        for fn in files
    ]
    # Use your page’s existing classes (.chart-img, .chart-block, etc.)
    return '<div class="carousel-container">\n' + "\n".join(items) + "\n</div>"

# ─────────────────────────────────────────────────────────────────────────────
# Segment Performance block: [carousel] then [table] (no tabs)
# ─────────────────────────────────────────────────────────────────────────────
def build_segment_performance_block(ticker: str, charts_dir_fs: str, charts_dir_web: str) -> str:
    table_html = get_segment_table_html(ticker, charts_dir_fs)
    carousel   = build_single_carousel_html(ticker, charts_dir_fs, charts_dir_web)

    # If no table exists, still show images so there’s visible content
    if "No segment data available" in table_html:
        if not carousel:
            return ""  # truly nothing to show
        parts = [
            '<div class="chart-block">\n',
            '<h2>Segment Performance</h2>\n',
            carousel, '\n',
            '<div class="table-wrap"><p>No segment table found for this ticker.</p></div>\n',
            '</div>'
        ]
        return "".join(parts)

    # Wrap table if needed (don’t double-wrap)
    has_wrap = ('class="seg-table"' in table_html) or ('class="table-wrap"' in table_html) \
               or ("class='seg-table'" in table_html) or ("class='table-wrap'" in table_html)
    safe_table = table_html if has_wrap else f'<div class="table-wrap">{table_html}</div>'

    # IMPORTANT: avoid f-string expressions containing backslashes; build in steps
    parts = [
        '<div class="chart-block">\n',
        '<h2>Segment Performance</h2>\n'
    ]
    if carousel:
        parts.append(carousel)
        parts.append('\n')
    parts.append(safe_table)
    parts.append('\n</div>')
    return "".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard helper (kept for main_remote.py compatibility)
# Returns keys your logging expects: Nicks_TTM_Value_Average, Nicks_Forward_Value_Average, Finviz_TTM_Value_Average
# ─────────────────────────────────────────────────────────────────────────────
def generate_dashboard_table(raw_rows):
    base_cols = [
        "Ticker", "Share Price",
        "Nick's TTM Value", "Nick's Forward Value",
        "Finviz TTM Value", "Finviz Forward Value"
    ]
    df = pd.DataFrame(raw_rows, columns=base_cols) if raw_rows else pd.DataFrame(columns=base_cols)

    if df.empty:
        html = (
            '<div class="table-wrap">'
            '<table border="1" class="dataframe"><thead><tr>'
            + "".join(f"<th>{h}</th>" for h in base_cols) +
            "</tr></thead><tbody></tbody></table></div>"
        )
        # Return the keys your logger expects, set to None
        return html, {
            "Nicks_TTM_Value_Average": None,
            "Nicks_Forward_Value_Average": None,
            "Finviz_TTM_Value_Average": None,
        }

    # numeric conversions
    def as_num(s):
        return pd.to_numeric(pd.Series(s), errors="coerce")

    ttm_vals   = as_num(df["Nick's TTM Value"])
    fwd_vals   = as_num(df["Nick's Forward Value"])
    finviz_ttm = as_num(df["Finviz TTM Value"])

    # display table (simple)
    body = []
    for i, r in df.iterrows():
        row = [str(r[c]) if pd.notnull(r[c]) else "–" for c in base_cols]
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in base_cols) + "</tr></thead>"
    table_html = '<div class="table-wrap"><table id="sortable-table" style="width:100%">' + thead + "<tbody>" + "".join(body) + "</tbody></table></div>"

    stats = {
        "Nicks_TTM_Value_Average":     float(ttm_vals.mean())     if not ttm_vals.dropna().empty   else None,
        "Nicks_Forward_Value_Average": float(fwd_vals.mean())     if not fwd_vals.dropna().empty   else None,
        "Finviz_TTM_Value_Average":    float(finviz_ttm.mean())   if not finviz_ttm.dropna().empty else None,
    }
    return table_html, stats

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

                # Existing fragments already used by your template
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

                # Provide both keys; your template can use either
                "segment_sections_html":         seg_block,
                "segment_table_html":            seg_block,

                # Chart paths (as your template already expects)
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
    """No-op: home page is built elsewhere in your pipeline."""
    pass

def html_generator2(tickers, financial_data, full_dashboard_html, avg_values, spy_qqq_growth_html=""):
    """Entrypoint used by main_remote.py."""
    prepare_and_generate_ticker_pages(tickers)
