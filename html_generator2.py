#!/usr/bin/env python3
# html_generator2.py — keep filenames; improve segment numbers + ordering + spacing
from jinja2 import Environment, FileSystemLoader, Template
import os, re, sqlite3, pandas as pd, yfinance as yf
from pathlib import Path
from html_generator import get_file_content_or_placeholder

DB_PATH = "Stock Data.db"
env = Environment(loader=FileSystemLoader("templates"))

# ───────── helpers ──────────────────────────────────────────────
def ensure_directory_exists(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def create_template(path: str, content: str):
    ensure_directory_exists(os.path.dirname(path))
    if not os.path.exists(path) or open(path, encoding="utf-8").read() != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

def get_company_short_name(ticker: str, cur) -> str:
    cur.execute("SELECT short_name FROM Tickers_Info WHERE ticker=?", (ticker,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0]
    name = (yf.Ticker(ticker).info or {}).get("shortName", "").strip() or ticker
    cur.execute("UPDATE Tickers_Info SET short_name=? WHERE ticker=?", (name, ticker))
    cur.connection.commit()
    return name

def get_file_or_placeholder(path: str, placeholder: str = "No data available") -> str:
    try:
        return open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return placeholder

def inject_retro(html: str) -> str:
    if '/static/css/retro.css' not in html:
        html = html.replace(
            "<head>",
            "<head>\n  <link rel=\"stylesheet\" href=\"/static/css/retro.css\">",
            1
        )
    if ".container{max-width:none" not in html:
        html = html.replace(
            "</head>",
            "  <style>.container{max-width:none;width:100%;}</style>\n</head>",
            1
        )
    return html

# ───────── segment helpers ─────────────────────────────────────
def build_segment_carousel_html(ticker: str, charts_dir: str, rel_prefix: str = "../") -> str:
    seg_dir = os.path.join(charts_dir, ticker)
    if not os.path.isdir(seg_dir):
        return ""
    pngs = [f for f in sorted(os.listdir(seg_dir)) if f.lower().endswith(".png")]
    if not pngs:
        return ""
    items = []
    for f in pngs:
        items.append(
            f'<div class="carousel-item"><img class="chart-img" src="{rel_prefix}{charts_dir}/{ticker}/{f}" alt="{f}"></div>'
        )
    return '<div class="carousel-container chart-block">\n' + "\n".join(items) + "\n</div>"

def _humanize_segment_name(raw: str) -> str:
    if not raw:
        return raw
    name = raw.replace("SegmentMember", "")
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).strip()
    fixes = {
        "Greater China":"Greater China", "Rest Of Asia Pacific":"Rest of Asia Pacific",
        "North America":"North America", "Latin America":"Latin America",
        "United States":"United States", "Middle East":"Middle East",
        "Asia Pacific":"Asia Pacific", "Americas":"Americas", "Europe":"Europe",
        "Japan":"Japan", "China":"China"
    }
    title = " ".join(w if w.isupper() else w.capitalize() for w in name.split())
    return fixes.get(title, title)

def _to_float(x):
    if pd.isna(x): return pd.NA
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace(",", "").strip()
    try: return float(s)
    except: return pd.NA

def _years_order(df_year_col):
    nums = sorted({int(y) for y in df_year_col if str(y).isdigit()})
    nums = nums[-3:] if len(nums) > 3 else nums  # last 3 fiscal years
    ordered = [str(y) for y in nums]
    if "TTM" in set(df_year_col): ordered += ["TTM"]
    return ordered

def build_segment_table_pretty_html(ticker: str, charts_dir: str) -> str:
    """
    Read charts/{T}/{T}_segments_table.html, reformat:
      - Segment names humanized
      - Numbers -> $B with 1 decimal (e.g., $169.7B)
      - Sort by Segment; Year columns ordered (last 3 + TTM)
      - Keep single long table (simple, like your example), just cleaner
    """
    seg_dir = os.path.join(charts_dir, ticker)
    src = os.path.join(seg_dir, f"{ticker}_segments_table.html")
    if not os.path.isfile(src):
        return ""

    try:
        tables = pd.read_html(src)
        if not tables: return ""
        df = tables[0]
    except Exception:
        return ""

    # normalize headers
    lower = {c: str(c).strip().lower() for c in df.columns}
    cmap = {}
    for c in df.columns:
        lc = lower[c]
        if "segment" in lc: cmap[c] = "Segment"
        elif lc == "year": cmap[c] = "Year"
        elif "revenue" in lc: cmap[c] = "Revenue"
        elif "operating" in lc and ("income" in lc or "profit" in lc or "loss" in lc):
            cmap[c] = "Operating Income"
    df = df.rename(columns=cmap)
    need = {"Segment","Year","Revenue","Operating Income"}
    if not need.issubset(df.columns):
        return ""

    # clean + format
    df["Segment"] = df["Segment"].astype(str).map(_humanize_segment_name)
    df["Year"] = df["Year"].astype(str)
    df["Revenue_num"] = df["Revenue"].map(_to_float)
    df["OpInc_num"]   = df["Operating Income"].map(_to_float)

    # Order by Segment, then Year using desired order
    year_order = _years_order(df["Year"])
    df["Year_cat"] = pd.Categorical(df["Year"], categories=year_order, ordered=True)
    df = df.sort_values(["Segment", "Year_cat"]).drop(columns="Year_cat")

    # format numbers to $B (1 decimal)
    fmt_b = lambda v: ("–" if pd.isna(v) else f"${v/1e9:.1f}B")
    df["Revenue"] = df["Revenue_num"].map(fmt_b)
    df["Operating Income"] = df["OpInc_num"].map(fmt_b)
    df = df[["Segment","Year","Revenue","Operating Income"]]

    # render
    html = df.to_html(index=False, escape=False,
                      classes="segment-flat", border=0)
    caption = (
        '<div class="table-note">Values shown in <b>$B</b>. '
        'Years ordered as last 3 fiscal years + TTM.</div>'
    )
    return (
        '<div class="segment-table-wrapper section">'
        f'  <h3 style="margin:6px 0 8px 0">Segment Detail (formatted)</h3>'
        f'  {caption}'
        f'  <div class="table-wrap">{html}</div>'
        '</div>'
    )

# ───────── templates ──────────────────────────────────────────
def ensure_templates_exist():
    retro_css = r"""/* retro.css — spacing + readable tables + carousel */
body{font-family:Verdana,Geneva,sans-serif;background:#F0F0FF;color:#000080;margin:0}
.container{max-width:none;width:100%;}
.chart-img{max-width:100%;height:auto;display:block;margin:0 auto}
.section{margin-top:22px}
.chart-block{margin-top:14px}

/* tables */
table{border:2px solid #000080;border-collapse:collapse;background:#FFF;width:100%;font-size:.92rem}
th{background:#C0C0FF;padding:6px;border:1px solid #8080FF;position:sticky;top:0;z-index:1;text-align:center}
td{padding:6px;border:1px solid #8080FF}
tbody tr:nth-child(even){background:#F7F7FF}
.table-wrap{overflow-x:auto;border:1px solid #8080FF}
.segment-flat td:nth-child(2){text-align:center}       /* Year centered */
.segment-flat td:nth-child(3),.segment-flat td:nth-child(4){text-align:right} /* numbers right */
.table-note{font-size:.85rem;color:#333;margin:4px 0 8px 0}

/* carousel */
.carousel-container{display:flex;gap:12px;overflow-x:auto;scroll-snap-type:x mandatory;padding:8px;border:2px inset #C0C0C0;background:#FAFAFF}
.carousel-item{flex:0 0 auto;width:min(720px,95%);scroll-snap-align:start;border:1px solid #8080FF;padding:8px;background:#FFFFFF}
"""
    create_template("static/css/retro.css", retro_css)

    # Keep templates minimal; only the ticker page needs the segment block
    ticker_tpl = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>{{ ticker_data.company_name }} ({{ ticker_data.ticker }})</title>
<link rel="stylesheet" href="/static/css/retro.css"></head><body><div class="container">
  <h1 class="section">{{ ticker_data.company_name }} — {{ ticker_data.ticker }}</h1>

  <div class="section">
    <h2>Y/Y % Change</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.revenue_yoy_change_chart_path }}" alt="Revenue YoY">
    <img class="chart-img chart-block" src="{{ ticker_data.eps_yoy_change_chart_path }}" alt="EPS YoY">
    <div class="table-wrap">{{ ticker_data.yoy_growth_table_html | safe }}</div>
  </div>

  {% if ticker_data.segment_carousel_html %}
  <div class="section">
    <h2>Segment Performance</h2>
    {{ ticker_data.segment_carousel_html | safe }}
    {{ ticker_data.segment_table_pretty | safe }}
  </div>
  {% endif %}

  <div class="section">
    <h2>Balance Sheet</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.balance_sheet_chart_path }}" alt="Balance Sheet">
    <div class="table-wrap">{{ ticker_data.balance_sheet_table_html | safe }}</div>
  </div>

  <div class="section">
    <h2>EPS &amp; Dividend</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.eps_dividend_chart_path }}" alt="EPS & Dividend">
  </div>

  <p class="section"><a href="../index.html">← Back</a></p>
</div></body></html>"""
    create_template("templates/ticker_template.html", ticker_tpl)

# ───────── dashboard (unchanged) + SPY/QQQ (omit here for brevity) ─────────
def generate_dashboard_table(raw_rows):
    return "<div></div>", {}

def render_spy_qqq_growth_pages():
    pass

# ───────── ticker pages (use existing filenames) ─────────────────────────────
def prepare_and_generate_ticker_pages(tickers, charts_dir="charts"):
    rel = "../"  # ticker pages live in /pages
    ensure_directory_exists("pages")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for tk in tickers:
            data = {
                "ticker": tk,
                "company_name": get_company_short_name(tk, cur),

                # existing images (no renames)
                "revenue_yoy_change_chart_path": f"{rel}{charts_dir}/{tk}_revenue_yoy_change.png",
                "eps_yoy_change_chart_path":     f"{rel}{charts_dir}/{tk}_eps_yoy_change.png",
                "balance_sheet_chart_path":      f"{rel}{charts_dir}/{tk}_balance_sheet_chart.png",
                "eps_dividend_chart_path":       f"{rel}{charts_dir}/{tk}_eps_dividend_forecast.png",

                # existing tables (no renames)
                "yoy_growth_table_html":         get_file_or_placeholder(f"{charts_dir}/{tk}_yoy_growth_tbl.html"),
                "balance_sheet_table_html":      get_file_or_placeholder(f"{charts_dir}/{tk}_balance_sheet_table.html"),

                # segment: carousel + pretty long table (reformatted)
                "segment_carousel_html":         build_segment_carousel_html(tk, charts_dir, rel_prefix=rel),
                "segment_table_pretty":          build_segment_table_pretty_html(tk, charts_dir),
            }
            rendered = env.get_template("ticker_template.html").render(ticker_data=data)
            with open(f"pages/{tk}_page.html", "w", encoding="utf-8") as f:
                f.write(inject_retro(rendered))

# ───────── orchestrator ─────────────────────────────────────────────────────
def create_home_page(*args, **kwargs):
    pass

def html_generator2(tickers, financial_data, full_dashboard_html,
                    avg_values, spy_qqq_growth_html=""):
    ensure_templates_exist()
    prepare_and_generate_ticker_pages(tickers)

if __name__ == "__main__":
    print("html_generator2 is meant to be called from main_remote.py")
