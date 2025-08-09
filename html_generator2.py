#!/usr/bin/env python3
# html_generator2.py – retro fix + SEGMENT CAROUSEL + layout polish (restored)
# ---------------------------------------------------------------------
from jinja2 import Environment, FileSystemLoader, Template
import os, sqlite3, pandas as pd, yfinance as yf
from pathlib import Path
from html_generator import get_file_content_or_placeholder

DB_PATH = "Stock Data.db"
env = Environment(loader=FileSystemLoader("templates"))

# ───────── helpers ──────────────────────────────────────────────
def ensure_directory_exists(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def create_template(path: str, content: str):
    """Idempotently write a template file when missing or different."""
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

# Inject retro CSS + override container width
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
    """
    Build an HTML carousel for all PNG charts under charts_dir/{ticker}.
    Ticker pages live in /pages, so default rel_prefix is '../'.
    """
    seg_dir = os.path.join(charts_dir, ticker)
    if not os.path.isdir(seg_dir):
        return ""
    pngs = [f for f in sorted(os.listdir(seg_dir)) if f.lower().endswith(".png")]
    if not pngs:
        return ""
    items = []
    for f in pngs:
        src = f"{rel_prefix}{charts_dir}/{ticker}/{f}"
        items.append(
            f'<div class="carousel-item"><img class="chart-img" src="{src}" alt="{f}"></div>'
        )
    return '<div class="carousel-container chart-block">\n' + "\n".join(items) + "\n</div>"

# ───────── template creation ───────────────────────────────────
def ensure_templates_exist():
    # 1) retro.css (carousel + table polish + section spacing)
    retro_css = r"""/* === retro.css — late-90s / early-2000s style + layout polish === */
body{font-family:Verdana,Geneva,sans-serif;background:#F0F0FF url("../images/retro_bg.gif");color:#000080;margin:0}
a{color:#0000FF}a:visited{color:#800080}a:hover{text-decoration:underline}
h1,h2,h3{color:#FF0000;text-shadow:1px 1px #000080;margin:8px 0}
.navbar{background:#C0C0C0;border:2px outset #FFF;padding:6px;text-align:center}
.button,.navbar a{display:inline-block;border:2px outset #C0C0C0;background:#E0E0E0;padding:3px 8px;font-weight:bold;margin:2px}
.container{max-width:none;width:100%;}
.chart-img{max-width:100%;height:auto;display:block;margin:0 auto}

/* Section spacing so charts don't collide vertically */
.section{margin-top:22px}
.section h2{margin:10px 0 8px 0}
.chart-block{margin-top:14px} /* extra space before a block of charts */

/* Generic table polish */
table{border:2px solid #000080;border-collapse:collapse;background:#FFF;width:100%;font-size:.92rem}
th{background:#C0C0FF;padding:6px;border:1px solid #8080FF;position:sticky;top:0;z-index:1}
td{padding:6px;border:1px solid #8080FF}
tbody tr:nth-child(even){background:#F7F7FF}
.table-wrap{overflow-x:auto;border:1px solid #8080FF}

/* Segment table: center Year, right-align numbers */
.segment-table-wrapper .table-wrap table td:nth-child(2){text-align:center}
.segment-table-wrapper .table-wrap table td:nth-child(3),
.segment-table-wrapper .table-wrap table td:nth-child(4){text-align:right}

/* Pivot/pretty variants */
.segment-pivot.compact th{background:#C0C0FF;position:sticky;top:0;text-align:center}
.segment-pivot.compact td{text-align:right}
.segment-pivot.compact td:first-child{text-align:left}
.table-note{font-size:.85rem;color:#333;margin:4px 0 8px 0}

/* simple, touch-friendly horizontal scroller for charts */
.carousel-container{display:flex;gap:12px;overflow-x:auto;scroll-snap-type:x mandatory;padding:8px;border:2px inset #C0C0C0;background:#FAFAFF}
.carousel-item{flex:0 0 auto;width:min(720px,95%);scroll-snap-align:start;border:1px solid #8080FF;padding:8px;background:#FFFFFF}
"""
    create_template("static/css/retro.css", retro_css)

    # 2) Home page template
    home_tpl = """<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="UTF-8"><title>Nick's Stock Financials</title>

  <!-- retro + existing -->
  <link rel="stylesheet" href="/static/css/retro.css">
  <link rel="stylesheet" href="/style.css">

  <!-- DataTables -->
  <link rel="stylesheet" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
  <style>
    td.positive{color:green;} td.negative{color:red;}
    td.pct::after{content:'%';}
    .center-table{margin:0 auto;width:100%%}
  </style>
  <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
  <script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
  <script>
    $(function(){
      $('#sortable-table').DataTable({
        pageLength:100,scrollX:true,
        createdRow:function(row){
          $('td',row).each(function(){
            if(!$(this).attr('data-order')) return;
            var n=parseFloat($(this).data('order'));if(isNaN(n)) return;
            var col=$(this).index();
            if(col===6){$(this).addClass(n<50?'negative':'positive');}
            else if(col>=2&&col<=5){$(this).addClass(n<0?'negative':'positive');}
          });
        }
      });
    });
  </script>
</head><body>
<div class="container">

  <div class="marquee-wrapper" style="background:#000080;color:#FFFF00;padding:4px;font-weight:bold">
    <marquee behavior="scroll" direction="left" scrollamount="6">
      Nick's Stock Financials — Surfacing Under-Priced Stocks Since 2025
    </marquee>
  </div>

  <nav class="navbar">
    {% for t in tickers %}
      <a href="pages/{{t}}_page.html" class="button">{{t}}</a>{% if not loop.last %} | {% endif %}
    {% endfor %}
  </nav>

  <header class="section"><h1>Financial Overview</h1></header>

  <div id="spy-qqq-growth" class="center-table section">
    <h2>SPY vs QQQ Overview</h2>
    {{ spy_qqq_growth | safe }}
  </div>

  <div class="center-table section">
    <h2>Key U.S. Economic Indicators</h2>
    {{ econ_table | safe }}
  </div>

  <div class="center-table section">
    <h2>Past Earnings (Last 7 Days)</h2>
    {{ earnings_past | safe }}
    <h2>Upcoming Earnings</h2>
    {{ earnings_upcoming | safe }}
  </div>

  <div class="section">{{ dashboard_table | safe }}</div>

  <footer class="section"><p>Nick's Financial Data Dashboard</p></footer>
</div></body></html>"""
    create_template("templates/home_template.html", home_tpl)

    # 3) SPY/QQQ growth pages (root-level pages)
    spy_tpl = """<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="UTF-8"><title>SPY Growth &amp; P/E History</title>
  <link rel="stylesheet" href="/static/css/retro.css">
</head><body><div class="container section">
  <h1>SPY — Implied Growth &amp; P/E Ratio</h1>

  <div class="section">
    <h2>Implied Growth (TTM)</h2>
    <img src="charts/spy_growth_chart.png" alt="SPY growth chart" class="chart-img chart-block">
    {{ spy_growth_summary | safe }}
  </div>

  <div class="section">
    <h2>P/E Ratio (TTM)</h2>
    <img src="charts/spy_pe_chart.png" alt="SPY P/E chart" class="chart-img chart-block">
    {{ spy_pe_summary | safe }}
  </div>

  <p class="section"><a href="index.html">← Back to Dashboard</a></p>
</div></body></html>"""
    create_template("templates/spy_growth_template.html", spy_tpl)
    qqq_tpl = spy_tpl.replace("SPY","QQQ").replace("spy_","qqq_")
    create_template("templates/qqq_growth_template.html", qqq_tpl)

    # 4) Ticker page template (carousel below Y/Y, table right under it)
    ticker_tpl = """<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="UTF-8"><title>{{ ticker_data.company_name }} ({{ ticker_data.ticker }})</title>
  <link rel="stylesheet" href="/static/css/retro.css">
</head><body><div class="container">

  <h1 class="section">{{ ticker_data.company_name }} — {{ ticker_data.ticker }}</h1>

  <div class="section">
    <h2>Y/Y % Change</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.revenue_yoy_change_chart_path }}" alt="Revenue YoY Change">
    <img class="chart-img chart-block" src="{{ ticker_data.eps_yoy_change_chart_path }}" alt="EPS YoY Change">
    <div class="table-wrap">{{ ticker_data.yoy_growth_table_html | safe }}</div>
  </div>

  {% if ticker_data.segment_carousel_html %}
  <div class="section">
    <h2>Segment Performance</h2>
    {{ ticker_data.segment_carousel_html | safe }}
    <div class="segment-table-wrapper">
      <div class="table-wrap">
        {{ ticker_data.segment_table_html | safe }}
      </div>
    </div>
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

# ───────── ancillary page-builders ───────────────────────────
def render_spy_qqq_growth_pages():
    chart_dir, out_dir = "charts", "."
    for key in ("spy", "qqq"):
        tpl = Template(get_file_or_placeholder(f"templates/{key}_growth_template.html"))
        rendered = tpl.render(
            **{
                f"{key}_growth_summary": get_file_or_placeholder(f"{chart_dir}/{key}_growth_summary.html"),
                f"{key}_pe_summary":     get_file_or_placeholder(f"{chart_dir}/{key}_pe_summary.html"),
            }
        )
        with open(f"{out_dir}/{key}_growth.html", "w", encoding="utf-8") as f:
            f.write(inject_retro(rendered))

# ───────── ticker pages (segment carousel + table restored) ───
def prepare_and_generate_ticker_pages(tickers, charts_dir="charts"):
    """
    Ticker pages are written into /pages, so all image src paths must be prefixed
    with '../' to reach /charts.
    """
    rel = "../"
    ensure_directory_exists("pages")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for tk in tickers:
            data = {
                "ticker":                       tk,
                "company_name":                 get_company_short_name(tk, cur),
                "revenue_yoy_change_chart_path":f"{rel}{charts_dir}/{tk}_revenue_yoy_change.png",
                "eps_yoy_change_chart_path":    f"{rel}{charts_dir}/{tk}_eps_yoy_change.png",
                "balance_sheet_chart_path":     f"{rel}{charts_dir}/{tk}_balance_sheet_chart.png",
                "eps_dividend_chart_path":      f"{rel}{charts_dir}/{tk}_eps_dividend_forecast.png",
                "yoy_growth_table_html":        get_file_or_placeholder(f"{charts_dir}/{tk}_yoy_growth_tbl.html"),
                "balance_sheet_table_html":     get_file_or_placeholder(f"{charts_dir}/{tk}_balance_sheet_table.html"),
                # Segment content (exact filenames/paths preserved)
                "segment_carousel_html":        build_segment_carousel_html(tk, charts_dir, rel_prefix=rel),
                "segment_table_html":           get_file_or_placeholder(f"{charts_dir}/{tk}/{tk}_segments_table.html",
                                                                        "No segment data available."),
            }
            rendered = env.get_template("ticker_template.html").render(ticker_data=data)
            with open(f"pages/{tk}_page.html", "w", encoding="utf-8") as f:
                f.write(inject_retro(rendered))

# ───────── home-page builder (unchanged) ─────────────────────
def create_home_page(tickers, dashboard_html, avg_vals, spy_qqq_html,
                     econ_table="", earnings_past="", earnings_upcoming=""):
    tpl = env.get_template("home_template.html")
    rendered = tpl.render(
        tickers=tickers,
        dashboard_table=dashboard_html,
        dashboard_data=avg_vals,
        spy_qqq_growth=spy_qqq_html,
        econ_table=econ_table,
        earnings_past=earnings_past,
        earnings_upcoming=earnings_upcoming
    )
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(rendered)

# ───────── orchestrator (exported) ──────────────────────────
def html_generator2(tickers, financial_data, full_dashboard_html,
                    avg_values, spy_qqq_growth_html=""):
    ensure_templates_exist()
    # If you use economic & earnings HTML, keep reading them like this:
    charts_path = Path("charts")
    econ_html = get_file_content_or_placeholder(
        charts_path / "economic_data.html",
        placeholder="<!-- Economic data not available -->"
    )
    # It's fine if econ is missing; don't assert-break here.
    create_home_page(
        tickers, full_dashboard_html, avg_values, spy_qqq_growth_html,
        econ_html,
        get_file_or_placeholder("charts/earnings_past.html"),
        get_file_or_placeholder("charts/earnings_upcoming.html")
    )
    prepare_and_generate_ticker_pages(tickers)
    render_spy_qqq_growth_pages()

if __name__ == "__main__":
    print("html_generator2 is meant to be called from main_remote.py")
