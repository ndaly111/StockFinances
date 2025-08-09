#!/usr/bin/env python3
# html_generator2.py – retro fix: inject CSS + show economic data + (minimal) segment carousel + table
# ----------------------------------------------------------------
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

# ───────── segment helpers (NEW – minimal) ─────────────────────
def build_segment_carousel_html(ticker: str, charts_dir: str, rel_prefix: str = "") -> str:
    """
    Build an HTML carousel for all PNG charts under charts_dir/{ticker}.
    Minimal helper; keeps existing relative path behavior unchanged.
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
    # 1.  retro.css  (late-90s vibes) + minimal additions for carousel/table
    retro_css = r"""/* === retro.css — late-90s / early-2000s style === */
body{font-family:Verdana,Geneva,sans-serif;background:#F0F0FF url("../images/retro_bg.gif");color:#000080;margin:0}
a{color:#0000FF}a:visited{color:#800080}a:hover{text-decoration:underline}
h1,h2,h3{color:#FF0000;text-shadow:1px 1px #000080;margin:8px 0}
.navbar{background:#C0C0C0;border:2px outset #FFF;padding:6px;text-align:center}
.button,.navbar a{display:inline-block;border:2px outset #C0C0C0;background:#E0E0E0;padding:3px 8px;font-weight:bold;margin:2px}
table{border:2px solid #000080;border-collapse:collapse;background:#FFF;width:100%;font-size:.85rem}
th{background:#C0C0FF;padding:4px;border:1px solid #8080FF}
td{padding:4px;border:1px solid #8080FF}
.marquee-wrapper{background:#000080;color:#FFFF00;padding:4px;font-weight:bold}
.blink{animation:blink 1s steps(5,start) infinite}@keyframes blink{to{visibility:hidden}}
.container{max-width:none;width:100%;}

/* --- minimal additions for spacing & segment UI --- */
.chart-img{max-width:100%;height:auto;display:block;margin:0 auto}
.chart-block{margin-top:14px}
.table-wrap{overflow-x:auto;border:1px solid #8080FF}
.segment-table-wrapper .table-wrap table td:nth-child(2){text-align:center}
.segment-table-wrapper .table-wrap table td:nth-child(3),
.segment-table-wrapper .table-wrap table td:nth-child(4){text-align:right}

/* simple, touch-friendly horizontal scroller for segment charts */
.carousel-container{display:flex;gap:12px;overflow-x:auto;scroll-snap-type:x mandatory;padding:8px;border:2px inset #C0C0C0;background:#FAFAFF}
.carousel-item{flex:0 0 auto;width:min(720px,95%);scroll-snap-align:start;border:1px solid #8080FF;padding:8px;background:#FFFFFF}
"""
    create_template("static/css/retro.css", retro_css)

    # 2.  Home page template  (unchanged apart from retro.css link)
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
            var n=parseFloat($(this).data-order'));if(isNaN(n)) return;
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

  <div class="marquee-wrapper">
    <marquee behavior="scroll" direction="left" scrollamount="6">
      Nick's Stock Financials — Surfacing Under-Priced Stocks Since 2025
    </marquee>
  </div>

  <nav class="navbar">
    {% for t in tickers %}
      <a href="pages/{{t}}_page.html" class="button">{{t}}</a>{% if not loop.last %} | {% endif %}
    {% endfor %}
  </nav>

  <header><h1>Financial Overview</h1></header>

  <div id="spy-qqq-growth" class="center-table">
    <h2>SPY vs QQQ Overview</h2>
    {{ spy_qqq_growth | safe }}
  </div>

  <div class="center-table">
    <h2>Key U.S. Economic Indicators</h2>
    {{ econ_table | safe }}
  </div>

  <div class="center-table">
    <h2>Past Earnings (Last 7 Days)</h2>
    {{ earnings_past | safe }}
    <h2>Upcoming Earnings</h2>
    {{ earnings_upcoming | safe }}
  </div>

  <div>{{ dashboard_table | safe }}</div>

  <footer><p>Nick's Financial Data Dashboard</p></footer>
</div></body></html>"""
    create_template("templates/home_template.html", home_tpl)

    # 3.  SPY & QQQ growth-detail pages — **correct, lowercase filenames**
    spy_tpl = """<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="UTF-8"><title>SPY Growth &amp; P/E History</title>
  <link rel="stylesheet" href="/static/css/retro.css">
</head><body><div class="container">
  <h1>SPY — Implied Growth &amp; P/E Ratio</h1>

  <h2>Implied Growth (TTM)</h2>
  <img src="../charts/spy_growth_chart.png" alt="SPY growth chart" style="max-width:100%;">
  {{ spy_growth_summary | safe }}

  <h2>P/E Ratio (TTM)</h2>
  <img src="../charts/spy_pe_chart.png" alt="SPY P/E chart" style="max-width:100%;">
  {{ spy_pe_summary | safe }}

  <p><a href="../index.html">← Back to Dashboard</a></p>
</div></body></html>"""
    create_template("templates/spy_growth_template.html", spy_tpl)

    qqq_tpl = spy_tpl.replace("SPY","QQQ").replace("spy_","qqq_")
    create_template("templates/qqq_growth_template.html", qqq_tpl)

# ───────── dashboard-builder ─────────────────────────────────
def generate_dashboard_table(raw_rows):
    base_cols = [
        "Ticker", "Share Price",
        "Nick's TTM Value", "Nick's Forward Value",
        "Finviz TTM Value", "Finviz Forward Value"
    ]
    df = pd.DataFrame(raw_rows, columns=base_cols)

    with sqlite3.connect(DB_PATH) as conn:
        pct = pd.read_sql_query(
            """SELECT Ticker, Percentile
                 FROM Index_Growth_Pctile
                WHERE Growth_Type='TTM'
                  AND Date = (SELECT MAX(Date) FROM Index_Growth_Pctile)""",
            conn
        )

    df = df.merge(pct, how="left", on="Ticker")

    # numeric / display cols
    sp_num = pd.to_numeric(df["Share Price"], errors="coerce")
    df["Share Price_num"]  = sp_num
    df["Share Price_disp"] = sp_num.map(lambda x: f"{x:.2f}" if pd.notnull(x) else "–")

    pct_cols = base_cols[2:]
    for col in pct_cols:
        num = pd.to_numeric(df[col].astype(str).str.rstrip('%'), errors="coerce")
        df[col + "_num"]  = num
        df[col + "_disp"] = num.map(lambda x: f"{x:.1f}" if pd.notnull(x) else "–")

    df["Implied-Growth Pctile_num"]  = df["Percentile"]
    df["Implied-Growth Pctile_disp"] = df["Percentile"].map(
        lambda x: f"{x:.0f}" if pd.notnull(x) else "–"
    )
    df.drop(columns="Percentile", inplace=True)

    def link(tk):
        if tk == "SPY":
            return '<a href="spy_growth.html">SPY</a>'
        if tk == "QQQ":
            return '<a href="qqq_growth.html">QQQ</a>'
        return f'<a href="pages/{tk}_page.html">{tk}</a>'

    df["Ticker"] = df["Ticker"].apply(link)
    df.sort_values("Nick's TTM Value_num", ascending=False, inplace=True)

    # table rows
    body = []
    for _, r in df.iterrows():
        cells = [
            f"<td>{r['Ticker']}</td>",
            f'<td data-order="{r["Share Price_num"] if pd.notnull(r["Share Price_num"]) else -999}">'
            f'{r["Share Price_disp"]}</td>'
        ]
        for col in pct_cols:
            num, disp = r[col + "_num"], r[col + "_disp"]
            cells.append(
                f'<td class="pct" data-order="{num if pd.notnull(num) else -999}">{disp}</td>'
            )
        num, disp = r["Implied-Growth Pctile_num"], r["Implied-Growth Pctile_disp"]
        cells.append(
            f'<td class="pct" data-order="{num if pd.notnull(num) else -999}">{disp}</td>'
        )
        body.append("<tr>" + "".join(cells) + "</tr>")

    headers = base_cols + ["Implied-Growth Pctile"]
    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    dash_html = (
        '<table id="sortable-table" style="width:100%">' +
        thead + "<tbody>" + "".join(body) + "</tbody></table>"
    )

    # summary stats
    pc = lambda s: f"{s:.1f}" if pd.notnull(s) else "–"
    ttm, fwd = df["Nick's TTM Value_num"].dropna(), df["Nick's Forward Value_num"].dropna()
    fttm, ffwd = df["Finviz TTM Value_num"].dropna(), df["Finviz Forward Value_num"].dropna()

    summary = [
        ["Average", pc(ttm.mean()), pc(fwd.mean()), pc(fttm.mean()), pc(ffwd.mean())],
        ["Median",  pc(ttm.median()), pc(fwd.median()), pc(fttm.median()), pc(ffwd.median())]
    ]
    avg_html = pd.DataFrame(summary, columns=["Metric"] + pct_cols).to_html(index=False, escape=False)

    ensure_directory_exists("charts")
    with open("charts/dashboard.html", "w", encoding="utf-8") as f:
        f.write(avg_html + dash_html)

    return avg_html + dash_html, {
        "Nicks_TTM_Value_Average":       ttm.mean(),
        "Nicks_TTM_Value_Median":        ttm.median(),
        "Nicks_Forward_Value_Average":   fwd.mean(),
        "Nicks_Forward_Value_Median":    fwd.median(),
        "Finviz_TTM_Value_Average":      fttm.mean() if not fttm.empty else None,
        "Finviz_TTM_Value_Median":       fttm.median() if not fttm.empty else None,
        "Finviz_Forward_Value_Average":  ffwd.mean() if not ffwd.empty else None,
        "Finviz_Forward_Value_Median":   ffwd.median() if not ffwd.empty else None
    }

# ───────── ancillary page-builders ───────────────────────────
def render_spy_qqq_growth_pages():
    chart_dir, out_dir = "charts", "."
    for key in ("spy", "qqq"):
        tpl = Template(get_file_content_or_placeholder(f"templates/{key}_growth_template.html"))
        rendered = tpl.render(
            **{
                f"{key}_growth_summary": get_file_content_or_placeholder(f"{chart_dir}/{key}_growth_summary.html", ""),
                f"{key}_pe_summary":     get_file_content_or_placeholder(f"{chart_dir}/{key}_pe_summary.html", ""),
            }
        )
        with open(f"{out_dir}/{key}_growth.html", "w", encoding="utf-8") as f:
            f.write(inject_retro(rendered))

def prepare_and_generate_ticker_pages(tickers, charts_dir="charts"):
    ensure_directory_exists("pages")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for tk in tickers:
            data = {
                "ticker":                       tk,
                "company_name":                 get_company_short_name(tk, cur),
                "ticker_info":                  get_file_or_placeholder(f"{charts_dir}/{tk}_ticker_info.html"),
                "revenue_net_income_chart_path":f"{charts_dir}/{tk}_revenue_net_income_chart.png",
                "eps_chart_path":               f"{charts_dir}/{tk}_eps_chart.png",
                "financial_table":              get_file_or_placeholder(f"{charts_dir}/{tk}_rev_net_table.html"),
                "forecast_rev_net_chart_path":  f"{charts_dir}/{tk}_Revenue_Net_Income_Forecast.png",
                "forecast_eps_chart_path":      f"{charts_dir}/{tk}_EPS_Forecast.png",
                "yoy_growth_table_html":        get_file_or_placeholder(f"{charts_dir}/{tk}_yoy_growth_tbl.html"),
                "expense_chart_path":           f"{charts_dir}/{tk}_rev_expense_chart.png",
                "expense_percent_chart_path":   f"{charts_dir}/{tk}_expense_percent_chart.png",
                "expense_abs_html":             get_file_or_placeholder(f"{charts_dir}/{tk}_expense_absolute.html"),
                "expense_yoy_html":             get_file_or_placeholder(f"{charts_dir}/{tk}_yoy_expense_change.html"),
                "balance_sheet_chart_path":     f"{charts_dir}/{tk}_balance_sheet_chart.png",
                "balance_sheet_table_html":     get_file_or_placeholder(f"{charts_dir}/{tk}_balance_sheet_table.html"),
                "revenue_yoy_change_chart_path":f"{charts_dir}/{tk}_revenue_yoy_change.png",
                "eps_yoy_change_chart_path":    f"{charts_dir}/{tk}_eps_yoy_change.png",
                "valuation_chart":              f"{charts_dir}/{tk}_valuation_chart.png",
                "valuation_info_table":         get_file_or_placeholder(f"{charts_dir}/{tk}_valuation_info.html"),
                "valuation_data_table":         get_file_or_placeholder(f"{charts_dir}/{tk}_valuation_table.html"),
                "unmapped_expense_html":        get_file_or_placeholder(f"{charts_dir}/{tk}_unmapped_fields.html",
                                                                       "No unmapped expenses."),
                "eps_dividend_chart_path":      f"{charts_dir}/{tk}_eps_dividend_forecast.png",
                "implied_growth_chart_path":    f"{charts_dir}/{tk}_implied_growth_plot.png",
                "implied_growth_table_html":    get_file_or_placeholder(f"{charts_dir}/{tk}_implied_growth_summary.html",
                                                                       "No implied growth data available."),
                # NEW (minimal): Segment carousel + table below it
                "segment_carousel_html":        build_segment_carousel_html(tk, charts_dir, rel_prefix=""),
                "segment_table_html":           get_file_or_placeholder(f"{charts_dir}/{tk}/{tk}_segments_table.html",
                                                                        "No segment data available."),
            }
            # Minimal template: add a 'Segment Performance' block under Y/Y if content exists
            ticker_tpl = """<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="UTF-8"><title>{{ ticker_data.company_name }} ({{ ticker_data.ticker }})</title>
  <link rel="stylesheet" href="/static/css/retro.css">
</head><body><div class="container">
  <h1>{{ ticker_data.company_name }} — {{ ticker_data.ticker }}</h1>

  <div>
    <h2>Y/Y % Change</h2>
    <img src="{{ ticker_data.revenue_yoy_change_chart_path }}" alt="Revenue YoY Change" class="chart-img chart-block">
    <img src="{{ ticker_data.eps_yoy_change_chart_path }}" alt="EPS YoY Change" class="chart-img chart-block">
    <div class="table-wrap">{{ ticker_data.yoy_growth_table_html | safe }}</div>
  </div>

  {% if ticker_data.segment_carousel_html %}
  <div class="chart-block">
    <h2>Segment Performance</h2>
    {{ ticker_data.segment_carousel_html | safe }}
    <div class="segment-table-wrapper">
      <div class="table-wrap">
        {{ ticker_data.segment_table_html | safe }}
      </div>
    </div>
  </div>
  {% endif %}

  <div class="chart-block">
    <h2>Balance Sheet</h2>
    <img src="{{ ticker_data.balance_sheet_chart_path }}" alt="Balance Sheet" class="chart-img chart-block">
    <div class="table-wrap">{{ ticker_data.balance_sheet_table_html | safe }}</div>
  </div>

  <div class="chart-block">
    <h2>EPS &amp; Dividend</h2>
    <img src="{{ ticker_data.eps_dividend_chart_path }}" alt="EPS & Dividend" class="chart-img chart-block">
  </div>

  <p class="chart-block"><a href="../index.html">← Back</a></p>
</div></body></html>"""
            rendered = Template(ticker_tpl).render(ticker_data=data)
            with open(f"pages/{tk}_page.html", "w", encoding="utf-8") as f:
                f.write(inject_retro(rendered))

# ───────── home-page builder ────────────────────────────────
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
    charts_path = Path("charts")
    econ_html = get_file_content_or_placeholder(
                charts_path / "economic_data.html",
                placeholder="<!-- Economic data not available -->"
            )
    assert "Economic data not available" not in econ_html[:60], \
       "[ECON]  placeholder detected – file missing"
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
