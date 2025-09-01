#!/usr/bin/env python3
# html_generator2.py – retro fix: economic data + per-axis segment blocks (carousel + table) + dividend
from jinja2 import Environment, FileSystemLoader, Template
import os, sqlite3, pandas as pd, yfinance as yf, re

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

def get_company_short_name(tk: str, cur) -> str:
    cur.execute("SELECT short_name FROM Tickers_Info WHERE ticker=?", (tk,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0]
    name = (yf.Ticker(tk).info or {}).get("shortName", "").strip() or tk
    cur.execute("UPDATE Tickers_Info SET short_name=? WHERE ticker=?", (name, tk))
    cur.connection.commit()
    return name

def get_file_or_placeholder(path: str, ph: str = "No data available") -> str:
    try:
        return open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return ph

def get_first_file(paths, placeholder="No data available") -> str:
    import glob
    for p in paths:
        for m in glob.glob(p):
            try:
                return open(m, encoding="utf-8").read()
            except FileNotFoundError:
                continue
    return placeholder

def inject_retro(html: str) -> str:
    if '/static/css/retro.css' not in html:
        html = html.replace(
            "<head>", "<head>\n  <link rel=\"stylesheet\" href=\"/static/css/retro.css\">", 1
        )
    if ".container{max-width:none" not in html:
        html = html.replace(
            "</head>", "  <style>.container{max-width:none;width:100%;}</style>\n</head>", 1
        )
    return html

# ────── segment helpers ────────────────────────────────
def build_segment_carousel_html(ticker: str, charts_dir_fs: str, charts_dir_web: str) -> str:
    """Build Business Segment carousels grouped by type id."""

    sub_dir = os.path.join(charts_dir_fs, ticker.upper())
    if not os.path.isdir(sub_dir):
        return ""

    pat = re.compile(
        rf"^(\d+){ticker.upper()}_bisseg_(.+)\.png$",
        re.IGNORECASE,
    )

    grouped: dict[int, list[str]] = {}
    for fname in os.listdir(sub_dir):
        m = pat.match(fname)
        if not m:
            continue
        type_id = int(m.group(1))
        grouped.setdefault(type_id, []).append(
            f"{charts_dir_web}/{ticker.upper()}/{fname}"
        )

    if not grouped:
        return ""

    parts: list[str] = []
    for type_id in sorted(grouped):
        items = [
            f'<div class="carousel-item"><img class="chart-img" src="{src}" alt="{ticker.upper()} Business Segments Type {type_id}"></div>'
            for src in grouped[type_id]
        ]
        parts.append(
            f'<h3>Business Segments (Type {type_id})</h3>\n'
            f'<div class="carousel-container chart-block">\n' + "\n".join(items) + "\n</div>"
        )

    return "\n".join(parts)

def _split_h3_sections(html: str, wanted_class: str = None):
    """
    Split HTML into [(title, body_html)] sections keyed by <h3>...</h3>.
    If wanted_class is provided, capture only the first matching wrapper in that section.
    """
    if not html:
        return []
    heads = list(re.finditer(r"<h3[^>]*>(.*?)</h3>", html, flags=re.IGNORECASE | re.DOTALL))
    sections = []
    for i, m in enumerate(heads):
        title = re.sub(r"<.*?>", "", m.group(1)).strip()
        start = m.end()
        end = heads[i+1].start() if i+1 < len(heads) else len(html)
        blob = html[start:end]
        if wanted_class:
            mm = re.search(
                rf"<div[^>]*class=['\"][^'\"]*{wanted_class}[^'\"]*['\"][^>]*>.*?</div>",
                blob, flags=re.IGNORECASE | re.DOTALL
            )
            if mm:
                blob = mm.group(0)
        sections.append((title, blob.strip()))
    return sections

def interleave_segment_blocks(carousel_html: str, table_html: str) -> str:
    """
    For each axis title: render [charts row] then [matching table].
    Falls back to an inline notice if a table section is missing.
    """
    car = _split_h3_sections(carousel_html, wanted_class="carousel-container")
    tab = _split_h3_sections(table_html,   wanted_class="table-wrap")

    car_map = {}
    order = []
    for title, body in car:
        if title not in car_map:
            order.append(title)
            car_map[title] = []
        car_map[title].append(body)

    tab_map = {title: body for title, body in tab}

    blocks = []
    for title in order:
        table_part = tab_map.get(title, '<div class="table-wrap"><p>No table for this axis.</p></div>')
        for body in car_map[title]:
            blocks.append(f'<div class="seg-axis-block">\n<h3>{title}</h3>\n{body}\n{table_part}\n</div>')

    # tables that have no charts
    for title, body in tab_map.items():
        if title not in car_map:
            blocks.append(f'<div class="seg-axis-block">\n<h3>{title}</h3>\n{body}\n</div>')

    html = "\n".join(blocks).strip()
    return html or (table_html or "")

# ───────── template creation ────────────────────────────────────
def ensure_templates_exist():
    retro_css = r"""/* === retro.css — keep your existing look === */
body{font-family:Verdana,Geneva,sans-serif;background:#F0F0FF url("../images/retro_bg.gif");color:#000080;margin:0}
a{color:#0000FF}a:visited{color:#800080}a:hover{text-decoration:underline}
h1,h2,h3{color:#FF0000;text-shadow:1px 1px #000080;margin:8px 0}
.navbar{background:#C0C0C0;border:2px outset #FFF;padding:6px;text-align:center}
.button,.navbar a{display:inline-block;border:2px outset #C0C0C0;background:#E0E0E0;padding:3px 8px;font-weight:bold;margin:2px}
table{border:2px solid #000080;border-collapse:collapse;background:#FFF;width:100%;font-size:.85rem}
th{background:#C0C0FF;padding:4px;border:1px solid #8080FF}
td{padding:4px;border:1px solid #8080FF}
.marquee-wrapper{background:#000080;color:#FFFF00;padding:4px;font-weight:bold}
.container{max-width:none;width:100%;}
.chart-img{max-width:100%;height:auto;display:block;margin:0 auto}
.chart-block{margin-top:14px}
.table-wrap{overflow-x:auto;border:1px solid #8080FF}
.carousel-container{display:flex;gap:12px;overflow-x:auto;scroll-snap-type:x mandatory;padding:8px;border:2px inset #C0C0C0;background:#FAFAFF}
.carousel-item{flex:0 0 auto;width:min(720px,95%);scroll-snap-align:start;border:1px solid #8080FF;padding:8px;background:#FFFFFF}
.seg-axis-block{margin-bottom:16px}
"""
    create_template("static/css/retro.css", retro_css)

    home_tpl = """<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="UTF-8"><title>Nick's Stock Financials</title>
  <link rel="stylesheet" href="/static/css/retro.css">
  <link rel="stylesheet" href="/style.css">
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

  <div id="economic-data" class="center-table">
    <h2>Economic Data</h2>
    {{ economic_data | safe }}
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

    # Ticker page: NO TABS. Interleaved segments only.
    ticker_tpl = """<!DOCTYPE html><html lang="en"><head>
  <meta charset="UTF-8"><title>{{ ticker_data.company_name }} ({{ ticker_data.ticker }})</title>
  <link rel="stylesheet" href="/static/css/retro.css">
</head><body><div class="container">
  <h1>{{ ticker_data.company_name }} — {{ ticker_data.ticker }}</h1>

  <div class="chart-block">
    {{ ticker_data.ticker_info | safe }}
  </div>

  <div class="chart-block">
    <h2>Revenue &amp; Net Income</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.revenue_net_income_chart_path }}" alt="Revenue & Net Income">
    <div class="table-wrap">{{ ticker_data.financial_table | safe }}</div>
  </div>

  <div class="chart-block">
    <h2>EPS</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.eps_chart_path }}" alt="EPS">
  </div>

  <div class="chart-block">
    <h2>Forecasts</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.forecast_rev_net_chart_path }}" alt="Revenue/Net Income Forecast">
    <img class="chart-img chart-block" src="{{ ticker_data.forecast_eps_chart_path }}" alt="EPS Forecast">
  </div>

  <div class="chart-block">
    <h2>Y/Y % Change</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.revenue_yoy_change_chart_path }}" alt="Revenue YoY Change">
    <img class="chart-img chart-block" src="{{ ticker_data.eps_yoy_change_chart_path }}" alt="EPS YoY Change">
    <div class="table-wrap">{{ ticker_data.yoy_growth_table_html | safe }}</div>
  </div>

  <div class="chart-block">
    <h2>Expenses</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.expense_chart_path }}" alt="Revenue vs Expenses">
    <img class="chart-img chart-block" src="{{ ticker_data.expense_percent_chart_path }}" alt="Expenses % of Revenue">
    <div class="table-wrap">{{ ticker_data.expense_abs_html | safe }}</div>
    <div class="table-wrap">{{ ticker_data.expense_yoy_html | safe }}</div>
    <div class="table-wrap">{{ ticker_data.unmapped_expense_html | safe }}</div>
  </div>

  {% if ticker_data.segment_interleaved_html %}
  <div class="chart-block">
    <h2>Segment Performance</h2>
    {{ ticker_data.segment_interleaved_html | safe }}
  </div>
  {% endif %}

  <div class="chart-block">
    <h2>Balance Sheet</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.balance_sheet_chart_path }}" alt="Balance Sheet">
    <div class="table-wrap">{{ ticker_data.balance_sheet_table_html | safe }}</div>
  </div>

  <div class="chart-block">
    <h2>Valuation</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.valuation_chart }}" alt="Valuation">
    <div class="table-wrap">{{ ticker_data.valuation_info_table | safe }}</div>
    <div class="table-wrap">{{ ticker_data.valuation_data_table | safe }}</div>
  </div>

  <div class="chart-block">
    <h2>Implied Growth</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.implied_growth_chart_path }}" alt="Implied Growth">
    <div class="table-wrap">{{ ticker_data.implied_growth_table_html | safe }}</div>
  </div>

  <div class="chart-block">
    <h2>EPS &amp; Dividend</h2>
    <img class="chart-img chart-block" src="{{ ticker_data.eps_dividend_chart_path }}" alt="EPS & Dividend">
  </div>

  <p class="chart-block"><a href="../index.html">← Back</a></p>
</div></body></html>"""
    create_template("templates/ticker_template.html", ticker_tpl)

# ───────── dashboard builder (exported) ────────────────────────
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

    sp_num = pd.to_numeric(df["Share Price"], errors="coerce")
    df["Share Price_num"]  = sp_num
    df["Share Price_disp"] = sp_num.map(lambda x: f"{x:.2f}" if pd.notnull(x) else "–")

    pct_cols = base_cols[2:]
    for col in pct_cols:
        num = pd.to_numeric(df[col].astype(str).str.rstrip('%'), errors="coerce")
        df[col + "_num"]  = num
        df[col + "_disp"] = num.map(lambda x: f"{x:.1f}" if pd.notnull(x) else "–")

    df["Implied-Growth Pctile_num"]  = df["Percentile"]
    df["Implied-Growth Pctile_disp"] = df["Percentile"].map(lambda x: f"{x:.0f}" if pd.notnull(x) else "–")
    df.drop(columns="Percentile", inplace=True)

    def link(t):
        if t == "SPY":
            return '<a href="spy_growth.html">SPY</a>'
        if t == "QQQ":
            return '<a href="qqq_growth.html">QQQ</a>'
        return f'<a href="pages/{t}_page.html">{t}</a>'

    df["Ticker"] = df["Ticker"].apply(link)
    df.sort_values("Nick's TTM Value_num", ascending=False, inplace=True)

    body = []
    for _, r in df.iterrows():
        cells = [
            f"<td>{r['Ticker']}</td>",
            f'<td data-order="{r["Share Price_num"] if pd.notnull(r["Share Price_num"]) else -999}">{r["Share Price_disp"]}</td>'
        ]
        for col in pct_cols:
            num, disp = r[col + "_num"], r[col + "_disp"]
            cells.append(f'<td class="pct" data-order="{num if pd.notnull(num) else -999}">{disp}</td>')
        num, disp = r["Implied-Growth Pctile_num"], r["Implied-Growth Pctile_disp"]
        cells.append(f'<td class="pct" data-order="{num if pd.notnull(num) else -999}">{disp}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")

    headers = base_cols + ["Implied-Growth Pctile"]
    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    dash_html = (
        '<table id="sortable-table" style="width:100%">' +
        thead + "<tbody>" + "".join(body) + "</tbody></table>"
    )

    pc = lambda s: f"{s:.1f}" if pd.notnull(s) else "–"
    ttm  = df["Nick's TTM Value_num"].dropna()
    fwd  = df["Nick's Forward Value_num"].dropna()
    fttm = df["Finviz TTM Value_num"].dropna()
    ffwd = df["Finviz Forward Value_num"].dropna()

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
        "Finviz_TTM Value_Average":      fttm.mean() if not fttm.empty else None,
        "Finviz_TTM Value_Median":       fttm.median() if not fttm.empty else None,
        "Finviz_Forward Value_Average":  ffwd.mean() if not ffwd.empty else None,
        "Finviz_Forward Value_Median":   ffwd.median() if not ffwd.empty else None
    }

# ───────── ancillary page builders ──────────────────────────
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

def prepare_and_generate_ticker_pages(tickers, charts_dir_fs="charts"):
    charts_dir_web = "../" + charts_dir_fs
    ensure_directory_exists("pages")

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for t in tickers:
            # Read canonical table (with safe fallback to helper flat file)
            raw_table = get_first_file(
                [
                    f"{charts_dir_fs}/{t}/{t}_segments_table.html",  # canonical
                    f"{charts_dir_fs}/{t}_segments.html",            # optional helper flat file
                ],
                f"No segment data available for {t}."
            )
            raw_carousels = build_segment_carousel_html(t, charts_dir_fs, charts_dir_web)
            interleaved   = interleave_segment_blocks(raw_carousels, raw_table)

            d = {
                "ticker":                        t,
                "company_name":                  get_company_short_name(t, cur),

                # HTML fragments
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

                # Segments (interleaved: charts for each axis, then that axis table)
                "segment_interleaved_html":      interleaved,

                # Images (web paths)
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
            rendered = env.get_template("ticker_template.html").render(ticker_data=d)
            with open(f"pages/{t}_page.html", "w", encoding="utf-8") as f:
                f.write(inject_retro(rendered))

def create_home_page(tickers, dashboard_html, avg_vals, spy_qqq_html,
                     earnings_past="", earnings_upcoming="", economic_html=""):
    tpl = env.get_template("home_template.html")
    rendered = tpl.render(
        tickers=tickers,
        dashboard_table=dashboard_html,
        dashboard_data=avg_vals,
        spy_qqq_growth=spy_qqq_html,
        earnings_past=earnings_past,
        earnings_upcoming=earnings_upcoming,
        economic_data=economic_html
    )
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(rendered)

# ───────── orchestrator ────────────────────────────────────
def html_generator2(tickers, financial_data, full_dashboard_html,
                    avg_values, spy_qqq_growth_html=""):
    ensure_templates_exist()
    create_home_page(
        tickers,
        full_dashboard_html,
        avg_values,
        spy_qqq_growth_html,
        get_file_or_placeholder("charts/earnings_past.html"),
        get_file_or_placeholder("charts/earnings_upcoming.html"),
        get_file_or_placeholder("charts/economic_data.html", "No economic data available.")
    )
    prepare_and_generate_ticker_pages(tickers)
    render_spy_qqq_growth_pages()

if __name__ == "__main__":
    print("html_generator2 is meant to be called from main_remote.py")
