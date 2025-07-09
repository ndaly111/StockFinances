#!/usr/bin/env python3
# html_generator2.py — builds index.html, per-ticker pages, SPY/QQQ pages
# (colour coding restored, NaN% shown as “–”)
# ───────────────────────────────────────────────────────────
from jinja2 import Environment, FileSystemLoader, Template
import os, sqlite3, numpy as np, pandas as pd, yfinance as yf

# ─── Basic setup ───────────────────────────────────────────
db_path = "Stock Data.db"
env = Environment(loader=FileSystemLoader("templates"))

# ─── Generic helpers ──────────────────────────────────────
def ensure_directory_exists(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path)

def create_template(path: str, content: str):
    """Write template only if missing or content differs."""
    ensure_directory_exists(os.path.dirname(path))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            if f.read() == content:
                return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ─── DB & file helpers (restored) ─────────────────────────
def get_company_short_name(ticker: str, cur):
    """
    Pull short name from cache; fetch from yfinance on first use.
    """
    cur.execute("SELECT short_name FROM Tickers_Info WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0]

    info = yf.Ticker(ticker).info or {}
    name = info.get("shortName", "").strip() or ticker
    cur.execute("UPDATE Tickers_Info SET short_name = ? WHERE ticker = ?", (name, ticker))
    cur.connection.commit()
    return name

def get_file_content_or_placeholder(path: str, placeholder="No data available"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return placeholder

# ───────────────────────────────────────────────────────────
# Template setup (home, ticker, spy, qqq)
# ───────────────────────────────────────────────────────────
def ensure_templates_exist():
    # ─── Home template ───
    home_template_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Nick's Stock Financials</title>
  <link rel="stylesheet" href="style.css">
  <link rel="stylesheet" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
  <style>
    td.positive { color: green; }
    td.negative { color: red; }
    .center-table { margin: 0 auto; width: 80%%; }
  </style>
  <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
  <script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
  <script>
    $(document).ready(function () {
      $('#sortable-table').DataTable({
        pageLength: 100,
        createdRow: function (row) {
          $('td', row).each(function () {
            var txt = $(this).text();
            if (txt.includes('%')) {                // add colour classes
              var n = parseFloat(txt.replace('%', ''));
              if (!isNaN(n)) {
                $(this).addClass(n < 0 ? 'negative' : 'positive');
              }
            }
          });
        }
      });
    });
  </script>
</head>
<body>
  <header><h1>Financial Overview</h1></header>

  <nav class="navigation">
    {% for t in tickers %}
      <a href="pages/{{t}}_page.html" class="home-button">{{t}}</a> |
    {% endfor %}
  </nav>

  <div id="spy-qqq-growth" class="center-table">
    <h2>SPY vs QQQ Overview</h2>
    {{ spy_qqq_growth | safe }}
  </div>

  <div class="center-table">
    <h2>Past Earnings (Last 7 Days)</h2>
    {{ earnings_past | safe }}
    <h2>Upcoming Earnings</h2>
    {{ earnings_upcoming | safe }}
  </div>

  <div>{{ dashboard_table | safe }}</div>

  <footer><p>Nick's Financial Data Dashboard</p></footer>
</body>
</html>
"""
    # ─── Ticker template (unchanged) ───
    ticker_template_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ ticker_data.company_name }} – Financial Overview</title>
  <link rel="stylesheet" href="../style.css">
</head>
<body>
  <header>
    <a href="../index.html" class="home-button">Home</a>
    <h1>{{ ticker_data.company_name }} – Financial Overview</h1>
    <h2>Ticker: {{ ticker_data.ticker }}</h2>
  </header>

  <section>{{ ticker_data.ticker_info | safe }}</section>

  <div>
    <img src="../{{ ticker_data.revenue_net_income_chart_path }}" alt="Rev vs NI">
    <img src="../{{ ticker_data.eps_chart_path }}"             alt="EPS">
    {{ ticker_data.financial_table | safe }}
  </div>

  <h1>{{ ticker_data.ticker }} – Forecast Data</h1>
  <div class="carousel-container">
    <div class="carousel-item"><img src="../{{ ticker_data.forecast_rev_net_chart_path }}" alt="Rev/NI Forecast"></div>
    <div class="carousel-item"><img src="../{{ ticker_data.forecast_eps_chart_path }}"     alt="EPS Forecast"></div>
  </div>

  <h1>{{ ticker_data.ticker }} – Y/Y %% Change</h1>
  <div class="carousel-container">
    <img class="carousel-item" src="../{{ ticker_data.revenue_yoy_change_chart_path }}" alt="Rev YoY">
    <img class="carousel-item" src="../{{ ticker_data.eps_yoy_change_chart_path }}"     alt="EPS YoY">
    <div class="carousel-item">{{ ticker_data.yoy_growth_table_html | safe }}</div>
  </div>

  <div class="balance-sheet-container">
    <div class="balance-sheet-table">{{ ticker_data.balance_sheet_table_html | safe }}</div>
    <div class="balance-sheet-chart"><img src="../{{ ticker_data.balance_sheet_chart_path }}" alt="BS Chart"></div>
  </div>

  <h1>{{ ticker_data.ticker }} – Expense Overview</h1>
  <div class="carousel-container">
    <img class="carousel-item" src="../{{ ticker_data.expense_chart_path }}"         alt="Rev vs Exp">
    <img class="carousel-item" src="../{{ ticker_data.expense_percent_chart_path }}" alt="Exp %% of Rev">
    <div class="carousel-item">{{ ticker_data.expense_abs_html | safe }}</div>
    <div class="carousel-item">{{ ticker_data.expense_yoy_html | safe }}</div>
  </div>

  {% if ticker_data.implied_growth_chart_path %}
  <h1>{{ ticker_data.ticker }} – Implied Growth Summary</h1>
  <img src="../{{ ticker_data.implied_growth_chart_path }}" alt="Implied Growth Chart">
  <div class="implied-growth-table">{{ ticker_data.implied_growth_table_html | safe }}</div>
  {% endif %}

  <footer><a href="../index.html" class="home-button">Back to Home</a></footer>
</body>
</html>
"""
    create_template("templates/home_template.html",   home_template_content)
    create_template("templates/ticker_template.html", ticker_template_content)

    # ─── SPY & QQQ templates (unchanged) ───
    spy_tpl = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>SPY – Implied Growth & P/E</title>
<link rel="stylesheet" href="style.css"></head>
<body>
  <header><a href="index.html">← Home</a></header>

  <h1>SPY – Implied Growth Summary</h1>
  <img src="charts/spy_growth_chart.png" style="max-width:100%%;" alt="SPY Growth Chart">
  {{ spy_growth_summary | safe }}

  <h1>SPY – P/E Ratio Summary</h1>
  <img src="charts/spy_pe_chart.png" style="max-width:100%%;" alt="SPY P/E Chart">
  {{ spy_pe_summary | safe }}
</body>
</html>
"""
    qqq_tpl = spy_tpl.replace("SPY", "QQQ").replace("spy_", "qqq_")
    create_template("templates/spy_growth_template.html", spy_tpl)
    create_template("templates/qqq_growth_template.html", qqq_tpl)

# ───────────────────────────────────────────────────────────
# Render SPY & QQQ pages
# ───────────────────────────────────────────────────────────
def render_spy_qqq_growth_pages():
    chart_dir, out_dir = "charts", "."
    mapping = {
        "spy": ("spy_growth_template.html",
                "spy_growth_summary.html", "spy_pe_summary.html",
                "spy_growth.html"),
        "qqq": ("qqq_growth_template.html",
                "qqq_growth_summary.html", "qqq_pe_summary.html",
                "qqq_growth.html")
    }
    for key, (tpl_name, growth_file, pe_file, output_file) in mapping.items():
        tpl = Template(open(os.path.join("templates", tpl_name), encoding="utf-8").read())
        rendered = tpl.render(**{
            f"{key}_growth_summary": get_file_content_or_placeholder(os.path.join(chart_dir, growth_file)),
            f"{key}_pe_summary":     get_file_content_or_placeholder(os.path.join(chart_dir, pe_file))
        })
        with open(os.path.join(out_dir, output_file), "w", encoding="utf-8") as f:
            f.write(rendered)
        print(f"[html_generator2] Rendered {output_file}")

# ───────────────────────────────────────────────────────────
# Build index.html
# ───────────────────────────────────────────────────────────
def create_home_page(tickers, output_dir,
                     dashboard_table, avg_values,
                     spy_qqq_growth="", earnings_past="", earnings_upcoming=""):
    tpl = env.get_template("home_template.html")
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(tpl.render(
            tickers=tickers,
            dashboard_table=dashboard_table,
            dashboard_data=avg_values,
            spy_qqq_growth=spy_qqq_growth,
            earnings_past=earnings_past,
            earnings_upcoming=earnings_upcoming
        ))

# ───────────────────────────────────────────────────────────
# Generate per-ticker pages
# ───────────────────────────────────────────────────────────
def prepare_and_generate_ticker_pages(tickers, output_dir, charts_dir):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for t in tickers:
            d = {
                "ticker": t,
                "company_name": get_company_short_name(t, cur),
                "ticker_info": get_file_content_or_placeholder(f"{charts_dir}/{t}_ticker_info.html"),
                "revenue_net_income_chart_path": f"{charts_dir}/{t}_revenue_net_income_chart.png",
                "eps_chart_path":               f"{charts_dir}/{t}_eps_chart.png",
                "financial_table": get_file_content_or_placeholder(f"{charts_dir}/{t}_rev_net_table.html"),
                "forecast_rev_net_chart_path": f"{charts_dir}/{t}_Revenue_Net_Income_Forecast.png",
                "forecast_eps_chart_path":     f"{charts_dir}/{t}_EPS_Forecast.png",
                "yoy_growth_table_html":       get_file_content_or_placeholder(f"{charts_dir}/{t}_yoy_growth_tbl.html"),
                "expense_chart_path":          f"{charts_dir}/{t}_rev_expense_chart.png",
                "expense_percent_chart_path":  f"{charts_dir}/{t}_expense_percent_chart.png",
                "expense_abs_html": get_file_content_or_placeholder(f"{charts_dir}/{t}_expense_absolute.html"),
                "expense_yoy_html": get_file_content_or_placeholder(f"{charts_dir}/{t}_yoy_expense_change.html"),
                "balance_sheet_chart_path":    f"{charts_dir}/{t}_balance_sheet_chart.png",
                "balance_sheet_table_html":    get_file_content_or_placeholder(f"{charts_dir}/{t}_balance_sheet_table.html"),
                "revenue_yoy_change_chart_path": f"{charts_dir}/{t}_revenue_yoy_change.png",
                "eps_yoy_change_chart_path":      f"{charts_dir}/{t}_eps_yoy_change.png",
                "valuation_chart":              f"{charts_dir}/{t}_valuation_chart.png",
                "valuation_info_table":         get_file_content_or_placeholder(f"{charts_dir}/{t}_valuation_info.html"),
                "valuation_data_table":         get_file_content_or_placeholder(f"{charts_dir}/{t}_valuation_table.html"),
                "unmapped_expense_html":        get_file_content_or_placeholder(f"{charts_dir}/{t}_unmapped_fields.html", "No unmapped expenses."),
                "eps_dividend_chart_path":      f"{charts_dir}/{t}_eps_dividend_forecast.png",
                "implied_growth_chart_path":    f"{charts_dir}/{t}_implied_growth_plot.png",
                "implied_growth_table_html":    get_file_content_or_placeholder(
                    f"{charts_dir}/{t}_implied_growth_summary.html", "No implied growth data available."
                )
            }
            out = os.path.join(output_dir, "pages", f"{t}_page.html")
            ensure_directory_exists(os.path.dirname(out))
            with open(out, "w", encoding="utf-8") as f:
                f.write(env.get_template("ticker_template.html").render(ticker_data=d))

# ───────────────────────────────────────────────────────────
# Dashboard table + summary (NaN→dash & colour classes)
# ───────────────────────────────────────────────────────────
def generate_dashboard_table(dashboard_data):
    df = pd.DataFrame(
        dashboard_data,
        columns=["Ticker", "Share Price",
                 "Nick's TTM Value", "Nick's Forward Value",
                 "Finviz TTM Value", "Finviz Forward Value"]
    )

    def link(t):
        if t == "SPY": return '<a href="spy_growth.html">SPY</a>'
        if t == "QQQ": return '<a href="qqq_growth.html">QQQ</a>'
        return f'<a href="pages/{t}_page.html">{t}</a>'
    df["Ticker"] = df["Ticker"].apply(link)

    for col in ["Nick's TTM Value", "Nick's Forward Value",
                "Finviz TTM Value", "Finviz Forward Value"]:
        num = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")
        df[col + "_num"] = num
        df[col] = num.apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "–")

    df.sort_values("Nick's TTM Value_num", ascending=False, inplace=True)

    ttm   = df["Nick's TTM Value_num"].dropna()
    fwd   = df["Nick's Forward Value_num"].dropna()
    fttm  = df["Finviz TTM Value_num"].dropna()
    ffwd  = df["Finviz Forward Value_num"].dropna()

    pc = lambda s: f"{s:.1f}%" if pd.notnull(s) else "–"
    summary_rows = [
        ["Average", pc(ttm.mean()), pc(fwd.mean()), pc(fttm.mean()), pc(ffwd.mean())],
        ["Median",  pc(ttm.median()), pc(fwd.median()), pc(fttm.median()), pc(ffwd.median())]
    ]
    avg_html = pd.DataFrame(
        summary_rows,
        columns=["Metric", "Nick's TTM Value", "Nick's Forward Value",
                 "Finviz TTM Value", "Finviz Forward Value"]
    ).to_html(index=False, classes="table table-striped", escape=False)

    dash_html = df[[
        "Ticker", "Share Price", "Nick's TTM Value", "Nick's Forward Value",
        "Finviz TTM Value", "Finviz Forward Value"
    ]].to_html(index=False, classes="table table-striped", table_id="sortable-table", escape=False)

    ensure_directory_exists("charts")
    with open("charts/dashboard.html", "w", encoding="utf-8") as f:
        f.write(avg_html + dash_html)

    return avg_html + dash_html, {
        "Nicks_TTM_Value_Average": ttm.mean(),   "Nicks_TTM_Value_Median": ttm.median(),
        "Nicks_Forward_Value_Average": fwd.mean(), "Nicks_Forward_Value_Median": fwd.median(),
        "Finviz_TTM_Value_Average": fttm.mean() if not fttm.empty else None,
        "Finviz_TTM_Value_Median": fttm.median() if not fttm.empty else None,
        "Finviz_Forward_Value_Average": ffwd.mean() if not ffwd.empty else None,
        "Finviz_Forward_Value_Median": ffwd.median() if not ffwd.empty else None
    }

# ───────────────────────────────────────────────────────────
# MAIN WRAPPER
# ───────────────────────────────────────────────────────────
def html_generator2(tickers, financial_data,
                    full_dashboard_html, avg_values,
                    spy_qqq_growth_html=""):

    ensure_templates_exist()

    earnings_past     = get_file_content_or_placeholder("charts/earnings_past.html")
    earnings_upcoming = get_file_content_or_placeholder("charts/earnings_upcoming.html")

    create_home_page(
        tickers=tickers,
        output_dir=".",
        dashboard_table=full_dashboard_html,
        avg_values=avg_values,
        spy_qqq_growth=spy_qqq_growth_html,
        earnings_past=earnings_past,
        earnings_upcoming=earnings_upcoming
    )

    prepare_and_generate_ticker_pages(tickers, ".", "charts/")
    render_spy_qqq_growth_pages()

# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("html_generator2 is intended to be invoked from main_remote.py")
