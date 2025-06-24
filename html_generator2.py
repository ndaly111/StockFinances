from jinja2 import Environment, FileSystemLoader
import os
import pandas as pd
import sqlite3
import yfinance as yf
import numpy as np

# Path to the database file
db_path = 'Stock Data.db'
env = Environment(loader=FileSystemLoader('templates'))

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_template(template_path, content):
    ensure_directory_exists(os.path.dirname(template_path))
    # overwrite only if the content has changed
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            if f.read() == content:
                return
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(content)

def get_company_short_name(ticker, cursor):
    cursor.execute("SELECT short_name FROM Tickers_Info WHERE ticker = ?", (ticker,))
    row = cursor.fetchone()
    if row and row[0]:
        return row[0]
    stock = yf.Ticker(ticker)
    short_name = stock.info.get('shortName', '').strip()
    if short_name:
        cursor.execute(
            "UPDATE Tickers_Info SET short_name = ? WHERE ticker = ?",
            (short_name, ticker)
        )
        cursor.connection.commit()
        return short_name
    return ticker

def ensure_templates_exist():
    # home_template.html
    home_template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Nick's Stock Financials</title>
  <link rel="stylesheet" href="style.css">
  <link rel="stylesheet" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
  <style>
    .positive { color: green; }
    .negative { color: red; }
    .center-table { margin: 0 auto; width: 80%; }
    .highlight-soon { background-color: #fff3cd; }
  </style>
  <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
  <script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
  <script>
    $(document).ready(function() {
      $('#sortable-table').DataTable({
        "pageLength": 100,
        "createdRow": function(row, data, dataIndex) {
          $('td', row).each(function() {
            var v = $(this).text();
            if (v.includes('%')) {
              var n = parseFloat(v.replace('%',''));
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

  <div>
    {{ dashboard_table | safe }}
  </div>

  <footer><p>Nick's Financial Data Dashboard</p></footer>
</body>
</html>
"""

    # ticker_template.html
    ticker_template_content = """
<!DOCTYPE html>
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
    <div class="carousel-item">
      <img src="../{{ ticker_data.forecast_rev_net_chart_path }}" alt="Rev/NI Forecast">
    </div>
    <div class="carousel-item">
      <img src="../{{ ticker_data.forecast_eps_chart_path }}"     alt="EPS Forecast">
    </div>
  </div>

  <h1>{{ ticker_data.ticker }} – Y/Y % Change</h1>
  <div class="carousel-container">
    <img class="carousel-item" src="../{{ ticker_data.revenue_yoy_change_chart_path }}" alt="Rev YoY">
    <img class="carousel-item" src="../{{ ticker_data.eps_yoy_change_chart_path }}"     alt="EPS YoY">
    <div class="carousel-item">{{ ticker_data.yoy_growth_table_html | safe }}</div>
  </div>

  <div class="balance-sheet-container">
    <div class="balance-sheet-table">{{ ticker_data.balance_sheet_table_html | safe }}</div>
    <div class="balance-sheet-chart">
      <img src="../{{ ticker_data.balance_sheet_chart_path }}" alt="BS Chart">
    </div>
  </div>

  <h1>{{ ticker_data.ticker }} – Expense Overview</h1>
  <div class="carousel-container">
    <img class="carousel-item" src="../{{ ticker_data.expense_chart_path }}"         alt="Rev vs Exp">
    <img class="carousel-item" src="../{{ ticker_data.expense_percent_chart_path }}" alt="Exp % of Rev">
    <div class="carousel-item">{{ ticker_data.expense_table_html | safe }}</div>
    <div class="carousel-item">{{ ticker_data.expense_yoy_table_html | safe }}</div>
  </div>

  {% if ticker_data.unmapped_expense_html %}
  <h1>{{ ticker_data.ticker }} – Unmapped Items</h1>
  <div>{{ ticker_data.unmapped_expense_html | safe }}</div>
  {% endif %}

  {% if ticker_data.valuation_chart %}
  <h1>{{ ticker_data.ticker }} – Valuation Chart</h1>
  <img src="../{{ ticker_data.valuation_chart }}" alt="Valuation">
  <div class="valuation-tables">
    {{ ticker_data.valuation_info_table | safe }}
    {{ ticker_data.valuation_data_table | safe }}
  </div>
  {% endif %}

  <footer>
    <a href="../index.html" class="home-button">Back to Home</a>
  </footer>
</body>
</html>
"""

    create_template('templates/home_template.html',   home_template_content)
    create_template('templates/ticker_template.html', ticker_template_content)


def get_file_content_or_placeholder(path, placeholder="No data available"):
    try:
        return open(path, 'r', encoding='utf-8').read()
    except FileNotFoundError:
        return placeholder


def create_home_page(tickers, output_dir,
                     dashboard_table, avg_values,
                     spy_qqq_growth="", earnings_past="", earnings_upcoming=""):
    tpl = env.get_template('home_template.html')
    out = os.path.join(output_dir, 'index.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(tpl.render(
            tickers=tickers,
            dashboard_table=dashboard_table,
            dashboard_data=avg_values,
            spy_qqq_growth=spy_qqq_growth,
            earnings_past=earnings_past,
            earnings_upcoming=earnings_upcoming
        ))


def prepare_and_generate_ticker_pages(tickers, output_dir, charts_dir):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for t in tickers:
            name = get_company_short_name(t, cur)
            d = {
                'ticker': t,
                'company_name': name,
                'ticker_info':      get_file_content_or_placeholder(f"{charts_dir}/{t}_ticker_info.html"),
                'revenue_net_income_chart_path': f"{charts_dir}/{t}_revenue_net_income_chart.png",
                'eps_chart_path':                f"{charts_dir}/{t}_eps_chart.png",
                'financial_table':   get_file_content_or_placeholder(f"{charts_dir}/{t}_rev_net_table.html"),
                'forecast_rev_net_chart_path': f"{charts_dir}/{t}_Revenue_Net_Income_Forecast.png",
                'forecast_eps_chart_path':     f"{charts_dir}/{t}_EPS_Forecast.png",
                'yoy_growth_table_html':       get_file_content_or_placeholder(f"{charts_dir}/{t}_yoy_growth_tbl.html"),
                'expense_chart_path':          f"{charts_dir}/{t}_rev_expense_chart.png",
                'expense_percent_chart_path':  f"{charts_dir}/{t}_expense_percent_chart.png",
                'expense_table_html':          get_file_content_or_placeholder(f"{charts_dir}/{t}_yearly_financials.html"),
                'expense_yoy_table_html':      get_file_content_or_placeholder(f"{charts_dir}/{t}_yoy_expense_change.html"),
                'balance_sheet_chart_path':    f"{charts_dir}/{t}_balance_sheet_chart.png",
                'balance_sheet_table_html':    get_file_content_or_placeholder(f"{charts_dir}/{t}_balance_sheet_table.html"),
                'revenue_yoy_change_chart_path': f"{charts_dir}/{t}_revenue_yoy_change.png",
                'eps_yoy_change_chart_path':      f"{charts_dir}/{t}_eps_yoy_change.png",
                'valuation_chart':              f"{charts_dir}/{t}_valuation_chart.png",
                'valuation_info_table':         get_file_content_or_placeholder(f"{charts_dir}/{t}_valuation_info.html"),
                'valuation_data_table':         get_file_content_or_placeholder(f"{charts_dir}/{t}_valuation_table.html"),
                'unmapped_expense_html':        get_file_content_or_placeholder(f"{charts_dir}/{t}_unmapped_fields.html", "No unmapped expenses."),
                'eps_dividend_chart_path':      f"{charts_dir}/{t}_eps_dividend_forecast.png"
            }
            tpl = env.get_template('ticker_template.html')
            out = os.path.join(output_dir, 'pages', f"{t}_page.html")
            ensure_directory_exists(os.path.dirname(out))
            with open(out, 'w', encoding='utf-8') as f:
                f.write(tpl.render(ticker_data=d))


# ─────────────────────────────────────────────────────────────────────────────
# build exactly six columns, compute avg/med, strip helper "_num"
# ─────────────────────────────────────────────────────────────────────────────
def generate_dashboard_table(dashboard_data):
    # dashboard_data: list of rows [ticker, price, nick_ttm%, nick_fwd%, finviz_ttm%, finviz_fwd%]
    df = pd.DataFrame(dashboard_data, columns=[
        "Ticker",
        "Share Price",
        "Nick's TTM Value",
        "Nick's Forward Value",
        "Finviz TTM Value",
        "Finviz Forward Value"
    ])

    # hyperlink tickers
    df["Ticker"] = df["Ticker"].apply(lambda t: f'<a href="pages/{t}_page.html">{t}</a>')

    # numeric twins
    for col in ["Nick's TTM Value", "Nick's Forward Value", "Finviz TTM Value", "Finviz Forward Value"]:
        df[col + "_num"] = (
            df[col].astype(str)
                   .str.rstrip("%")
                   .replace("-", np.nan)
                   .astype(float)
        )

    # sort by Nick's TTM
    df.sort_values("Nick's TTM Value_num", ascending=False, inplace=True)

    # compute metrics
    ttm = df["Nick's TTM Value_num"].dropna()
    fwd = df["Nick's Forward Value_num"].dropna()
    fttm = df["Finviz TTM Value_num"].dropna()
    ffwd = df["Finviz Forward Value_num"].dropna()

    ttm_avg, ttm_med = ttm.mean(), ttm.median()
    fwd_avg, fwd_med = fwd.mean(), fwd.median()
    fttm_avg = fttm.mean() if not fttm.empty else None
    fttm_med = fttm.median() if not fttm.empty else None
    ffwd_avg = ffwd.mean() if not ffwd.empty else None
    ffwd_med = ffwd.median() if not ffwd.empty else None

    def fmt(x):
        return f"{x:.1f}%" if pd.notnull(x) else "–"

    summary = [
        ["Average", fmt(ttm_avg), fmt(fwd_avg), fmt(fttm_avg), fmt(ffwd_avg)],
        ["Median",  fmt(ttm_med), fmt(fwd_med), fmt(fttm_med), fmt(ffwd_med)]
    ]
    avg_table = pd.DataFrame(
        summary,
        columns=["Metric", "Nick's TTM Value", "Nick's Forward Value", "Finviz TTM Value", "Finviz Forward Value"]
    ).to_html(index=False, escape=False, classes='table table-striped')

    # display only visible six columns
    display = df[[
        "Ticker",
        "Share Price",
        "Nick's TTM Value",
        "Nick's Forward Value",
        "Finviz TTM Value",
        "Finviz Forward Value"
    ]]
    dash_table = display.to_html(
        index=False, escape=False,
        classes='table table-striped', table_id="sortable-table"
    )

    # write out
    with open('charts/dashboard.html','w', encoding='utf-8') as f:
        f.write(avg_table + dash_table)

    return avg_table + dash_table, {
        'Nicks_TTM_Avg': ttm_avg,
        'Nicks_TTM_Med': ttm_med,
        'Nicks_FWD_Avg': fwd_avg,
        'Nicks_FWD_Med': fwd_med,
        'Finviz_TTM_Avg': fttm_avg,
        'Finviz_TTM_Med': fttm_med,
        'Finviz_FWD_Avg': ffwd_avg,
        'Finviz_FWD_Med': ffwd_med
    }


def html_generator2(tickers, financial_data, full_dashboard_html, avg_values, spy_qqq_growth_html=""):
    ensure_templates_exist()
    past     = get_file_content_or_placeholder("charts/earnings_past.html")
    upcoming = get_file_content_or_placeholder("charts/earnings_upcoming.html")

    create_home_page(
        tickers            = tickers,
        output_dir         = '.',
        dashboard_table    = full_dashboard_html,
        avg_values         = avg_values,
        spy_qqq_growth     = spy_qqq_growth_html,
        earnings_past      = past,
        earnings_upcoming  = upcoming
    )

    prepare_and_generate_ticker_pages(tickers, '.', 'charts/')
