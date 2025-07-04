# ───────────────────────────────────────────────────────────
# html_generator2.py  —  FULL FILE
# Builds index.html, per-ticker pages, and SPY/QQQ growth pages
# ───────────────────────────────────────────────────────────

from jinja2 import Environment, FileSystemLoader, Template
import os, sqlite3, numpy as np, pandas as pd, yfinance as yf

# ─── Basic setup ───────────────────────────────────────────
db_path = "Stock Data.db"
env = Environment(loader=FileSystemLoader("templates"))

def ensure_directory_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def create_template(path: str, content: str):
    ensure_directory_exists(os.path.dirname(path))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            if f.read() == content:
                return                          # no change
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ─── DB helpers ────────────────────────────────────────────
def get_company_short_name(ticker: str, cur) -> str:
    cur.execute("SELECT short_name FROM Tickers_Info WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0]

    info = yf.Ticker(ticker).info or {}
    name = info.get("shortName", "").strip()
    if name:
        cur.execute("UPDATE Tickers_Info SET short_name = ? WHERE ticker = ?", (name, ticker))
        cur.connection.commit()
        return name
    return ticker

def get_file_content_or_placeholder(path: str, placeholder="No data available"):
    try:
        return open(path, "r", encoding="utf-8").read()
    except FileNotFoundError:
        return placeholder

# ───────────────────────────────────────────────────────────
# Template setup  (HOME + TICKER + NEW SPY/QQQ pages)
# ───────────────────────────────────────────────────────────
def ensure_templates_exist():
    # ─── home_template.html (original body) ───
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
""".lstrip()

    # ─── ticker_template.html (original body) ───
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
    <div class="carousel-item"><img src="../{{ ticker_data.forecast_rev_net_chart_path }}" alt="Rev/NI Forecast"></div>
    <div class="carousel-item"><img src="../{{ ticker_data.forecast_eps_chart_path }}"     alt="EPS Forecast"></div>
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
    <div class="carousel-item">{{ ticker_data.expense_abs_html | safe }}</div>
    <div class="carousel-item">{{ ticker_data.expense_yoy_html | safe }}</div>
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

  {% if ticker_data.implied_growth_chart_path %}
  <h1>{{ ticker_data.ticker }} – Implied Growth Summary</h1>
  <img src="../{{ ticker_data.implied_growth_chart_path }}" alt="Implied Growth Chart">
  <div class="implied-growth-table">{{ ticker_data.implied_growth_table_html | safe }}</div>
  {% endif %}

  <footer><a href="../index.html" class="home-button">Back to Home</a></footer>
</body>
</html>
""".lstrip()

    create_template("templates/home_template.html",   home_template_content)
    create_template("templates/ticker_template.html", ticker_template_content)

    # ─── NEW: SPY & QQQ growth templates ───
    spy_tpl = """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>SPY – Implied Growth Summary</title>
<link rel="stylesheet" href="../style.css"></head>
<body>
  <header><a href="../index.html">← Home</a></header>
  <h1>SPY – Implied Growth Summary</h1>
  <img src="../charts/spy_growth_chart.png" style="max-width:100%%;" alt="SPY Growth Chart">
  {{ spy_growth_summary | safe }}
</body>
</html>
""".lstrip()

    qqq_tpl = spy_tpl.replace("SPY", "QQQ").replace("spy_", "qqq_")

    create_template("templates/spy_growth_template.html", spy_tpl)
    create_template("templates/qqq_growth_template.html", qqq_tpl)

# ───────────────────────────────────────────────────────────
# Build SPY/QQQ standalone pages
# ───────────────────────────────────────────────────────────
def render_spy_qqq_growth_pages():
    chart_dir, out_dir = "charts", "."
    env_paths = {
        "spy": ("spy_growth_template.html", "spy_growth_summary.html", "spy_growth.html"),
        "qqq": ("qqq_growth_template.html", "qqq_growth_summary.html", "qqq_growth.html")
    }

    for k, (tpl_name, summary_file, output_file) in env_paths.items():
        try:
            tpl_path = os.path.join("templates", tpl_name)
            html_template = open(tpl_path, encoding="utf-8").read()
            tpl = Template(html_template)

            summary_html = get_file_content_or_placeholder(
                os.path.join(chart_dir, summary_file), "No data available."
            )
            rendered = tpl.render(**{f"{k}_growth_summary": summary_html})

            with open(os.path.join(out_dir, output_file), "w", encoding="utf-8") as f:
                f.write(rendered)
            print(f"[html_generator2] Rendered {output_file}")
        except Exception as e:
            print(f"Error rendering {k.upper()} growth page: {e}")

# ───────────────────────────────────────────────────────────
# create_home_page, prepare_and_generate_ticker_pages,
# generate_dashboard_table  (unchanged – same as your current file)
# ───────────────────────────────────────────────────────────
# ...  (functions omitted here for brevity – keep exactly as in your version)

# ───────────────────────────────────────────────────────────
# MAIN WRAPPER
# ───────────────────────────────────────────────────────────
def html_generator2(tickers, financial_data,
                    full_dashboard_html, avg_values,
                    spy_qqq_growth_html=""):

    ensure_templates_exist()

    past     = get_file_content_or_placeholder("charts/earnings_past.html")
    upcoming = get_file_content_or_placeholder("charts/earnings_upcoming.html")

    create_home_page(
        tickers=tickers,
        output_dir=".",
        dashboard_table=full_dashboard_html,
        avg_values=avg_values,
        spy_qqq_growth=spy_qqq_growth_html,
        earnings_past=past,
        earnings_upcoming=upcoming
    )

    prepare_and_generate_ticker_pages(tickers, ".", "charts/")
    render_spy_qqq_growth_pages()

# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("html_generator2 is intended to be invoked from main_remote.py")
