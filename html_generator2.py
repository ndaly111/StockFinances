from jinja2 import Environment, FileSystemLoader
import os
import pandas as pd
import sqlite3
import yfinance as yf

# Path to the database file
db_path = 'Stock Data.db'

env = Environment(loader=FileSystemLoader('templates'))

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_template(template_path, content):
    ensure_directory_exists(os.path.dirname(template_path))
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
        cursor.execute("UPDATE Tickers_Info SET short_name = ? WHERE ticker = ?", (short_name, ticker))
        cursor.connection.commit()
        return short_name
    return ticker

def ensure_templates_exist():
    ticker_template_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>{{ ticker_data.company_name }} - Financial Overview</title>
      <link rel="stylesheet" href="../style.css">
    </head>
    <body>
      <header>
        <a href="../index.html" class="home-button">Home</a>
        <h1>{{ ticker_data.company_name }} - Financial Overview</h1>
        <h2>Ticker - {{ ticker_data.ticker }}</h2>
      </header>

      <section>
        <p>{{ ticker_data.ticker_info | safe }}</p>
      </section>

      <div>
        <img src="../{{ ticker_data.revenue_net_income_chart_path }}" alt="Revenue and Net Income Chart">
        <img src="../{{ ticker_data.eps_chart_path }}" alt="EPS Chart">
        {{ ticker_data.financial_table | safe }}
      </div>

      <div><br><br><hr><br><h1>{{ ticker_data.ticker }} - Forecast Data</h1></div>
      <div class="carousel-container">
        <div class="carousel-item">
          <img src="../{{ ticker_data.forecast_rev_net_chart_path }}" alt="Revenue and Net Income Forecast Chart">
        </div>
        <div class="carousel-item">
          <img src="../{{ ticker_data.forecast_eps_chart_path }}" alt="EPS Forecast Chart">
        </div>
      </div>

      <div><br><br><h1>{{ ticker_data.ticker }} - Y/Y % Change</h1></div>
      <div class="carousel-container">
        <div class="carousel-item">
          <img src="../{{ ticker_data.revenue_yoy_change_chart_path }}" alt="Revenue Year-over-Year Change Chart">
        </div>
        <div class="carousel-item">
          <img src="../{{ ticker_data.eps_yoy_change_chart_path }}" alt="EPS Year-over-Year Change Chart">
        </div>
        <div class="carousel-item">
          {{ ticker_data.yoy_growth_table_html | safe }}
        </div>
      </div>

      <div class="balance-sheet-container">
        <div class="balance-sheet-table">
          {{ ticker_data.balance_sheet_table_html | safe }}
        </div>
        <div class="balance-sheet-chart">
          <img src="../{{ ticker_data.balance_sheet_chart_path }}" alt="{{ ticker_data.ticker }} Balance Sheet Chart">
        </div>
      </div>

      <div><br><br><h1>{{ ticker_data.ticker }} - Expense Overview</h1></div>
      <div class="carousel-container">
        <div class="carousel-item">
          <img src="../{{ ticker_data.expense_chart_path }}" alt="Revenue vs Expense Chart">
        </div>
        <div class="carousel-item">
          <img src="../{{ ticker_data.expense_percent_chart_path }}" alt="Expense % of Revenue Chart">
        </div>
        <div class="carousel-item">
          {{ ticker_data.expense_table_html | safe }}
        </div>
        <div class="carousel-item">
          {{ ticker_data.expense_yoy_table_html | safe }}
        </div>
      </div>

      <!-- NEW: Unmapped Line Items -->
      <div><br><br><h1>{{ ticker_data.ticker }} - Unmapped Expense Line Items</h1></div>
      <div>
        {{ ticker_data.unmapped_expense_html | safe }}
      </div>

      <hr>
      {% if ticker_data.valuation_chart %}
      <div><br><br><h1>{{ ticker_data.ticker }} - Valuation Chart</h1></div>
      <div><br>
        <img src="../{{ ticker_data.valuation_chart }}" alt="Valuation Chart">
        <br><br>
        <div class="valuation-tables">
          {{ ticker_data.valuation_info_table | safe }}
          {{ ticker_data.valuation_data_table | safe }}
        </div>
        <br><br><br><hr>
      </div>
      {% endif %}

      <footer>
        <a href="../index.html" class="home-button">Back to Home</a>
        <br><br><br><br><br>
      </footer>
    </body>
    </html>
    """
    create_template(os.path.join('templates', 'ticker_template.html'), ticker_template_content)

def get_file_content_or_placeholder(file_path, placeholder="No data available"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return placeholder

def create_home_page(tickers, output_dir, dashboard_table, avg_values,
                     spy_qqq_growth="", earnings_past="", earnings_upcoming=""):
    template = env.get_template('home_template.html')
    out_path = os.path.join(output_dir, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(template.render(
            tickers=tickers,
            dashboard_table=dashboard_table,
            dashboard_data=avg_values,
            spy_qqq_growth=spy_qqq_growth,
            earnings_past=earnings_past,
            earnings_upcoming=earnings_upcoming
        ))

def prepare_and_generate_ticker_pages(tickers, output_dir, charts_output_dir):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for ticker in tickers:
            company_name = get_company_short_name(ticker, cursor)
            ticker_data = {
                'ticker': ticker,
                'company_name': company_name,
                'ticker_info': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_ticker_info.html"),
                'revenue_net_income_chart_path': f"{charts_output_dir}/{ticker}_revenue_net_income_chart.png",
                'eps_chart_path': f"{charts_output_dir}/{ticker}_eps_chart.png",
                'financial_table': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_rev_net_table.html"),
                'forecast_rev_net_chart_path': f"{charts_output_dir}/{ticker}_Revenue_Net_Income_Forecast.png",
                'forecast_eps_chart_path': f"{charts_output_dir}/{ticker}_EPS_Forecast.png",
                'yoy_growth_table_html': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_yoy_growth_tbl.html"),
                'expense_chart_path': f"{charts_output_dir}/{ticker}_rev_expense_chart.png",
                'expense_percent_chart_path': f"{charts_output_dir}/{ticker}_expense_percent_chart.png",
                'expense_table_html': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_yearly_financials.html"),
                'expense_yoy_table_html': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_yoy_expense_change.html"),
                'balance_sheet_chart_path': f"{charts_output_dir}/{ticker}_balance_sheet_chart.png",
                'balance_sheet_table_html': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_balance_sheet_table.html"),
                'revenue_yoy_change_chart_path': f"{charts_output_dir}/{ticker}_revenue_yoy_change.png",
                'eps_yoy_change_chart_path': f"{charts_output_dir}/{ticker}_eps_yoy_change.png",
                'valuation_chart': f"{charts_output_dir}/{ticker}_valuation_chart.png",
                'valuation_info_table': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_valuation_info.html"),
                'valuation_data_table': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_valuation_table.html"),
                'unmapped_expense_html': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_unmapped_fields.html", "No unmapped expenses.")
            }
            create_ticker_page(ticker, ticker_data, output_dir)

def create_ticker_page(ticker, ticker_data, output_dir):
    template = env.get_template('ticker_template.html')
    page_path = os.path.join(output_dir, 'pages', f'{ticker}_page.html')
    ensure_directory_exists(os.path.dirname(page_path))
    with open(page_path, 'w', encoding='utf-8') as f:
        f.write(template.render(ticker_data=ticker_data))

def html_generator2(tickers, financial_data, full_dashboard_html,
                    avg_values, spy_qqq_growth_html=""):
    ensure_templates_exist()

    past = get_file_content_or_placeholder("charts/earnings_past.html")
    upcoming = get_file_content_or_placeholder("charts/earnings_upcoming.html")

    create_home_page(
        tickers=tickers,
        output_dir='.',
        dashboard_table=full_dashboard_html,
        avg_values=avg_values,
        spy_qqq_growth=spy_qqq_growth_html,
        earnings_past=past,
        earnings_upcoming=upcoming
    )

    prepare_and_generate_ticker_pages(tickers, '.', 'charts/')
