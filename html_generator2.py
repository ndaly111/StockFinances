from jinja2 import Environment, FileSystemLoader
import os
import pandas as pd
import sqlite3
import yfinance as yf

#push
db_path = 'Stock Data.db'

env = Environment(loader=FileSystemLoader('templates'))

def ensure_directory_exists(directory):
    print(f"Checking if directory {directory} exists...")
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Creating directory...")
        os.makedirs(directory)
    else:
        print(f"Directory {directory} already exists.")

def create_template(template_path, content):
    print(f"Creating/updating template at: {template_path}")
    ensure_directory_exists(os.path.dirname(template_path))
    if os.path.exists(template_path):
        with open(template_path, 'r') as file:
            current_content = file.read()
        if current_content == content:
            print(f"No changes needed for template {template_path}.")
            return
    with open(template_path, 'w') as file:
        file.write(content)
    print(f"Template {template_path} has been updated or created.")

def get_company_short_name(ticker, cursor):
    cursor.execute("SELECT short_name FROM Tickers_Info WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()
    if result and result[0]:
        return result[0]
    else:
        stock = yf.Ticker(ticker)
        short_name = stock.info.get('shortName', '').strip()
        if short_name:
            cursor.execute("UPDATE Tickers_Info SET short_name = ? WHERE ticker = ?", (short_name, ticker))
            cursor.connection.commit()
            return short_name
        else:
            return ticker

def ensure_templates_exist():
    print("Ensuring that all necessary templates exist...")
    home_template_content = """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <title>Nick's Stock Financials</title>
        <link rel=\"stylesheet\" href=\"style.css\">
        <link rel=\"stylesheet\" href=\"https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css\">
        <style>
            .positive { color: green; }
            .negative { color: red; }
        </style>
        <script src=\"https://code.jquery.com/jquery-3.5.1.js\"></script>
        <script src=\"https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js\"></script>
        <script>
            $(document).ready(function() {
                $('#sortable-table').DataTable({
                    "pageLength": 100,
                    "createdRow": function(row, data, dataIndex) {
                        $('td', row).each(function() {
                            var cellValue = $(this).text();
                            if (cellValue.includes('%')) {
                                var value = parseFloat(cellValue.replace('%', ''));
                                if (!isNaN(value)) {
                                    if (value < 0) {
                                        $(this).addClass('negative');
                                    } else {
                                        $(this).addClass('positive');
                                    }
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
        <nav class=\"navigation\">
            {% for ticker in tickers %}
            <a href=\"pages/{{ ticker }}_page.html\" class=\"home-button\">{{ ticker }}</a> |
            {% endfor %}
        </nav>

        <br><br><br>
        <div>{{ dashboard_table | safe }}</div>

        <br><br><hr><br>
        <div id=\"spy-qqq-growth\">
            <h2>SPY & QQQ Growth Metrics</h2>
            {{ spy_qqq_growth | safe }}
        </div>

        <footer>
            <p>Nick's Financial Data Dashboard</p>
        </footer>
    </body>
    </html>
    """

    ticker_template_content = """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <title>{{ ticker_data.ticker }} - Financial Overview</title>
        <link rel=\"stylesheet\" href=\"../style.css\">
    </head>
    <body>
        <header>
            <a href=\"../index.html\" class=\"home-button\">Home</a>
            <h1>{{ ticker_data.company_name }} - Financial Overview</h1>
            <h2>Ticker - {{ ticker_data.ticker }}</h2>
        </header>
        <section><p>{{ ticker_data.ticker_info | safe }}</p></section>
        <div>
            <img src=\"../{{ ticker_data.revenue_net_income_chart_path }}\" alt=\"Revenue and Net Income Chart\">
            <img src=\"../{{ ticker_data.eps_chart_path }}\" alt=\"EPS Chart\">
            {{ ticker_data.financial_table | safe }}
        </div>
        <div><br><br><hr><br><h1>{{ ticker_data.ticker }} - Forecast Data</h1></div>
        <div class=\"carousel-container\">
            <div class=\"carousel-item\">
                <img src=\"../{{ ticker_data.forecast_rev_net_chart_path }}\" alt=\"Revenue and Net Income Forecast Chart\">
            </div>
            <div class=\"carousel-item\">
                <img src=\"../{{ ticker_data.forecast_eps_chart_path }}\" alt=\"EPS Forecast Chart\">
            </div>
            <div class=\"carousel-item\">
                {{ ticker_data.yoy_growth_table_html | safe }}
            </div>
        </div>
        <div><br><br><h1>{{ ticker_data.ticker }} - Y/Y % Change</h1></div>
        <div class=\"carousel-container\">
            <div class=\"carousel-item\">
                <img src=\"../{{ ticker_data.revenue_yoy_change_chart_path }}\" alt=\"Revenue Year-over-Year Change Chart\">
            </div>
            <div class=\"carousel-item\">
                <img src=\"../{{ ticker_data.eps_yoy_change_chart_path }}\" alt=\"EPS Year-over-Year Change Chart\">
            </div>
        </div>
        <div class=\"balance-sheet-container\">
            <div class=\"balance-sheet-table\">{{ ticker_data.balance_sheet_table_html | safe }}</div>
            <div class=\"balance-sheet-chart\">
                <img src=\"../{{ ticker_data.balance_sheet_chart_path }}\" alt=\"{{ ticker_data.ticker }} Balance Sheet Chart\">
            </div>
        </div>
        <hr>
        {% if ticker_data.valuation_chart %}
        <div><br><br><h1>{{ ticker_data.ticker }} - Valuation Chart</h1></div>
        <div><br>
            <img src=\"../{{ ticker_data.valuation_chart }}\" alt=\"Valuation Chart\">
            <br><br>
            <div class=\"valuation-tables\">
                {{ ticker_data.valuation_info_table | safe }}
                {{ ticker_data.valuation_data_table | safe }}
            </div>
            <br><br><br><hr>
        </div>
        {% endif %}
        <footer>
            <a href=\"../index.html\" class=\"home-button\">Back to Home</a>
            <br><br><br><br><br>
        </footer>
    </body>
    </html>
    """

    templates_dir = 'templates'
    create_template(os.path.join(templates_dir, 'home_template.html'), home_template_content)
    create_template(os.path.join(templates_dir, 'ticker_template.html'), ticker_template_content)

def create_home_page(tickers, output_dir, dashboard_html, avg_values, spy_qqq_growth_html=""):
    print(f"Creating home page in {output_dir}...")
    template = env.get_template('home_template.html')
    home_page_path = os.path.join(output_dir, 'index.html')
    with open(home_page_path, 'w') as file:
        file.write(template.render(
            tickers=tickers,
            dashboard_table=dashboard_html,
            dashboard_data=avg_values,
            spy_qqq_growth=spy_qqq_growth_html
        ))
    print(f"Home page created at {home_page_path}")
