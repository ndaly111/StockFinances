from jinja2 import Environment, FileSystemLoader
import os
import pandas as pd
import sqlite3
import yfinance as yf

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
    # Check if the template exists and compare its content
    if os.path.exists(template_path):
        with open(template_path, 'r') as file:
            current_content = file.read()
        if current_content == content:
            print(f"No changes needed for template {template_path}.")
            return
    # Update or create template file
    with open(template_path, 'w') as file:
        file.write(content)
    print(f"Template {template_path} has been updated or created.")


def ensure_templates_exist():
    print("Ensuring that all necessary templates exist...")
    home_template_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Nick's Stock Financials</title>
        <link rel="stylesheet" href="style.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
        <style>
            .positive {
                color: green;
            }
            .negative {
                color: red;
            }
        </style>
        <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
        <script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
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
        <header>
            <h1>Financial Overview</h1>
        </header>
        <nav class="navigation">
            {% for ticker in tickers %}
            <a href="pages/{{ ticker }}_page.html" class="home-button">{{ ticker }}</a> |
            {% endfor %}
        </nav>
        
        <br><br><br>
        <div>
            <!-- Main sortable table -->
            {{ dashboard_table | safe }}
        </div>
        <footer>
            <p>Nick's Financial Data Dashboard</p>
        </footer>
    </body>
    </html>

    """

    # Define the ticker-specific template content
    ticker_template_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{{ ticker_data.ticker }} - Financial Overview</title>
        <link rel="stylesheet" href="../style.css">
    </head>
    <body>
        <header>
            <a href="../index.html" class="home-button">Home</a>
            <h1>{{ ticker_data.company_name }} - Financial Overview</h1>
            <h2>Ticker - {{ ticker_data.ticker }}</h2>
        </header>

        <!-- Section for ticker information and summary -->
        <section>
            <p>{{ ticker_data.ticker_info | safe }}</p>
        </section>

        <!-- Section for financial charts and tables -->
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
            <div class="carousel-item">
                {{ ticker_data.yoy_growth_table_html | safe }}
            </div>
        </div>

        <!-- New Carousel for YoY Growth Charts -->
        <div><br><br><h1>{{ ticker_data.ticker }} - Y/Y % Change</h1></div>
        <div class="carousel-container">
            <div class="carousel-item">
                <img src="../{{ ticker_data.revenue_yoy_change_chart_path }}" alt="Revenue Year-over-Year Change Chart">
            </div>
            <div class="carousel-item">
                <img src="../{{ ticker_data.eps_yoy_change_chart_path }}" alt="EPS Year-over-Year Change Chart">
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

    templates_dir = 'templates'
    create_template(os.path.join(templates_dir, 'home_template.html'), home_template_content)
    create_template(os.path.join(templates_dir, 'ticker_template.html'), ticker_template_content)


def create_home_page(tickers, output_dir, full_dashboard_html, avg_values):
    print(f"Creating home page in {output_dir}...")
    template = env.get_template('home_template.html')
    home_page_path = os.path.join(output_dir, 'index.html')
    with open(home_page_path, 'w') as file:
        file.write(template.render(tickers=tickers, dashboard_table=full_dashboard_html, dashboard_data=avg_values))
    print(f"Home page created at {home_page_path}")



def prepare_and_generate_ticker_pages(tickers, output_dir, charts_output_dir):
    """Prepares and generates individual HTML pages for each ticker using provided charts."""
    print(f"Preparing and generating pages for tickers in {output_dir} using charts from {charts_output_dir}...")
    if not charts_output_dir.endswith('/'):
        charts_output_dir += '/'

    # Establish a connection to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for ticker in tickers:
        company_name = get_company_short_name(ticker, cursor)

        valuation_chart_path = f"{charts_output_dir}{ticker}_valuation_chart.png"
        # Check if the valuation chart exists
        if os.path.exists(valuation_chart_path):
            valuation_chart = valuation_chart_path
        else:
            valuation_chart = None  # Or provide a path to a default placeholder image

        ticker_data = {
            'ticker': ticker,
            'company_name': company_name,
            'ticker_info': get_file_content_or_placeholder(f"{charts_output_dir}{ticker}_ticker_info.html",
                                                           "Ticker info not available"),
            'revenue_net_income_chart_path': f"{charts_output_dir}{ticker}_revenue_net_income_chart.png",
            'eps_chart_path': f"{charts_output_dir}{ticker}_eps_chart.png",
            'financial_table': get_file_content_or_placeholder(f"{charts_output_dir}{ticker}_rev_net_table.html",
                                                               "Financial table not available"),
            'forecast_rev_net_chart_path': f"{charts_output_dir}{ticker}_Revenue_Net_Income_Forecast.png",
            'forecast_eps_chart_path': f"{charts_output_dir}{ticker}_EPS_Forecast.png",
            'yoy_growth_table_html': get_file_content_or_placeholder(f"{charts_output_dir}{ticker}_yoy_growth_tbl.html",
                                                                     "No Year-Over-Year Growth data available"),
            'balance_sheet_chart_path': f"{charts_output_dir}{ticker}_balance_sheet_chart.png",
            'balance_sheet_table_html': get_file_content_or_placeholder(
                f"{charts_output_dir}{ticker}_balance_sheet_table.html", "Balance sheet data not available"),
            'revenue_yoy_change_chart_path': f"{charts_output_dir}{ticker}_revenue_yoy_change.png",
            'eps_yoy_change_chart_path': f"{charts_output_dir}{ticker}_eps_yoy_change.png",
            'valuation_chart': valuation_chart,
            'valuation_info_table': get_file_content_or_placeholder(f"{charts_output_dir}{ticker}_valuation_info.html",
                                                                    "Valuation info not available"),
            'valuation_data_table': get_file_content_or_placeholder(f"{charts_output_dir}{ticker}_valuation_table.html",
                                                                    "Valuation data not available")
        }

        create_ticker_page(ticker, ticker_data, output_dir)


def get_file_content_or_placeholder(file_path, placeholder="No data available"):
    print(f"Retrieving content from {file_path} or using placeholder...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File {file_path} not found. Using placeholder.")
        return placeholder


def create_ticker_page(ticker, ticker_data, output_dir):
    print(f"Creating page for ticker: {ticker}")
    template = env.get_template('ticker_template.html')
    page_path = os.path.join(output_dir, 'pages', f'{ticker}_page.html')
    ensure_directory_exists(os.path.dirname(page_path))
    with open(page_path, 'w') as file:
        file.write(template.render(ticker_data=ticker_data))
    print(f"Generated page for {ticker} at {page_path}")


def generate_dashboard_table(dashboard_data):
    dashboard_df = pd.DataFrame(dashboard_data, columns=[
        "Ticker", "Share Price", "Nicks TTM Valuation", "Nicks TTM Value",
        "Nicks Forward Valuation", "Nicks Forward Value", "Finviz TTM Valuation",
        "Finviz TTM Value", "Finviz Forward Valuation", "Finviz Forward Value"
    ])

    def parse_percentage(value):
        try:
            return float(value.strip('%')) if value != "-" else None
        except ValueError:
            return None

    # Calculate average TTM and Forward Valuation values
    avg_nicks_ttm = dashboard_df["Nicks TTM Value"].apply(parse_percentage).mean()
    avg_nicks_forward = dashboard_df["Nicks Forward Value"].apply(parse_percentage).mean()
    avg_finviz_ttm = dashboard_df["Finviz TTM Value"].apply(parse_percentage).mean()
    avg_finviz_forward = dashboard_df["Finviz Forward Value"].apply(parse_percentage).mean()

    # Calculate medians
    median_nicks_ttm = dashboard_df["Nicks TTM Value"].apply(parse_percentage).median()
    median_nicks_forward = dashboard_df["Nicks Forward Value"].apply(parse_percentage).median()
    median_finviz_ttm = dashboard_df["Finviz TTM Value"].apply(parse_percentage).median()
    median_finviz_forward = dashboard_df["Finviz Forward Value"].apply(parse_percentage).median()

    avg_values = {
        'Nicks_TTM_Value_Average': avg_nicks_ttm,
        'Nicks_Forward_Value_Average': avg_nicks_forward,
        'Finviz_TTM_Value_Average': avg_finviz_ttm,
        'Nicks_TTM_Value_Median': median_nicks_ttm,
        'Nicks_Forward_Value_Median': median_nicks_forward,
        'Finviz_TTM_Value_Median': median_finviz_ttm,
        'Finviz_Forward_Value_Median': median_finviz_forward
    }

    avg_values_df = pd.DataFrame([
        ["Average", f"{avg_nicks_ttm:.1f}%", f"{avg_nicks_forward:.1f}%", f"{avg_finviz_ttm:.1f}%", f"{avg_finviz_forward:.1f}%"],
        ["Median", f"{median_nicks_ttm:.1f}%", f"{median_nicks_forward:.1f}%", f"{median_finviz_ttm:.1f}%", f"{median_finviz_forward:.1f}%"]
    ], columns=["Metric", "Nicks TTM Value", "Nicks Forward Value", "Finviz TTM Value", "Finviz Forward Value"])

    avg_values_html = avg_values_df.to_html(index=False, escape=False, classes='table table-striped', justify='left')

    # Append the main dashboard data table
    dashboard_html = dashboard_df.to_html(index=False, escape=False, classes='table table-striped', justify='left', table_id="sortable-table")

    full_dashboard_html = avg_values_html + dashboard_html

    dashboard_path = os.path.join('charts', "dashboard.html")
    with open(dashboard_path, "w") as file:
        file.write(full_dashboard_html)

    print(f"Dashboard saved to {dashboard_path}")

    return full_dashboard_html, avg_values



def create_home_page(tickers, output_dir, dashboard_html, avg_values):
    print(f"Creating home page in {output_dir}...")
    template = env.get_template('home_template.html')
    home_page_path = os.path.join(output_dir, 'index.html')
    with open(home_page_path, 'w') as file:
        file.write(template.render(tickers=tickers, dashboard_table=dashboard_html, dashboard_data=avg_values))
    print(f"Home page created at {home_page_path}")


def get_company_short_name(ticker, cursor):
    """Fetch the company short name for a given ticker."""
    cursor.execute("SELECT short_name FROM Tickers_Info WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()

    if result and result[0]:  # Check if result exists and is not empty
        return result[0]
    else:
        stock = yf.Ticker(ticker)
        short_name = stock.info.get('shortName', '').strip()  # Fetch and strip any surrounding whitespace

        if short_name:  # Check if short name is not empty
            # Update the database with the fetched short name
            cursor.execute("UPDATE Tickers_Info SET short_name = ? WHERE ticker = ?", (short_name, ticker))
            cursor.connection.commit()
            return short_name
        else:
            return ticker

def html_generator2(tickers, financial_data, full_dashboard_html, avg_values):
    output_dir = '.'  # Define the main directory for output
    print("Starting HTML generation process...")
    ensure_templates_exist()
    create_home_page(tickers, output_dir, full_dashboard_html, avg_values)
    for ticker in tickers:
        prepare_and_generate_ticker_pages([ticker], output_dir, 'charts/')
    else:
        print(f"No data available for ticker: {ticker}")