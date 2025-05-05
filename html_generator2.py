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
        with open(template_path, 'r', encoding='utf-8') as file:
            current_content = file.read()
        if current_content == content:
            return
    with open(template_path, 'w', encoding='utf-8') as file:
        file.write(content)

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

    <footer>
        <p>Nick's Financial Data Dashboard</p>
    </footer>
    </body>
    </html>
    """

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

def create_home_page(tickers, output_dir, dashboard_html, avg_values,
                     spy_qqq_growth_html="", earnings_past_html="", earnings_upcoming_html=""):
    template = env.get_template('home_template.html')
    home_page_path = os.path.join(output_dir, 'index.html')
    with open(home_page_path, 'w', encoding='utf-8') as file:
        file.write(template.render(
            tickers=tickers,
            dashboard_table=dashboard_html,
            dashboard_data=avg_values,
            spy_qqq_growth=spy_qqq_growth_html,
            earnings_past=earnings_past_html,
            earnings_upcoming=earnings_upcoming_html
        ))

def prepare_and_generate_ticker_pages(tickers, output_dir, charts_output_dir):
    conn = sqlite3.connect(db_path)
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
            'balance_sheet_chart_path': f"{charts_output_dir}/{ticker}_balance_sheet_chart.png",
            'balance_sheet_table_html': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_balance_sheet_table.html"),
            'revenue_yoy_change_chart_path': f"{charts_output_dir}/{ticker}_revenue_yoy_change.png",
            'eps_yoy_change_chart_path': f"{charts_output_dir}/{ticker}_eps_yoy_change.png",
            'valuation_chart': f"{charts_output_dir}/{ticker}_valuation_chart.png",
            'valuation_info_table': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_valuation_info.html"),
            'valuation_data_table': get_file_content_or_placeholder(f"{charts_output_dir}/{ticker}_valuation_table.html")
        }
        create_ticker_page(ticker, ticker_data, output_dir)
    conn.close()

def get_file_content_or_placeholder(file_path, placeholder="No data available"):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return placeholder

def create_ticker_page(ticker, ticker_data, output_dir):
    template = env.get_template('ticker_template.html')
    page_path = os.path.join(output_dir, 'pages', f'{ticker}_page.html')
    ensure_directory_exists(os.path.dirname(page_path))
    with open(page_path, 'w', encoding='utf-8') as file:
        file.write(template.render(ticker_data=ticker_data))

def generate_dashboard_table(dashboard_data):
    dashboard_df = pd.DataFrame(dashboard_data, columns=[
        "Ticker", "Share Price", "Nicks TTM Valuation", "Nicks TTM Value",
        "Nicks Forward Valuation", "Nicks Forward Value", "Finviz TTM Valuation",
        "Finviz TTM Value", "Finviz Forward Valuation", "Finviz Forward Value"
    ])
    dashboard_df.drop(columns=[
        "Nicks TTM Valuation", "Nicks Forward Valuation", 
        "Finviz TTM Valuation", "Finviz Forward Valuation"
    ], inplace=True)
    dashboard_df["Ticker"] = dashboard_df["Ticker"].apply(
        lambda ticker: f'<a href="pages/{ticker}_page.html">{ticker}</a>'
    )

    def parse_percentage(value):
        try:
            return float(value.strip('%')) if value != "-" else None
        except ValueError:
            return None

    avg_values = {
        'Nicks_TTM_Value_Average': dashboard_df["Nicks TTM Value"].apply(parse_percentage).mean(),
        'Nicks_Forward_Value_Average': dashboard_df["Nicks Forward Value"].apply(parse_percentage).mean(),
        'Finviz_TTM_Value_Average': dashboard_df["Finviz TTM Value"].apply(parse_percentage).mean(),
        'Nicks_TTM_Value_Median': dashboard_df["Nicks TTM Value"].apply(parse_percentage).median(),
        'Nicks_Forward_Value_Median': dashboard_df["Nicks Forward Value"].apply(parse_percentage).median(),
        'Finviz_TTM_Value_Median': dashboard_df["Finviz TTM Value"].apply(parse_percentage).median(),
        'Finviz_Forward_Value_Median': dashboard_df["Finviz Forward Value"].apply(parse_percentage).median()
    }

    avg_values_df = pd.DataFrame([
        ["Average", f"{avg_values['Nicks_TTM_Value_Average']:.1f}%", f"{avg_values['Nicks_Forward_Value_Average']:.1f}%", f"{avg_values['Finviz_TTM_Value_Average']:.1f}%", f"{avg_values['Finviz_Forward_Value_Median']:.1f}%"],
        ["Median", f"{avg_values['Nicks_TTM_Value_Median']:.1f}%", f"{avg_values['Nicks_Forward_Value_Median']:.1f}%", f"{avg_values['Finviz_TTM_Value_Median']:.1f}%", f"{avg_values['Finviz_Forward_Value_Median']:.1f}%"]
    ], columns=["Metric", "Nicks TTM Value", "Nicks Forward Value", "Finviz TTM Value", "Finviz Forward Value"])

    avg_values_html = avg_values_df.to_html(index=False, escape=False, classes='table table-striped')
    dashboard_html = dashboard_df.to_html(index=False, escape=False, classes='table table-striped', table_id="sortable-table")
    with open('charts/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(avg_values_html + dashboard_html)
    return avg_values_html + dashboard_html, avg_values

def html_generator2(tickers, financial_data, full_dashboard_html, avg_values, spy_qqq_growth_html=""):
    print("Starting HTML generation process...")
    ensure_templates_exist()

    earnings_past_html = get_file_content_or_placeholder("charts/earnings_past.html")
    earnings_upcoming_html = get_file_content_or_placeholder("charts/earnings_upcoming.html")

    create_home_page(
        tickers=tickers,
        output_dir='.',
        dashboard_html=full_dashboard_html,
        avg_values=avg_values,
        spy_qqq_growth_html=spy_qqq_growth_html,
        earnings_past_html=earnings_past_html,
        earnings_upcoming_html=earnings_upcoming_html
    )

    for ticker in tickers:
        prepare_and_generate_ticker_pages([ticker], '.', 'charts/')
