# render_earnings_to_dashboard.py

from ticker_manager import read_tickers, modify_tickers
from html_generator2 import html_generator2, get_file_or_placeholder
from generate_earnings_tables import generate_earnings_tables
import os

# Constants
TICKERS_FILE_PATH       = 'tickers.csv'
DASHBOARD_HTML_PATH     = 'charts/dashboard.html'
SPY_QQQ_GROWTH_PATH     = 'charts/spy_qqq_growth.html'

# Step 1: Load tickers
tickers = modify_tickers(read_tickers(TICKERS_FILE_PATH), is_remote=True)

# Refresh earnings tables so the dashboard shows current data
# Use CI environment flag to bypass interactive prompts.
os.environ.setdefault("CI", "1")
generate_earnings_tables()

# Step 2: Load base dashboard HTML
dashboard_html = get_file_or_placeholder(DASHBOARD_HTML_PATH)

# Step 3: Load SPY/QQQ growth fragment
spy_qqq_growth_html = get_file_or_placeholder(SPY_QQQ_GROWTH_PATH)

# Step 4: Placeholder valuation stats
avg_values = {
    'Nicks_TTM_Value_Average': 0,
    'Nicks_Forward_Value_Average': 0,
    'Finviz_TTM_Value_Average': 0,
    'Nicks_TTM_Value_Median': 0,
    'Nicks_Forward_Value_Median': 0,
    'Finviz_TTM_Value_Median': 0,
    'Finviz_Forward_Value_Median': 0
}

# Step 5: Render homepage (earnings HTML will be loaded internally)
html_generator2(
    tickers=tickers,
    financial_data=None,
    full_dashboard_html=dashboard_html,
    avg_values=avg_values,
    spy_qqq_growth_html=spy_qqq_growth_html
)