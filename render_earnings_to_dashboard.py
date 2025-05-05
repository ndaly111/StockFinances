from ticker_manager import read_tickers, modify_tickers
from html_generator2 import html_generator2, get_file_content_or_placeholder
import os

# Constants
TICKERS_FILE_PATH = 'tickers.csv'
DASHBOARD_HTML_PATH = 'charts/dashboard.html'
SPY_QQQ_GROWTH_PATH = 'charts/spy_qqq_growth.html'
PAST_EARNINGS_PATH = 'charts/earnings_past.html'
UPCOMING_EARNINGS_PATH = 'charts/earnings_upcoming.html'

# Step 1: Load tickers
tickers = modify_tickers(read_tickers(TICKERS_FILE_PATH), is_remote=True)

# Step 2: Load base HTML files
dashboard_html = get_file_content_or_placeholder(DASHBOARD_HTML_PATH)
spy_qqq_growth_html = get_file_content_or_placeholder(SPY_QQQ_GROWTH_PATH)
past_earnings_html = get_file_content_or_placeholder(PAST_EARNINGS_PATH)
upcoming_earnings_html = get_file_content_or_placeholder(UPCOMING_EARNINGS_PATH)

# Step 3: Placeholder average valuation stats
avg_values = {
    'Nicks_TTM_Value_Average': 0,
    'Nicks_Forward_Value_Average': 0,
    'Finviz_TTM_Value_Average': 0,
    'Nicks_TTM_Value_Median': 0,
    'Nicks_Forward_Value_Median': 0,
    'Finviz_TTM_Value_Median': 0,
    'Finviz_Forward_Value_Median': 0
}

# Step 4: Render homepage with earnings tables included
html_generator2(
    tickers=tickers,
    financial_data=None,
    full_dashboard_html=dashboard_html,
    avg_values=avg_values,
    spy_qqq_growth_html=spy_qqq_growth_html,
    earnings_past_html=past_earnings_html,
    earnings_upcoming_html=upcoming_earnings_html
)
