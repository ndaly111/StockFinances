# implied_growth_summary.py
# -----------------------------------------------------------
# Build per-ticker “Implied Growth Summary” HTML table + plot
# -----------------------------------------------------------
import os, sqlite3, numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from itertools import product          # ← for column re-ordering

# ───────────────────────────────────
# Configuration
# ───────────────────────────────────
DB_PATH        = 'Stock Data.db'
TABLE_NAME     = 'Implied_Growth_History'

CHARTS_DIR     = 'charts'
HTML_TEMPLATE  = os.path.join(CHARTS_DIR, '{ticker}_implied_growth_summary.html')
CHART_TEMPLATE = os.path.join(CHARTS_DIR, '{ticker}_implied_growth_plot.png')

TIME_FRAMES = {
    '1 Year':   365,
    '3 Years':  365 * 3,
    '5 Years':  365 * 5,
    '10 Years': 365 * 10,
}

ROW_ORDER   = ['1 Year', '3 Years', '5 Years', '10 Years']   # nicer order
COL_METRICS = ['Average', 'Median', 'Std Dev', 'Current', 'Percentile']
COL_TYPES   = ['TTM', 'Forward']                             # TTM first!

# ───────────────────────────────────
# Helpers
# ───────────────────────────────────
def ensure_output_directory():
    os.makedirs(CHARTS_DIR, exist_ok=True)

def load_growth_data() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        except
