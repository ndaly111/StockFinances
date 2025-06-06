#!/usr/bin/env python
"""
expense_reports.py  (2025-06-06 refactor, table-only wipe)
---------------------------------------------------------
• Drops incorrect expense tables but keeps the rest of Stock Data.db
• Imports label lists from expense_labels.py
• Sums duplicate labels, stores zeros, fills missing years
• Generates the same two PNG charts
"""

import os, sqlite3
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import FuncFormatter

# ──────────────────────────────────────────
#  Config
# ──────────────────────────────────────────
DB_PATH   = "Stock Data.db"
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def reset_expense_tables():
    """Drop only the old expense tables, leave other data intact."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS QuarterlyIncomeStatement;")
    cur.execute("DROP TABLE IF EXISTS IncomeStatement;")
    conn.commit(); conn.close()

reset_expense_tables()

# ──────────────────────────────────────────
#  Import master label lists
# ──────────────────────────────────────────
from expense_labels import (
    COST_OF_REVENUE,
    RESEARCH_AND_DEVELOPMENT,
    SELLING_AND_MARKETING,
    GENERAL_AND_ADMIN,
    SGA_COMBINED,
)

BUCKET_MAP = {
    "cost_of_revenue":          COST_OF_REVENUE,
    "research_and_development": RESEARCH_AND_DEVELOPMENT,
    "selling_and_marketing":    SELLING_AND_MARKETING,
    "general_and_admin":        GENERAL_AND_ADMIN,
    "sga_combined":             SGA_COMBINED,
}
DB_COLS = list(BUCKET_MAP.keys()) + ["total_revenue"]

# ── [remaining code unchanged: helper funcs, SQLite I/O, charts, orchestrator] ──
# ... (use the same body as in the previous full file I provided) ...
