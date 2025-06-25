# implied_growth_summary.py

from jinja2 import Environment, FileSystemLoader
import sqlite3
import os
from datetime import datetime

TEMPLATE_DIR = 'templates'
TEMPLATE_FILE = 'implied_growth_template.html'
DB_PATH = 'Stock Data.db'
OUTPUT_HTML = 'charts/implied_growth_summary.html'

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

def generate_all_summaries():
    if not os.path.exists(DB_PATH):
        print("Database not found:", DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT ticker, date, type, growth_rate FROM implied_growth ORDER BY date DESC")
        rows = cursor.fetchall()
    except Exception as e:
        print("Error querying implied_growth table:", e)
        conn.close()
        return

    data = {}
    for ticker, date, typ, val in rows:
        if isinstance(val, complex) or val is None:
            print(f"Skipping complex or None value for {ticker}: {val}")
            continue
        if abs(val) > 1e6:
            print(f"Skipping unrealistic value for {ticker}: {val}")
            continue

        if ticker not in data:
            data[ticker] = {"ttm": None, "fwd": None}
        if typ == "TTM":
            data[ticker]["ttm"] = val
        elif typ == "Forward":
            data[ticker]["fwd"] = val

    final = []
    for ticker, vals in data.items():
        if vals["ttm"] is not None and vals["fwd"] is not None:
            final.append({
                "ticker": ticker,
                "ttm_growth": round(vals["ttm"] * 100, 2),
                "fwd_growth": round(vals["fwd"] * 100, 2)
            })

    template = env.get_template(TEMPLATE_FILE)
    html = template.render(records=final, last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("Implied growth summary HTML written successfully.")

# DO NOT CHANGE THE MINI MAIN NAME
generate_all_summaries()
