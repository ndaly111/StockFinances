# index_growth_table.py
# ---------------------------------------------------------------------
# Exposes index_growth() for main_remote.py
#   • Logs today’s SPY/QQQ implied growth
#   • Regenerates charts + summary tables
#   • Returns a single HTML snippet combining SPY & QQQ summaries
# ---------------------------------------------------------------------

import os
from datetime import datetime
from log_index_growth import log_index_growth       # stores today’s values
from index_growth_charts import render_index_growth_charts

CHART_DIR = "charts"
SUMMARY_FILES = {
    "SPY":  "spy_growth_summary.html",
    "QQQ":  "qqq_growth_summary.html",
}

def _read_html(path: str, placeholder: str = "No data available.") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"<p>{placeholder}</p>"

def index_growth() -> str:
    """
    Runs the full SPY/QQQ pipeline and returns an HTML block
    ready to embed in the dashboard home page.
    """
    # 1) Update DB with today’s data
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Logging index growth …")
    log_index_growth()

    # 2) Regenerate charts + summary tables
    print("Building growth charts & summary tables …")
    render_index_growth_charts()

    # 3) Combine SPY & QQQ summary tables into one HTML snippet
    blocks = []
    for ticker, fname in SUMMARY_FILES.items():
        full_path = os.path.join(CHART_DIR, fname)
        blocks.append(f"<h3>{ticker}</h3>")
        blocks.append(_read_html(full_path))

    combined_html = "\n".join(blocks)
    return combined_html

# Mini-main for standalone testing
if __name__ == "__main__":
    html_snippet = index_growth()
    out_path = os.path.join(CHART_DIR, "spy_qqq_combined_summary.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_snippet)
    print(f"Wrote combined summary to {out_path}")
