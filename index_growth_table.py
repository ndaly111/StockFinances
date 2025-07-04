# index_growth_table.py
# ---------------------------------------------------------------------
# Provides the public mini-main  index_growth()
#   • Logs today’s SPY + QQQ implied growth to the DB
#   • Regenerates charts & summary HTML tables
#   • Returns one combined HTML snippet (SPY + QQQ) for the dashboard
# ---------------------------------------------------------------------

import os
from datetime import datetime

from log_index_growth    import log_index_growth          # writes today's data
from index_growth_charts import render_index_growth_charts

CHART_DIR = "charts"
SUMMARY_FILES = {
    "SPY": "spy_growth_summary.html",
    "QQQ": "qqq_growth_summary.html",
}

def _read_html(path: str, fallback: str = "<p>No data available.</p>") -> str:
    """Utility: load HTML file or return fallback text."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return fallback

# ------------------------------------------------------------------ #
#  mini-main (public) – this is what main_remote.py imports & calls  #
# ------------------------------------------------------------------ #
def index_growth() -> str:
    """
    Mini-main: orchestrates the full SPY & QQQ pipeline and returns
    an HTML snippet ready for embedding in the home page.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Updating SPY & QQQ implied-growth data …")

    # 1️⃣  Append today’s data
    log_index_growth()

    # 2️⃣  Rebuild charts & summary tables
    render_index_growth_charts()

    # 3️⃣  Collate both summaries into a single block
    blocks = []
    for ticker, fname in SUMMARY_FILES.items():
        blocks.append(f"<h3>{ticker}</h3>")
        blocks.append(_read_html(os.path.join(CHART_DIR, fname)))

    return "\n".join(blocks)

# ------------------------------------------------------------------ #
#  Stand-alone execution (optional)                                  #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    html_snippet = index_growth()            # run mini-main
    out_path = os.path.join(CHART_DIR, "spy_qqq_combined_summary.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_snippet)
    print(f"Combined summary written to {out_path}")
