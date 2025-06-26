"""
html_generator2.py
──────────────────────────────────────────────────────────────────────────────
Builds a dictionary per-ticker that your Jinja template consumes to render
all charts and tables, including the new *absolute-dollar expense table*.

Assumptions
───────────
• All HTML/PNG outputs live in the `charts/` directory.
• You have a helper called `get_file_content_or_placeholder(...)` that
  returns the file’s content (or a “no data” stub).

Edit the TICKERS list or import it from elsewhere if needed.
"""

from pathlib import Path
import json

CHART_DIR = Path("charts")
TICKERS   = ["AAPL", "MSFT", "GOOGL"]   # ← or however you pass tickers


def get_file_content_or_placeholder(filename: str) -> str:
    """Read file text, or return a polite placeholder if it’s missing."""
    f = CHART_DIR / filename
    if f.exists():
        return f.read_text(encoding="utf-8")
    return f'<p class="placeholder">No data available for {filename}</p>'


def build_ticker_dict(ticker: str) -> dict[str, str]:
    """Collect all HTML/PNG snippets for one ticker."""
    d: dict[str, str] = {}

    # ─────────── Charts ───────────
    d["chart_exp_abs"]   = f"{ticker}_expenses_vs_revenue.png"
    d["chart_exp_pct"]   = f"{ticker}_expenses_pct_of_rev.png"

    # ─────────── Expense tables ───────────
    d["expense_abs_html"] = get_file_content_or_placeholder(
        f"{ticker}_expense_absolute.html"
    )                                  # ← NEW absolute-dollar table

    d["expense_yoy_html"] = get_file_content_or_placeholder(
        f"{ticker}_yoy_expense_change.html"
    )                                  # ← YoY % table (unchanged)

    # … load any other tables/charts your site needs here …

    return d


def main() -> None:
    all_data = {tkr: build_ticker_dict(tkr) for tkr in TICKERS}
    Path("ticker_data.json").write_text(
        json.dumps(all_data, indent=2), encoding="utf-8"
    )
    print("✅ ticker_data.json written.")


if __name__ == "__main__":
    main()
