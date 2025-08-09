#!/usr/bin/env python3
"""
generate_segment_charts.py
-----------------------------------

This script reads a list of tickers from a CSV file (``tickers.csv``) and
produces two artifacts for each ticker:

1. **Segment charts** – for every business segment reported by the company,
   the script creates a bar chart showing revenue and operating income for
   each of the last three fiscal years plus the trailing‑twelve‑month (TTM)
   period.  All charts for a given ticker share the same y‑axis scale so
   that comparisons across segments are meaningful.  The figures are saved
   as PNG files in a ``charts`` subdirectory.

2. **Segment values table** – an HTML file summarizing the underlying
   numbers used in the charts.  Each row contains a segment name, year
   (including ``TTM``), and the corresponding revenue and operating income.
   The table is saved as ``charts/{ticker}_segments_table.html``.

The script uses the ``sec_segment_data_arelle`` module to fetch segment
data directly from the SEC’s iXBRL filings.  It assumes that
``tickers.csv`` is present in the current working directory and has a
column named ``ticker`` (case insensitive) listing one ticker per row.

Dependencies: pandas, matplotlib.

Usage:
    python generate_segment_charts.py [--tickers_csv path/to/tickers.csv]

If the ``tickers.csv`` file is not found, the script will default to a
small list of demonstration tickers (``AAPL``, ``MSFT``, and ``AMZN``).

"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from sec_segment_data_arelle import get_segment_data


def read_tickers(csv_path: Path) -> List[str]:
    """Read a CSV file and return a list of tickers.

    The CSV must have a column named ``ticker`` (case insensitive).
    Returns an empty list if the file does not exist or does not contain
    a ticker column.
    """
    if not csv_path.is_file():
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    cols = [c for c in df.columns if c.lower() == "ticker"]
    if not cols:
        return []
    return [str(t).upper().strip() for t in df[cols[0]].dropna().tolist()]


def sort_years(years: List[str]) -> List[str]:
    """Sort a list of years (strings) so that numeric years are ascending and
    'TTM' appears last.  Unknown labels are placed at the end in
    alphabetical order.
    """
    def key(y: str) -> Tuple[int, str]:
        if y == "TTM":
            return (2, "")
        try:
            return (0, int(y))
        except Exception:
            return (1, y)
    return [y for _, y in sorted([(key(y), y) for y in years], key=lambda x: x[0])]


def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def generate_segment_charts_for_ticker(ticker: str, out_dir: Path) -> None:
    """Generate charts and an HTML table for a single ticker.

    The function fetches segment data for the given ticker, computes a
    consistent y‑axis scale across all segments, generates a bar chart
    for each segment, and writes an HTML table summarizing the values.
    Charts and the table are stored in ``out_dir``.
    """
    # Fetch segment data; if an error occurs, log and return
    try:
        df = get_segment_data(ticker)
    except Exception as fetch_err:
        print(f"Error fetching segment data for {ticker}: {fetch_err}")
        # Write an HTML placeholder indicating the failure
        ensure_dir(out_dir)
        table_path = out_dir / f"{ticker}_segments_table.html"
        table_path.write_text(
            f"<p>Error fetching segment data for {ticker}: {fetch_err}</p>",
            encoding="utf-8",
        )
        return
    # Ensure the output directory exists
    ensure_dir(out_dir)
    # If no data returned, write a simple table and return
    if df is None or df.empty:
        table_path = out_dir / f"{ticker}_segments_table.html"
        html = f"<p>No segment data available for {ticker}.</p>"
        table_path.write_text(html, encoding="utf-8")
        return
    # Compute y‑axis range (min and max across all segments and both metrics)
    rev_vals = df["Revenue"].dropna().tolist()
    op_vals = df["OpIncome"].dropna().tolist()
    all_vals = rev_vals + op_vals
    if not all_vals:
        min_y, max_y = 0.0, 0.0
    else:
        min_y = min(all_vals)
        max_y = max(all_vals)
        # If values are all positive or all negative, include zero for context
        if min_y > 0:
            min_y = 0.0
        if max_y < 0:
            max_y = 0.0
    # Add a small margin (10%) to the y‑axis range for visual clarity
    margin = (max_y - min_y) * 0.1
    max_y_plot = max_y + margin
    min_y_plot = min_y - margin
    # Determine the sorted list of years
    years = sort_years(sorted(set(df["Year"].astype(str).tolist())))
    # Build a lookup for each segment
    segments = sorted(set(df["Segment"].tolist()))
    # Build a table rows list for HTML rendering
    table_rows = []
    for seg in segments:
        seg_df = df[df["Segment"] == seg]
        # Ensure we have all year entries; fill missing with zeros
        seg_dict = {(str(row["Year"]), "Revenue"): row["Revenue"] for _, row in seg_df.iterrows()}
        seg_dict.update({(str(row["Year"]), "OpIncome"): row["OpIncome"] for _, row in seg_df.iterrows()})
        # Prepare values for plotting: revenue and op income per year
        revenues = []
        op_incomes = []
        for yr in years:
            revenues.append(seg_dict.get((yr, "Revenue"), 0.0))
            op_incomes.append(seg_dict.get((yr, "OpIncome"), 0.0))
            # Add to table rows if not already; we'll add each unique row
            table_rows.append((seg, yr, seg_dict.get((yr, "Revenue"), 0.0), seg_dict.get((yr, "OpIncome"), 0.0)))
        # Create the chart
        # Convert values to billions to improve readability and label the y-axis accordingly
        revenues_b = [v / 1e9 for v in revenues]
        op_incomes_b = [v / 1e9 for v in op_incomes]
        min_y_plot_b = min_y_plot / 1e9
        max_y_plot_b = max_y_plot / 1e9
        # Create a figure with a larger size to improve readability
        fig, ax = plt.subplots(figsize=(8, 5))
        x_indices = range(len(years))
        bar_width = 0.35
        # Plot revenue and operating income side by side (in billions)
        ax.bar([x - bar_width / 2 for x in x_indices], revenues_b, width=bar_width, label="Revenue")
        ax.bar([x + bar_width / 2 for x in x_indices], op_incomes_b, width=bar_width, label="Operating Income")
        # Configure x-axis
        ax.set_xticks(list(x_indices))
        ax.set_xticklabels(years)
        # Configure y-axis with consistent limits (in billions)
        ax.set_ylim(min_y_plot_b, max_y_plot_b)
        ax.set_ylabel("Value ($B)")
        # Add a title equal to the segment name
        ax.set_title(seg)
        # Add horizontal gridlines for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        # Position the legend above the chart
        ax.legend(loc='upper left')
        # Tighten the layout to minimize whitespace
        plt.tight_layout()
        # Save the figure
        safe_seg_name = seg.replace("/", "_").replace(" ", "_")
        fig_path = out_dir / f"{ticker}_{safe_seg_name}.png"
        plt.savefig(fig_path)
        plt.close(fig)
    # Build HTML table
    # We may have duplicate entries in table_rows; group by (segment, year)
    unique_rows: Dict[Tuple[str, str], Tuple[str, str, float, float]] = {}
    for seg, yr, rev, op in table_rows:
        key = (seg, yr)
        if key not in unique_rows:
            unique_rows[key] = (seg, yr, rev, op)
        else:
            # Sum values if duplicates occur
            prev = unique_rows[key]
            unique_rows[key] = (seg, yr, prev[2] + rev, prev[3] + op)
    # Sort table rows by year ascending and segment name
    sorted_table_rows = sorted(unique_rows.values(), key=lambda r: (sort_years([r[1]])[0], r[0]))
    # Render HTML table
    table_html = ["<table class='segment-table' border='1' cellpadding='4' cellspacing='0'>"]
    table_html.append("<thead><tr><th>Segment</th><th>Year</th><th>Revenue</th><th>Operating Income</th></tr></thead>")
    table_html.append("<tbody>")
    for seg, yr, rev, op in sorted_table_rows:
        table_html.append(
            f"<tr><td>{seg}</td><td>{yr}</td>"
            f"<td>{rev:,.2f}</td><td>{op:,.2f}</td></tr>"
        )
    table_html.append("</tbody></table>")
    table_content = "\n".join(table_html)
    table_path = out_dir / f"{ticker}_segments_table.html"
    table_path.write_text(table_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate segment charts and tables for a list of tickers.")
    parser.add_argument(
        "--tickers_csv",
        type=str,
        default="tickers.csv",
        help="Path to the CSV file containing tickers.  Defaults to 'tickers.csv' in the current directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="charts",
        help="Directory where charts and tables will be saved.  Defaults to 'charts'."
    )
    args = parser.parse_args()
    csv_path = Path(args.tickers_csv)
    tickers = read_tickers(csv_path)
    if not tickers:
        # fallback to demonstration tickers
        tickers = ["AAPL", "MSFT", "AMZN"]
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx}/{len(tickers)}] Processing {ticker}…")
        ticker_dir = output_dir / ticker
        generate_segment_charts_for_ticker(ticker, ticker_dir)
    print("Done.")


if __name__ == "__main__":
    main()
