#!/usr/bin/env python3
"""
generate_segment_charts.py
-----------------------------------

Reads tickers from tickers.csv and for each ticker:

1) Creates segment bar charts (Revenue & Operating Income) for each segment,
   using a consistent y-axis scale across all segments for the ticker.

2) Writes a compact pivot HTML table per ticker at:
   charts/{TICKER}/{TICKER}_segments_table.html

   - One row per segment
   - Columns: last 3 fiscal years + TTM, each with Rev & OI side-by-side
   - Single, consistent unit across the whole table ($K/$M/$B/$T)
   - TTM columns bold
   - “% of Total (TTM)” mix column
   - Sorted by TTM Revenue descending

Data source: sec_segment_data_arelle.get_segment_data(ticker)
Expected columns: Segment, Year, Revenue, OpIncome
"""

from __future__ import annotations

import os
import argparse
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from sec_segment_data_arelle import get_segment_data


# ─────────────────────────── utilities ───────────────────────────

def read_tickers(csv_path: Path) -> List[str]:
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
    """Numeric years ascending; 'TTM' last; others after numeric."""
    def key(y: str) -> Tuple[int, int | str]:
        if y == "TTM":
            return (2, 0)
        try:
            return (0, int(y))
        except Exception:
            return (1, y)
    return [y for _, y in sorted([(key(y), y) for y in years], key=lambda x: x[0])]

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _humanize_segment_name(raw: str) -> str:
    if not isinstance(raw, str) or not raw:
        return raw
    name = raw.replace("SegmentMember", "")
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).strip()
    fixes = {
        "Greater China": "Greater China",
        "Rest Of Asia Pacific": "Rest of Asia Pacific",
        "North America": "North America",
        "Latin America": "Latin America",
        "United States": "United States",
        "Middle East": "Middle East",
        "Asia Pacific": "Asia Pacific",
        "Americas": "Americas",
        "Europe": "Europe",
        "Japan": "Japan",
        "China": "China",
    }
    title = " ".join(w if w.isupper() else w.capitalize() for w in name.split())
    return fixes.get(title, title)

def _to_float(x):
    if pd.isna(x): return pd.NA
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace(",", "").strip()
    try: return float(s)
    except: return pd.NA

def _choose_scale(max_abs_value: float) -> Tuple[float, str]:
    """Choose one unit for the whole table."""
    if max_abs_value is None or (isinstance(max_abs_value, float) and math.isnan(max_abs_value)) or max_abs_value == 0:
        return (1.0, "$")
    v = abs(max_abs_value)
    if v >= 1e12: return (1e12, "$T")
    if v >= 1e9:  return (1e9,  "$B")
    if v >= 1e6:  return (1e6,  "$M")
    if v >= 1e3:  return (1e3,  "$K")
    return (1.0, "$")

def _fmt_scaled(x, div, unit) -> str:
    if pd.isna(x): return "–"
    return f"{unit}{(x/div):.1f}"

def _last3_plus_ttm(years: List[str]) -> List[str]:
    nums = sorted({int(y) for y in years if str(y).isdigit()})
    keep = nums[-3:] if len(nums) > 3 else nums
    out = [str(y) for y in keep]
    if "TTM" in set(years):
        out.append("TTM")
    return out


# ───────────────────── main per-ticker routine ────────────────────

def generate_segment_charts_for_ticker(ticker: str, out_dir: Path) -> None:
    """Generate charts and a compact pivot HTML table for a single ticker."""
    # Fetch
    try:
        df = get_segment_data(ticker)
    except Exception as fetch_err:
        print(f"Error fetching segment data for {ticker}: {fetch_err}")
        ensure_dir(out_dir)
        (out_dir / f"{ticker}_segments_table.html").write_text(
            f"<p>Error fetching segment data for {ticker}: {fetch_err}</p>",
            encoding="utf-8",
        )
        return

    ensure_dir(out_dir)

    if df is None or df.empty:
        (out_dir / f"{ticker}_segments_table.html").write_text(
            f"<p>No segment data available for {ticker}.</p>", encoding="utf-8"
        )
        return

    # Clean + normalize
    df = df.copy()
    df["Segment"] = df["Segment"].astype(str).map(_humanize_segment_name)
    df["Year"] = df["Year"].astype(str)
    df["Revenue"] = df["Revenue"].map(_to_float)
    df["OpIncome"] = df["OpIncome"].map(_to_float)

    # Compute y-axis range for charts
    all_vals = pd.concat([df["Revenue"].dropna(), df["OpIncome"].dropna()], ignore_index=True)
    if all_vals.empty:
        min_y, max_y = 0.0, 0.0
    else:
        min_y, max_y = float(all_vals.min()), float(all_vals.max())
        if min_y > 0: min_y = 0.0
        if max_y < 0: max_y = 0.0
    margin = (max_y - min_y) * 0.1
    min_y_plot, max_y_plot = min_y - margin, max_y + margin

    # Years for charts (keep them all, ordered), and years for the table (last3 + TTM)
    years_all = sort_years(sorted(set(df["Year"].tolist())))
    years_tbl = _last3_plus_ttm(df["Year"].tolist())

    # Segment list
    segments = sorted(set(df["Segment"].tolist()))

    # ── Charts per segment (y in $B) ──
    for seg in segments:
        seg_df = df[df["Segment"] == seg]
        # Build value lists in the order of years_all
        revenues = [seg_df.loc[seg_df["Year"] == y, "Revenue"].sum() for y in years_all]
        op_incomes = [seg_df.loc[seg_df["Year"] == y, "OpIncome"].sum() for y in years_all]

        revenues_b = [0.0 if pd.isna(v) else v / 1e9 for v in revenues]
        op_incomes_b = [0.0 if pd.isna(v) else v / 1e9 for v in op_incomes]
        min_y_plot_b = min_y_plot / 1e9
        max_y_plot_b = max_y_plot / 1e9

        fig, ax = plt.subplots(figsize=(8, 5))
        x_indices = list(range(len(years_all)))
        bar_width = 0.35

        ax.bar([x - bar_width / 2 for x in x_indices], revenues_b, width=bar_width, label="Revenue")
        ax.bar([x + bar_width / 2 for x in x_indices], op_incomes_b, width=bar_width, label="Operating Income")

        ax.set_xticks(x_indices)
        ax.set_xticklabels(years_all)
        ax.set_ylim(min_y_plot_b, max_y_plot_b)
        ax.set_ylabel("Value ($B)")
        ax.set_title(seg)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")
        plt.tight_layout()

        safe_seg_name = seg.replace("/", "_").replace(" ", "_")
        fig_path = out_dir / f"{ticker}_{safe_seg_name}.png"
        plt.savefig(fig_path)
        plt.close(fig)

    # ── Compact pivot table (Wireframe A) ──
    # Keep only last 3 numeric years + TTM for the table columns
    use_years = years_tbl
    # Build pivots
    def pv(col):
        p = df[df["Year"].isin(use_years)].pivot_table(index="Segment", columns="Year", values=col, aggfunc="sum")
        return p.reindex(columns=[y for y in use_years if y in p.columns])

    rev_p = pv("Revenue")
    oi_p  = pv("OpIncome")

    # Sort rows by TTM Rev if available
    sort_col = "TTM" if "TTM" in rev_p.columns else (rev_p.columns[-1] if len(rev_p.columns) else None)
    if sort_col:
        rev_p = rev_p.sort_values(by=sort_col, ascending=False)
        oi_p = oi_p.reindex(index=rev_p.index)

    # % of Total (TTM)
    pct_series = None
    if "TTM" in rev_p.columns:
        total_ttm = rev_p["TTM"].sum(skipna=True)
        if total_ttm and total_ttm != 0:
            pct_series = (rev_p["TTM"] / total_ttm) * 100.0

    # Determine single scale for the whole table
    max_val = pd.concat([rev_p, oi_p]).abs().max().max()
    div, unit = _choose_scale(float(max_val) if pd.notna(max_val) else 0.0)

    # Assemble interleaved columns: 20xx Rev, 20xx OI, ..., TTM Rev, TTM OI
    cols: List[Tuple[str, str]] = []
    for y in [c for c in use_years if c != "TTM"]:
        cols += [(y, "Rev"), (y, "OI")]
    if "TTM" in use_years:
        cols += [("TTM", "Rev"), ("TTM", "OI")]

    out = pd.DataFrame(index=rev_p.index)
    for (y, kind) in cols:
        src = rev_p.get(y) if kind == "Rev" else oi_p.get(y)
        label = f"{y} {'Rev' if kind=='Rev' else 'OI'}"
        out[label] = src

    if pct_series is not None:
        out["% of Total (TTM)"] = pct_series

    # Format values using the chosen unit; bold TTM columns
    for c in out.columns:
        if c == "% of Total (TTM)":
            out[c] = out[c].map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "–")
        else:
            out[c] = out[c].map(lambda x, d=div, u=unit: _fmt_scaled(x, d, u))

    for ttm_col in [c for c in out.columns if c.startswith("TTM ")]:
        out[ttm_col] = out[ttm_col].map(lambda s: f"<strong>{s}</strong>" if s != "–" else s)

    out.index.name = "Segment"
    out_disp = out.reset_index()

    caption = (
        '<div class="table-note">Values scaled to '
        f'<b>{unit}</b> for the entire table. TTM values are <b>bold</b>. '
        '“% of Total (TTM)” shows revenue mix.</div>'
    )
    html = out_disp.to_html(index=False, escape=False, classes="segment-pivot compact", border=0)
    table_content = caption + f"\n<div class='table-wrap'>{html}</div>"

    (out_dir / f"{ticker}_segments_table.html").write_text(table_content, encoding="utf-8")


# ─────────────────────────── CLI wrapper ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate segment charts and tables for a list of tickers.")
    parser.add_argument(
        "--tickers_csv",
        type=str,
        default="tickers.csv",
        help="Path to the CSV file containing tickers. Defaults to 'tickers.csv' in the current directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="charts",
        help="Directory where charts and tables will be saved. Defaults to 'charts'."
    )
    args = parser.parse_args()

    csv_path = Path(args.tickers_csv)
    tickers = read_tickers(csv_path)
    if not tickers:
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
