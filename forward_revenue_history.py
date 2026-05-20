"""Forward revenue projection history per fiscal year.

Sibling of forward_eps_history.py — reads the same Forward_EPS_FY_History
table (which gained forward_revenue and revenue_analysts columns on
2026-05-20) and plots one line per fiscal year so you can see how each
year's revenue projection evolved before that fiscal year rolled over.

Data starts sparse for any given ticker — the chart fills in as the daily
snapshot pipeline accumulates rows. There is no historical backfill
available from yfinance.
"""

from __future__ import annotations

import csv
import logging
import os
import sqlite3

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from Forward_data import ensure_forward_schema

DB_PATH = "Stock Data.db"
CHARTS_DIR = "charts"
FY_TABLE = "Forward_EPS_FY_History"

logger = logging.getLogger(__name__)


def ensure_output_directory():
    os.makedirs(CHARTS_DIR, exist_ok=True)


def _get_all_fy_revenue_history(ticker):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        ensure_forward_schema(conn=conn)
        df = pd.read_sql_query(
            f"""
            SELECT date_recorded, period_end, forward_revenue, fiscal_year, period_label
            FROM {FY_TABLE}
            WHERE ticker = ?
            ORDER BY period_end ASC, date_recorded ASC
            """,
            conn,
            params=(ticker,),
        )
    if df.empty:
        return df
    df["date_recorded"] = pd.to_datetime(df["date_recorded"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["forward_revenue"] = pd.to_numeric(df["forward_revenue"], errors="coerce")
    df = df.dropna(subset=["date_recorded", "period_end", "forward_revenue"]).copy()
    return df


def _format_revenue(value):
    """Compact unit label for the y-axis (1.2B, 950M, etc.)."""
    if value is None:
        return ""
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"{value / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"{value / 1e6:.0f}M"
    return f"{value:,.0f}"


def plot_forward_revenue_history_by_fy(ticker):
    ensure_output_directory()
    df = _get_all_fy_revenue_history(ticker)
    output_path = f"{CHARTS_DIR}/{ticker}_forward_revenue_history_by_fy.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty:
        ax.text(0.5, 0.55, f"No revenue projection history for {ticker} yet",
                ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(0.5, 0.42, "Daily snapshots accumulate here; this chart fills in over time.",
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_axis_off()
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
        return output_path

    period_ends = sorted(df["period_end"].dropna().unique())
    for period_end in period_ends:
        line_df = df[df["period_end"] == period_end].sort_values("date_recorded")
        if line_df.empty:
            continue
        fy = int(pd.Timestamp(period_end).year)
        ax.plot(line_df["date_recorded"], line_df["forward_revenue"],
                marker="o", linewidth=1.6, markersize=4, label=f"FY{fy}")

    ax.set_title(f"Forward revenue projection history by fiscal year - {ticker}")
    ax.set_xlabel("Date projection was recorded")
    ax.set_ylabel("Forward revenue estimate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _format_revenue(v)))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    return output_path


def generate_all_forward_revenue_assets(tickers=None):
    ensure_output_directory()
    if tickers is None:
        if not os.path.exists("tickers.csv"):
            logger.warning("No tickers provided and tickers.csv not found")
            return
        with open("tickers.csv", "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "Ticker" in (reader.fieldnames or []):
                tickers = [row["Ticker"].strip().upper() for row in reader if row.get("Ticker")]
            else:
                handle.seek(0)
                tickers = [line.strip().upper() for line in handle if line.strip()]

    success_count = 0
    empty_count = 0
    error_count = 0
    for ticker in tickers:
        try:
            plot_forward_revenue_history_by_fy(ticker)
            df = _get_all_fy_revenue_history(ticker)
            if df.empty:
                empty_count += 1
            else:
                success_count += 1
        except Exception as exc:
            error_count += 1
            logger.error(f"[{ticker}] forward revenue chart failed: {exc}")
    logger.info(
        f"Forward revenue generation complete: {success_count} with data, "
        f"{empty_count} empty, {error_count} errors"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_all_forward_revenue_assets()
