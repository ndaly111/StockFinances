import csv
import os
import sqlite3
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from Forward_data import ensure_forward_schema
DB_PATH = "Stock Data.db"
CHARTS_DIR = "charts"
FY_TABLE = "Forward_EPS_FY_History"
NTM_TABLE = "Forward_EPS_History"


def ensure_output_directory():
    os.makedirs(CHARTS_DIR, exist_ok=True)


def ensure_schema(conn):
    ensure_forward_schema(conn=conn)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Forward_EPS_History (
            date_recorded TEXT,
            ticker TEXT,
            forward_eps REAL,
            source TEXT,
            PRIMARY KEY (date_recorded, ticker)
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_forward_eps_hist_ticker_date "
        "ON Forward_EPS_History (ticker, date_recorded)"
    )
    conn.commit()


def _get_fy_history(ticker):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        ensure_schema(conn)
        df = pd.read_sql_query(
            f"""
            SELECT date_recorded, period_end, forward_eps, eps_analysts, period_label
            FROM {FY_TABLE}
            WHERE ticker = ?
            ORDER BY date_recorded ASC
            """,
            conn,
            params=(ticker,),
        )

    if df.empty:
        return df

    df["date_recorded"] = pd.to_datetime(df["date_recorded"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["forward_eps"] = pd.to_numeric(df["forward_eps"], errors="coerce")
    df = df.dropna(subset=["date_recorded", "period_end", "forward_eps"]).copy()
    return df


def _select_active_period_end(df_fy):
    if df_fy.empty:
        return None
    today = pd.Timestamp(datetime.now().date())
    period_ends = sorted(df_fy["period_end"].dropna().unique())
    for period_end in period_ends:
        if period_end >= today:
            return period_end
    return period_ends[-1] if period_ends else None


def _get_ntm_history(ticker):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        ensure_schema(conn)
        df = pd.read_sql_query(
            f"""
            SELECT date_recorded, forward_eps
            FROM {NTM_TABLE}
            WHERE ticker = ?
            ORDER BY date_recorded ASC
            """,
            conn,
            params=(ticker,),
        )

    if df.empty:
        return df

    df["date_recorded"] = pd.to_datetime(df["date_recorded"], errors="coerce")
    df["forward_eps"] = pd.to_numeric(df["forward_eps"], errors="coerce")
    df = df.dropna(subset=["date_recorded", "forward_eps"]).copy()
    return df


def get_forward_eps_history_for_display(ticker):
    df_fy = _get_fy_history(ticker)
    active_period_end = _select_active_period_end(df_fy) if not df_fy.empty else None
    if active_period_end is not None:
        df = df_fy[df_fy["period_end"] == active_period_end].copy()
        meta = {
            "mode": "FY",
            "period_end": active_period_end,
            "period_year": int(pd.Timestamp(active_period_end).year),
        }
        return df, meta

    df_ntm = _get_ntm_history(ticker)
    meta = {"mode": "NTM", "period_end": None, "period_year": None}
    return df_ntm, meta


def value_asof(df, target_date):
    if df.empty:
        return None
    subset = df[df["date_recorded"] <= pd.Timestamp(target_date)]
    if subset.empty:
        return None
    return float(subset.iloc[-1]["forward_eps"])


def write_forward_eps_summary_html(ticker, df, meta):
    ensure_output_directory()
    output_path = f"{CHARTS_DIR}/{ticker}_forward_eps_revision_summary.html"

    if df.empty:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("<p>No forward EPS history available yet.</p>")
        return output_path

    last_date = df["date_recorded"].iloc[-1].to_pydatetime()
    last_eps = float(df["forward_eps"].iloc[-1])

    eps_7d = value_asof(df, last_date - timedelta(days=7))
    eps_30d = value_asof(df, last_date - timedelta(days=30))
    eps_90d = value_asof(df, last_date - timedelta(days=90))

    def fmt_delta(current, previous):
        if previous is None:
            return "-"
        delta = current - previous
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.2f}"

    if meta.get("mode") == "FY":
        fy_year = meta.get("period_year")
        period_end = meta.get("period_end")
        forecast_label = f"FY{fy_year}" if fy_year else "FY"
        period_end_str = pd.Timestamp(period_end).strftime("%Y-%m-%d") if period_end is not None else "-"
    else:
        forecast_label = "NTM"
        period_end_str = "-"

    analysts_val = "-"
    if "eps_analysts" in df.columns and not df["eps_analysts"].dropna().empty:
        try:
            analysts_val = int(df["eps_analysts"].dropna().iloc[-1])
        except Exception:
            analysts_val = "-"

    summary = pd.DataFrame(
        [
            {
                "Last Updated": last_date.strftime("%Y-%m-%d"),
                "Forecast": forecast_label,
                "Period End": period_end_str,
                "Forward EPS": f"{last_eps:.2f}",
                "Change vs 7D": fmt_delta(last_eps, eps_7d),
                "Change vs 30D": fmt_delta(last_eps, eps_30d),
                "Change vs 90D": fmt_delta(last_eps, eps_90d),
                "Observations": int(len(df)),
                "Analysts": analysts_val,
            }
        ]
    )

    summary.to_html(
        output_path,
        index=False,
        escape=False,
        classes=["table", "table-striped"],
    )
    return output_path


def plot_forward_eps_revision_chart(ticker):
    ensure_output_directory()
    df, meta = get_forward_eps_history_for_display(ticker)
    output_path = f"{CHARTS_DIR}/{ticker}_forward_eps_revision.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty:
        ax.text(0.5, 0.5, "No forward EPS history yet", ha="center", va="center")
        ax.set_axis_off()
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
        return output_path

    ax.plot(df["date_recorded"], df["forward_eps"], marker="o")
    if meta.get("mode") == "FY":
        fy_year = meta.get("period_year")
        period_end = meta.get("period_end")
        period_end_str = pd.Timestamp(period_end).strftime("%Y-%m-%d") if period_end is not None else ""
        ax.set_title(f"Forward EPS Revisions – FY{fy_year} (ends {period_end_str}) – {ticker}")
    else:
        ax.set_title(f"Forward EPS (NTM) – Revisions – {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Forward EPS")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.autofmt_xdate()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    return output_path


def generate_all_forward_eps_assets(tickers=None):
    ensure_output_directory()
    if tickers is None:
        if not os.path.exists("tickers.csv"):
            print("[forward_eps_history] No tickers provided and tickers.csv not found.")
            return
        with open("tickers.csv", "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "Ticker" in (reader.fieldnames or []):
                tickers = [row["Ticker"].strip().upper() for row in reader if row.get("Ticker")]
            else:
                handle.seek(0)
                tickers = [line.strip().upper() for line in handle if line.strip()]

    for ticker in tickers:
        try:
            plot_forward_eps_revision_chart(ticker)
            df, meta = get_forward_eps_history_for_display(ticker)
            write_forward_eps_summary_html(ticker, df, meta)
            print(f"[forward_eps_history] {ticker} assets generated")
        except Exception as exc:
            print(f"[forward_eps_history] Failed for {ticker}: {exc}")


def forward_eps_history_main():
    generate_all_forward_eps_assets()


if __name__ == "__main__":
    forward_eps_history_main()
