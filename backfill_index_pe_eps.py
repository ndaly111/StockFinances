#!/usr/bin/env python3
"""
backfill_index_pe_eps.py

Backfill Index_PE_History + Index_EPS_History for SPY/QQQ earlier than your current DB coverage.

Strategy (practical + consistent):
- Use Trendonify monthly trailing PE history as the valuation anchor:
    SPY proxy: S&P 500 PE ratio page
    QQQ proxy: Nasdaq 100 PE ratio page
- Use Yahoo Finance (yfinance) for daily Close prices of SPY/QQQ.
- Compute monthly implied TTM EPS at month-end:
    EPS_month = (month_end_close) / (PE_month)
- Then build daily rows:
    EPS_daily(day) = EPS_month(day's month)      (constant within month)
    PE_daily(day)  = Close(day) / EPS_month      (moves daily with price)

Optional simulation before ETF inception:
- For SPY pre-1993: use ^GSPC index close, scaled to SPY level using overlapping months.
- For QQQ pre-1999: use ^NDX index close, scaled to QQQ level using overlapping months.

This is a "one-time" backfill. It only inserts dates earlier than the earliest date already present
per (ticker, type) in each table, so re-running should be a no-op.

Dependencies: pandas, numpy, requests, lxml, yfinance
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

TRENDONIFY_PE_URL = {
    # trailing PE histories
    "SPY": "https://trendonify.com/united-states/stock-market/pe-ratio",
    "QQQ": "https://trendonify.com/united-states/stock-market/nasdaq-100/pe-ratio",
}

# Used only if --simulate-pre-etf is enabled
INDEX_PROXY = {
    "SPY": "^GSPC",  # S&P 500 index
    "QQQ": "^NDX",  # Nasdaq-100 index
}

# Approx inception-ish; used as a guard when simulating
ETF_INCEPTION = {
    "SPY": pd.Timestamp("1993-01-29"),
    "QQQ": pd.Timestamp("1999-03-10"),
}


@dataclass(frozen=True)
class TableSpec:
    name: str
    date_col: str
    ticker_col: str
    type_col: str
    value_col: str


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)
    )
    return cur.fetchone() is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]  # row[1] = column name


def _ensure_tables(conn: sqlite3.Connection) -> tuple[TableSpec, TableSpec]:
    """
    Ensure Index_PE_History and Index_EPS_History exist (create if missing).
    Also pick the correct column names if tables exist but differ slightly.
    """
    # ---- PE TABLE ----
    pe_table = "Index_PE_History"
    if not _table_exists(conn, pe_table):
        conn.execute(
            """
            CREATE TABLE Index_PE_History (
                Date TEXT NOT NULL,
                Ticker TEXT NOT NULL,
                PE_Type TEXT NOT NULL,
                PE_Ratio REAL,
                PRIMARY KEY (Ticker, Date, PE_Type)
            )
            """
        )
        conn.commit()

    pe_cols = set(_table_columns(conn, pe_table))
    pe_date = "Date" if "Date" in pe_cols else "date"
    pe_ticker = "Ticker" if "Ticker" in pe_cols else "ticker"
    pe_type = "PE_Type" if "PE_Type" in pe_cols else (
        "pe_type" if "pe_type" in pe_cols else "Type"
    )
    # value column
    pe_val_candidates = ["PE_Ratio", "pe_ratio", "PE", "pe"]
    pe_val = next((c for c in pe_val_candidates if c in pe_cols), None)
    if pe_val is None:
        raise RuntimeError(
            f"{pe_table} exists but I can't find a PE value column. "
            f"Expected one of {pe_val_candidates}, found: {sorted(pe_cols)}"
        )
    pe_spec = TableSpec(pe_table, pe_date, pe_ticker, pe_type, pe_val)

    # Ensure uniqueness even if the table pre-existed without a PK/UNIQUE constraint.
    # If this fails, your table likely already contains duplicates.
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_index_pe_history_uq "
            f"ON {pe_table} ({pe_ticker}, {pe_date}, {pe_type})"
        )
    except sqlite3.IntegrityError as e:
        raise RuntimeError(
            f"Cannot create UNIQUE index for {pe_table} on ({pe_ticker},{pe_date},{pe_type}). "
            "Duplicates exist; dedupe the table before running backfill."
        ) from e

    # ---- EPS TABLE ----
    eps_table = "Index_EPS_History"
    if not _table_exists(conn, eps_table):
        conn.execute(
            """
            CREATE TABLE Index_EPS_History (
                Date TEXT NOT NULL,
                Ticker TEXT NOT NULL,
                EPS_Type TEXT NOT NULL,
                EPS REAL,
                PRIMARY KEY (Ticker, Date, EPS_Type)
            )
            """
        )
        conn.commit()

    eps_cols = set(_table_columns(conn, eps_table))
    eps_date = "Date" if "Date" in eps_cols else "date"
    eps_ticker = "Ticker" if "Ticker" in eps_cols else "ticker"
    eps_type = "EPS_Type" if "EPS_Type" in eps_cols else (
        "eps_type" if "eps_type" in eps_cols else "Type"
    )
    eps_val_candidates = ["EPS", "EPS_TTM", "eps", "eps_ttm", "EPS_Value", "eps_value"]
    eps_val = next((c for c in eps_val_candidates if c in eps_cols), None)
    if eps_val is None:
        raise RuntimeError(
            f"{eps_table} exists but I can't find an EPS value column. "
            f"Expected one of {eps_val_candidates}, found: {sorted(eps_cols)}"
        )
    eps_spec = TableSpec(eps_table, eps_date, eps_ticker, eps_type, eps_val)

    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_index_eps_history_uq "
            f"ON {eps_table} ({eps_ticker}, {eps_date}, {eps_type})"
        )
    except sqlite3.IntegrityError as e:
        raise RuntimeError(
            f"Cannot create UNIQUE index for {eps_table} on ({eps_ticker},{eps_date},{eps_type}). "
            "Duplicates exist; dedupe the table before running backfill."
        ) from e

    return pe_spec, eps_spec


def _min_date(
    conn: sqlite3.Connection, spec: TableSpec, ticker: str, type_value: str
) -> Optional[pd.Timestamp]:
    if not _table_exists(conn, spec.name):
        return None
    sql = f"""
        SELECT MIN({spec.date_col})
        FROM {spec.name}
        WHERE {spec.ticker_col}=? AND {spec.type_col}=?
    """
    cur = conn.execute(sql, (ticker, type_value))
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    # robust parse
    ts = pd.to_datetime(row[0], errors="coerce")
    if pd.isna(ts):
        return None
    return ts.normalize()


def _fetch_trendonify_pe_monthly(url: str) -> pd.Series:
    """
    Returns monthly PE series indexed by PeriodIndex (M), ascending.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StockFinancesBackfill/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    }
    session = requests.Session()
    last_err: Exception | None = None
    resp = None
    for attempt in range(1, 4):
        try:
            resp = session.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    if resp is None or last_err is not None:
        raise RuntimeError(
            f"Failed to fetch Trendonify PE page after retries: {url}"
        ) from last_err

    # Trendonify pages include an HTML table with columns:
    #   Date | Current PE Ratio
    tables = pd.read_html(resp.text)
    target = None
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "Date" in cols and any("PE" in c for c in cols):
            # Prefer the history table specifically
            if "Current PE Ratio" in cols:
                target = t
                break
            target = t

    if target is None:
        raise RuntimeError(f"Couldn't find a PE history table on: {url}")

    # normalize columns
    df = target.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # pick a PE column
    pe_col = (
        "Current PE Ratio"
        if "Current PE Ratio" in df.columns
        else next((c for c in df.columns if "PE" in c), None)
    )
    if pe_col is None or "Date" not in df.columns:
        raise RuntimeError(
            f"Unexpected table format on: {url}. Columns: {df.columns.tolist()}"
        )

    # parse month strings like "Jan 2026" (fallback if Trendonify ever changes formatting)
    raw_dates = df["Date"].astype(str).str.strip()
    month_dt = pd.to_datetime(raw_dates, format="%b %Y", errors="coerce")
    if month_dt.isna().all():
        month_dt = pd.to_datetime(raw_dates, errors="coerce")
    pe = pd.to_numeric(df[pe_col], errors="coerce")

    out = pd.DataFrame({"month": month_dt.dt.to_period("M"), "pe": pe})
    out = out.dropna(subset=["month", "pe"])
    out = out[out["pe"] > 0]  # skip non-positive
    out = out.drop_duplicates(subset=["month"], keep="last").sort_values("month")
    return out.set_index("month")["pe"]


def _densify_monthly_pe(pe_monthly: pd.Series) -> pd.Series:
    """
    Expand PE series to a full monthly PeriodIndex and forward-fill.
    This prevents large historical gaps when Trendonify provides only sparse older points.
    """
    if pe_monthly is None or pe_monthly.empty:
        return pd.Series(dtype=float)
    full_idx = pd.period_range(pe_monthly.index.min(), pe_monthly.index.max(), freq="M")
    densified = pe_monthly.reindex(full_idx).ffill()
    return densified.dropna()


def _yf_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Daily Close series, timezone-naive, sorted.
    """
    df = None
    for attempt in range(1, 4):
        df = yf.download(
            symbol,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            progress=False,
            auto_adjust=False,
            actions=False,
            threads=True,
        )
        if df is not None and not df.empty:
            break
        if attempt < 3:
            time.sleep(2 ** (attempt - 1))
    if df is None or df.empty:
        return pd.Series(dtype=float)
    close = df["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    return close


def _build_effective_close(
    ticker: str,
    end: pd.Timestamp,
    simulate_pre_etf: bool,
) -> pd.Series:
    """
    Returns a daily close series for the ticker. If simulate_pre_etf=True, extend earlier dates
    using an index proxy scaled to the ETF's price level.
    """
    # Fetch ETF close back as far as possible
    etf_close = _yf_close(ticker, pd.Timestamp("1900-01-01"), end)
    if etf_close.empty:
        return etf_close

    if not simulate_pre_etf:
        return etf_close

    proxy = INDEX_PROXY.get(ticker)
    if not proxy:
        return etf_close

    proxy_close = _yf_close(proxy, pd.Timestamp("1900-01-01"), end)
    if proxy_close.empty:
        return etf_close

    # compute a stable scaling ratio using overlapping MONTH-END closes
    etf_m = etf_close.resample("M").last()
    proxy_m = proxy_close.resample("M").last()

    overlap = pd.concat([proxy_m.rename("proxy"), etf_m.rename("etf")], axis=1).dropna()
    if overlap.empty:
        return etf_close

    # Use median ratio over first ~12 overlapping months to reduce noise
    overlap = overlap.sort_index()
    first_block = overlap.iloc[:12] if len(overlap) >= 12 else overlap
    ratio = (first_block["proxy"] / first_block["etf"]).median()
    if not np.isfinite(ratio) or ratio <= 0:
        return etf_close

    inception = ETF_INCEPTION.get(ticker, etf_close.index.min())
    pre_mask = proxy_close.index < etf_close.index.min()
    # For safety, only simulate pre-ETF dates (before inception)
    pre_mask = pre_mask & (proxy_close.index < inception)

    if not pre_mask.any():
        return etf_close

    simulated = (proxy_close.loc[pre_mask] / ratio).rename("Close")
    combined = pd.concat([simulated, etf_close]).sort_index()
    return combined


def _compute_daily_series(
    effective_close: pd.Series,
    pe_monthly: pd.Series,
) -> pd.DataFrame:
    """
    Build daily EPS (monthly-constant) + daily PE (price/EPS).
    Returns DataFrame with index = daily dates, columns: close, pe, eps
    """
    df = pd.DataFrame({"close": effective_close}).dropna()
    df["month"] = df.index.to_period("M")

    # month-end close by month
    month_end_close = df["close"].resample("M").last()
    month_end_close.index = month_end_close.index.to_period("M")

    monthly = pd.concat(
        [month_end_close.rename("month_end_close"), pe_monthly.rename("pe_month")],
        axis=1,
    ).dropna()

    # EPS anchor per month
    monthly = monthly[monthly["pe_month"] > 0]
    monthly["eps_month"] = monthly["month_end_close"] / monthly["pe_month"]
    monthly = monthly.replace([np.inf, -np.inf], np.nan).dropna(subset=["eps_month"])
    monthly = monthly[monthly["eps_month"] > 0]

    # map EPS per day by month
    eps_by_month = monthly["eps_month"]
    df["eps"] = df["month"].map(eps_by_month)

    # daily PE derived from daily close and monthly EPS
    df["pe"] = df["close"] / df["eps"]
    df = df.replace([np.inf, -np.inf], np.nan)

    return df[["close", "pe", "eps"]]


def _insert_series(
    conn: sqlite3.Connection,
    spec: TableSpec,
    ticker: str,
    type_value: str,
    series: pd.Series,
    cutoff_exclusive: Optional[pd.Timestamp],
    dry_run: bool,
) -> int:
    """
    Insert series into spec table for dates < cutoff_exclusive (if provided).
    Dates stored as YYYY-MM-DD.
    Returns inserted row count (attempted inserts).
    """
    if series.empty:
        return 0

    s = series.dropna().copy()
    s.index = pd.to_datetime(s.index).normalize()

    if cutoff_exclusive is not None:
        s = s[s.index < cutoff_exclusive.normalize()]

    if s.empty:
        return 0

    rows = [(d.strftime("%Y-%m-%d"), ticker, type_value, float(v)) for d, v in s.items()]
    sql = f"""
        INSERT OR IGNORE INTO {spec.name} ({spec.date_col}, {spec.ticker_col}, {spec.type_col}, {spec.value_col})
        VALUES (?, ?, ?, ?)
    """

    if dry_run:
        return len(rows)

    before = conn.total_changes
    conn.executemany(sql, rows)
    return conn.total_changes - before


def main() -> int:
    p = argparse.ArgumentParser(
        description="Backfill SPY/QQQ PE + EPS history into Stock Data.db"
    )
    p.add_argument(
        "--db", default="Stock Data.db", help="Path to SQLite DB (default: Stock Data.db)"
    )
    p.add_argument(
        "--tickers",
        nargs="*",
        default=["SPY", "QQQ"],
        help="Tickers to backfill (default: SPY QQQ)",
    )
    p.add_argument(
        "--simulate-pre-etf",
        action="store_true",
        help="Simulate pre-ETF history via ^GSPC/^NDX scaling",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write; just report how many rows would be inserted",
    )
    args = p.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: DB not found: {args.db}")
        return 2

    with sqlite3.connect(args.db) as conn:
        pe_spec, eps_spec = _ensure_tables(conn)

        for tk in args.tickers:
            tk = tk.upper().strip()
            if tk not in TRENDONIFY_PE_URL:
                print(f"Skipping {tk}: no Trendonify URL mapping")
                continue

            print(f"\n=== {tk} ===")
            pe_url = TRENDONIFY_PE_URL[tk]
            print(f"Fetching PE history: {pe_url}")
            pe_monthly = _fetch_trendonify_pe_monthly(pe_url)
            pe_monthly = _densify_monthly_pe(pe_monthly)

            # Determine earliest existing dates (we only insert earlier than these)
            min_pe = _min_date(conn, pe_spec, tk, "TTM")
            min_eps = _min_date(conn, eps_spec, tk, "TTM")
            print(f"Earliest existing PE date:  {min_pe if min_pe is not None else 'None'}")
            print(
                f"Earliest existing EPS date: {min_eps if min_eps is not None else 'None'}"
            )

            # If both exist, backfill only before the earliest of each table.
            # If one is missing, we backfill that table as far as the other allows.
            end_pe = (
                min_pe - pd.Timedelta(days=1)
                if min_pe is not None
                else pd.Timestamp.today().normalize()
            )
            end_eps = (
                min_eps - pd.Timedelta(days=1)
                if min_eps is not None
                else pd.Timestamp.today().normalize()
            )
            end = max(end_pe, end_eps).normalize()

            # Build a daily close series to that end date (with optional pre-ETF simulation)
            print(
                f"Fetching prices through: {end.date().isoformat()}  "
                f"(simulate_pre_etf={args.simulate_pre_etf})"
            )
            close = _build_effective_close(
                tk, end=end, simulate_pre_etf=args.simulate_pre_etf
            )
            if close.empty:
                print(f"  !! No price data for {tk}; skipping.")
                continue

            # Compute daily PE/EPS series
            daily = _compute_daily_series(close, pe_monthly)

            # Insert PE rows only for dates earlier than min_pe (if it exists)
            pe_rows = _insert_series(
                conn=conn,
                spec=pe_spec,
                ticker=tk,
                type_value="TTM",
                series=daily["pe"],
                cutoff_exclusive=min_pe,
                dry_run=args.dry_run,
            )

            # Insert EPS rows only for dates earlier than min_eps (if it exists)
            eps_rows = _insert_series(
                conn=conn,
                spec=eps_spec,
                ticker=tk,
                type_value="TTM",
                series=daily["eps"],
                cutoff_exclusive=min_eps,
                dry_run=args.dry_run,
            )

            if args.dry_run:
                print(
                    f"DRY RUN: would insert {pe_rows:,} PE rows and {eps_rows:,} "
                    f"EPS rows for {tk}."
                )
            else:
                conn.commit()
                print(f"Inserted {pe_rows:,} PE rows and {eps_rows:,} EPS rows for {tk}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
