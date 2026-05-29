"""Generate the 4 watchlist-style charts for each top microcap candidate.

For each of the top N candidates the screener surfaces:
  1. Revenue YoY Change      (charts/microcap/{T}_revenue_yoy_change.png)
  2. EPS YoY Change          (charts/microcap/{T}_eps_yoy_change.png)
  3. Balance Sheet Breakdown (charts/microcap/{T}_BS_chart.png)
  4. Revenue / Net Income Forecast (charts/microcap/{T}_Revenue_Net_Income_Forecast.png)

Strategy: build a throwaway temp SQLite DB with the same schemas as
Stock Data.db (Annual_Data, TTM_Data, BalanceSheetData,
ForwardFinancialData), populate it from EDGAR + yfinance for each
candidate, then call the existing chart generators
(forecasted_earnings_chart.generate_forecast_charts_and_tables and
balancesheet_chart.balancesheet_chart) pointed at that DB. Outputs
land in charts/microcap/ so they don't conflict with watchlist
artifacts at charts/. After we're done, the temp DB is deleted.

This avoids any pollution of the canonical Stock Data.db with
candidate data we don't intend to persist between weekly runs.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# Re-use the screener's EDGAR provider helper.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microcap_screener import _edgar_annual_series  # noqa: E402

# Import the chart generators from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import balancesheet_chart  # noqa: E402
from forecasted_earnings_chart import generate_forecast_charts_and_tables  # noqa: E402

log = logging.getLogger("microcap_charts")

OUTPUT_DIR = REPO_ROOT / "charts" / "microcap"

SCHEMA_SQL = [
    """CREATE TABLE Annual_Data(
        Symbol TEXT,
        Date TEXT,
        Revenue REAL,
        Net_Income REAL,
        EPS REAL,
        Last_Updated TEXT,
        PRIMARY KEY(Symbol, Date)
    )""",
    """CREATE TABLE TTM_Data(
        Symbol TEXT PRIMARY KEY,
        TTM_Revenue REAL,
        TTM_Net_Income REAL,
        TTM_EPS REAL,
        Shares_Outstanding REAL,
        Quarter TEXT,
        Last_Updated TEXT
    )""",
    """CREATE TABLE BalanceSheetData(
        Symbol TEXT PRIMARY KEY,
        Date TEXT,
        Cash_and_Cash_Equivalents REAL,
        Total_Assets REAL,
        Total_Liabilities REAL,
        Total_Debt REAL,
        Total_Shareholder_Equity REAL,
        Last_Updated TEXT
    )""",
    """CREATE TABLE ForwardFinancialData(
        Ticker TEXT,
        Date TEXT,
        ForwardRevenue REAL,
        ForwardEPS REAL,
        ForwardRevenueAnalysts INTEGER,
        ForwardEPSAnalysts INTEGER,
        LastUpdated TEXT,
        Source TEXT,
        PRIMARY KEY(Ticker, Date)
    )""",
]


def _build_temp_db() -> str:
    """Create a fresh temp SQLite file with the necessary tables."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="microcap_charts_")
    os.close(fd)
    conn = sqlite3.connect(path)
    for stmt in SCHEMA_SQL:
        conn.execute(stmt)
    conn.commit()
    conn.close()
    return path


def _populate_annual(conn: sqlite3.Connection, ticker: str) -> int:
    """Pull annual EPS/Revenue/Net_Income from EDGAR and insert into
    Annual_Data. Returns the number of rows inserted."""
    eps_series = _edgar_annual_series(ticker, "EPS")
    rev_series = _edgar_annual_series(ticker, "Revenue")
    # Net Income isn't directly exposed by _edgar_annual_series but the
    # EDGAR provider stores it alongside EPS — re-fetch via the provider.
    from microcap_screener import _get_edgar
    try:
        records = _get_edgar().fetch_annual_financials(ticker)
    except Exception:
        records = []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = 0
    for r in records:
        date = r.get("Date")
        if not date:
            continue
        try:
            conn.execute(
                "INSERT OR REPLACE INTO Annual_Data "
                "(Symbol, Date, Revenue, Net_Income, EPS, Last_Updated) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ticker,
                    date,
                    float(r["Revenue"]) if r.get("Revenue") is not None else None,
                    float(r["Net_Income"]) if r.get("Net_Income") is not None else None,
                    float(r["EPS"]) if r.get("EPS") is not None else None,
                    now,
                ),
            )
            rows += 1
        except (TypeError, ValueError):
            continue
    return rows


def _populate_ttm(conn: sqlite3.Connection, ticker: str) -> bool:
    """Fetch the trailing-12-month financials from yfinance and insert into
    TTM_Data. Returns True if we wrote anything."""
    try:
        yt = yf.Ticker(ticker)
        q = yt.quarterly_financials
        info = yt.info or {}
    except Exception as exc:
        log.debug(f"{ticker}: yfinance TTM fetch failed: {exc}")
        return False
    if q is None or q.empty:
        return False

    def _sum_last4(row_names):
        for name in row_names:
            if name in q.index:
                series = q.loc[name].dropna().astype(float)
                if len(series) >= 4:
                    return float(series.iloc[:4].sum())
                elif len(series) > 0:
                    return float(series.sum())
        return None

    ttm_rev = _sum_last4(["Total Revenue", "Revenue"])
    ttm_ni = _sum_last4(["Net Income", "Net Income Common Stockholders"])
    shares = None
    for k in ("sharesOutstanding", "impliedSharesOutstanding"):
        v = info.get(k)
        if v:
            try:
                shares = float(v)
                break
            except (TypeError, ValueError):
                pass
    ttm_eps = None
    if shares and ttm_ni is not None and shares > 0:
        ttm_eps = ttm_ni / shares

    # The most recent quarter end (label for the chart).
    try:
        quarter = str(q.columns[0])[:10]
    except (IndexError, AttributeError):
        quarter = ""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT OR REPLACE INTO TTM_Data "
        "(Symbol, TTM_Revenue, TTM_Net_Income, TTM_EPS, Shares_Outstanding, Quarter, Last_Updated) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ticker, ttm_rev, ttm_ni, ttm_eps, shares, quarter, now),
    )
    return True


def _populate_balance_sheet(conn: sqlite3.Connection, ticker: str) -> bool:
    """Fetch the most recent balance sheet from yfinance and insert into
    BalanceSheetData. Returns True if we wrote anything."""
    try:
        yt = yf.Ticker(ticker)
        bs = yt.quarterly_balance_sheet
    except Exception as exc:
        log.debug(f"{ticker}: yfinance balance sheet fetch failed: {exc}")
        return False
    if bs is None or bs.empty:
        return False

    most_recent = bs.iloc[:, 0]
    def _get(*names):
        for n in names:
            if n in most_recent.index:
                v = most_recent.loc[n]
                try:
                    fv = float(v)
                    if fv == fv:  # not NaN
                        return fv
                except (TypeError, ValueError):
                    continue
        return None

    cash = _get("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    total_assets = _get("Total Assets")
    total_liabilities = _get(
        "Total Liabilities Net Minority Interest",
        "Total Liab",
        "Total Liabilities",
    )
    total_equity = _get(
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
        "Total Stockholder Equity",
    )
    total_debt = _get("Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation")

    if total_assets is None or total_liabilities is None or total_equity is None:
        log.debug(f"{ticker}: balance sheet missing required fields; skipping")
        return False

    date_label = str(bs.columns[0])[:10]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT OR REPLACE INTO BalanceSheetData "
        "(Symbol, Date, Cash_and_Cash_Equivalents, Total_Assets, Total_Liabilities, Total_Debt, Total_Shareholder_Equity, Last_Updated) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (ticker, date_label, cash, total_assets, total_liabilities, total_debt, total_equity, now),
    )
    return True


def _populate_forward(conn: sqlite3.Connection, ticker: str) -> bool:
    """Fetch forward EPS/revenue estimates from yfinance and insert into
    ForwardFinancialData. Re-uses the helpers already used by the existing
    Forward_data module so the schema lines up. Returns True if rows
    were written."""
    try:
        from Forward_data import yahoo_annual_estimates
    except Exception as exc:
        log.debug(f"{ticker}: Forward_data import failed: {exc}")
        return False

    try:
        df = yahoo_annual_estimates(ticker)
    except Exception as exc:
        log.debug(f"{ticker}: forward estimates fetch failed: {exc}")
        return False
    if df is None or df.empty:
        return False

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = 0
    for _, r in df.iterrows():
        try:
            conn.execute(
                "INSERT OR REPLACE INTO ForwardFinancialData "
                "(Ticker, Date, ForwardRevenue, ForwardEPS, ForwardRevenueAnalysts, ForwardEPSAnalysts, LastUpdated, Source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ticker,
                    r.get("Date"),
                    float(r["ForwardRevenue"]) if pd.notna(r.get("ForwardRevenue")) else None,
                    float(r["ForwardEPS"]) if pd.notna(r.get("ForwardEPS")) else None,
                    int(r["ForwardRevenueAnalysts"]) if pd.notna(r.get("ForwardRevenueAnalysts")) else None,
                    int(r["ForwardEPSAnalysts"]) if pd.notna(r.get("ForwardEPSAnalysts")) else None,
                    now,
                    "yahoo",
                ),
            )
            rows += 1
        except (TypeError, ValueError, KeyError):
            continue
    return rows > 0


def generate_for_ticker(ticker: str, temp_db_path: str) -> dict:
    """Populate the temp DB with everything we have for one ticker, then
    call the chart generators against it. Returns a dict of which chart
    files were produced (relative web paths under /charts/microcap/)."""
    ticker = ticker.strip().upper()
    if not ticker:
        return {}

    conn = sqlite3.connect(temp_db_path)
    annual_rows = 0
    have_ttm = False
    have_bs = False
    have_fwd = False
    try:
        annual_rows = _populate_annual(conn, ticker)
        have_ttm = _populate_ttm(conn, ticker)
        have_bs = _populate_balance_sheet(conn, ticker)
        have_fwd = _populate_forward(conn, ticker)
        conn.commit()
    finally:
        conn.close()

    if annual_rows == 0:
        log.warning(f"{ticker}: no Annual_Data; skipping chart generation")
        return {}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Revenue YoY, EPS YoY, Revenue/NI Forecast, EPS Forecast — all from
    # forecasted_earnings_chart.generate_forecast_charts_and_tables. It
    # accepts a db_path argument and a charts_output_dir.
    out: dict = {}
    try:
        generate_forecast_charts_and_tables(ticker, temp_db_path, str(OUTPUT_DIR) + "/")
    except Exception as exc:
        log.warning(f"{ticker}: forecast charts failed: {exc}")

    # 2) Balance sheet chart. balancesheet_chart hardcodes DB_PATH and the
    # output dir at module level — monkey-patch them around the call.
    if have_bs:
        prev_db = balancesheet_chart.DB_PATH
        prev_dir = balancesheet_chart.charts_output_dir
        try:
            balancesheet_chart.DB_PATH = temp_db_path
            balancesheet_chart.charts_output_dir = str(OUTPUT_DIR) + "/"
            data = balancesheet_chart.fetch_balance_sheet_data(ticker)
            if data:
                balancesheet_chart.plot_chart(data, str(OUTPUT_DIR) + "/", ticker)
        except Exception as exc:
            log.warning(f"{ticker}: balance sheet chart failed: {exc}")
        finally:
            balancesheet_chart.DB_PATH = prev_db
            balancesheet_chart.charts_output_dir = prev_dir

    # Discover which output files actually got created and report relative
    # web paths.
    candidates = {
        "revenue_yoy":   f"{ticker}_revenue_yoy_change.png",
        "eps_yoy":       f"{ticker}_eps_yoy_change.png",
        "forecast_rni":  f"{ticker}_Revenue_Net_Income_Forecast.png",
        "forecast_eps":  f"{ticker}_EPS_Forecast.png",
        "balance_sheet": f"{ticker}_balance_sheet_chart.png",
    }
    for key, fname in list(candidates.items()):
        if (OUTPUT_DIR / fname).exists():
            out[key] = f"/charts/microcap/{fname}"
    return out


def generate_for_tickers(tickers: list[str]) -> dict[str, dict]:
    """Run the per-ticker generator for each input. Returns
    {ticker: {chart_key: web_path}}. One shared temp DB is reused across
    all tickers so we pay the table-create cost only once."""
    temp_db = _build_temp_db()
    log.info(f"Temp chart DB: {temp_db}")
    results: dict[str, dict] = {}
    t0 = time.time()
    try:
        for i, t in enumerate(tickers, 1):
            try:
                out = generate_for_ticker(t, temp_db)
                results[t.strip().upper()] = out
                log.info(f"  [{i}/{len(tickers)}] {t}: {len(out)} charts")
            except Exception as exc:
                log.warning(f"  [{i}/{len(tickers)}] {t}: chart pipeline crashed: {exc}")
                results[t.strip().upper()] = {}
    finally:
        try:
            os.unlink(temp_db)
        except OSError:
            pass
    log.info(f"Chart pipeline finished in {time.time()-t0:.0f}s for {len(tickers)} tickers")
    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="+", help="Tickers to render charts for")
    args = p.parse_args()
    r = generate_for_tickers(args.tickers)
    for t, paths in r.items():
        print(f"{t}: {paths}")
