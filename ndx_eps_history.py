"""Nasdaq-100 trailing EPS history (monthly, 1990+) for the QQQ growth page.

EPS = NDX monthly close (Yahoo ^NDX) / NDX trailing P/E (worldperatio.com,
which embeds its full monthly P/E series in the page as `detailPE_data`).

Why not the IMPLIED_FROM_PE series in Index_EPS_History: it only starts in
2016 and its 2017-2020 calendar years are wrong (+68% in 2018, -11% in 2019
and 2020; WSJ year-end P/Es put 2018 near +34%). Checked 2026-10-16.

Known break: worldperatio's P/E jumps 13.7 -> 17.4 between Feb and Mar 2015
with the price flat (a source/method change, not earnings collapsing 23%).
The pre-break segment is chain-linked so that one month carries no change.

A snapshot lives in data/ndx_pe_monthly_worldperatio.csv; a failed fetch
falls back to it so the page never loses the history.
"""

import os
import re

import pandas as pd

_URL = "https://worldperatio.com/index/nasdaq-100/"
_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "ndx_pe_monthly_worldperatio.csv")
_BREAK = pd.Timestamp("2015-03-01")


def _fetch_pe() -> pd.Series:
    import requests
    html = requests.get(_URL, timeout=20,
                        headers={"User-Agent": "Mozilla/5.0"}).text
    m = re.search(r"detailPE_data\s*=\s*\[(.*?)\];", html, re.S)
    if not m:
        raise ValueError("detailPE_data not found")
    pts = re.findall(r"Date\.UTC\((\d+),\s*(\d+),\s*(\d+)\),([\d.]+)\]", m.group(1))
    pe = pd.Series({pd.Timestamp(int(y), int(mo) + 1, 1): float(v) for y, mo, _d, v in pts})
    if len(pe) < 300:
        raise ValueError(f"only {len(pe)} P/E points")
    return pe.sort_index()


def _fetch_close() -> pd.Series:
    import yfinance as yf
    px = yf.Ticker("^NDX").history(start="1989-12-01", interval="1mo",
                                   auto_adjust=False)["Close"]
    px.index = px.index.tz_localize(None).to_period("M").to_timestamp()
    px = px[~px.index.duplicated(keep="last")]
    if len(px) < 300:
        raise ValueError(f"only {len(px)} ^NDX months")
    return px


def load_ndx_monthly() -> pd.DataFrame:
    """DataFrame[pe, close] indexed by month start; live, else the CSV snapshot."""
    try:
        df = pd.DataFrame({"pe": _fetch_pe(), "close": _fetch_close()}).dropna()
        try:
            df.to_csv(_CSV, index_label="month")
        except Exception:
            pass
        return df
    except Exception as e:
        print(f"[ndx_eps] live fetch failed ({e}); using {os.path.basename(_CSV)}")
        if os.path.exists(_CSV):
            return pd.read_csv(_CSV, index_col="month", parse_dates=["month"])
        return pd.DataFrame(columns=["pe", "close"])


def ndx_monthly_eps() -> pd.Series:
    """Monthly NDX trailing EPS (index points), break-adjusted, at month end."""
    df = load_ndx_monthly()
    if df.empty:
        return pd.Series(dtype=float)
    eps = (df["close"] / df["pe"]).astype(float)
    if _BREAK in eps.index:
        prev = eps[eps.index < _BREAK]
        if not prev.empty:
            eps[eps.index < _BREAK] *= eps[_BREAK] / prev.iloc[-1]
    # Label each month by its last day, but never in the future.
    idx = eps.index.to_period("M").to_timestamp(how="end").normalize()
    today = pd.Timestamp.today().normalize()
    eps.index = idx.where(idx <= today, today)
    return eps
