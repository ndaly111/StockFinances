"""Does the implied-vs-delivered growth gap predict forward returns?

For each month-end t:
    gap_t  = trend_t - implied_t
    trend_t = log-trend EPS growth fitted on the trailing <=20Y (min 8Y) of
              TTM EPS available AT t (point-in-time, no hindsight)
    implied_t = the page's implied growth at t
and the index's annualized PRICE return over the next 1, 3 and 5 years
(dividends excluded; ~1-2%/yr, roughly constant across buckets).

Output: rows per gap band (the verdict labels) with n months, median forward
returns, % of 5Y windows positive, plus an all-months base rate. Monthly
windows overlap heavily, so n is NOT independent evidence — the number of
non-overlapping 5Y windows is reported alongside.
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_PX_CSV = {"SPY": os.path.join(_DATA, "sp500_monthly_close.csv")}

BANDS = [("Expensive", -np.inf, -2.0), ("Modestly expensive", -2.0, -0.5),
         ("Fair", -0.5, 0.5), ("Modestly cheap", 0.5, 2.0), ("Cheap", 2.0, np.inf)]


def monthly_index_close(ticker: str) -> pd.Series:
    """Month-end index close. QQQ -> ^NDX (via ndx_eps_history), SPY -> ^GSPC
    (yfinance, CSV snapshot fallback)."""
    if ticker.upper() == "QQQ":
        from ndx_eps_history import load_ndx_monthly
        px = load_ndx_monthly()["close"]
    else:
        csv = _PX_CSV.get(ticker.upper())
        try:
            import yfinance as yf
            px = yf.Ticker("^GSPC").history(start="1950-01-01", interval="1mo",
                                            auto_adjust=False)["Close"]
            px.index = px.index.tz_localize(None).to_period("M").to_timestamp()
            px = px[~px.index.duplicated(keep="last")]
            if len(px) < 300:
                raise ValueError("short ^GSPC history")
            if csv:
                px.rename("close").to_csv(csv, index_label="month")
        except Exception as e:
            print(f"[gap study] ^GSPC fetch failed ({e}); using snapshot")
            if not csv or not os.path.exists(csv):
                return pd.Series(dtype=float)
            px = pd.read_csv(csv, index_col="month", parse_dates=["month"])["close"]
    px = px.astype(float)
    px.index = pd.DatetimeIndex(px.index).to_period("M").to_timestamp(how="end").normalize()
    return px.sort_index()


def point_in_time_trend(eps_m: pd.Series, max_years: int = 20, min_years: int = 8) -> pd.Series:
    """Rolling log-trend growth (%/yr) using only data up to each month."""
    e = eps_m.dropna(); e = e[e > 0]
    logs = np.log(e.values)
    out = {}
    for i in range(len(e)):
        lo = max(0, i + 1 - 12 * max_years)
        n = i + 1 - lo
        if n < 12 * min_years:
            continue
        x = np.arange(n) / 12.0
        slope = np.polyfit(x, logs[lo:i + 1], 1)[0]
        out[e.index[i]] = (np.exp(slope) - 1.0) * 100.0
    return pd.Series(out, dtype=float)


def _to_month_end(s: pd.Series) -> pd.Series:
    s = s.dropna().copy()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    s.index = idx
    s = s.resample("ME").last().dropna()
    s.index = s.index.normalize()
    return s


def gap_study(ticker: str, ig: pd.Series, eps: pd.Series, horizons=(1, 3, 5)) -> Optional[Dict]:
    px = monthly_index_close(ticker)
    if px.empty:
        return None
    ig_m, eps_m = _to_month_end(ig), _to_month_end(eps)
    trend = point_in_time_trend(eps_m)
    df = pd.DataFrame({"ig": ig_m, "trend": trend, "px": px}).dropna(subset=["ig", "trend"])
    df["px"] = df["px"].ffill()
    df = df.dropna(subset=["px"])
    if len(df) < 60:
        return None
    df["gap"] = df["trend"] - df["ig"]
    for h in horizons:
        fwd = df["px"].shift(-12 * h)
        df[f"r{h}"] = ((fwd / df["px"]) ** (1.0 / h) - 1.0) * 100.0
    today_gap = float(df["gap"].iloc[-1])
    rows = []

    def _row(name, sub):
        r = {"band": name, "n": int(len(sub))}
        for h in horizons:
            v = sub[f"r{h}"].dropna()
            r[f"med{h}"] = float(v.median()) if len(v) else None
            r[f"n{h}"] = int(len(v))
        v5 = sub[f"r{max(horizons)}"].dropna()
        r["pos"] = float((v5 > 0).mean() * 100.0) if len(v5) else None
        r["today"] = False
        return r

    for name, lo, hi in BANDS:
        sub = df[(df["gap"] >= lo) & (df["gap"] < hi)]
        r = _row(name, sub)
        r["today"] = lo <= today_gap < hi
        rows.append(r)
    rows.append(_row("All months", df))
    h5 = max(horizons)
    valid = df.dropna(subset=[f"r{h5}"])
    rho = float(valid["gap"].corr(valid[f"r{h5}"], method="spearman")) if len(valid) > 24 else None
    return {
        "rows": rows, "horizons": list(horizons), "today_gap": today_gap,
        "start": df.index[0].date().isoformat(), "end": df.index[-1].date().isoformat(),
        "n_months": int(len(df)), "n_indep": int(len(valid) // (12 * h5)) if len(valid) else 0,
        "rho": rho,
        "scatter": valid[["gap", f"r{h5}"]].rename(columns={f"r{h5}": "fwd"}),
    }


if __name__ == "__main__":
    import sqlite3
    import generate_index_growth_pages as g
    conn = sqlite3.connect("Stock Data.db")
    for tk in ("QQQ", "SPY"):
        d = g._load_series(conn, tk)
        res = gap_study(tk, d["ig"], d["eps"])
        if not res:
            print(tk, "no result"); continue
        print(f"\n{tk}: {res['start']}..{res['end']}, {res['n_months']} months, "
              f"~{res['n_indep']} independent 5Y windows, today gap {res['today_gap']:+.1f}, "
              f"spearman(gap, 5Y fwd) = {res['rho']:+.2f}")
        def f(v):
            return "   n/a" if v is None else f"{v:6.1f}"
        for r in res["rows"]:
            print(f"  {r['band']:<20} n={r['n']:<4} 1Y {f(r['med1'])} 3Y {f(r['med3'])} 5Y {f(r['med5'])}  "
                  f"%pos5Y {'n/a' if r['pos'] is None else round(r['pos'])}{'  <-- today' if r['today'] else ''}")
