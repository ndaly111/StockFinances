"""Index membership + weights from slickcharts (Wikipedia fallback) and the
weight-prioritized auto-extend of the forward-EPS scrape universe."""
from __future__ import annotations
import io, logging, sqlite3
from datetime import date
from typing import List, Tuple
import pandas as pd, requests

logger = logging.getLogger(__name__)
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Referer": "https://www.google.com/"}
_SLICK = {"QQQ": "https://www.slickcharts.com/nasdaq100",
          "SPY": "https://www.slickcharts.com/sp500"}
_MIN_ROWS = {"QQQ": 99, "SPY": 490}


def parse_slickcharts(html: str) -> List[Tuple[str, float]]:
    tables = pd.read_html(io.StringIO(html))
    target = next((t for t in tables if "Symbol" in t.columns and "Weight" in t.columns), None)
    if target is None:
        raise ValueError("slickcharts table structure changed")
    out = []
    for _, row in target.iterrows():
        tk = str(row["Symbol"]).strip()
        w = str(row["Weight"]).strip().rstrip("%").replace(",", "")
        try:
            out.append((tk, float(w)))
        except ValueError:
            continue
    return out


def fetch_holdings(index_name: str) -> List[Tuple[str, float]]:
    url = _SLICK[index_name.upper()]
    r = requests.get(url, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    rows = parse_slickcharts(r.text)
    if len(rows) < _MIN_ROWS[index_name.upper()]:
        raise ValueError(f"{index_name}: only {len(rows)} holdings parsed (partial load?)")
    return rows


def persist_holdings(conn: sqlite3.Connection, index_name: str, rows: List[Tuple[str, float]], today=None) -> None:
    import index_forward_eps as ife
    ife.ensure_constituents_table(conn)
    today = today or date.today().isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO index_constituents (date_recorded, index_name, ticker, weight) VALUES (?,?,?,?)",
        [(today, index_name.upper(), tk, w) for tk, w in rows])
    conn.commit()


def managed_scrape_tickers(conn: sqlite3.Connection) -> set:
    """Return the set of constituent tickers from the latest index_constituents snapshot.

    Used to auto-extend the forward-EPS scrape universe without modifying tickers.csv.
    Returns an empty set if the table is missing or empty.
    """
    try:
        rows = conn.execute(
            """SELECT DISTINCT ticker FROM index_constituents
               WHERE date_recorded = (SELECT MAX(date_recorded) FROM index_constituents)"""
        ).fetchall()
    except sqlite3.Error:
        return set()
    return {r[0].upper() for r in rows}


def uncovered_for_target(holdings: List[Tuple[str, float]], covered, target_pct: float = 90.0) -> List[str]:
    """Highest-weight uncovered tickers to add until cumulative covered weight >= target."""
    covered = {c.upper() for c in covered}
    have = sum(w for tk, w in holdings if tk.upper() in covered)
    total = sum(w for _, w in holdings) or 1.0
    add = []
    for tk, w in sorted(holdings, key=lambda x: x[1], reverse=True):
        if have / total * 100.0 >= target_pct:
            break
        if tk.upper() in covered:
            continue
        add.append(tk); have += w
    return add
