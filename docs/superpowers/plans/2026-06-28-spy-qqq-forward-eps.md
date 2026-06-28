# SPY / QQQ Forward EPS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show future estimated EPS growth for SPY and QQQ via a daily-snapshotted consensus forward-EPS point on the existing index growth-page charts.

**Architecture:** A new `index_forward_eps.py` fetches a consensus forward EPS for each index (stockanalysis.com primary, yfinance fallback), scales ETF→index level, sanity-checks it, and upserts a daily row into a new `Index_Forward_EPS_History` table. `index_growth_charts.py` reads the latest row and draws a dashed forward point bending off the end of the existing EPS line, plus a callout giving the expected growth %. The homepage overview table is untouched.

**Tech Stack:** Python 3, SQLite, requests + BeautifulSoup/pandas (scrape), yfinance (fallback), Bokeh (charts), pytest (`Test/`).

**Spec:** `docs/superpowers/specs/2026-06-28-spy-qqq-forward-eps-design.md`

---

## File Structure

- **Create:** `index_forward_eps.py` — fetch + scale + sanity-check + snapshot consensus forward EPS for SPY/QQQ. Owns the new table.
- **Create:** `Test/test_index_forward_eps.py` — unit tests for fetch/fallback/guards/snapshot.
- **Create:** `Test/fixtures/stockanalysis_spy.html`, `Test/fixtures/stockanalysis_qqq.html` — saved page fixtures for the parser test.
- **Modify:** `index_growth_charts.py` — add a forward-EPS read helper, a callout helper, an overlay helper, and wire them into `render_index_growth_charts`.
- **Modify:** `Test/test_index_growth_charts.py` — add a test that the forward overlay/callout is applied when data exists.
- **Modify:** `main_remote.py` — call `snapshot_forward_eps()` in the daily run, right after `index_growth(treasury)`.

**Run tests from repo root** (`C:\Users\ndaly\projects\sf-fix`) with: `python -m pytest <path> -v`

---

## Task 1: New table + ensure function in `index_forward_eps.py`

**Files:**
- Create: `index_forward_eps.py`
- Test: `Test/test_index_forward_eps.py`

- [ ] **Step 1: Write the failing test**

```python
# Test/test_index_forward_eps.py
import pathlib, sqlite3, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import index_forward_eps as ife


def test_ensure_table_creates_schema(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        ife.ensure_forward_eps_table(conn)
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(Index_Forward_EPS_History)")}
    assert cols == {
        "date_recorded", "ticker", "forward_eps_etf",
        "forward_eps_index", "forward_pe", "horizon_date", "source",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Test/test_index_forward_eps.py::test_ensure_table_creates_schema -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'index_forward_eps'`

- [ ] **Step 3: Write minimal implementation**

```python
# index_forward_eps.py
"""Consensus forward EPS for SPY/QQQ (index level), snapshotted daily.

Primary source: stockanalysis.com ETF pages (holdings-weighted forward P/E +
forward EPS). Fallback: yfinance ETF forwardPE/forwardEps. Values are scaled
from ETF level to index level so they line up with index_growth_charts EPS.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "Stock Data.db"
TABLE = "Index_Forward_EPS_History"
IDXES = ["SPY", "QQQ"]

# Mirror of index_growth_charts._INDEX_EPS_DIVISOR. A drift-guard test in
# Test/test_index_forward_eps.py asserts these stay equal.
INDEX_EPS_DIVISOR = {"SPY": 10.0, "QQQ": 4.0}


def ensure_forward_eps_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            date_recorded     TEXT NOT NULL,
            ticker            TEXT NOT NULL,
            forward_eps_etf   REAL,
            forward_eps_index REAL,
            forward_pe        REAL,
            horizon_date      TEXT,
            source            TEXT,
            PRIMARY KEY (date_recorded, ticker)
        )
        """
    )
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Test/test_index_forward_eps.py::test_ensure_table_creates_schema -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add index_forward_eps.py Test/test_index_forward_eps.py
git commit -m "feat(forward-eps): add Index_Forward_EPS_History table + ensure fn"
```

---

## Task 2: Divisor drift-guard test

**Files:**
- Test: `Test/test_index_forward_eps.py`

- [ ] **Step 1: Write the failing test**

```python
def test_divisor_matches_chart_module():
    import index_growth_charts as igc
    assert ife.INDEX_EPS_DIVISOR == igc._INDEX_EPS_DIVISOR
```

- [ ] **Step 2: Run test**

Run: `python -m pytest Test/test_index_forward_eps.py::test_divisor_matches_chart_module -v`
Expected: PASS (both already `{"SPY": 10.0, "QQQ": 4.0}`). If it FAILS, the chart module changed — update `INDEX_EPS_DIVISOR` to match, do not edit the test.

- [ ] **Step 3: Commit**

```bash
git add Test/test_index_forward_eps.py
git commit -m "test(forward-eps): guard divisor against drift from chart module"
```

---

## Task 3: yfinance fallback + scaling

**Files:**
- Modify: `index_forward_eps.py`
- Test: `Test/test_index_forward_eps.py`

- [ ] **Step 1: Write the failing test**

```python
def test_forward_from_yf_info_scales_to_index():
    info = {"forwardPE": 22.0, "forwardEps": 25.0, "regularMarketPrice": 550.0}
    fe = ife._forward_from_yf("SPY", info)
    assert fe is not None
    assert fe.forward_pe == 22.0
    assert fe.forward_eps_etf == 25.0
    assert fe.forward_eps_index == 250.0   # 25.0 * divisor(10)
    assert fe.source == "yfinance"


def test_forward_from_yf_derives_eps_when_missing():
    # No forwardEps -> derive from price / forwardPE
    info = {"forwardPE": 20.0, "regularMarketPrice": 500.0}
    fe = ife._forward_from_yf("QQQ", info)
    assert fe is not None
    assert round(fe.forward_eps_etf, 4) == 25.0      # 500/20
    assert round(fe.forward_eps_index, 4) == 100.0   # 25 * divisor(4)


def test_forward_from_yf_returns_none_without_pe():
    assert ife._forward_from_yf("SPY", {"regularMarketPrice": 500.0}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Test/test_index_forward_eps.py -k forward_from_yf -v`
Expected: FAIL with `AttributeError: module 'index_forward_eps' has no attribute '_forward_from_yf'`

- [ ] **Step 3: Write minimal implementation**

Add to `index_forward_eps.py`:

```python
@dataclass
class ForwardEPS:
    ticker: str
    forward_pe: float
    forward_eps_etf: float
    forward_eps_index: float
    horizon_date: str
    source: str


def _default_horizon() -> str:
    """NTM estimates ~ 12 months out; used when the source gives no date."""
    return (date.today() + timedelta(days=365)).isoformat()


def _scale_index(tk: str, eps_etf: float) -> float:
    return eps_etf * INDEX_EPS_DIVISOR.get(tk.upper(), 1.0)


def _forward_from_yf(tk: str, info: dict) -> Optional[ForwardEPS]:
    pe = info.get("forwardPE")
    eps_etf = info.get("forwardEps")
    price = info.get("regularMarketPrice")
    try:
        pe = float(pe) if pe is not None else None
    except (TypeError, ValueError):
        pe = None
    if pe is None or pe <= 0:
        return None
    if eps_etf is None and price:
        try:
            eps_etf = float(price) / pe
        except (TypeError, ValueError, ZeroDivisionError):
            eps_etf = None
    if eps_etf is None:
        return None
    eps_etf = float(eps_etf)
    return ForwardEPS(
        ticker=tk.upper(),
        forward_pe=pe,
        forward_eps_etf=eps_etf,
        forward_eps_index=_scale_index(tk, eps_etf),
        horizon_date=_default_horizon(),
        source="yfinance",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Test/test_index_forward_eps.py -k forward_from_yf -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add index_forward_eps.py Test/test_index_forward_eps.py
git commit -m "feat(forward-eps): yfinance fallback fetch with ETF->index scaling"
```

---

## Task 4: Sanity guard

**Files:**
- Modify: `index_forward_eps.py`
- Test: `Test/test_index_forward_eps.py`

- [ ] **Step 1: Write the failing test**

```python
def test_sanity_accepts_reasonable():
    # forward index EPS 250 vs latest 230 -> +8.7% growth, scale ~1.09x
    assert ife._passes_sanity(forward_pe=22.0, forward_eps_index=250.0,
                              latest_hist_eps=230.0) is True


def test_sanity_rejects_nonpositive_pe():
    assert ife._passes_sanity(0.0, 250.0, 230.0) is False
    assert ife._passes_sanity(-5.0, 250.0, 230.0) is False


def test_sanity_rejects_out_of_band_growth():
    # +200% implied growth is absurd for an index
    assert ife._passes_sanity(22.0, 690.0, 230.0) is False
    # -60% collapse
    assert ife._passes_sanity(22.0, 90.0, 230.0) is False


def test_sanity_rejects_bad_scale():
    # forward index EPS 3x the latest -> scaling/source error
    assert ife._passes_sanity(22.0, 800.0, 230.0) is False


def test_sanity_allows_missing_history():
    # No history to compare against -> only the P/E and absolute checks apply
    assert ife._passes_sanity(22.0, 250.0, None) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Test/test_index_forward_eps.py -k sanity -v`
Expected: FAIL with `AttributeError: ... '_passes_sanity'`

- [ ] **Step 3: Write minimal implementation**

```python
# Sane bounds (per spec §6). Growth band -50%..+80%; scale band 0.5x..2.0x.
_GROWTH_MIN, _GROWTH_MAX = -0.50, 0.80
_SCALE_MIN, _SCALE_MAX = 0.5, 2.0


def _passes_sanity(forward_pe, forward_eps_index, latest_hist_eps) -> bool:
    try:
        forward_pe = float(forward_pe)
        forward_eps_index = float(forward_eps_index)
    except (TypeError, ValueError):
        return False
    if forward_pe <= 0 or forward_eps_index <= 0:
        return False
    if latest_hist_eps is None:
        return True
    try:
        latest = float(latest_hist_eps)
    except (TypeError, ValueError):
        return True
    if latest <= 0:
        return True
    growth = forward_eps_index / latest - 1.0
    if not (_GROWTH_MIN <= growth <= _GROWTH_MAX):
        return False
    scale = forward_eps_index / latest
    if not (_SCALE_MIN <= scale <= _SCALE_MAX):
        return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Test/test_index_forward_eps.py -k sanity -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add index_forward_eps.py Test/test_index_forward_eps.py
git commit -m "feat(forward-eps): sanity guard (PE>0, growth band, scale band)"
```

---

## Task 5: stockanalysis.com parser (primary source)

**Files:**
- Modify: `index_forward_eps.py`
- Create: `Test/fixtures/stockanalysis_spy.html`
- Test: `Test/test_index_forward_eps.py`

- [ ] **Step 1: Create the fixture**

Fetch the live page once and save raw HTML (the parser is tested against this, so it must be the real layout):

```bash
python -c "import requests; open('Test/fixtures/stockanalysis_spy.html','w',encoding='utf-8').write(requests.get('https://stockanalysis.com/etf/spy/', headers={'User-Agent':'Mozilla/5.0'}, timeout=20).text)"
```

If the fetch is blocked (non-200 / Cloudflare), STOP and tell the user — the primary source is unavailable and we ship on the yfinance fallback only (skip Tasks 5 and the scrape branch in Task 6; the feature still works). Otherwise continue.

Open the saved file and confirm it contains a "Forward PE" (or "PE Ratio (Forward)") label with a numeric value. Note the exact label text — you'll match it in Step 3.

- [ ] **Step 2: Write the failing test**

```python
def test_parse_stockanalysis_extracts_forward_pe():
    html = (ROOT / "Test" / "fixtures" / "stockanalysis_spy.html").read_text(
        encoding="utf-8")
    pe = ife._parse_forward_pe(html)
    assert pe is not None
    assert 5.0 < pe < 60.0     # sane forward P/E for SPY
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest Test/test_index_forward_eps.py -k parse_stockanalysis -v`
Expected: FAIL with `AttributeError: ... '_parse_forward_pe'`

- [ ] **Step 4: Write minimal implementation**

```python
import re

import pandas as pd
import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
_FWD_PE_RE = re.compile(r"forward.*p/?e|p/?e.*forward", re.IGNORECASE)


def _parse_forward_pe(html: str) -> Optional[float]:
    """Find a 'Forward PE' label/value in any table on the page."""
    try:
        tables = pd.read_html(html)
    except (ValueError, Exception):
        return None
    for tbl in tables:
        if tbl.shape[1] < 2:
            continue
        labels = tbl.iloc[:, 0].astype(str)
        mask = labels.str.contains(_FWD_PE_RE, na=False)
        if not mask.any():
            continue
        raw = str(tbl.loc[mask].iloc[0, 1])
        m = re.search(r"-?\d+(?:\.\d+)?", raw.replace(",", ""))
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def _fetch_stockanalysis_pe(tk: str, session: requests.Session) -> Optional[float]:
    url = f"https://stockanalysis.com/etf/{tk.lower()}/"
    try:
        r = session.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("[%s] stockanalysis fetch failed: %s", tk, e)
        return None
    pe = _parse_forward_pe(r.text)
    if pe is None:
        logger.warning("[%s] stockanalysis: forward P/E not found (layout drift?)", tk)
    return pe
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest Test/test_index_forward_eps.py -k parse_stockanalysis -v`
Expected: PASS

If the value the parser finds is wrong, adjust `_FWD_PE_RE` to the exact label you noted in Step 1 — do not weaken the assertion.

- [ ] **Step 6: Commit**

```bash
git add index_forward_eps.py Test/test_index_forward_eps.py Test/fixtures/stockanalysis_spy.html
git commit -m "feat(forward-eps): stockanalysis.com forward P/E parser + fixture"
```

---

## Task 6: `fetch_forward_eps` orchestration (primary -> fallback -> guard)

**Files:**
- Modify: `index_forward_eps.py`
- Test: `Test/test_index_forward_eps.py`

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import patch


def test_fetch_prefers_stockanalysis_with_price():
    # stockanalysis gives forward PE; price comes from yfinance info.
    with (
        patch.object(ife, "_fetch_stockanalysis_pe", return_value=20.0),
        patch.object(ife, "_yf_info", return_value={"regularMarketPrice": 460.0}),
    ):
        fe = ife.fetch_forward_eps("SPY", session=object(), latest_hist_eps=230.0)
    assert fe is not None
    assert fe.source == "stockanalysis"
    assert fe.forward_pe == 20.0
    assert round(fe.forward_eps_etf, 4) == 23.0       # 460/20
    assert round(fe.forward_eps_index, 4) == 230.0    # *10


def test_fetch_falls_back_to_yfinance():
    with (
        patch.object(ife, "_fetch_stockanalysis_pe", return_value=None),
        patch.object(ife, "_yf_info",
                     return_value={"forwardPE": 22.0, "forwardEps": 24.0,
                                   "regularMarketPrice": 528.0}),
    ):
        fe = ife.fetch_forward_eps("SPY", session=object(), latest_hist_eps=230.0)
    assert fe is not None
    assert fe.source == "yfinance"
    assert fe.forward_eps_index == 240.0              # 24 * 10


def test_fetch_returns_none_when_sanity_fails():
    # forward PE 2.0 -> EPS 230 etf -> 2300 index -> absurd scale, rejected
    with (
        patch.object(ife, "_fetch_stockanalysis_pe", return_value=2.0),
        patch.object(ife, "_yf_info", return_value={"regularMarketPrice": 460.0}),
    ):
        fe = ife.fetch_forward_eps("SPY", session=object(), latest_hist_eps=230.0)
    assert fe is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Test/test_index_forward_eps.py -k "fetch_" -v`
Expected: FAIL with `AttributeError: ... 'fetch_forward_eps'` / `'_yf_info'`

- [ ] **Step 3: Write minimal implementation**

```python
import yfinance as yf


def _yf_info(tk: str) -> dict:
    try:
        info = yf.Ticker(tk).info
        return info if isinstance(info, dict) else {}
    except Exception as e:        # noqa: BLE001 - yfinance raises many types
        logger.warning("[%s] yfinance info failed: %s", tk, e)
        return {}


def fetch_forward_eps(tk, session, latest_hist_eps) -> Optional[ForwardEPS]:
    """Primary stockanalysis.com -> yfinance fallback -> sanity guard."""
    tk = tk.upper()
    info = _yf_info(tk)
    price = info.get("regularMarketPrice")

    # Primary: stockanalysis forward P/E + price for the EPS dollar value.
    pe = _fetch_stockanalysis_pe(tk, session)
    if pe is not None and pe > 0 and price:
        eps_etf = float(price) / pe
        cand = ForwardEPS(
            ticker=tk, forward_pe=pe, forward_eps_etf=eps_etf,
            forward_eps_index=_scale_index(tk, eps_etf),
            horizon_date=_default_horizon(), source="stockanalysis",
        )
        if _passes_sanity(cand.forward_pe, cand.forward_eps_index, latest_hist_eps):
            return cand
        logger.warning("[%s] stockanalysis value failed sanity; trying yfinance", tk)

    # Fallback: yfinance.
    cand = _forward_from_yf(tk, info)
    if cand and _passes_sanity(cand.forward_pe, cand.forward_eps_index, latest_hist_eps):
        return cand
    logger.warning("[%s] no usable forward EPS (primary+fallback failed/rejected)", tk)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Test/test_index_forward_eps.py -k "fetch_" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add index_forward_eps.py Test/test_index_forward_eps.py
git commit -m "feat(forward-eps): fetch orchestration with fallback + guard"
```

---

## Task 7: `snapshot_forward_eps` (read history, fetch, upsert)

**Files:**
- Modify: `index_forward_eps.py`
- Test: `Test/test_index_forward_eps.py`

- [ ] **Step 1: Write the failing test**

```python
def test_snapshot_upserts_and_is_idempotent(tmp_path):
    db = tmp_path / "t.db"
    fe = ife.ForwardEPS("SPY", 20.0, 23.0, 230.0, "2027-06-28", "yfinance")

    def fake_fetch(tk, session, latest_hist_eps):
        return fe if tk == "SPY" else None

    with sqlite3.connect(db) as conn:
        ife.ensure_forward_eps_table(conn)
        with patch.object(ife, "fetch_forward_eps", side_effect=fake_fetch), \
             patch.object(ife, "_latest_hist_eps", return_value=225.0):
            ife.snapshot_forward_eps(conn)
            ife.snapshot_forward_eps(conn)   # same day -> overwrite, not dup
        rows = list(conn.execute(
            f"SELECT ticker, forward_eps_index, source FROM {ife.TABLE}"))
    assert rows == [("SPY", 230.0, "yfinance")]   # QQQ skipped (None), no dup
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Test/test_index_forward_eps.py -k snapshot -v`
Expected: FAIL with `AttributeError: ... 'snapshot_forward_eps'` / `'_latest_hist_eps'`

- [ ] **Step 3: Write minimal implementation**

```python
def _latest_hist_eps(conn: sqlite3.Connection, tk: str) -> Optional[float]:
    """Latest index-level historical EPS for scale/growth sanity checks.

    Mirrors index_growth_charts._series_eps source priority loosely: prefer
    TTM_REPORTED, else TTM_DAILY, else IMPLIED_FROM_PE*divisor.
    """
    try:
        row = conn.execute(
            """SELECT EPS, EPS_Type FROM Index_EPS_History
                WHERE Ticker=? AND EPS_Type IN
                      ('TTM_REPORTED','TTM_DAILY','IMPLIED_FROM_PE')
             ORDER BY CASE EPS_Type
                        WHEN 'TTM_REPORTED' THEN 0
                        WHEN 'TTM_DAILY'    THEN 1
                        ELSE 2 END, Date DESC
                LIMIT 1""",
            (tk,),
        ).fetchone()
    except sqlite3.Error:
        return None
    if not row or row[0] is None:
        return None
    eps, eps_type = float(row[0]), row[1]
    if eps_type == "IMPLIED_FROM_PE":
        eps *= INDEX_EPS_DIVISOR.get(tk.upper(), 1.0)
    return eps


def snapshot_forward_eps(conn: sqlite3.Connection) -> int:
    """Fetch + upsert today's forward EPS for each index. Returns rows written."""
    ensure_forward_eps_table(conn)
    today = date.today().isoformat()
    session = requests.Session()
    written = 0
    for tk in IDXES:
        latest = _latest_hist_eps(conn, tk)
        fe = fetch_forward_eps(tk, session, latest)
        if fe is None:
            continue
        conn.execute(
            f"""INSERT OR REPLACE INTO {TABLE}
                (date_recorded, ticker, forward_eps_etf, forward_eps_index,
                 forward_pe, horizon_date, source)
                VALUES (?,?,?,?,?,?,?)""",
            (today, fe.ticker, fe.forward_eps_etf, fe.forward_eps_index,
             fe.forward_pe, fe.horizon_date, fe.source),
        )
        written += 1
    conn.commit()
    logger.info("[forward-eps] snapshot wrote %d row(s) for %s", written, today)
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Test/test_index_forward_eps.py -k snapshot -v`
Expected: PASS

- [ ] **Step 5: Run the whole module suite**

Run: `python -m pytest Test/test_index_forward_eps.py -v`
Expected: PASS (all tasks 1-7)

- [ ] **Step 6: Commit**

```bash
git add index_forward_eps.py Test/test_index_forward_eps.py
git commit -m "feat(forward-eps): daily snapshot upsert into Index_Forward_EPS_History"
```

---

## Task 8: Wire daily snapshot into `main_remote.py`

**Files:**
- Modify: `main_remote.py` (import near the other index imports; call after `index_growth(treasury)` at the `spy_qqq_html = index_growth(treasury)` line, ~498)

- [ ] **Step 1: Add the import**

Find the existing index-related imports (near `from index_growth_table import index_growth`) and add:

```python
from index_forward_eps import snapshot_forward_eps
```

- [ ] **Step 2: Add the daily call**

Locate this line (~498):

```python
        spy_qqq_html = index_growth(treasury)
```

Immediately after it, add:

```python
        # Consensus forward EPS for SPY/QQQ (stockanalysis.com -> yfinance),
        # snapshotted daily so the growth-page EPS chart gets a forward point.
        try:
            snapshot_forward_eps(conn)
        except Exception as exc:  # never let this break the daily build
            print(f"[WARN] forward EPS snapshot failed: {exc}")
```

(`conn` is the open connection used throughout `mini_main`; it is in scope here.)

- [ ] **Step 3: Smoke-test the import path**

Run: `python -c "import main_remote; print('import ok')"`
Expected: prints `import ok` with no ImportError.

- [ ] **Step 4: Commit**

```bash
git add main_remote.py
git commit -m "feat(forward-eps): snapshot forward EPS in the daily index run"
```

---

## Task 9: Forward read helper + callout helper in `index_growth_charts.py`

**Files:**
- Modify: `index_growth_charts.py` (add helpers after `_INDEX_EPS_DIVISOR` definition, ~line 48)
- Test: `Test/test_index_growth_charts.py`

- [ ] **Step 1: Write the failing test**

```python
# append to Test/test_index_growth_charts.py
import sqlite3


def test_latest_forward_eps_reads_newest_row(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE Index_Forward_EPS_History (
                 date_recorded TEXT, ticker TEXT, forward_eps_etf REAL,
                 forward_eps_index REAL, forward_pe REAL, horizon_date TEXT,
                 source TEXT, PRIMARY KEY (date_recorded, ticker))""")
        conn.executemany(
            "INSERT INTO Index_Forward_EPS_History VALUES (?,?,?,?,?,?,?)",
            [("2026-06-01", "SPY", 23.0, 230.0, 20.0, "2027-06-01", "yfinance"),
             ("2026-06-28", "SPY", 25.0, 250.0, 22.0, "2027-06-28", "stockanalysis")])
        row = igc._latest_forward_eps(conn, "SPY")
    assert row["forward_eps_index"] == 250.0
    assert row["horizon_date"] == "2027-06-28"


def test_forward_eps_callout_text():
    txt = igc._forward_eps_callout(
        forward_eps_index=250.0, latest_hist_eps=230.0,
        horizon_date="2027-06-28", source="stockanalysis",
        forward_implied_growth=0.124)
    assert "8.7%" in txt           # 250/230 - 1 (expected EPS growth)
    assert "stockanalysis" in txt
    assert "$250" in txt
    assert "12.4%" in txt          # forward implied growth (valuation model)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Test/test_index_growth_charts.py -k "forward" -v`
Expected: FAIL with `AttributeError: ... '_latest_forward_eps'`

- [ ] **Step 3: Write minimal implementation**

Add to `index_growth_charts.py` (after `_INDEX_EPS_DIVISOR`):

```python
def _latest_forward_eps(conn, tk):
    """Return the newest Index_Forward_EPS_History row for *tk* as a dict, or None."""
    try:
        cur = conn.execute(
            """SELECT date_recorded, forward_eps_index, forward_pe,
                      horizon_date, source
                 FROM Index_Forward_EPS_History
                WHERE ticker=? ORDER BY date_recorded DESC LIMIT 1""",
            (tk.upper(),),
        )
        row = cur.fetchone()
    except Exception:
        return None
    if not row or row[1] is None:
        return None
    return {
        "date_recorded": row[0], "forward_eps_index": float(row[1]),
        "forward_pe": row[2], "horizon_date": row[3], "source": row[4],
    }


def _latest_forward_implied_growth(conn, tk):
    """Latest forward implied growth (valuation model) from Index_Growth_History.

    This value is already logged daily by index_growth_table._log_today
    (Growth_Type='Forward') but is not otherwise shown on the page.
    """
    try:
        row = conn.execute(
            """SELECT Implied_Growth FROM Index_Growth_History
                WHERE Ticker=? AND Growth_Type='Forward'
             ORDER BY Date DESC LIMIT 1""",
            (tk.upper(),),
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def _forward_eps_callout(forward_eps_index, latest_hist_eps, horizon_date, source,
                         forward_implied_growth=None):
    """One-line consensus sentence for the EPS block."""
    parts = [f"Consensus forward EPS ≈ ${forward_eps_index:,.0f} (index level)"]
    if latest_hist_eps and latest_hist_eps > 0:
        growth = forward_eps_index / latest_hist_eps - 1.0
        parts.append(f"→ <b>{growth:+.1%}</b> expected earnings growth")
    tail = f" (source: {source}"
    if horizon_date:
        tail += f", target {horizon_date}"
    tail += ")."
    sentence = " ".join(parts) + tail
    if forward_implied_growth is not None:
        sentence += (f" Forward implied growth (valuation model): "
                     f"{forward_implied_growth:.1%}.")
    return sentence
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Test/test_index_growth_charts.py -k "forward" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add index_growth_charts.py Test/test_index_growth_charts.py
git commit -m "feat(forward-eps): chart read helper + callout text"
```

---

## Task 10: Overlay glyph helper

**Files:**
- Modify: `index_growth_charts.py`
- Test: `Test/test_index_growth_charts.py`

- [ ] **Step 1: Write the failing test**

```python
def test_add_forward_eps_overlay_adds_renderers(tmp_path):
    from bokeh.plotting import figure as _figure
    fig = _figure(x_axis_type="datetime")
    before = len(fig.renderers)
    igc._add_forward_eps_overlay(
        fig,
        last_date=pd.Timestamp("2026-06-01"),
        last_eps=230.0,
        forward_date=pd.Timestamp("2027-06-01"),
        forward_eps_index=250.0,
    )
    # dashed connector line + a marker = 2 new renderers
    assert len(fig.renderers) == before + 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Test/test_index_growth_charts.py -k overlay -v`
Expected: FAIL with `AttributeError: ... '_add_forward_eps_overlay'`

- [ ] **Step 3: Write minimal implementation**

```python
def _add_forward_eps_overlay(fig, last_date, last_eps, forward_date,
                             forward_eps_index, color="#ff8800"):
    """Dashed connector + diamond marker from last historical EPS to forward."""
    xs = [pd.Timestamp(last_date).to_pydatetime(),
          pd.Timestamp(forward_date).to_pydatetime()]
    ys = [float(last_eps), float(forward_eps_index)]
    fig.line(xs, ys, line_width=2, line_dash="dashed", color=color)
    # Bokeh 3.x: use scatter(marker=...) rather than the removed .diamond().
    fig.scatter([xs[1]], [ys[1]], marker="diamond", size=12, color=color,
                line_color=color, fill_alpha=0.85)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Test/test_index_growth_charts.py -k overlay -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add index_growth_charts.py Test/test_index_growth_charts.py
git commit -m "feat(forward-eps): bokeh forward-EPS overlay glyph helper"
```

---

## Task 11: Integrate overlay + callout into `render_index_growth_charts`

**Files:**
- Modify: `index_growth_charts.py` (`render_index_growth_charts`)
- Test: `Test/test_index_growth_charts.py`

- [ ] **Step 1: Read the forward row alongside the other series**

In `render_index_growth_charts`, inside the `with sqlite3.connect(DB_PATH) as conn:` block (where `ig_s/pe_s/eps_s` are read, ~line 774-778), add:

```python
        fwd_row = _latest_forward_eps(conn, tk)
        fwd_impl_g = _latest_forward_implied_growth(conn, tk)
```

- [ ] **Step 2: Extend the axis max so the forward point is visible**

Find where `max_date` is computed (~line 835):

```python
        max_date = max(s.index.max() for s in non_empty_series)
```

Replace with:

```python
        max_date = max(s.index.max() for s in non_empty_series)
        if fwd_row and fwd_row.get("horizon_date"):
            try:
                fwd_dt = pd.Timestamp(fwd_row["horizon_date"])
                if fwd_dt > max_date:
                    max_date = fwd_dt
            except Exception:
                pass
```

- [ ] **Step 3: Add the forward sentence to the EPS callout**

Find the EPS block setup (~line 965-973):

```python
    eps_callout = None
    if eps_has_nonpositive:
        eps_callout = "EPS includes non-positive values, so the chart uses a linear scale."
```

Replace with:

```python
    eps_callout = None
    if eps_has_nonpositive:
        eps_callout = "EPS includes non-positive values, so the chart uses a linear scale."
    latest_hist_eps = float(eps_s.iloc[-1]) if not eps_s.empty else None
    if fwd_row:
        fwd_sentence = _forward_eps_callout(
            fwd_row["forward_eps_index"], latest_hist_eps,
            fwd_row.get("horizon_date"), fwd_row.get("source"),
            forward_implied_growth=fwd_impl_g)
        eps_callout = f"{eps_callout} {fwd_sentence}" if eps_callout else fwd_sentence
```

- [ ] **Step 4: Draw the overlay after the EPS block is built**

The EPS block is the last one appended (`blocks.append(_build_chart_block(eps_s, ...))`, ~line 974-989). Immediately **after** that `blocks.append(...)` call, add:

```python
    eps_block = blocks[-1]
    if fwd_row and eps_block.fig is not None and not eps_s.empty:
        try:
            _add_forward_eps_overlay(
                eps_block.fig,
                last_date=eps_s.index[-1],
                last_eps=float(eps_s.iloc[-1]),
                forward_date=pd.Timestamp(fwd_row["horizon_date"]),
                forward_eps_index=fwd_row["forward_eps_index"],
            )
        except Exception as exc:
            print(f"[WARN] forward EPS overlay failed for {tk}: {exc}")
```

- [ ] **Step 5: Write the integration test**

```python
def test_render_applies_forward_overlay_when_data_present():
    dates = pd.date_range("2024-03-31", periods=3, freq="Q-DEC")
    eps = pd.Series([220.0, 225.0, 230.0], index=dates)
    fwd = {"forward_eps_index": 250.0, "forward_pe": 22.0,
           "horizon_date": "2027-06-28", "source": "stockanalysis",
           "date_recorded": "2026-06-28"}

    captured = {}
    with (
        patch.object(igc, "sqlite3") as mock_sqlite,
        patch.object(igc, "_series_growth", return_value=pd.Series(dtype=float)),
        patch.object(igc, "_series_pe", return_value=pd.Series(dtype=float)),
        patch.object(igc, "_series_pe_monthly_derived", return_value=pd.Series(dtype=float)),
        patch.object(igc, "_series_eps", return_value=eps),
        patch.object(igc, "_latest_forward_eps", return_value=fwd),
        patch.object(igc, "_add_forward_eps_overlay") as mock_overlay,
        patch.object(igc, "_build_chart_block") as mock_block,
        patch.object(igc, "_write_chart_assets"),
        patch.object(igc, "_extend_eps_csv"),
    ):
        mock_sqlite.connect.return_value.__enter__.return_value = object()

        def capture(series, title, ylabel, percent_axis, x_range,
                    callout_text=None, **kwargs):
            if "EPS" in title:
                captured["eps_callout"] = callout_text
            return igc.ChartBlock(layout=igc.Div(text="x"),
                                   fig=object(), source=None, log_axis=False,
                                   window_div=None, percent_axis=False,
                                   window_mode="ratio")
        mock_block.side_effect = capture
        igc.render_index_growth_charts("QQQ")

    assert mock_overlay.called
    assert "8.7%" in captured["eps_callout"]   # 250/230 - 1
```

- [ ] **Step 6: Run the test**

Run: `python -m pytest Test/test_index_growth_charts.py -k "forward or overlay" -v`
Expected: PASS

- [ ] **Step 7: Run the full chart-test file to check no regressions**

Run: `python -m pytest Test/test_index_growth_charts.py -v`
Expected: PASS (existing tests still green — overlay/callout are additive and guarded by `if fwd_row`)

- [ ] **Step 8: Commit**

```bash
git add index_growth_charts.py Test/test_index_growth_charts.py
git commit -m "feat(forward-eps): draw forward point + callout on index EPS chart"
```

---

## Task 12: Full suite + manual end-to-end check

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `python -m pytest Test/ -v`
Expected: PASS (no regressions). If pre-existing unrelated failures exist, confirm they fail identically on `main` before continuing.

- [ ] **Step 2: Real snapshot smoke test**

Against a scratch copy of the DB (do NOT mutate the live `Stock Data.db`):

```bash
python -c "import shutil, sqlite3, index_forward_eps as ife; shutil.copy('Stock Data.db','_scratch.db'); ife.DB_PATH='_scratch.db'; conn=sqlite3.connect('_scratch.db'); print('rows written:', ife.snapshot_forward_eps(conn)); print(list(conn.execute('SELECT ticker, forward_pe, forward_eps_index, source FROM Index_Forward_EPS_History')))"
```

Expected: 1-2 rows with sane forward P/E (SPY ~18-26, QQQ ~22-34), `forward_eps_index` within ~0.5-2x the latest historical index EPS, source `stockanalysis` or `yfinance`. Delete `_scratch.db` afterward.

- [ ] **Step 3: Report results to the user**

Summarize: rows written, the forward P/E + implied growth % per index, and which source was used (so we know if the stockanalysis scrape is working or we're on the yfinance fallback). Do NOT deploy — deployment (committing artifacts / dispatching `site-refresh.yml`) is a separate, user-approved step.

---

## Notes for the implementer

- **Deploy is out of scope for this plan.** The growth-page chart artifacts rebuild on the weekly `site-refresh.yml` (or a manual `workflow_dispatch`). Pushing/dispatching is done by Nick (Windows git has the only push auth). Surface the branch and let him decide.
- **DB lives on the `data` branch.** `ensure_forward_eps_table` uses `CREATE TABLE IF NOT EXISTS`, so it self-creates on hydrate. No migration needed.
- **Never write placeholder/zero rows** — every guard path returns `None` and writes nothing, per the project's data-quality rules.
