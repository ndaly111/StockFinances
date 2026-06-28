# SPY/QQQ Forward EPS — Bottom-Up Build Plan (rev. 3)

> **For agentic workers:** Use superpowers:subagent-driven-development to execute task-by-task. Steps use `- [ ]`.

**Goal:** Compute SPY & QQQ forward EPS growth bottom-up from our own constituent data (membership+weights from slickcharts), snapshot daily, display on the index growth pages, validation-gated, future-proof to index membership changes.

**Architecture:** New `index_holdings.py` (slickcharts → `index_constituents` + weight-prioritized scrape auto-extend), `forward_eps_bottom_up.py` (aggregate constituent forward EPS × shares), `forward_eps_validate.py` (benchmark gate). `index_forward_eps.py`'s snapshot is rewired to use these; the dead stockanalysis/yfinance fetch functions are removed. Chart overlay/callout reused, callout updated to show current-FY/next-FY growth + coverage.

**Tech:** Python, SQLite, requests + pandas.read_html, pytest. Run tests from repo root: `python -m pytest <path> -v`. Git: `git -C "C:/Users/ndaly/projects/sf-fix"`.

**Spec:** `docs/superpowers/specs/2026-06-28-spy-qqq-forward-eps-design.md` (rev 3).

**Verified data facts:**
- slickcharts `https://www.slickcharts.com/nasdaq100` & `/sp500` → HTML table, columns `Symbol`,`Weight` (e.g. "12.33%"); needs browser UA + Referer; `pandas.read_html(io.StringIO(text))`. QQQ 101 rows (GOOG+GOOGL both), SPY ~503.
- `Forward_EPS_FY_History(date_recorded, ticker, period_end, period_label, forward_eps, eps_analysts, fiscal_year, ...)` — `period_label` ∈ {'This FY','Next FY'}.
- `TTM_Data(Symbol, TTM_EPS, Shares_Outstanding, ...)`.
- Existing `index_forward_eps.py`: `ensure_forward_eps_table`, `snapshot_forward_eps`, `_passes_sanity`, `ForwardEPS`, `INDEX_EPS_DIVISOR={'SPY':10,'QQQ':4}`, `_latest_hist_eps`. Existing `index_growth_charts.py`: `_latest_forward_eps`, `_forward_eps_callout`, `_add_forward_eps_overlay`.

---

## Task 1: Schema — extend snapshot table + add `index_constituents`

**Files:** Modify `index_forward_eps.py`; Test `Test/test_index_forward_eps.py`.

- [ ] **Step 1 — failing test** (append):

```python
def test_ensure_table_has_bottomup_columns(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        ife.ensure_forward_eps_table(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(Index_Forward_EPS_History)")}
    for c in ("coverage_weight", "growth_this_fy", "growth_next_fy", "method", "displayable"):
        assert c in cols


def test_ensure_constituents_table(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        ife.ensure_constituents_table(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(index_constituents)")}
    assert {"date_recorded", "index_name", "ticker", "weight"} <= cols
```

- [ ] **Step 2 — run, expect FAIL** (`-k "bottomup_columns or constituents_table"`).

- [ ] **Step 3 — implement.** In `ensure_forward_eps_table`, after the existing CREATE TABLE, add idempotent ALTERs and the new table helper:

```python
def _add_columns(conn, table, cols):
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in cols:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def ensure_constituents_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS index_constituents (
             date_recorded TEXT NOT NULL, index_name TEXT NOT NULL,
             ticker TEXT NOT NULL, weight REAL,
             PRIMARY KEY (date_recorded, index_name, ticker))""")
    conn.commit()
```

In `ensure_forward_eps_table`, before `conn.commit()`, add:

```python
    _add_columns(conn, TABLE, [
        ("coverage_weight", "REAL"), ("growth_this_fy", "REAL"),
        ("growth_next_fy", "REAL"), ("method", "TEXT"), ("displayable", "INTEGER")])
```

- [ ] **Step 4 — run, expect PASS** (both new tests + the existing schema tests still pass).
- [ ] **Step 5 — commit** `feat(forward-eps): schema for bottom-up (coverage/growth cols + index_constituents)`.

---

## Task 2: `index_holdings.py` — slickcharts parser (fixture-tested)

**Files:** Create `index_holdings.py`, `Test/fixtures/slickcharts_nasdaq100.html`; Test `Test/test_index_holdings.py`.

- [ ] **Step 1 — capture fixture.** Save the live page (network confirmed working):

```bash
python -c "import requests,os; os.makedirs('Test/fixtures',exist_ok=True); h={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36','Referer':'https://www.google.com/'}; open('Test/fixtures/slickcharts_nasdaq100.html','w',encoding='utf-8').write(requests.get('https://www.slickcharts.com/nasdaq100',headers=h,timeout=20).text)"
```
Confirm the saved file contains a table with Symbol/Weight. If the fetch is blocked, STOP and report (the whole approach depends on this source).

- [ ] **Step 2 — failing test:**

```python
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import index_holdings as ih

def test_parse_slickcharts_holdings():
    html = (ROOT/"Test"/"fixtures"/"slickcharts_nasdaq100.html").read_text(encoding="utf-8")
    rows = ih.parse_slickcharts(html)
    assert len(rows) >= 99
    d = dict(rows)
    assert "NVDA" in d and "AAPL" in d
    assert 0 < d["NVDA"] < 100          # weight is a percent
    assert 95 < sum(w for _, w in rows) < 105   # weights ~sum to 100
```

- [ ] **Step 3 — run, expect FAIL.**

- [ ] **Step 4 — implement** `index_holdings.py`:

```python
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
```

- [ ] **Step 5 — run, expect PASS.** If a real-fetch test is desired keep it `@pytest.mark.skip` to avoid network in CI.
- [ ] **Step 6 — commit** `feat(holdings): slickcharts index membership+weight parser`.

---

## Task 3: persist holdings to `index_constituents`

**Files:** Modify `index_holdings.py`; Test `Test/test_index_holdings.py`.

- [ ] **Step 1 — failing test:**

```python
import sqlite3
def test_persist_holdings(tmp_path):
    db = tmp_path/"t.db"
    with sqlite3.connect(db) as conn:
        import index_forward_eps as ife; ife.ensure_constituents_table(conn)
        ih.persist_holdings(conn, "QQQ", [("NVDA",12.3),("AAPL",11.0)], today="2026-06-28")
        ih.persist_holdings(conn, "QQQ", [("NVDA",12.4),("AAPL",11.1)], today="2026-06-28")  # idempotent
        rows = list(conn.execute("SELECT ticker, weight FROM index_constituents WHERE index_name='QQQ' ORDER BY ticker"))
    assert rows == [("AAPL",11.1),("NVDA",12.4)]
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** (append to `index_holdings.py`):

```python
def persist_holdings(conn, index_name, rows, today=None):
    import index_forward_eps as ife
    ife.ensure_constituents_table(conn)
    today = today or date.today().isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO index_constituents (date_recorded, index_name, ticker, weight) VALUES (?,?,?,?)",
        [(today, index_name.upper(), tk, w) for tk, w in rows])
    conn.commit()
```

- [ ] **Step 4 — run, expect PASS. Step 5 — commit** `feat(holdings): persist constituents (idempotent upsert)`.

---

## Task 4: weight-prioritized scrape auto-extend

**Files:** Modify `index_holdings.py`; Test `Test/test_index_holdings.py`.

- [ ] **Step 1 — failing test:**

```python
def test_uncovered_to_reach_target():
    # holdings sorted by weight desc; covered = names we already have fwd EPS for
    holdings = [("A",40.0),("B",30.0),("C",20.0),("D",10.0)]
    covered = {"B"}                      # we have B (30%)
    # need to reach 90% weight: have 30, add A(40)->70, add C(20)->90 stop. D not needed.
    add = ih.uncovered_for_target(holdings, covered, target_pct=90.0)
    assert add == ["A", "C"]

def test_uncovered_none_when_already_covered():
    holdings = [("A",95.0),("B",5.0)]
    assert ih.uncovered_for_target(holdings, {"A","B"}, 90.0) == []
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement:**

```python
def uncovered_for_target(holdings, covered, target_pct=90.0):
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
```

- [ ] **Step 4 — run, expect PASS. Step 5 — commit** `feat(holdings): weight-prioritized uncovered-ticker selection`.

---

## Task 5: bottom-up constituent loaders

**Files:** Create `forward_eps_bottom_up.py`; Test `Test/test_forward_eps_bottom_up.py`.

- [ ] **Step 1 — failing test** (uses a real temp sqlite with the two source tables):

```python
import pathlib, sqlite3, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import forward_eps_bottom_up as bu

def _seed(conn):
    conn.execute("CREATE TABLE TTM_Data (Symbol TEXT, TTM_EPS REAL, Shares_Outstanding REAL)")
    conn.executemany("INSERT INTO TTM_Data VALUES (?,?,?)",
        [("AAA",10.0,100.0),("BBB",5.0,200.0)])
    conn.execute("""CREATE TABLE Forward_EPS_FY_History (date_recorded TEXT, ticker TEXT,
                    period_label TEXT, forward_eps REAL)""")
    conn.executemany("INSERT INTO Forward_EPS_FY_History VALUES (?,?,?,?)",
        [("2026-06-28","AAA","This FY",12.0),("2026-06-28","AAA","Next FY",14.0),
         ("2026-06-28","BBB","This FY",5.5),("2026-06-28","BBB","Next FY",6.0)])
    conn.commit()

def test_load_constituent_financials():
    conn = sqlite3.connect(":memory:"); _seed(conn)
    fin = bu.load_constituent_financials(conn, ["AAA","BBB","CCC"])
    assert fin["AAA"] == {"ttm_eps":10.0,"shares":100.0,"this_fy":12.0,"next_fy":14.0}
    assert "CCC" not in fin                     # no data -> excluded
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement:**

```python
"""Bottom-up index forward EPS from constituent forward EPS x shares."""
from __future__ import annotations
import logging, sqlite3
from typing import Dict, List
logger = logging.getLogger(__name__)


def load_constituent_financials(conn, tickers) -> Dict[str, dict]:
    out = {}
    for tk in tickers:
        row = conn.execute("SELECT TTM_EPS, Shares_Outstanding FROM TTM_Data WHERE Symbol=?",
                           (tk,)).fetchone()
        if not row or row[0] is None or row[1] is None:
            continue
        fy = {}
        for label in ("This FY", "Next FY"):
            r = conn.execute("""SELECT forward_eps FROM Forward_EPS_FY_History
                                WHERE ticker=? AND period_label=? AND forward_eps IS NOT NULL
                                ORDER BY date_recorded DESC LIMIT 1""", (tk, label)).fetchone()
            fy[label] = r[0] if r else None
        if fy["This FY"] is None:
            continue
        out[tk.upper()] = {"ttm_eps": float(row[0]), "shares": float(row[1]),
                           "this_fy": float(fy["This FY"]),
                           "next_fy": float(fy["Next FY"]) if fy["Next FY"] is not None else None}
    return out
```

- [ ] **Step 4 — run, expect PASS. Step 5 — commit** `feat(bottom-up): constituent financials loader`.

---

## Task 6: bottom-up aggregation + coverage

**Files:** Modify `forward_eps_bottom_up.py`; Test same file.

- [ ] **Step 1 — failing test:**

```python
def test_aggregate_growth_and_coverage():
    # holdings: AAA 60%, BBB 30%, CCC 10%. We have AAA,BBB (cover 90%); CCC missing.
    holdings = [("AAA",60.0),("BBB",30.0),("CCC",10.0)]
    fin = {"AAA":{"ttm_eps":10.0,"shares":100.0,"this_fy":12.0,"next_fy":14.0},
           "BBB":{"ttm_eps":5.0,"shares":200.0,"this_fy":5.5,"next_fy":6.0}}
    res = bu.aggregate(holdings, fin)
    # trailing$ = 10*100 + 5*200 = 2000 ; thisFY$ = 12*100 + 5.5*200 = 2300
    # nextFY$ = 14*100 + 6*200 = 2600
    assert round(res["growth_this_fy"], 4) == round(2300/2000 - 1, 4)   # +15%
    assert round(res["growth_next_fy"], 4) == round(2600/2300 - 1, 4)
    assert round(res["coverage_weight"], 4) == 0.90
    assert res["fwd_earnings_this_fy"] == 2300.0
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement:**

```python
def aggregate(holdings, fin) -> dict:
    covered = {tk.upper() for tk, _ in holdings if tk.upper() in fin}
    total_w = sum(w for _, w in holdings) or 1.0
    cov_w = sum(w for tk, w in holdings if tk.upper() in covered) / total_w
    ttm = thisfy = nextfy = 0.0
    nextfy_ok = True
    for tk in covered:
        f = fin[tk]
        ttm += f["ttm_eps"] * f["shares"]
        thisfy += f["this_fy"] * f["shares"]
        if f["next_fy"] is None:
            nextfy_ok = False
        else:
            nextfy += f["next_fy"] * f["shares"]
    if ttm <= 0 or not covered:
        return {"growth_this_fy": None, "growth_next_fy": None,
                "coverage_weight": cov_w, "fwd_earnings_this_fy": thisfy,
                "trailing_earnings": ttm}
    return {"growth_this_fy": thisfy / ttm - 1.0,
            "growth_next_fy": (nextfy / thisfy - 1.0) if (nextfy_ok and thisfy > 0) else None,
            "coverage_weight": cov_w, "fwd_earnings_this_fy": thisfy,
            "trailing_earnings": ttm}
```

- [ ] **Step 4 — run, expect PASS. Step 5 — commit** `feat(bottom-up): aggregate growth + coverage`.

---

## Task 7: `forward_eps_validate.py` — benchmark gate

**Files:** Create `forward_eps_validate.py`; Test `Test/test_forward_eps_validate.py`.

- [ ] **Step 1 — failing test:**

```python
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import forward_eps_validate as v

def test_displayable_when_in_range():
    # QQQ benchmark this-fy growth ~0.19; coverage 0.92
    assert v.is_displayable("QQQ", growth_this_fy=0.20, coverage_weight=0.92) is True

def test_withheld_low_coverage():
    assert v.is_displayable("QQQ", growth_this_fy=0.19, coverage_weight=0.70) is False

def test_withheld_out_of_tolerance():
    assert v.is_displayable("QQQ", growth_this_fy=0.45, coverage_weight=0.95) is False

def test_none_growth_withheld():
    assert v.is_displayable("SPY", growth_this_fy=None, coverage_weight=0.95) is False
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement:**

```python
"""Validation gate: only display a bottom-up number if coverage is high enough and it
lands near published consensus. Benchmarks are coarse anchors, updated occasionally."""
_MIN_COVERAGE = 0.85
_TOL = 0.08   # +/- 8 percentage points on the growth rate
# Published consensus anchors (current-FY YoY EPS growth), refresh periodically.
_BENCH = {"SPY": 0.21, "QQQ": 0.19}


def is_displayable(index_name, growth_this_fy, coverage_weight) -> bool:
    if growth_this_fy is None or coverage_weight is None:
        return False
    if coverage_weight < _MIN_COVERAGE:
        return False
    bench = _BENCH.get(index_name.upper())
    if bench is not None and abs(growth_this_fy - bench) > _TOL:
        return False
    return True
```

- [ ] **Step 4 — run, expect PASS. Step 5 — commit** `feat(validate): coverage + consensus gate`.

---

## Task 8: scrape-universe union in `Forward_data`

**Files:** Modify the scrape-universe source. First `grep -n "tickers.csv\|read_tickers\|def.*ticker" Forward_data.py main_remote.py scripts/*.py` to find where the forward-data scrape list is built. Add a helper that unions the curated `tickers.csv` with the auto-managed `index_constituents` set, WITHOUT modifying `tickers.csv`.

- [ ] **Step 1 — failing test** `Test/test_scrape_universe.py`:

```python
import pathlib, sqlite3, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import index_holdings as ih

def test_scrape_universe_unions_constituents(tmp_path):
    db = tmp_path/"t.db"
    with sqlite3.connect(db) as conn:
        import index_forward_eps as ife; ife.ensure_constituents_table(conn)
        conn.executemany("INSERT INTO index_constituents VALUES (?,?,?,?)",
            [("2026-06-28","QQQ","NEWCO",1.0),("2026-06-28","SPY","AAPL",6.0)])
        conn.commit()
        extra = ih.managed_scrape_tickers(conn)
    assert "NEWCO" in extra and "AAPL" in extra
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `managed_scrape_tickers` in `index_holdings.py` (latest date_recorded distinct tickers across indices):

```python
def managed_scrape_tickers(conn) -> set:
    try:
        rows = conn.execute(
            """SELECT DISTINCT ticker FROM index_constituents
               WHERE date_recorded = (SELECT MAX(date_recorded) FROM index_constituents)""").fetchall()
    except sqlite3.Error:
        return set()
    return {r[0].upper() for r in rows}
```

Then, at the scrape-universe build site found above, union `managed_scrape_tickers(conn)` into the ticker list (dedupe). Keep the change minimal and additive; do not write to `tickers.csv`. Add a one-line comment referencing the auto-extend design.

- [ ] **Step 4 — run, expect PASS; smoke `python -c "import Forward_data"`. Step 5 — commit** `feat(forward-eps): union auto-managed constituents into scrape universe`.

---

## Task 9: rewire `snapshot_forward_eps` to bottom-up + remove dead fetchers

**Files:** Modify `index_forward_eps.py`; Modify `Test/test_index_forward_eps.py`.

- [ ] **Step 1 — remove** the dead `_fetch_stockanalysis_pe`, `_parse_forward_pe`, `_forward_from_yf`, `_yf_info`, `fetch_forward_eps`, `_HEADERS`, `_FWD_PE_RE` and their tests (`-k "parse_stockanalysis or forward_from_yf or fetch_"`), plus the `Test/fixtures/stockanalysis_spy.html` fixture. Keep `ForwardEPS`, `_passes_sanity`, `_default_horizon`, `_scale_index`, `_latest_hist_eps`, `INDEX_EPS_DIVISOR`, table helpers.

- [ ] **Step 2 — failing test** for the new snapshot wiring:

```python
def test_snapshot_bottom_up(tmp_path, monkeypatch):
    db = tmp_path/"t.db"
    import index_forward_eps as ife, index_holdings as ih, forward_eps_bottom_up as bu, forward_eps_validate as v
    holdings = [("AAA",60.0),("BBB",30.0),("CCC",10.0)]
    fin = {"AAA":{"ttm_eps":10.0,"shares":100.0,"this_fy":12.0,"next_fy":14.0},
           "BBB":{"ttm_eps":5.0,"shares":200.0,"this_fy":5.5,"next_fy":6.0}}
    monkeypatch.setattr(ih, "fetch_holdings", lambda idx: holdings)
    monkeypatch.setattr(ih, "uncovered_for_target", lambda h,c,target_pct=90.0: [])
    monkeypatch.setattr(bu, "load_constituent_financials", lambda conn, tks: fin)
    monkeypatch.setattr(v, "is_displayable", lambda idx,growth_this_fy,coverage_weight: True)
    with sqlite3.connect(db) as conn:
        n = ife.snapshot_forward_eps(conn)
        rows = list(conn.execute("SELECT ticker, growth_this_fy, coverage_weight, displayable, method FROM Index_Forward_EPS_History"))
    assert n == 2                         # SPY + QQQ both written
    g = {r[0]: r for r in rows}
    assert round(g["QQQ"][1],4) == round(2300/2000-1,4)
    assert g["QQQ"][3] == 1 and g["QQQ"][4] == "bottom_up"
```

- [ ] **Step 3 — run, expect FAIL.**
- [ ] **Step 4 — implement** the new `snapshot_forward_eps`:

```python
def snapshot_forward_eps(conn, today=None):
    import index_holdings as ih, forward_eps_bottom_up as bu, forward_eps_validate as v
    ensure_forward_eps_table(conn)
    ensure_constituents_table(conn)
    today = today or date.today().isoformat()
    written = 0
    for idx in IDXES:                      # ["SPY","QQQ"]
        try:
            holdings = ih.fetch_holdings(idx)
        except Exception as e:
            logger.warning("[%s] holdings fetch failed: %s", idx, e); continue
        ih.persist_holdings(conn, idx, holdings, today=today)
        fin = bu.load_constituent_financials(conn, [tk for tk, _ in holdings])
        # weight-prioritized auto-extend (next run collects them)
        add = ih.uncovered_for_target(holdings, set(fin), target_pct=90.0)
        if add:
            logger.info("[%s] auto-extend scrape: %d names (%s...)", idx, len(add), add[:5])
        res = bu.aggregate(holdings, fin)
        if res["growth_this_fy"] is None:
            logger.warning("[%s] no usable bottom-up growth; skip", idx); continue
        divisor = INDEX_EPS_DIVISOR.get(idx, 1.0)
        # index-level forward EPS proxy for the chart point: scale this-FY $earnings into
        # the same per-share level as _series_eps via the latest historical EPS ratio.
        latest_hist = _latest_hist_eps(conn, idx)
        fwd_eps_index = (latest_hist * (1 + res["growth_this_fy"])) if latest_hist else None
        displayable = bool(v.is_displayable(idx, res["growth_this_fy"], res["coverage_weight"]))
        if fwd_eps_index is not None and not _passes_sanity(20.0, fwd_eps_index, latest_hist):
            displayable = False
        conn.execute(
            f"""INSERT OR REPLACE INTO {TABLE}
                (date_recorded, ticker, forward_eps_etf, forward_eps_index, forward_pe,
                 horizon_date, source, coverage_weight, growth_this_fy, growth_next_fy,
                 method, displayable)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (today, idx, None, fwd_eps_index, None, _default_horizon(), "bottom_up",
             res["coverage_weight"], res["growth_this_fy"], res["growth_next_fy"],
             "bottom_up", int(displayable)))
        written += 1
    conn.commit()
    logger.info("[forward-eps] snapshot wrote %d row(s) for %s", written, today)
    return written
```

Note: `forward_eps_index` is derived from the latest historical index EPS × (1+thisFY growth) so the chart's forward point sits on the same scale as `_series_eps` without needing the index divisor/share-base. Document this in a comment.

- [ ] **Step 5 — run** `-k "snapshot_bottom_up or sanity or ensure_table or divisor or latest"` expect PASS; then full file.
- [ ] **Step 6 — commit** `feat(forward-eps): rewire snapshot to bottom-up; remove dead ETF fetchers`.

---

## Task 10: chart callout shows current-FY/next-FY growth + coverage; gate on displayable

**Files:** Modify `index_growth_charts.py`; Test `Test/test_index_growth_charts.py`.

- [ ] **Step 1 — update `_latest_forward_eps`** to also return the new columns (`growth_this_fy`, `growth_next_fy`, `coverage_weight`, `displayable`) and to return None when `displayable=0`. Failing test:

```python
def test_latest_forward_eps_respects_displayable(tmp_path):
    db = tmp_path/"t.db"
    with sqlite3.connect(db) as conn:
        import index_forward_eps as ife; ife.ensure_forward_eps_table(conn)
        conn.execute(f"""INSERT INTO Index_Forward_EPS_History
          (date_recorded,ticker,forward_eps_index,horizon_date,source,coverage_weight,
           growth_this_fy,growth_next_fy,method,displayable)
          VALUES ('2026-06-28','QQQ',250.0,'2027-06-28','bottom_up',0.93,0.19,0.14,'bottom_up',0)""")
        conn.commit()
        assert igc._latest_forward_eps(conn,"QQQ") is None     # displayable=0 -> hidden
        conn.execute("UPDATE Index_Forward_EPS_History SET displayable=1")
        conn.commit()
        row = igc._latest_forward_eps(conn,"QQQ")
    assert row["growth_this_fy"]==0.19 and row["coverage_weight"]==0.93
```

- [ ] **Step 2 — update `_forward_eps_callout`** to a bottom-up sentence. New failing test:

```python
def test_forward_eps_callout_bottomup():
    txt = igc._forward_eps_callout_bottomup(growth_this_fy=0.19, growth_next_fy=0.14,
            coverage_weight=0.93, horizon_date="2027-06-28")
    assert "+19.0%" in txt and "+14.0%" in txt and "93%" in txt
```

- [ ] **Step 3 — run, expect FAIL. Step 4 — implement** both (read new cols in `_latest_forward_eps`, return None if `displayable` falsy; add `_forward_eps_callout_bottomup`):

```python
def _forward_eps_callout_bottomup(growth_this_fy, growth_next_fy, coverage_weight, horizon_date=None):
    parts = [f"Forward earnings growth (bottom-up): <b>{growth_this_fy:+.1%}</b> this fiscal year"]
    if growth_next_fy is not None:
        parts.append(f", <b>{growth_next_fy:+.1%}</b> next")
    parts.append(f". Based on {coverage_weight:.0%} of index weight.")
    return "".join(parts)
```

- [ ] **Step 5 — update the integration** in `render_index_growth_charts`: where it builds the EPS callout/overlay (the `if fwd_row:` blocks from the earlier build), use `_forward_eps_callout_bottomup(fwd_row["growth_this_fy"], fwd_row["growth_next_fy"], fwd_row["coverage_weight"], fwd_row.get("horizon_date"))`. The overlay still uses `fwd_row["forward_eps_index"]`. Update the existing `test_render_applies_forward_overlay_when_data_present` fwd dict to include the new keys + `displayable`.
- [ ] **Step 6 — run** full `Test/test_index_growth_charts.py` expect PASS. **Step 7 — commit** `feat(forward-eps): bottom-up callout (this/next FY growth + coverage) on chart`.

---

## Task 11: full suite + real end-to-end

**Files:** none (verify only).

- [ ] **Step 1 —** `python -m pytest Test/ -v`; confirm all forward-EPS/holdings/bottom-up/validate tests pass; classify any other failure as pre-existing (untouched files).
- [ ] **Step 2 — real e2e** against a scratch DB copy (do NOT mutate `Stock Data.db`):

```bash
python -c "import shutil, sqlite3, index_forward_eps as ife; shutil.copy('Stock Data.db','_e2e.db'); conn=sqlite3.connect('_e2e.db'); print('rows:', ife.snapshot_forward_eps(conn)); print(list(conn.execute('SELECT ticker, growth_this_fy, growth_next_fy, coverage_weight, displayable FROM Index_Forward_EPS_History'))); conn.close()"
```
Report per index: growth_this_fy, growth_next_fy, coverage_weight, displayable. **Sanity-check against benchmarks**: SPY this-FY growth should land near ~+21% (and coverage may be low on first run before auto-extend fills in — note it); QQQ near ~+19%. If coverage is low (auto-extend hasn't populated forward EPS for newly added names yet), that's expected on first run — note that a few `Forward_data` cycles are needed to reach 90%. Delete `_e2e.db` (+ wal/shm).
- [ ] **Step 3 —** `git status --porcelain` clean; report. Do NOT deploy (Nick pushes).

---

## Notes
- First-run coverage will be below target until `Forward_data` scrapes the newly auto-added constituents over a day or two; the validation gate correctly withholds display until coverage clears 85%. This is expected, not a bug — report it as such.
- Deploy stays with Nick. Index growth pages render on the weekly `site-refresh.yml`/dispatch.
- Benchmarks in `forward_eps_validate.py` are coarse anchors; note they need occasional refresh.
