# SPY / QQQ Forward EPS — Consensus Forward Earnings on the Index Growth Pages

**Date:** 2026-06-28
**Status:** Design — awaiting review
**Repo:** StockFinances (`C:\Users\ndaly\projects\sf-fix`, tracks `origin/main`)

---

## 1. Goal

Let Nick see the **future estimated EPS growth of SPY and QQQ** — i.e. how fast
index earnings are expected to grow — by adding a **forward-EPS trajectory** to
the dedicated index growth pages (`spy_growth.html` / `qqq_growth.html`),
sourced from a reputable analyst-consensus value and snapshotted daily so a
revision history accrues over time. Conceptually this is the index-level analog
of the existing per-ticker EPS forecast charts.

## 2. Non-goals (YAGNI)

- **No homepage overview-table changes.** The SPY vs QQQ table stays exactly as
  it is (decided 2026-06-28: "chart-first, leave table as-is").
- **No multi-year FY1/FY2/FY3 curve in v1.** v1 shows a single near-term (NTM,
  ~12-month) consensus point. A true multi-year quarterly curve for SPY from the
  S&P Global official EPS spreadsheet is noted as a *future enrichment* only.
- **No new pages, no new workflows.** Reuse the existing index growth pages and
  the existing daily index-data run.
- **No per-side / no overfit filtering** — this is a display feature, not a
  trading signal.

## 3. Background — what already exists

The forward plumbing is half-built already:

- `index_growth_table.py::_fetch_pe()` already pulls **forward P/E** for SPY and
  QQQ every day (yfinance `forwardPE`, or `price / forwardEps`).
- `_log_today()` already writes **forward implied growth** (`Index_Growth_History`,
  `Growth_Type='Forward'`) and **forward P/E** (`Index_PE_History`,
  `PE_Type='Forward'`) daily.
- `index_growth_charts.py::_series_eps()` builds an **index-level** EPS history
  series (TTM_REPORTED monthly → TTM_DAILY → IMPLIED_FROM_PE scaled by
  `_INDEX_EPS_DIVISOR = {SPY: 10, QQQ: 4}`), rendered as the EPS panel of the
  Bokeh "valuation bundle" loaded by the growth-page shell.

Key consequence: the **expected EPS growth rate** is derivable for free, because
price cancels:

> forward EPS growth = (TTM P/E ÷ Forward P/E) − 1

So the *numbers* need no new source; only the **trajectory chart** (which needs
an actual forward-EPS dollar value plotted over time) benefits from a named
consensus feed.

## 4. Data source (decided: Approach B)

**Primary:** stockanalysis.com ETF pages — `https://stockanalysis.com/etf/spy/`
and `/etf/qqq/` — which publish a holdings-weighted **forward P/E** and
**forward EPS** (ETF-level). One reputable, named source covering *both* indices
symmetrically, scrapeable with the same `requests.Session` + `pandas.read_html` /
BeautifulSoup pattern already used in `Forward_data.py`.

**Fallback:** yfinance ETF `forwardPE` / `forwardEps` (already fetched in
`index_growth_table.py`). If the scrape fails or fails sanity checks, derive
forward EPS = `price / forwardPE`.

**Scaling:** stockanalysis.com / yfinance give **ETF-level** EPS. Multiply by
`_INDEX_EPS_DIVISOR[tk]` to get **index-level** EPS so the forward point sits on
the same scale as `_series_eps()`'s historical line.

> Future enrichment (not in scope): S&P Global `sp-500-eps-est.xlsx` gives an
> authoritative multi-year quarterly forward-EPS curve for SPY. QQQ has no clean
> equivalent, so adding it now would create asymmetry; defer.

## 5. Architecture

### 5.1 New module — `index_forward_eps.py`

Responsibilities:

1. `fetch_forward_eps(tk) -> ForwardEPS | None`
   - Scrape stockanalysis.com ETF page for `tk` → `(forward_pe, forward_eps_etf,
     as_of_date?)`.
   - On failure/garbage → yfinance fallback (`forwardPE`, `forwardEps`, price).
   - Compute `forward_eps_index = forward_eps_etf * divisor`.
   - Return `None` if both sources fail or sanity checks reject.
2. `snapshot_forward_eps(conn)` — fetch both indices, upsert today's row(s).
3. `latest_forward_eps(conn, tk)` — read helper for the chart layer.

Reuse `Forward_data.py` conventions: shared `HEADERS`, single `requests.Session`,
layout-drift tolerance, `PRAGMA busy_timeout`.

### 5.2 New table — `Index_Forward_EPS_History`

Mirrors `Forward_EPS_History`:

```sql
CREATE TABLE IF NOT EXISTS Index_Forward_EPS_History (
    date_recorded     TEXT NOT NULL,   -- snapshot date (YYYY-MM-DD)
    ticker            TEXT NOT NULL,   -- 'SPY' | 'QQQ'
    forward_eps_etf   REAL,            -- ETF-level consensus forward EPS
    forward_eps_index REAL,            -- scaled to index level (×divisor)
    forward_pe        REAL,
    horizon_date      TEXT,            -- estimate as-of/target date if known (nullable)
    source            TEXT,            -- 'stockanalysis' | 'yfinance'
    PRIMARY KEY (date_recorded, ticker)
);
```

`INSERT OR REPLACE` so same-day re-runs overwrite. `CREATE TABLE IF NOT EXISTS`
so it is compatible with the `data`-branch hydrate model (DB lives on `data`).

### 5.3 Daily wiring

Call `snapshot_forward_eps()` once per daily run, **right after**
`index_growth(treasury)` in `main_remote.py` (that path already runs
`_log_today` daily and already has forward P/E in hand for the fallback). One new
network call per index; fully graceful on failure (chart simply omits the
forward point until data exists).

### 5.4 Chart integration — `index_growth_charts.py`

Extend the **EPS block** (the one fed by `_series_eps()`):

- Add a **forward overlay**: a dashed segment + distinct marker from the last
  historical index-level EPS point to the consensus `forward_eps_index`, in a
  distinct accent (e.g. `#ff8800`, `line_dash="dashed"`).
- Hover tooltip on the forward marker: forward EPS (index-level), expected growth
  vs latest TTM EPS (`forward/TTM − 1`), source, as-of date.
- Add a one-line **callout** under the EPS block (reuse the existing
  `callout_text` Div mechanism in `_build_chart_block`):
  > "Consensus forward EPS ≈ $X (index-level) → **+Y%** expected earnings growth
  > over the next ~12 months. Forward implied growth: Z%. Source: stockanalysis.com,
  > as of DATE."
  This delivers the *expected EPS growth rate* and *forward implied growth*
  numbers Nick asked for, on the page (since the table is untouched).
- As daily snapshots accrue, a later iteration can draw the
  `Index_Forward_EPS_History` series as a faint "estimate revision" line. v1
  minimum = single forward point + connecting dashed segment + callout.
- Verify whether `_series_growth()` already includes `Growth_Type='Forward'`; if
  not, surface forward implied growth in the growth block/callout too.

## 6. Edge cases & data-quality guards

(Per the project's data-quality lessons — never write 0/placeholder, never let a
silent error masquerade as data.)

- **Both sources fail** → skip the overlay, render historical EPS only, log a
  warning, write **no** row.
- **Sanity bounds:** reject if `forward_pe <= 0`, or implied growth outside a
  sane band (e.g. −50%…+80%), or `forward_eps_index` not within 0.5×–2× the
  latest historical index EPS (catches scrape garbage and scaling errors — the
  MU-style P/E-blowup failure mode). Rejected → log + skip, no row.
- **Stale estimate:** store `horizon_date`/as-of; if the latest snapshot is older
  than 5 calendar days, the callout reads "(estimate as of DATE)".
- **Scrape drift:** if stockanalysis.com layout changes and parsing returns
  nothing, the yfinance fallback keeps the feature alive; an explicit
  parse-miss is logged (so we know the scraper needs maintenance).

## 7. Testing

Follow existing `Test/` patterns (`test_index_growth_charts.py`,
`test_main_remote_spy_eps.py`):

- **Parser unit:** saved stockanalysis.com HTML fixtures for SPY & QQQ → correct
  `(forward_pe, forward_eps)`.
- **Fallback unit:** mocked yfinance `info` dict → forward EPS via `price/PE`.
- **Math unit:** ETF→index scaling and `TTM_PE/Fwd_PE − 1` growth correctness.
- **Guard unit:** garbage inputs (PE≤0, out-of-band growth, bad scale) → no row
  written.
- **Idempotency:** same-day re-run overwrites (`INSERT OR REPLACE`).
- **Chart:** forward overlay present when data exists, gracefully absent when
  not; the existing EPS chart is unchanged when the new table is empty.

## 8. Runtime / deploy notes

- New module + new table only; no schema migration beyond `CREATE TABLE IF NOT
  EXISTS`. DB lives on the `data` branch — table is created on hydrate.
- The index growth pages rebuild on the **weekly** `site-refresh.yml` (and
  `workflow_dispatch`), **not** the daily run. Forward snapshots accrue daily but
  the chart reflects them after a weekly/dispatch render — note so the chart not
  moving intraday isn't mistaken for a bug. Dispatch `site-refresh.yml` to see it
  sooner.

## 9. Rollout order

1. Table + `index_forward_eps.py` + daily wiring → start accruing history
   (can ship silently first to bank a few days of snapshots).
2. Chart overlay + callout on the growth pages.
3. Tests.

## 10. Open questions for review

- Sane-growth band: is −50%…+80% reasonable, or tighter?
- Callout wording / how much detail Nick wants inline.
- OK to ship step 1 silently for a few days before the chart goes live?
