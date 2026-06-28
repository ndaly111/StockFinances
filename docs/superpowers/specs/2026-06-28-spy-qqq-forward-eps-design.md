# SPY / QQQ Forward EPS — Consensus Forward Earnings on the Index Growth Pages

**Date:** 2026-06-28 (rev. 2 — data-source pivot after research)
**Status:** Design — awaiting review
**Repo:** StockFinances (`C:\Users\ndaly\projects\sf-fix`, branch `forward-index-eps`)

> **Revision note.** Rev. 1 of this spec sourced forward P/E from stockanalysis.com
> (primary) + yfinance (fallback). End-to-end testing proved BOTH dead for ETFs:
> yfinance returns `forwardPE/forwardEps = None` for SPY/QQQ, and stockanalysis is a
> JS SPA with no forward P/E in static HTML. Research (2026-06-28) found there is **no
> free turnkey forward-EPS source for the QQQ/Nasdaq-100 index** (Barchart, Invesco,
> Nasdaq all publish only *trailing* P/E at the ETF level; Barchart's forward data is
> per-*stock* only). Rev. 2 pivots to: **S&P official xlsx for SPY** + **bottom-up
> calculation from our own constituent data for QQQ**, with index membership driven by
> the fund's published holdings so it stays correct as companies enter/leave the index.
> See [[reference_index_forward_eps_sources]] for the full source landscape.

---

## 1. Goal

Show the **future estimated EPS growth of SPY and QQQ** on the index growth pages
(`spy_growth.html` / `qqq_growth.html`): a current-FY and next-FY forward EPS growth
figure plus a forward point on the EPS trajectory chart, snapshotted daily, sourced
accurately and kept correct as index membership changes over time.

## 2. Non-goals (YAGNI)

- No homepage overview-table changes (decided: chart-first).
- No new pages.
- No live intraday updates (rebuilds on the weekly `site-refresh.yml` / dispatch).
- No per-side / overfit logic — this is a display metric.

## 3. What already exists (built + unit-tested on branch `forward-index-eps`)

These are DONE and source-agnostic — they stay:
- **`Index_Forward_EPS_History`** table (`ensure_forward_eps_table`).
- **Daily snapshot** `snapshot_forward_eps(conn)` wired into `main_remote.py` after
  `index_growth(treasury)`.
- **Sanity guard** `_passes_sanity` (PE>0, growth band −50%..+80%).
- **Chart overlay** `_add_forward_eps_overlay` + **callout** `_forward_eps_callout`
  (incl. expected-growth % and forward implied growth) wired into
  `render_index_growth_charts` (`index_growth_charts.py`).
- 21 passing tests.

What must be **replaced/added**: the two dead fetch functions
(`_fetch_stockanalysis_pe`, `_forward_from_yf`) → an S&P-xlsx parser (SPY) and a
holdings-driven bottom-up aggregator (QQQ); plus a holdings fetcher, scrape-universe
auto-extend, and a validation step.

## 4. Data sources (rev. 2)

### 4.1 SPY → S&P Dow Jones Indices official xlsx
`https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx`
- Needs a browser `User-Agent` header (bare requests → HTTP 403). Free, no login,
  overwritten weekly in place.
- Parse with `pandas.read_excel(io.BytesIO(r.content), sheet_name=...)`. Sheets of
  interest: `ESTIMATES&PEs` / `FORWARD SCHEDULE` (bottom-up forward operating EPS by
  quarter + annual FY current / FY next).
- **Future-proof by construction:** S&P maintains the 500-name membership; the file
  always reflects current index composition. No constituent management needed for SPY.
- Build step 0: verify current sheet/column layout (download once, inspect) before
  writing the parser; parse defensively (locate rows/columns by label, not fixed cell).

### 4.2 QQQ → bottom-up from our own constituent data, membership from holdings
No free turnkey source exists. Compute it:

**Inputs already in `Stock Data.db` (fresh daily — verified 2026-06-28):**
- `Forward_EPS_FY_History` — Zacks forward EPS, "This FY" + "Next FY", ~98 tickers.
- `TTM_Data` — `TTM_EPS` and `Shares_Outstanding`, ~102 tickers.

**Membership + weights — from the fund's published holdings (NOT a hardcoded list):**
- Fetch **Invesco QQQ holdings** (constituents + weights), weekly cadence.
  Primary: Invesco holdings download; fallback: slickcharts.com/nasdaq100 (constituents
  + weights). Build step: verify the exact endpoint + parse format, with the fallback.
- This is the future-proofing: the aggregate runs over *today's* holdings, so index
  adds/drops/renames flow through automatically.

**Aggregation (over covered constituents):**
```
trailing_$_i = TTM_EPS_i      × Shares_Outstanding_i
forward_$_i  = Forward_EPS_i  × Shares_Outstanding_i      # This-FY and Next-FY
index_forward_growth_thisFY = Σ forward_$_thisFY / Σ trailing_$ − 1
index_forward_growth_nextFY = Σ forward_$_nextFY / Σ forward_$_thisFY − 1
```
- Summed **dollar earnings** (not averaged per-share growth) — the correct way to
  aggregate index earnings; naturally handles negative earners.
- Report **current-FY and next-FY** growth (mirrors Barchart's per-stock "Growth Rate
  Est. (YoY)", which Nick uses), plus a forward-EPS dollar value for the chart point.
- Coverage measured by **holdings weight**, not name count.

### 4.3 Auto-extend the scrape universe (the key future-proofing — decided)
Every QQQ run: diff today's holdings against the tickers we have forward EPS for.
Any constituent **missing** forward EPS is **auto-added to the forward-EPS scrape
universe** so the next `Forward_data` run collects it, and it joins the aggregate
automatically. Implementation: maintain an `index_constituents` set (e.g. a DB table
or a managed supplementary list) that is **unioned** into the scrape universe — do NOT
mutate Nick's curated `tickers.csv` (keep his list intact; add a separate
auto-managed source). New names self-heal within a day or two.

## 5. Self-monitoring & validation gate

- **Coverage %:** each run computes the fraction of index weight covered by names with
  available forward EPS. Stored alongside the snapshot.
- **Validation gate:** a snapshot is marked displayable only if (a) coverage ≥ threshold
  (e.g. 85% of index weight) AND (b) the computed growth/forward-P/E lands within a
  tolerance of published consensus benchmarks (QQQ ≈ +19% growth / ~23.5× fwd P/E;
  S&P CY2026 ≈ +21%). Otherwise the number is withheld (chart/callout omit it) and the
  reason logged. The existing `_passes_sanity` remains the first-line filter.
- Coverage % and source/as-of are surfaced in the callout (e.g. "based on 94% of QQQ
  weight"), so the number is self-labeling about its completeness.

## 6. Table / schema changes

Extend `Index_Forward_EPS_History` (or add columns) to record what's needed for
auditing the bottom-up number: `coverage_weight` REAL (fraction of index weight
covered), `growth_this_fy` REAL, `growth_next_fy` REAL, `method` TEXT
('sp_xlsx' | 'bottom_up'), in addition to the existing forward_eps_index / forward_pe /
horizon_date / source. New `index_constituents` table (ticker, index, weight,
date_recorded) to drive the QQQ aggregate + auto-extend.

## 7. Module layout

- `index_forward_eps.py` (exists) — keep table/snapshot/sanity/orchestration; replace
  the fetch internals.
- `forward_eps_sp_xlsx.py` (new) — download + parse the S&P xlsx → SPY forward EPS/growth.
- `forward_eps_bottom_up.py` (new) — holdings fetch + scrape-universe auto-extend +
  constituent aggregation → QQQ forward EPS/growth + coverage.
- `forward_eps_validate.py` (new) — benchmark comparison / validation gate.
- Chart layer (`index_growth_charts.py`) — unchanged (already reads the snapshot row).

## 8. Edge cases & data-quality guards

(Per project rules: never write 0/placeholder; never let a silent error look like data.)
- Source/holdings fetch fails → skip that index's snapshot, log, write no row.
- Coverage below threshold → withhold display, log the gap + the missing high-weight names.
- Negative-earner sign flips, fiscal-year smear (AAPL/MSFT/NVDA non-Dec FY) → documented
  approximation; the validation gate is the backstop.
- S&P xlsx layout drift → defensive label-based parsing; on parse miss, skip + log.
- Holdings endpoint change → fallback source; on total failure, reuse the last good
  holdings snapshot (with an age warning) rather than silently emptying the universe.

## 9. Testing

- xlsx parser against a saved fixture of the real sheet.
- bottom-up aggregator against a synthetic constituents+EPS set with a known answer
  (incl. a negative-earner and a missing-forward-EPS name → coverage math).
- auto-extend: a holdings set with a new ticker → the ticker is added to the managed
  scrape set (and `tickers.csv` is untouched).
- validation gate: in-tolerance → displayable; out-of-tolerance / low-coverage → withheld.
- idempotent daily upsert; chart overlay present only when a displayable row exists.

## 10. Runtime / deploy notes

- Index growth pages rebuild on the **weekly** `site-refresh.yml` (or dispatch); daily
  snapshots accrue but show after a weekly/dispatch render.
- Holdings refresh weekly; S&P xlsx weekly; both cheap (one HTTP GET each).
- DB lives on the `data` branch — new tables self-create via `CREATE TABLE IF NOT EXISTS`.
- Deploy/push stays with Nick (Windows git has the only push auth).

## 11. Open questions for review

- Coverage threshold for display (proposed 85% of index weight) — OK?
- Validation tolerance band vs published consensus (proposed ±a few pts) — OK?
- Where to store the auto-managed constituent/scrape set (new DB table vs supplementary
  CSV) — recommend a DB table `index_constituents`.
