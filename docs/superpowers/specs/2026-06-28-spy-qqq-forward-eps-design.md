# SPY / QQQ Forward EPS — Bottom-Up Forward Earnings on the Index Growth Pages

**Date:** 2026-06-28 (rev. 3 — uniform bottom-up after S&P file discontinued)
**Status:** Design — approved, ready to plan
**Repo:** StockFinances (`C:\Users\ndaly\projects\sf-fix`, branch `forward-index-eps`)

> **Revision history.**
> - **Rev 1:** stockanalysis.com + yfinance. DEAD — yfinance returns `None` for ETF
>   forwardPE/forwardEps; stockanalysis is a JS SPA.
> - **Rev 2:** S&P official xlsx (SPY) + bottom-up (QQQ). DEAD for SPY — the S&P EPS
>   file was **discontinued** when Howard Silverblatt retired 2026-01-31; the live URL
>   now hard-403s (Akamai), no mirror, only a frozen Jan-2026 Wayback snapshot loads.
> - **Rev 3 (this):** **uniform bottom-up for BOTH indices** from our own constituent
>   forward EPS, with membership + weights from **slickcharts** (Invesco is WAF-blocked).
> See [[reference_index_forward_eps_sources]] for the full landscape.

---

## 1. Goal

Show the **future estimated EPS growth of SPY and QQQ** on `spy_growth.html` /
`qqq_growth.html`: current-FY and next-FY forward EPS growth (Barchart-style) plus a
forward point on the EPS trajectory chart, snapshotted daily, computed bottom-up and
kept correct as index membership changes.

## 2. Non-goals (YAGNI)

- No homepage overview-table changes (chart-first).
- No new pages; no intraday updates (rebuilds on weekly `site-refresh.yml`/dispatch).
- No per-side/overfit logic — a display metric.

## 3. Already built + unit-tested on branch `forward-index-eps` (reused as-is)

- `Index_Forward_EPS_History` table + `ensure_forward_eps_table`.
- `snapshot_forward_eps(conn)` daily, wired into `main_remote.py`.
- `_passes_sanity` (PE>0, growth band −50%..+80%).
- Chart `_add_forward_eps_overlay` + `_forward_eps_callout`, wired into
  `render_index_growth_charts`. 21 passing tests.

**To replace/add:** the two dead fetch functions → a bottom-up aggregator; plus a
holdings fetcher, weight-prioritized scrape auto-extend, and a validation gate.

## 4. Data sources (rev. 3)

### 4.1 Index membership + weights → slickcharts (verified working 2026-06-28)
- **QQQ:** `https://www.slickcharts.com/nasdaq100` — 101 rows, ticker + weight, parses
  with `pandas.read_html` (browser UA + Referer). Note GOOG **and** GOOGL both present.
- **SPY:** `https://www.slickcharts.com/sp500` — same structure (~503 rows).
- Select the table by **column labels** (`Symbol`, `Weight`), not position. `.str.strip()`
  tickers; parse `Weight` as float (strip `%`). Sanity check `len(df) >= 99` (QQQ) /
  `>= 490` (SPY) to catch partial loads.
- **Fallback:** Wikipedia (`/wiki/Nasdaq-100`, `/wiki/List_of_S%26P_500_companies`) for
  membership cross-check (tickers only, no weights) — alert if the ticker sets diverge
  (signals slickcharts stale after a reconstitution). Invesco/Nasdaq OMX are blocked/JS.
- **Future-proof by construction:** the aggregate runs over *today's* holdings, so index
  adds/drops/renames flow through automatically.

### 4.2 Constituent forward EPS + shares → our own `Stock Data.db` (fresh daily)
- `Forward_EPS_FY_History` — Zacks forward EPS, "This FY" + "Next FY" (~98 tickers today).
- `TTM_Data` — `TTM_EPS` and `Shares_Outstanding` (~102 tickers).

### 4.3 Aggregation (uniform, both indices)
Over the covered constituents (those in today's holdings that have forward EPS):
```
trailing_$_i = TTM_EPS_i      × Shares_Outstanding_i
fwd_$_thisFY_i = ForwardEPS_thisFY_i × Shares_Outstanding_i
fwd_$_nextFY_i = ForwardEPS_nextFY_i × Shares_Outstanding_i
growth_this_fy = Σ fwd_$_thisFY / Σ trailing_$  − 1
growth_next_fy = Σ fwd_$_nextFY / Σ fwd_$_thisFY − 1
forward_eps_index = (Σ fwd_$_thisFY) / index_share_base   # for the chart point; scale
                                                          # to match _series_eps level
```
- Summed **dollar earnings** (not averaged growth) — correct index aggregation; handles
  negative earners. Report **current-FY and next-FY** growth (mirrors Barchart).
- Coverage measured by **holdings weight** of the covered set.
- GOOG+GOOGL: keep both (they have separate financials); don't double-count by company.

### 4.4 Weight-prioritized scrape auto-extend (decided: target ~90% coverage)
Each run, per index: diff today's holdings (with weights) against the tickers we have
forward EPS for. Walking constituents **highest-weight first**, add any uncovered name
to an auto-managed scrape set until cumulative covered weight ≥ **90%** (cap the additions
so we don't chase the long tail). Those names get forward EPS collected on the next
`Forward_data` run and join the aggregate automatically.
- QQQ: ~all 101 names (cheap). SPY: ~top 100–150 names (S&P 500 is top-heavy; ~90% of
  weight sits in the largest ~150). We do NOT scrape all 500.
- Storage: a new DB table `index_constituents` (ticker, index, weight, date_recorded)
  drives the aggregate; the auto-managed scrape set is **unioned** into the
  `Forward_data` universe — **do NOT mutate Nick's curated `tickers.csv`.**

## 5. Self-monitoring & validation gate

- **Coverage %** (of index weight) computed each run, stored with the snapshot.
- **Validation gate** — display only if (a) coverage ≥ **85%** AND (b) computed growth /
  forward P/E within tolerance (**±~5 pts**) of published consensus benchmarks
  (S&P CY2026 ≈ +21% / ~20.9× fwd P/E; QQQ ≈ +19% / ~23.5×). Else withhold + log the
  reason and the missing high-weight names. `_passes_sanity` stays the first-line filter.
- Coverage % + as-of surfaced in the callout ("based on 94% of QQQ weight").

## 6. Schema changes

Add columns to `Index_Forward_EPS_History`: `coverage_weight` REAL, `growth_this_fy`
REAL, `growth_next_fy` REAL, `method` TEXT (='bottom_up'), `displayable` INTEGER
(validation-gate result). New table `index_constituents` (ticker, index, weight,
date_recorded, PRIMARY KEY (date_recorded, index, ticker)).

## 7. Module layout

- `index_forward_eps.py` (exists) — keep table/snapshot/sanity/orchestration; the
  snapshot now calls the bottom-up aggregator for both indices (remove the
  stockanalysis/yfinance fetch internals).
- `index_holdings.py` (new) — slickcharts fetch (+ Wikipedia fallback) → {ticker, weight}
  per index; persist to `index_constituents`; weight-prioritized auto-extend of the
  scrape set.
- `forward_eps_bottom_up.py` (new) — join holdings × `Forward_EPS_FY_History` ×
  `TTM_Data` → index forward EPS, growth_this_fy, growth_next_fy, coverage.
- `forward_eps_validate.py` (new) — benchmark comparison → displayable flag.
- `index_growth_charts.py` (exists) — unchanged (reads the snapshot row).
- `Forward_data.py` (exists) — scrape universe = `tickers.csv` ∪ auto-managed
  `index_constituents` set (small, additive change).

## 8. Edge cases & data-quality guards

(Never write 0/placeholder; never let a silent error look like data.)
- Holdings fetch fails → Wikipedia membership fallback; if both fail, reuse last good
  `index_constituents` snapshot with an age warning; never empty the universe silently.
- Coverage < 85% → withhold display, log gap + missing high-weight names.
- Constituent missing forward EPS or shares → excluded from the sums, counted against
  coverage (not silently treated as zero).
- Negative-earner sign flips / fiscal-year smear (AAPL/MSFT/NVDA non-Dec FY) → documented
  approximation; validation gate is the backstop.
- slickcharts layout drift → label-based table selection; on miss, fallback + log.

## 9. Testing

- Holdings parser against a saved slickcharts fixture (QQQ + SPY) → {ticker, weight}.
- Auto-extend: holdings with weights + a covered/uncovered split → adds exactly the
  top-weighted uncovered names to reach ≥90%, stops, and never writes to `tickers.csv`.
- Aggregator against a synthetic constituents+EPS+shares set with a hand-computed answer,
  including a negative-earner and a missing-forward-EPS name (coverage math).
- Validation gate: in-tolerance+high-coverage → displayable; out-of-tolerance or
  low-coverage → withheld.
- Idempotent daily upsert; chart overlay present only when a displayable row exists.

## 10. Runtime / deploy

- Index growth pages rebuild on the **weekly** `site-refresh.yml` / dispatch; daily
  snapshots accrue, show after a weekly/dispatch render.
- Holdings refresh weekly (NDX/SPX reconstitute infrequently); cheap (one GET each).
- DB on the `data` branch — new tables self-create via `CREATE TABLE IF NOT EXISTS`.
- Deploy/push stays with Nick (Windows git has the only push auth).
- Scrape-universe growth is bounded by the 90% weight target (QQQ ~101, SPY ~top 150).

## 11. Decisions locked

- Both indices bottom-up; slickcharts holdings (Wikipedia fallback).
- Auto-extend scrape set, weight-prioritized to ~90% coverage; `tickers.csv` untouched.
- Coverage display threshold 85%; validation tolerance ~±5 pts vs published consensus.
- Constituent/scrape set in DB table `index_constituents`.
