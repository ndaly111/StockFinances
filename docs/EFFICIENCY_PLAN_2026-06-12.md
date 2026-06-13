# StockFinances — Efficiency Modernization Plan (verified)
**Date:** 2026-06-12 · **Method:** 6 parallel deep-dive investigations + adversarial verification of each against the code (4 of 6 verified; git-surgery and workflow-endstate corroborated by the other tracks but their dedicated verifiers were cut off by a rate limit). Supersedes the sequencing in `ARCHITECTURE_REVIEW_2026-06-12.md`; that review's *findings* stand, its *Phase 1* is now known to contain a landmine — see below.

---

## The one thing that changes everything: DB persistence

Four independent tracks converged on the same buried fact:

> **The site-refresh workflows never commit the database to `main` — they only push to gh-pages. The *only* thing persisting the pipeline's daily writes to `main` is the legacy `actions.yml` job's `git add .`** (weekdays 08:00 UTC, full pipeline).

The daily per-ticker history dots the ticker pages render — `Implied_Growth_History` (`ticker_info.py:256`) and `Forward_EPS_FY_History` (`Forward_data.py`) — plus Sunday's quarterly `TTM_Data`/`Annual_Data`, survive on `main` *only* because `actions.yml` happens to commit them. Every run checks out `main`; nothing else writes the DB back.

**Two consequences that invert the obvious plan:**

1. **🚫 LANDMINE — do NOT just "disable actions.yml" (the old Phase 1).** Kill it with no replacement and the ticker history charts develop a permanent, silently-growing hole, daily valuations freeze on stale `TTM_Data`, and the homepage earnings tables freeze. Verified by `kill-list-safety`, `git-db-surgery` (F2), and `workflow-endstate` (F1) independently.

2. **🔑 KEYSTONE — DB persistence is the prerequisite for almost everything else.** The incremental-JSON dirty-check, the earnings-window scheduler, and the `fetch_state` table all assume state survives between runs. It doesn't today. Until the DB persists, every "skip if already fetched" design degenerates to "fetch everything every run" (state wiped) or silently fails to deploy (`ticker_data/` is gitignored on `main`; the <90-file guard blocks partial deploys; `rsync --delete` wipes skipped tickers).

The fix (from `git-db-surgery`): a dedicated **orphan `data` branch** as the DB's single source of truth, fetched at job start and pushed at job end with idempotent re-run-on-conflict. It simultaneously (a) defuses the actions.yml landmine, (b) provides the persistence layer the whole roadmap needs, and (c) stops the DB committing to `main` daily (22.7 MB/day of git bloat).

## The other landmine: GEN_WORKERS=4 is a trap

The old Phase 1 said "bump `GEN_WORKERS` 1→4 for a 3.5× JSON speedup." **Three tracks independently refuted this.** The workflow pins workers=1 with a comment recording *why*: a parallel attempt rate-limited yfinance and built **4/102 tickers** — and the failure is silent (`|| true` + the script never exits nonzero + the <90 guard hides it). At 56 EDGAR requests over ~7s/ticker, 2 workers already exceed SEC's 10 req/s cap. **Parallelism is unlocked only by removing the network calls first, not before.** Strike that line.

---

## Where the time actually goes (measured, verified)

| Run | Dominant cost | Detail |
|---|---|---|
| **Morning/dispatch JSON build, 17m41s** | ~90% network | **56 EDGAR HTTP requests per ticker** (`fetch_concept` tries every fallback × 2 taxonomies, no early-exit, no connection reuse) → ~5,800 requests/build, single-threaded |
| **Daily "Generate pages", 5m43s** | per-ticker loop | ~520 redundant HTTP/run: 3 modules re-fetch yfinance `.info` ignoring the existing cache; Finviz scraped 103×/run for a ~monthly value; `^TNX` fetched 104× for one number |
| **Checkout, 3m43s** | 9.3 GiB history | `fetch-depth: 0` of a repo bloated by daily DB + artifact commits |

The 9.3 GiB is **not** mostly DB snapshots (those delta-compress) — it's ~960k small regenerated artifacts (`charts/` PNGs, `pages/`, HTML) recommitted across ~2,000 deploy commits.

---

## Verified plan, in dependency order

### Phase 0 — Safe immediate wins (no persistence dependency, low risk, ~half a day)
Pure-efficiency changes with identical output. None touch the DB-commit model.
- **Network hoists in the JSON build** (`build_aapl_mockup.py` / `gen_ticker_json.py`): fetch `^TNX` once in the parent not 104×; reuse one `yf.Ticker` instead of the duplicate `.info`; load `assumptions.json` once. ≈ −210 requests/build.
- **Wire the 3 cache-ignoring daily modules into `_yf_cache`** (`ticker_info.py:53`, `valuation_update.py:32`+`:393`, `Forward_data.py` worker): ≈ −300 HTTP/daily run. (daily-run-cost Fixes A–C, verified sound.)
- **Dedupe the double market-summary** (runs inside `main_remote.py:517` *and* as a separate workflow step), **FRED incremental fetch** (stop re-pulling 15–50 yrs × 26 series 4×/day — must remove the CPI/PCE `DELETE`s in the same commit), **Finviz staleness gate** (with the `INSERT OR IGNORE`+restructure fix so it engages for SPCX-style new tickers and failure paths). (Fixes E–G.)
- **Failure visibility:** drop `|| true` **and** add a built-count exit-code threshold in `gen_ticker_json.py` (removing `|| true` alone does nothing — the script can't currently fail), plus a `gh issue create` on failure (zero-infra, pattern already in `update-assumptions.yml`).

### Phase 1 — DB persistence keystone (the `data` branch) — UNBLOCKS EVERYTHING
- Create orphan `data` branch holding `Stock Data.db`; every workflow fetches it at job start, pushes at job end with **re-fetch-and-re-run** on conflict (binary DBs can't rebase-merge; writers are idempotent `INSERT OR REPLACE`).
- Add the DB-persist step to the **evening daily run** (post-close dots) and the **weekly run** (Sunday fundamentals) — this is the replacement for actions.yml's accidental persistence.
- `git rm --cached "Stock Data.db"` on main + `.gitignore`; add `--exclude "Stock Data.db"` (+ `*.db-wal/shm/journal`) to the gh-pages rsyncs; `git rm` the public DB copy.
- **Only then** retire `actions.yml` (keep `workflow_dispatch` a week as a parachute).
- Result: main stops gaining 22.7 MB/day; gh-pages stops gaining ~110 MB/day.

### Phase 2 — The real JSON speedup: pure render
- Swap the 56 per-ticker `companyconcept` calls for **1 `companyfacts` call** (intermediate win → ~6 min), then move toward a **pure DB render** (build reads tables, zero live HTTP → morning JSON step ~30–60s). Requires: a new `Edgar_Financials_Annual` table (deep history — today's `Annual_Data` is only ~6 yrs, too shallow for the 15Y/MAX buttons), a `Ticker_News` table, 4 low/high columns on `ForwardFinancialData` (the data is already fetched and **thrown away** — ~10-line fix), and reviving `MarketData`/`Daily_Prices`.
- ⚠️ The existing `data_providers/edgar.py` extractor is **too naive to reuse as-is** (us-gaap-only, 10-K-only, first-concept-wins, no FX) — it would silently truncate IFRS filers (BUD/SPOT) and tag-transition companies. Port `fetch_concept`'s cross-concept merge faithfully and **diff all 104 tickers' JSON before/after** before cutover.
- After pure render removes the network calls: **then** `GEN_WORKERS=4` is safe (CPU-bound).

### Phase 3 — Earnings-window fetch scheduler (your model)
- Now implementable *because Phase 1 persists state*. `fetch_state(ticker, dataset, fiscal_quarter_fetched, window_anchor_done, fetched_at)` + a gate driven by the existing `earnings_upcoming`/`earnings_past` tables (fallback: last + 91d; new IPOs like SPCX fetch unconditionally until first success). Weekly fundamentals fetch drops from 104 tickers to the ~0–12 in their earnings window.
- ⚠️ Needs: refresh `earnings_upcoming` daily (its writer is weekly today); per-dataset windows (income vs balance sheet land days apart); split adjustment stays at **render time** against the `Splits` table (storing split-adjusted EPS at ingest would corrupt up to 90 days of history after any split).

### Phase 4 — Workflow consolidation + repo de-bloat
- Fold `daily_index_data.yml` into the evening daily run (it duplicates `backfill_index_growth`); fix the cron-cancels-dispatch race (`cancel-in-progress: false`, JSON build as its own *job* with its own group); add `timeout-minutes` (a 78-min hang on 6/10); `fetch-depth: 1`. Delete the 9 disabled + 2 finished-one-off workflows (safe-verified); kill `cleanup.yml` (broken no-op) and `segment_charts.yml` (redundant, failing).
- **`git filter-repo`** to reclaim the 9.3 GiB (strip DB + `charts/` + `pages/` + generated HTML) — quiet weekend, fresh clones afterward. **A+B+gh-pages-reset alone deliver ~95% of the benefit at ~10% of the risk** — the full history rewrite is a someday-task.

---

## Live bugs the review surfaced (independent of the plan)

1. **`ME` is delisted** (23andMe — gone from SEC's ticker map) so its page fails to build *every run* (`[ME] FAILED: no CIK`) and `ME.json` is already missing from the live site. → remove from `tickers.csv`.
2. **Weekly workflow can wipe the site.** `site-refresh.yml`'s rsync lacks the `ticker_data/` exclude and count-guard the daily run has; a partial JSON build (the documented 4/102) + `rsync --delete` would delete ~100 live ticker JSONs in one Sunday run. F3 (ME.json's disappearance) proves the deletion path is live. → needs careful fix (trace its ticker_data deploy path first).
3. **TTM dividend overstated ~4×** on the EPS&Dividend chart: `div_a.tail(4).sum()` sums 4 *years* of annual dividends as a "TTM" bar (`build_aapl_mockup.py:1521`). → switch to `Dividends_Data.TTM_Dividend`.

## What's safe to delete (adversarially verified)
SAFE: `cleanup.yml`, `segment_charts.yml`, the 9 disabled workflows, `backfill-revenue-history.yml` + `probe_fmp.yml` (run any DB-history-mining backfills *before* filter-repo), DB tables `MarketData`/`pe_cache`/`Dividends`/`TTM_Data_OLD_*`. KILL-WITH-CHANGE: `actions.yml` (persist DB first), `pages/`+dead `charts/` (strip the producer in `refresh_one_ticker.py` + keep `charts/earnings_*.html` until persistence lands), `ValuationHistory` (delete its writer in the same commit, archive the 8,347 rows first).
