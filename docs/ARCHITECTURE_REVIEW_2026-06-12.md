# StockFinances — Deep Architecture Review & Modernization Plan
**Date:** 2026-06-12 · **Method:** four parallel code audits (pipeline trace, live-site consumption map, workflow inventory, DB/data-freshness analysis), all claims verified against code with file:line citations.

---

## 1. How the system actually works today

**Data flow:** `tickers.csv` (104 tickers) → `main_remote.py` orchestrates ~24 stages → writes `Stock Data.db` (31 tables, 21.7 MB, committed to git) + thousands of files under `charts/` → `html_generator2.py` builds `index.html` (dashboard) → separate `scripts/gen_ticker_json.py` runs `build_aapl_mockup.py` per ticker → `ticker_data/<T>.json` consumed by the modern `ticker.html` shell → everything rsynced to `gh-pages`.

**Mode gate:** `MODE=daily` (4×/weekday) skips the quarterly fetchers; `MODE=weekly` (Sunday) runs everything. This gate already solved the worst "re-fetch quarterly data daily" problem — the remaining waste is subtler.

**What the live site actually consumes** (everything else is dead output):
- `index.html` + `daily-market-summary.html` (iframe) + `economic_charts.html` + `spy_growth.html`/`qqq_growth.html` + `microcaps.html` + `guides`-free static pages
- `ticker_data/<T>.json` — fully self-contained (inline Plotly figures + HTML tables); reads DB tables `Tickers_Info`, `Implied_Growth_History`, `Forward_EPS_FY_History` + EDGAR/yfinance directly. **Consumes zero files from `charts/`.**
- Exactly 4 runtime files in `charts/`: `{spy,qqq}_valuation_bundle_chart_div.html` + `.js`
- `charts/microcap/<T>_*.png` (4 per ticker) + microcap CSVs
- 2 CSS files

## 2. Dead weight (verified)

### Workflows (22 files; ~483 scheduled runner-min/week)
| Verdict | Workflows |
|---|---|
| **KILL — active & redundant** | `actions.yml` (legacy full pipeline, weekdays 08:00 UTC, **no MODE → runs FULL weekly pipeline daily**, 22 min, `git add .` commits DB to main daily, raw `cp -r` deploy without the microcap/ticker_data excludes = clobber risk; ~110 min/wk = 23% of all compute) |
| **KILL — broken no-op** | `cleanup.yml` (runs DB cleanup then exits without committing — result discarded every month) |
| **KILL — redundant + failing** | `segment_charts.yml` (monthly duplicate of Sunday weekly segments; failed 2 of last 3 runs) |
| **DELETE — dead files** | 9 already-disabled: `QQQPE.yml`, `QQQ_Pe.yml`, `render_earnings_to_dashboard.yml`, `run_index_history.yml`, `db-migration.yml`, `Update_DB.yml`, `main.yml`, `run_stock_split.yml`, `segment_revenue_report.yml`; plus completed one-offs `backfill-revenue-history.yml`, `probe_fmp.yml` |
| **KEEP** | `site-refresh-daily.yml`, `site-refresh.yml`, `daily_index_data.yml`, `screen-microcaps.yml`, `update-assumptions.yml`, `monthly_spy_backfill.yml`; manual tools `db_structure.yml`, `Spyhistorical.yml` |

Also: dispatched runs share concurrency group `site-refresh` with `cancel-in-progress: true` — a late cron can (and did, 2026-06-12 16:02 UTC) cancel a dispatch mid-JSON-build, and `gen_ticker_json.py … || true` swallows the failure silently.

### Generated artifacts
~3,600 files in `charts/` + all 108 `pages/<T>_page.html` are unreachable from the live site. ~1,500 of them are **still regenerated every run** to feed the legacy per-ticker pages whose generator was retired (`html_generator2.py:614`). Producers of dead output: `chart_generator.py` (self-documented legacy), `balancesheet_chart.py`, `forecasted_earnings_chart.py` PNGs, `forward_eps_history.py`/`forward_revenue_history.py` PNGs (their **DB writes are needed**; the files are not), `expense_reports.py` PNGs, `generate_segment_charts.py` tables/PNGs, `valuation_update.py` charts, `implied_growth_summary.py` per-ticker assets, `ticker_info.py` HTML, 5 FRED PNGs, and a long tail of stale orphans (doubled-ticker PNGs, `charts/spy_pe.html` era files, `matchups.html`, `financial_charts.html`, `template.html`, `update_form.html`).

### Database
- 83% of payload = append-only daily history logs (fine); quarterly fundamentals <2%.
- Dead tables: `MarketData`, `pe_cache`, `Dividends` (ticker/year schema), `TTM_Data_OLD_1752326232`; `ValuationHistory` is write-only (8,347 rows, nothing reads it), appends 4 dup rows/ticker/day.
- **Git bloat is severe:** local `.git` = 9.3 GiB; ~150 full snapshots of the DB in history (2.7 GB) + recommitted multi-MB HTML/images. The DB is committed to main daily AND rsynced+committed to gh-pages 4×/weekday (rsync excludes microcaps/ticker_data but **not** `Stock Data.db`). This is why `Checkout` takes 3m43s per run.

### Redundant network traffic
- **Daily runs:** ~400–500 duplicate HTTP/run. Four modules ignore the existing prefetch caches and re-fetch yfinance `.info` per ticker: `ticker_info.py:53`, `valuation_update.py:32+393`, `Forward_data.py:312` (fetches `.info` per ticker just for `nextFiscalYearEnd`, a yearly value; the `info=` param exists and is unused), `expense_reports.py:147` (weekly). Finviz scraped per ticker every run for a ~monthly-changing value. FRED re-downloads 15–50 *years* of 26 series 4×/day. `generate_market_summary` runs twice per workflow (inside main_remote + as a separate step).
- **Weekly runs:** segments re-download full 10-K/10-Q iXBRL from SEC for every ticker (`sec_segment_data_arelle.py:200` writes a `.cache_ix/` it never checks); `Annual_Data` blind-upserted with no freshness check; `expense_reports` re-fetches what `prefetch_yfinance_bulk` already pulled.

## 3. Where the 25 minutes goes (measured)
Morning/dispatch run: Checkout 3m43s (9 GiB history) + Generate pages 5m13s + **Generate ticker JSONs 17m41s (GEN_WORKERS=1, single-threaded, all 104 tickers)** + deploy ~15s. Non-morning dailies: ~10–12 min.

## 4. Earnings-driven redesign (the owner's model) — building blocks vs gaps

**Vision:** quarterly statements fetched only around each ticker's earnings date (known from data, else estimated last+~91d), marked done-for-quarter once ingested; daily fetch = price + next-FY EPS/revenue estimates only; all derived metrics recomputed from those; charts auto-populate from the per-ticker JSON.

**Already exists:**
- `earnings_upcoming` (next dates per ticker, refreshed from `yf.get_earnings_dates`, `generate_earnings_tables.py:146,204-214`) and `earnings_past` (4,692 rows) — the exact calendar feed needed, currently used only for HTML tables.
- Per-quarter freshness gating pattern proven twice: `_is_ttm_fresh` (`annual_and_ttm_update.py:251-284`) and the balance-sheet 111-day gate (`balance_sheet_data_fetcher.py:81-104`).
- The JSON-feeds-everything model is already how `ticker.html` works.

**Missing:**
- Nothing gates any fetch on earnings dates; the calendar-quarter heuristic re-fetches for 3–7 weeks between quarter-end and the actual report.
- No `fetch_state(ticker, dataset, fiscal_quarter_fetched, fetched_at)` table / central scheduler; each module decides independently; `expense_reports` and segments have no gate at all.

## 5. Modernization plan (phased, each phase independently shippable)

**Phase 1 — stop the bleeding (an hour; ~45% compute cut, no behavior change):**
1. Disable `actions.yml` schedule; delete the 11 dead workflow files; delete `cleanup.yml` (or add its missing commit step); delete `segment_charts.yml`.
2. `GEN_WORKERS: 4` in both site-refresh workflows (17m41s → ~5m) + `timeout-minutes: 40`.
3. `fetch-depth: 1` on checkouts (3m43s → ~10s).
4. Fix the dispatch-vs-cron race: per-ref concurrency or drop `cancel-in-progress` for dispatches; remove the `|| true` on `gen_ticker_json.py` so JSON failures are loud.

**Phase 2 — stop generating dead output (half a day; cuts most of "Generate pages"):**
Gate every dead-artifact producer behind a `LEGACY_ASSETS=0` default (keep their DB writes); delete `pages/` and stale `charts/` orphans from main and gh-pages. Site renders identically.

**Phase 3 — earnings-driven fetch scheduler (the core; ~a weekend):**
New `fetch_state` table + one helper ("is T within ±N days of an earnings date, or past last+91d fallback, and not yet ingested this quarter?"). Wire `annual_and_ttm_update`, balance sheet, expense reports, segments, EDGAR through it. Weekly run becomes "scan all, fetch the ~8–12 tickers in their earnings window" instead of all 104. Daily run: price + forward estimates + derived metrics only (and wire the 4 cache-ignoring modules into `_yf_cache`; Finviz monthly; FRED once daily/incremental; market summary once).

**Phase 4 — repo hygiene (background):**
Stop rsyncing the DB to gh-pages (one `--exclude`); move DB commits to weekly or out of git entirely (artifact/branch); then `git filter-repo` to reclaim ~9 GiB of history; drop dead DB tables; VACUUM.

**Phase 5 — incremental ticker JSONs:**
Regenerate JSON only for tickers whose inputs changed that run (price/estimates always → light daily rebuild path; full rebuild only post-earnings). Combined with Phase 1's parallelism: morning runs ~5 min, midday ~3 min, weekly ~10 min.

**End state:** two real pipelines (daily light, weekly earnings-window), ~120 runner-min/week (from 483), runs measured in single-digit minutes, no dead artifacts, repo a fraction of its size — and the architecture finally matches reality: quarterly data moves quarterly, daily data moves daily.
