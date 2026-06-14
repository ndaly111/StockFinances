# StockFinances Modernization — Session Record (2026-06-12 → 06-13)

A multi-phase efficiency + architecture overhaul, motivated by one question: **"why does the pipeline take ~25 minutes?"** Every irreversible step was adversarially verified before shipping; the live site had zero disruption throughout.

## Results at a glance

| Metric | Before | After | Factor |
|---|---|---|---|
| Daily ticker-JSON build (CI step) | ~1115 s (~18.5 min) | **35 s** | ~32× |
| Weekly "Generate pages" (CI step) | ~930 s (~15.5 min) | **349 s** (~5.8 min) | ~2.7× |
| EDGAR HTTP calls per ticker | 56 | **1** | 56× |
| Repo growth to `main` | +22 MB DB **every run** | **0** (DB on `data` branch) | — |
| GitHub workflows | 22 | **9** | — |
| Redundant HTTP per daily run | ~400–500 | ~0 (cache-wired) | — |

Companion docs: `ARCHITECTURE_REVIEW_2026-06-12.md` (initial audit), `EFFICIENCY_PLAN_2026-06-12.md` (verified plan), `PHASE1_DB_BRANCH_RUNBOOK.md` (the DB-off-main runbook).

---

## How the work was structured

1. **Audit** — six parallel deep-dive investigations (pipeline trace, live-site consumption map, workflow inventory, DB/freshness) with adversarial verification of each finding against the code.
2. **Plan** — a phased plan ordered by dependency and risk; the audit overturned the "obvious" plan (two landmines, below).
3. **Execute** — phase by phase, each validated on a real CI run before the next; the single irreversible change (Phase 1 Stage C) ran through two independent adversarial reviewers that caught four real bugs before it shipped.

### The two landmines the audit caught (and how the plan avoided them)

- **Killing `actions.yml` first would have frozen the ticker history charts.** The site-refresh workflows only push to `gh-pages`, never `main` — so the legacy `actions.yml` job was, by accident, the *only* thing persisting the pipeline's DB writes (the daily implied-growth / forward-EPS dots, weekly TTM/Annual) to `main`. Disabling it cold would have silently rotted the charts. The fix: build DB persistence *first* (Phase 1), then retire it.
- **Bumping `GEN_WORKERS` 1→4 for a "free" speedup was a trap.** A prior attempt rate-limited yfinance to *4 of 102 tickers built*, silently. Parallelism was only safe **after** the request volume was cut (Phase 2's companyfacts swap), not before.

---

## Phase 0 — pipeline efficiency + failure visibility

*(commits: dedup+dividend, cache-wiring, failure-visibility)*

- **Deduped the per-ticker `.info` fetch** in `scripts/build_aapl_mockup.py` (two `yf.Ticker` objects → one). Byte-identical output.
- **Fixed a TTM-dividend ~4× overstatement** on the EPS/Dividend chart: `div_a.tail(4).sum()` summed *4 years* of annual dividends as a "TTM" bar. Now reads `Dividends_Data.TTM_Dividend` (AAPL 3.50 → 1.04).
- **Wired three daily modules into the existing prefetch cache** (`ticker_info.fetch_stock_data`, `valuation_update.get_current_price/fetch_financial_valuation_data`, the `Forward_data` batch worker) → ~300 fewer HTTP per daily run, output unchanged (the cache returns the same `.info` dict).
- **Made build failures loud:** `scripts/gen_ticker_json.py` now exits nonzero when too few tickers build (`GEN_FAIL_TOLERANCE`); dropped the `|| true` on both site-refresh JSON steps; added a deduped `gh issue` on workflow failure. (This is how the silent loss of `ME.json` was caught.)

## Phase 1 — move the database off `main` (the keystone)

*(commits: `deea4d2e` additive persist, `7a5c0880` hardening+runbook, `f3106de1` Stage A, `ca4f6630` Stage B, `a4be29e6` Stage C)*

**Why:** `Stock Data.db` (22.7 MB) was committed to `main` on every run, bloating the repo to ~9.3 GiB and making each checkout ~3.5 min. It's also the prerequisite for any "remember what we already fetched" logic — that state has to survive between runs.

**The new architecture:** an **orphan `data` branch** holds only `Stock Data.db` and is its single source of truth. Workflows hydrate it at job start and persist it at job end:

- **Hydrate:** `git fetch --depth 1 origin data && git restore --source=origin/data --worktree -- "Stock Data.db"`. `git restore --source` writes the working tree *without staging*, so it works even though the DB is now gitignored on `main`, and a stray `git add .` can never re-commit it.
- **Persist:** clone the `data` branch shallow, copy the freshly-built DB in, commit, push — guarded by a `<10 MB` size check (sqlite silently creates empty files; an empty push would wipe history) and a 5-attempt loop. `daily_index_data`'s loop additionally **re-runs its own idempotent writer** on a push race (it shares 21:30 UTC with the evening refresh) rather than clobbering.

**Shipped in four validated stages** so the live site was never at risk:
- **A** — refresh workflows *read* from `data`; gh-pages stops carrying the DB.
- **B** — persistence becomes mandatory (loud on failure) on every run.
- **C** — *the irreversible flip:* `git rm --cached` + `.gitignore` the DB; convert the dual-writer workflows; **retire `actions.yml`'s schedule** (the refresh workflows are the proven replacement); fold the index-snapshot collection into the evening refresh to eliminate the 21:30 race.

**Stage C verification** (two adversarial reviewers, verdict `FIX_FIRST`) caught four real bugs that were fixed before shipping: two enabled workflows (`cleanup`, `backfill-revenue-history`) that would have built on an empty DB / step-failed on the gitignored `git add`; a `db_structure` empty-artifact case; and a disjoint-table clobber on the 21:30 race.

## Cleanup — 22 → 9 workflows

*(commit `6db92db8`)* Removed 13 adversarially-verified-dead workflows (9 disabled, 2 completed one-offs, the broken-no-op `cleanup`, the redundant `segment_charts`). Survivors: the 3 active site/data workflows, `update-assumptions` (form), `monthly_spy_backfill`, the `daily_index_data`/`actions.yml` dispatch backstops, and `db_structure`/`Spyhistorical` manual tools.

## Phase 2 — the JSON-build speedup (1115 s → 35 s)

*(commits: `32704e42` companyfacts, `01044cdc` workers=4)*

- **One `companyfacts` call per ticker, not ~56 `companyconcept` calls.** `build_aapl_mockup.fetch_concept` now reads every concept from a single cached `companyfacts` JSON instead of making a per-concept × per-taxonomy HTTP request. Same underlying XBRL facts → **byte-identical output**, verified across the hard cases (AAPL tag-transition, BAC/JPM banks, SPOT/BUD/BABA IFRS filers, BRK-B derived-EPS). **1115 s → 136 s.**
- **`GEN_WORKERS` 1 → 4.** The companyfacts swap cut per-ticker requests ~9× (62 → ~7), so 4 concurrent workers no longer trip Yahoo's rate limit (the old 4/102 failure was a request-volume problem). The `<90`-build guard makes any regression loud + non-deploying. **136 s → 35 s.** The full "pure render" (moving Yahoo calls to DB tables) was deemed unnecessary — the target was already beaten.

## Phase 3 — remove dead weekly fetchers (~930 s → 349 s)

*(commits: `fb703ab6` remove fetchers, `1f22eb1` prefetch slim)*

The plan was an *earnings-window fetch scheduler* (fetch quarterly data only around each ticker's earnings date). Investigation found something better: **the Phase 2 companyfacts swap had orphaned the very fetchers the scheduler would gate.** The ticker JSON now extracts balance-sheet and expenses straight from EDGAR, so the weekly DB-writers were dead output:

- **`build_segments_for_ticker`** — downloaded the full 10-K + 10-Q iXBRL per ticker (~400 SEC requests/run) to write `charts/<T>/` files no JSON references (AAPL's segment figures are hardcoded in `build_aapl_mockup`; all JSONs have 0 `charts/` refs).
- **`fetch_and_update_balance_sheet_data` + `balancesheet_chart`** — `BalanceSheetData` read only by the dead PNG chart.
- **`generate_expense_reports`** — `IncomeStatement`/`QuarterlyIncomeStatement` read by nothing.
- Plus **dropped `quarterly_balance_sheet`** from the bulk prefetch (its only consumer was the removed balance fetcher).

Removed outright (better than gating). Verified the live JSON still renders all sections from companyfacts; weekly run green, homepage 200. `annual_and_ttm_update` was **kept** — `TTM_Data` feeds the live homepage valuation, it rides a cheap bulk cache, and gating it by earnings would risk a stale valuation for a marginal once-a-week gain.

---

## Operating the new architecture (for future maintainers)

- **The DB lives on the `data` branch, not `main`.** To get it locally: `git fetch origin data && git restore --source=origin/data --worktree -- "Stock Data.db"`. Your local working copy is gitignored, so it won't be committed.
- **Every workflow that reads the DB hydrates it first** (a "Hydrate DB from data branch" step). If you add a new DB-reading workflow, it needs that step or it will build on an empty auto-created DB.
- **Writers persist to `data`**, never to `main`. Copy the persist block from `daily_index_data.yml` (with the re-run-on-race loop if the writer can run concurrently with another).
- **`actions.yml` is retired** to dispatch-only (a parachute); the site-refresh workflows do its old job. Delete it once you're confident.
- **`GEN_WORKERS=4`** is safe *because* EDGAR is now one call/ticker. If you re-add many per-ticker network calls, re-check the rate limit before raising workers further.

## Deferred (optional, documented)

- **`git filter-repo`** to purge the 22 MB DB blob + dead `charts/`/`pages/` from `main` and `gh-pages` *history* (the flip stopped new bloat; ~9 GiB of old history remains). Destructive force-push, invalidates clones, needs all workflows paused — a deliberate quiet-weekend task. Runbook in `PHASE1_DB_BRANCH_RUNBOOK.md` §Stage C/filter-repo.
- **Monthly squash** of the `data` branch as it accumulates daily DB commits.
- **Delete the `actions.yml` parachute** after a stretch of clean refresh runs.

## Incidental fixes shipped alongside

- **SPCX (SpaceX) added** day-one with IPO handling: when EDGAR's structured API is empty (fresh IPO / foreign filer), the build falls back to Yahoo's income statement (the S-1 prospectus financials) and renders a reduced page; the normal EDGAR path resumes automatically once `companyfacts` populates.
- **Adaptive chart money units** ($B / $M / $K by company size) so small caps no longer render every bar as "0".
- **Homepage `viewport` meta tag** (mobile rendering); removed the delisted `ME` failure path is tracked (ME has no CIK).
