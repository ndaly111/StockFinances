# StockFinances pipeline rethink — Phase 1: kill dead per-ticker output

- **Date:** 2026-06-06
- **Status:** Approved (design); ready for implementation plan
- **Scope of this doc:** Phase 1 only. Phases 2–3 are summarized for context but specced separately.

## Background

The site (nicksstockfinancials.com) has already migrated its front end to a
**single static shell + per-ticker JSON** model:

- `index.html` links to `ticker.html?t=<TICKER>`.
- `ticker.html` is a static shell that loads `ticker_data/<TICKER>.json` and
  renders everything with Plotly. It references **only** `ticker_data/*.json`
  — no `charts/*.png`, no `pages/*.html`.
- The per-ticker JSON is produced by `scripts/gen_ticker_json.py`, which runs
  `scripts/build_aapl_mockup.py` once per ticker (TICKER substituted).

Meanwhile `main_remote.py` still runs a 102-ticker loop that renders the
**legacy** desktop artifacts — matplotlib `charts/*.png` and per-ticker
`pages/*_page.html` — that nothing on the live site links to anymore. This
legacy rendering is the bulk of the ~15–17 min "Generate pages" workflow step.

Verified during exploration:
- Nothing references `pages/*_page.html` (orphaned).
- `index.html`, `spy_growth.html`, `qqq_growth.html`, `matchups.html` reference
  zero `charts/*.png`.
- `build_aapl_mockup.py` emits zero `.png` (all Plotly) and does only 2 DB reads
  — it re-fetches most data live (EDGAR/yfinance) per ticker (the basis for
  Phase 2).

## Phased plan (context)

1. **Phase 1 (this doc): kill dead output.** Stop generating the orphaned
   per-ticker pages/charts. No data-flow change. Runtime win, low risk.
2. **Phase 2: DB-backed build.** Refactor `build_aapl_mockup.py` to read the DB
   (populated by the data layer) instead of re-fetching live. Removes duplicate
   fetching; makes the build near-instant and **parallel-safe** (no per-ticker
   network → no rate limits, which is what broke the earlier parallel attempt).
3. **Phase 3: re-cadence + parallelize.** Run the now-cheap build **daily** (the
   agreed freshness target — daily at market close), parallelize it safely, and
   tidy `main_remote.py` into data-only jobs.

## Relationship to the overall goal

The objective has two halves:
1. **Stop writing individual ticker pages.** Delivered by Phase 1.
2. **A data pipeline that stores into JSON so the charts show the right data.**
   Delivered by Phase 2 (build the JSON from the DB, the single source of truth,
   instead of re-fetching live and duplicating `main_remote`'s work).

Phase 1 is the safe enabler, not the payoff. Two concrete ways it pays into the
"right data" goal rather than being throwaway cleanup:
- The **verification harness** built here (regenerate all JSON + live pages,
  diff against a baseline) is **reused in Phase 2 as the correctness oracle** —
  it's how we'll prove the DB-backed build produces the *same or better* data,
  not subtly different charts.
- Removing the dead per-ticker render path shrinks `main_remote` so Phase 2's
  refactor has a smaller, clearer surface and the data-writes are easier to
  isolate from rendering.

## Phase 1 goal

Stop `main_remote.py` from producing per-ticker artifacts that no live page
consumes, **without changing any data** the JSON shells or live site-level pages
depend on. Reduce the "Generate pages" step substantially.

## Core safety principle

Classification ("this looks like rendering, drop it") is **not trustworthy**
here — confirmed during exploration:
- `generate_financial_charts` is dual-purpose (writes DB **and** renders).
- Several data functions (`valuation_update`, `annual_and_ttm_update`,
  `fetch_and_update_balance_sheet_data`, `build_segments_for_ticker`) write the
  DB via helpers / multi-line SQL that a naive scan misses.

Therefore the design is built around an **output-identity guardrail**, not
around classification. The acid test: because the shells read the DB + live data
(never the dead artifacts), removing dead rendering **must** leave the JSON and
the live site-level pages byte-identical. Any diff means that step was actually
feeding output and must be kept (or split).

## Approach

### 1. Verification harness (build first)

A reproducible script that regenerates the full output set and snapshots it:
- All `ticker_data/*.json` (run `gen_ticker_json.py` over `tickers.csv`).
- The live site-level pages main_remote emits: dashboard table / `index`
  fragment, `spy_growth`, `qqq_growth`, `daily-market-summary`, earnings tables.

It produces a normalized snapshot (stable JSON key ordering; ignore volatile
fields such as live price/timestamps that legitimately move between runs — these
are enumerated explicitly, not hand-waved) so before/after can be compared.

Run it twice on unchanged code first to confirm the harness itself is stable
(only the enumerated volatile fields differ). This calibrates the diff before
any change.

### 2. Remove legacy rendering behind a reversible flag

Introduce `RENDER_LEGACY` (env, default `0`). Gate the per-ticker legacy
rendering so the data path is untouched and rollback is instant:
- Per-ticker loop legacy renders to gate off: financial charts, forecast
  charts/tables, balance-sheet chart, expense reports, the old per-ticker tile
  HTML (`generate_html_table` → `charts/{t}_ticker_info.html`), and per-ticker
  page HTML (`create_html_for_tickers` → `pages/{t}_page.html`).
- Post-loop legacy renders to gate off: the old matplotlib implied-growth PNG
  summaries (`generate_all_summaries` in `implied_growth_summary.py`) and the
  EPS-dividend chart.

**Dual-purpose functions are split, not dropped.** Any function that both writes
the DB and renders (e.g. `generate_financial_charts`) is refactored so the DB
write always runs and only the rendering is gated. The percentile side-table
(`Index_Growth_Pctile`, written inside `generate_all_summaries`) is preserved —
its write is extracted from the PNG rendering.

**Keep (data + live site-level output), never gated:** `annual_and_ttm_update`,
`fetch_and_update_balance_sheet_data`, `build_segments_for_ticker`,
`prepare_data_for_display` (implied-growth + forward-EPS DB logging),
`valuation_update`, forward EPS/revenue history snapshots, dashboard table,
`index_growth`, index-growth backfill + SPY/QQQ growth charts (used by the live
spy/qqq pages), `generate_market_summary`, earnings tables.

### 3. Assert identity, then deploy

- Locally: harness snapshot with `RENDER_LEGACY=1` (baseline) vs `=0` (changed).
  Require **identical** JSON + live site-level pages (modulo enumerated volatile
  fields). Any unexpected diff → that step feeds output → ungate/split it.
- Deploy only after local identity passes. Then verify the live run builds
  **101/101** tickers (the known-good count) and spot-check GOOGL (real series)
  + a synthetic ticker before considering it done.

## Out of scope (Phase 1)

- No change to `build_aapl_mockup.py` data sourcing (Phase 2).
- No cadence change; daily JSON freshness is Phase 3.
- No parallelization (Phase 3, and only once Phase 2 makes the build
  network-free).
- No deletion of the legacy code yet — it is flag-gated so it can be removed in
  a later cleanup once Phases 2–3 prove the new path end to end.

## Risks & rollback

- **Risk:** a gated render also wrote the DB → shells lose data. **Mitigation:**
  identity test catches it (JSON changes); dual-purpose functions are split.
- **Risk:** harness misses a consumer of a legacy artifact. **Mitigation:** the
  harness covers every live page; `pages/*.html` confirmed orphaned.
- **Rollback:** set `RENDER_LEGACY=1` (or revert the one workflow env line) —
  instant, no code revert needed.

## Success criteria

1. Local: `RENDER_LEGACY=0` output is identical to `=1` (modulo enumerated
   volatile fields) for all ticker JSON + live site-level pages.
2. Live: weekly run still builds 101/101 tickers; GOOGL + a synthetic ticker
   verified correct; site visually unchanged.
3. "Generate pages" step time materially reduced.
4. Change is flag-gated and reversible.
