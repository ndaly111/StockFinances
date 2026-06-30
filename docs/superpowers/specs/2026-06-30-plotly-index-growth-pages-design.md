# Plotly SPY/QQQ Growth Pages — Replace the Bokeh Charts

**Date:** 2026-06-30
**Status:** Design — approved, ready to plan
**Repo:** StockFinances (`C:\Users\ndaly\projects\sf-fix`, branch `forward-index-eps`)

---

## 1. Goal

Replace the glitchy/heavy **Bokeh** valuation bundle on `spy_growth.html` /
`qqq_growth.html` with **Plotly** charts that match the rest of the site (the ticker
pages, economic page, etc. are all Plotly). Build on the existing-but-unwired
`generate_index_growth_pages.py` (already a clean Plotly implementation of these exact
pages), port in the forward-EPS forecast + the click-to-measure tool, and retire the
Bokeh module entirely.

## 2. Why

- Bokeh is the **lone non-Plotly holdout** on the site; every interaction needs
  hand-written CustomJS, which produced the "????" entity bug, the custom log-tick
  formatter, and the general glitchiness. The bundle is ~1.4 MB/page.
- Plotly gives zoom/pan/hover/rangeselector/rangeslider/log **natively** → the whole
  class of hand-rolled-interaction bugs disappears, and the pages look like the rest of
  the site.
- A near-complete Plotly version already exists (`generate_index_growth_pages.py`) — it
  was just never wired into `main_remote`.

## 3. Non-goals

- No change to the **data pipeline** (bottom-up forward EPS, `Index_Forward_EPS_History`,
  the validation gate). Purely a rendering-layer swap.
- No change to the homepage overview table, P/E or implied-growth *data*.
- Not adopting a third charting library (TradingView etc.) — consistency with the
  existing Plotly stack is the point.

## 4. Existing draft (`generate_index_growth_pages.py`) — what it already does

- `generate_index_growth_pages(db_path)` → for SPY & QQQ → `_load_series` →
  `_build_one` → builds Plotly figures via `plotly.io.to_html`, assembles a clean
  self-contained HTML page (`_page_html`: system fonts, card layout, stats table,
  mobile viewport) → writes `spy_growth.html` / `qqq_growth.html`.
- **Implied Growth (TTM + Forward)** figure with rangeselector (1M/3M/6M/All),
  rangeslider, and a Daily/Weekly/Monthly toggle (`_growth_figure`).
- **EPS** figure (`_eps_figure`) with `$` tick prefix, rangeselector/slider/toggle.
- **Stats table** 1/3/5/10-yr (avg/median/std/current/percentile) for TTM & Forward
  implied growth (`_timeframe_table_html`).
- Plotly via CDN (`include_plotlyjs="cdn"`), self-contained pages.

## 5. What we add / change

### 5.1 EPS figure → log axis + forward-EPS forecast
- Set the EPS y-axis `type="log"` (Plotly renders clean log ticks natively — no custom
  formatter, no "????").
- Read the latest **displayable** row from `Index_Forward_EPS_History` (port the
  `_latest_forward_eps` read from `index_growth_charts.py`: newest row, `displayable=1`,
  non-null `forward_eps_index`; returns `forward_eps_index`, `growth_next_fy`,
  `horizon_date`, `growth_this_fy`, `coverage_weight`).
- Add a dashed forecast trace from the last historical EPS through two diamond markers:
  **this-FY** (`forward_eps_index` at `horizon_date`, ~+1yr) and **next-FY**
  (`forward_eps_index*(1+growth_next_fy)` at ~+2yr). Skip gracefully if no displayable
  row.

### 5.2 New EPS-Growth-indexed-to-100 figure
- New `_eps_indexed_figure`: rebase the EPS series to 100 at its first point
  (`s/s.iloc[0]*100`, guard base>0), `type="log"`, same rangeselector/slider. Forward
  forecast points scaled the same way. Y-axis reads as cumulative % gain.

### 5.3 Click-to-measure tool (ported to Plotly)
- Render the EPS and indexed figures with **known `div_id`s** (e.g. `eps-chart`,
  `eps-indexed-chart`) via `to_html(..., div_id=...)`.
- Inject one small `<script>` into `_page_html` that, for each measured chart div, binds
  `gd.on('plotly_click', handler)`:
  - Click A → record (x=date ms, y=value), show marker (via `Plotly.relayout`
    annotation or a marker trace), readout "Anchor set. Click another point."
  - Click B → order chronologically (earlier=start), compute `pct = end/start - 1`,
    `years` span, `annualized = (end/start)^(1/years) - 1` (guard tiny spans), set a
    readout `<div>` to `+14.5% over 1.5 years (+9.6% annualized)`; third click restarts.
  - Hover (`plotly_hover`) → `% from first point: +X%` when not mid-measure.
  - A `Clear` `<button>` resets.
- **All strings plain ASCII** (no HTML entities, no unicode separators) — the "????"
  failure mode cannot recur.
- One readout `<div>` + `Clear` `<button>` per measured chart, placed under each.

### 5.4 Forward-growth callout
- A short text line under the EPS chart: "Forward earnings growth (bottom-up): +X% this
  fiscal year, +Y% next. Based on N% of index weight." (from the displayable row;
  omitted when withheld). Reuse the wording from `_forward_eps_callout_bottomup`.

### 5.5 Page styling / consistency
- Keep `_page_html`'s clean card layout; ensure it visually matches the ticker pages
  (fonts/colors already close). Keep the "← Back to Dashboard" link.

## 6. Integration

- `main_remote.mini_main`: replace the `for idx in ("SPY","QQQ"):
  render_index_growth_charts(idx)` block with a single
  `generate_index_growth_pages(DB_PATH)` call (wrapped in try/except like the other
  steps so a failure can't break the build).
- `html_generator2.render_spy_qqq_growth_pages()` (the Bokeh template shell) →
  removed/neutered; the Plotly generator now owns `spy_growth.html`/`qqq_growth.html`.
- **Deploy:** the generator writes the same output filenames that are rsynced to
  gh-pages, so the deploy path is unchanged. The `*_valuation_bundle_chart.*` artifacts
  stop being produced.

## 7. Retire

- `index_growth_charts.py` (Bokeh) — delete (and its tests
  `Test/test_index_growth_charts.py`), after moving any still-needed data helpers
  (`_series_eps` logic, `_INDEX_EPS_DIVISOR`, the forward-EPS read) into the Plotly
  module or a small shared helper.
- `templates/spy_growth_template.html` / `qqq_growth_template.html` (Bokeh shells).
- Bundle artifacts `charts/{spy,qqq}_valuation_bundle_chart.{js,html}` (and the other
  `{spy,qqq}_{growth,pe,eps}_chart*` Bokeh fragments) — stop generating; remove from the
  repo/gh-pages.
- Note: the in-flight Bokeh "????" fix (run #516) becomes moot once Bokeh is retired;
  fine to let it land so the current page isn't broken in the interim.

## 8. Data

- Reuse the draft's `_load_series` / `_load_eps_series` (verify in the plan that they
  read the same `Index_Growth_History` / `Index_PE_History` / `Index_EPS_History` data;
  adapt if the draft's table-detection diverges from the live schema).
- Add a `_latest_forward_eps(conn, ticker)` reader (ported) for the forecast points +
  callout.
- The EPS series should use the same index-level construction as today's
  `_series_eps` (TTM_REPORTED → TTM_DAILY → IMPLIED_FROM_PE×divisor); port that logic so
  the Plotly EPS chart matches the current data.

## 9. Edge cases

- No displayable forward row → render history only (no forecast diamonds, no callout);
  measure tool still works.
- Non-positive / empty EPS → skip log (linear) and/or skip the indexed chart
  (guard base>0); never divide by a zero base.
- Forward fetch/JS errors → never break the page; the script is defensive and the
  Python generation is try/except-wrapped in `main_remote`.
- Plotly CDN unavailable → charts won't render, but that's the same risk as the
  ticker pages (acceptable, consistent).

## 10. Testing

- **Python-unit:** the EPS log + forecast trace is added when a displayable row exists;
  the indexed transform rebases to 100; `_latest_forward_eps` read; the page HTML
  contains the measured chart `div_id`s, the readout div, the Clear button, and the
  measure `<script>`; figures have the expected traces. (Follow existing test patterns;
  these are constructive/HTML-substring assertions — Plotly figures are inspectable
  `go.Figure` objects.)
- **Not unit-tested (JS):** the `plotly_click`/hover/clear behavior — verified by
  rendering a page and eyeballing in a browser post-deploy.
- Remove the Bokeh `Test/test_index_growth_charts.py` when that module is deleted.

## 11. Deploy / runtime

- Growth pages rebuild on the weekly `site-refresh.yml` (or dispatch), same as now.
- No new data, no schema change. Plotly already a project dependency.
- Deploy/push stays with Nick.

## 12. Decisions locked

- Plotly, building on `generate_index_growth_pages.py`; retire Bokeh entirely.
- Charts: Implied Growth (TTM+Fwd), P/E, EPS (log, +forecast, +measure), EPS-indexed-100
  (log, +measure), stats table, forward-growth callout.
- Measure tool ported to Plotly via `plotly_click`, plain-ASCII, with Clear + hover
  "% from first point".
