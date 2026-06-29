# EPS Click-to-Measure Tool + Indexed-Growth Chart (SPY/QQQ)

**Date:** 2026-06-28
**Status:** Design — approved, ready to plan
**Repo:** StockFinances (`C:\Users\ndaly\projects\sf-fix`, branch `forward-index-eps`)

---

## 1. Goal

Add an interactive **click-to-measure %-change tool** to the SPY/QQQ EPS charts, plus a
new **EPS-growth-indexed-to-100** chart, so you can read the % change between any two
points (year-over-year = click points a year apart) and see cumulative growth from the
start. All interactive Bokeh in the existing valuation bundle — no static images, no
server.

## 2. Scope (decided)

- The tool goes on **two panels only**: the **EPS (level, $)** chart and a **new EPS
  Growth (indexed to 100)** chart.
- The **P/E** and **Implied Growth** panels are **untouched** (no measure tool — they're
  rates/ratios where a "% gain" and log axis don't cleanly apply).
- SPY and QQQ both.

## 3. Non-goals (YAGNI)

- No measure tool on P/E or Implied Growth panels.
- No window-relative "first point" in v1 — "first point" = first point of the full
  series (fixed base). Window-relative recompute is a possible later enhancement.
- No change to the forward-EPS data pipeline (that ships separately, already on main).

## 4. The two panels

### 4.1 EPS (level, $) — existing panel, upgraded
- **Log y-axis** with **clean tick labels** (`$30 · $40 · $60 · $100`, not `6×10¹`).
  Bokeh: a `FuncTickFormatter`/`CustomJSTickFormatter` (or `NumeralTickFormatter`
  `"$0,0"`) on a `LogAxis`. Fall back to linear if the series has non-positive values
  (index EPS is always positive; guard anyway).
- Keeps the forward-EPS point + dashed connector already built.
- Gets the measure tool (§5).

### 4.2 EPS Growth (indexed to 100) — NEW panel
- Series = `eps_s / eps_s.iloc[0] * 100` (rebased so the first point = 100). The y-axis
  then reads as cumulative gain: 100 = start, 150 = +50%, 200 = +100%.
- **Log y-axis**, clean tick labels (`100 · 150 · 200 · 300`).
- Forward point indexed the same way (`forward_eps_index / eps_s.iloc[0] * 100`),
  dashed connector off the end.
- Gets the measure tool (§5).
- Placed in the bundle right after the EPS panel. (Note: on a log axis this is the same
  curve shape as the EPS panel, relabeled — a deliberate, accepted reframing where the
  axis reads directly as % gain.)

## 5. The measure tool (identical on both EPS panels)

All client-side Bokeh `CustomJS` (works in the static gh-pages export).

- **Default / hover:** a readout `Div` under the chart shows
  `% from first point: +X%` — `(hovered_value / first_value − 1)`, updating as you
  hover. First point = first of the full series.
- **Click two points:**
  - Click A → A becomes the anchor; render a distinct marker on A.
  - Click B → readout shows `A → B: +Y%  ·  <span>` where `Y% = B/A − 1` and `<span>`
    is the elapsed time (e.g. "3.0 yrs" / "11 mo"); render a marker on B + a faint
    connector A–B.
  - A **third click** starts a new measurement (that point becomes the new A).
- **Clear button:** a Bokeh `Button` ("✕ Clear") that resets the anchor/markers and
  returns the readout to the default "% from first point" mode.
- Taps select the nearest data point via a `TapTool` on the panel's circle/scatter
  renderer (the EPS block already has a dot glyph backed by a `ColumnDataSource` with
  `date`/`value`).

## 6. Architecture

- **`_attach_measure_tool(fig, source, readout_div, clear_button, value_label)`** — new
  reusable helper in `index_growth_charts.py`. Wires the `TapTool`, the tap `CustomJS`
  (anchor/measure state + marker `ColumnDataSource` + readout text), the hover
  `CustomJS` (% from first point), and the Clear `Button` `CustomJS`. Returns nothing;
  mutates `fig`. Used by both EPS panels.
- **`_clean_log_formatter()`** — small helper returning the Bokeh tick formatter for a
  `$`-style or plain log axis (so both panels share clean labels).
- **Indexed series**: built inline in `render_index_growth_charts` from the existing
  `eps_s` (`_series_eps`), rebased to 100; a new chart block via the existing
  `_build_chart_block` (with `log_axis=True`), then `_attach_measure_tool` + the forward
  overlay (reuse `_add_forward_eps_overlay`, indexing the forward point).
- `_build_chart_block` may need a small extension to accept/emit the per-block measure
  widgets (readout Div + Clear Button) into the block's column layout, or the helper
  appends them to `fig`'s surrounding layout. Keep the block API change minimal.

## 7. Data flow

`render_index_growth_charts(tk)` → builds `eps_s` (existing) → (a) EPS block: set
`log_axis`, clean formatter, attach measure tool; (b) indexed block: `eps_s/first*100`,
log, clean formatter, forward point, attach measure tool. P/E + Implied Growth blocks
unchanged. Bundle = [Implied Growth, P/E, EPS (+tool), EPS-Indexed (+tool)].

## 8. Edge cases

- **Non-positive / empty EPS series** → skip log (linear) and/or skip the indexed panel;
  never divide by a zero/NaN first value (guard `eps_s.iloc[0] > 0`).
- **Single-point series** → no measurement possible; tool no-ops gracefully, readout
  shows nothing.
- **Forward point absent** (low-coverage row withheld) → panels still render history +
  measure tool; just no forward point. (Independent of the validation gate.)
- **Tap on empty canvas** (no nearest point) → ignore.

## 9. Testing

- **Python-unit:** indexed-series transform (`rebase to 100`, base-guard); the EPS and
  indexed blocks are constructed with `log_axis=True`; `_attach_measure_tool` adds the
  expected renderers/widgets (marker source, Clear Button, readout Div) to the figure;
  clean-log-formatter helper returns a formatter; the indexed forward point math.
- **Not unit-tested (JS):** the CustomJS tap/hover/clear *behavior* — verified by
  rendering the bundle and manual click-through (or a brief note in the plan to
  eyeball it on the built page).
- Follow existing `Test/test_index_growth_charts.py` patterns (mock `_build_chart_block`
  etc.).

## 10. Runtime / deploy

- Pure chart-layer change in `index_growth_charts.py`; rebuilds on the weekly
  `site-refresh.yml` / dispatch (same as the other growth-page charts).
- No new data, no schema, no pipeline change. Deploy/push stays with Nick.

## 11. Decisions locked

- Tool on EPS + new EPS-indexed-to-100 only; P/E and Implied Growth untouched.
- Log axis on both EPS panels, clean tick labels.
- Indexed base = first point of the full series (fixed); "% from first point" relative
  to it. Click A→B = `B/A−1`; third click restarts; Clear resets.
- All client-side Bokeh CustomJS.
