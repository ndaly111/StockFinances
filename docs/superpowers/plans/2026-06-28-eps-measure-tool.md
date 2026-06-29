# EPS Click-to-Measure Tool + Indexed-Growth Chart — Build Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Add a log-scale + click-to-measure %-change tool to the SPY/QQQ EPS chart, and a new EPS-indexed-to-100 chart with the same tool. Interactive Bokeh CustomJS, no server.

**Architecture:** All in `index_growth_charts.py`. A `_clean_log_formatter()` helper, an `_indexed_series()` transform, an `_attach_measure_tool(...)` helper (TapTool + selection CustomJS + marker glyph + readout Div + Clear Button + hover "% from first" CustomJS), and a `measure=True` path in `_build_chart_block` that includes the readout Div + Clear Button in the block's card. Wire both EPS panels in `render_index_growth_charts`. P/E + Implied Growth untouched.

**Tech:** Bokeh 3.x, pandas, pytest. Tests from repo root: `python -m pytest <p> -v`. Git: `git -C "C:/Users/ndaly/projects/sf-fix"`.

**Spec:** `docs/superpowers/specs/2026-06-28-eps-measure-tool-design.md`.

**Key existing facts:** `_build_chart_block(series, title, ylab, percent_axis, x_range, ..., log_axis=False, ..., marker_alpha, marker_size)` builds a `ColumnDataSource` `source` with `date`/`value` (+ yoy), a `fig` with a line + circle dots, tools, and a `column(...)` card (`block_children`); returns `ChartBlock(layout, fig, source, ...)`. The EPS block is built in `render_index_growth_charts` (last `blocks.append(...)`), already log-capable (`eps_log_axis`). `bokeh.models` imports in the file already include `CustomJS, Div, Button, ColumnDataSource, NumeralTickFormatter`. `TapTool` likely needs adding to the import.

---

## Task 1: clean log tick formatter

**Files:** `index_growth_charts.py`; `Test/test_index_growth_charts.py`.

- [ ] **Step 1 — failing test:**
```python
def test_clean_log_formatter_returns_formatter():
    f_money = igc._clean_log_formatter(money=True)
    f_plain = igc._clean_log_formatter(money=False)
    from bokeh.models import CustomJSTickFormatter
    assert isinstance(f_money, CustomJSTickFormatter)
    assert "$" in f_money.code
    assert isinstance(f_plain, CustomJSTickFormatter)
```
- [ ] **Step 2 — run, expect FAIL** (`-k clean_log_formatter`).
- [ ] **Step 3 — implement** (add near the other helpers; ensure `from bokeh.models import CustomJSTickFormatter` is imported):
```python
def _clean_log_formatter(money: bool = False):
    """Tick formatter for a log axis that prints plain numbers ($30, $60, 100, 150)
    instead of scientific notation (6x10^1)."""
    from bokeh.models import CustomJSTickFormatter
    pre = "'$'" if money else "''"
    return CustomJSTickFormatter(code=f"""
        const v = tick;
        if (v <= 0) return '';
        const s = (v >= 100) ? v.toFixed(0) : v.toFixed(0);
        return {pre} + Number(s).toLocaleString();
    """)
```
- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(charts): clean log-axis tick formatter`.

---

## Task 2: indexed-to-100 series transform

**Files:** `index_growth_charts.py`; `Test/test_index_growth_charts.py`.

- [ ] **Step 1 — failing test:**
```python
def test_indexed_series_rebases_to_100():
    s = pd.Series([50.0, 75.0, 100.0],
                  index=pd.to_datetime(["2024-01-01","2024-06-01","2024-12-01"]))
    out = igc._indexed_series(s)
    assert list(out.round(2)) == [100.0, 150.0, 200.0]

def test_indexed_series_guards_bad_base():
    import numpy as np
    assert igc._indexed_series(pd.Series(dtype=float)).empty
    s = pd.Series([0.0, 5.0], index=pd.to_datetime(["2024-01-01","2024-02-01"]))
    assert igc._indexed_series(s).empty   # base <= 0 -> empty (can't index)
```
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement:**
```python
def _indexed_series(series):
    """Rebase a positive series so its first value = 100 (cumulative-growth view)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or s.iloc[0] <= 0:
        return pd.Series(dtype=float)
    return s / s.iloc[0] * 100.0
```
- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(charts): EPS indexed-to-100 transform`.

---

## Task 3: `_attach_measure_tool` helper

**Files:** `index_growth_charts.py`; `Test/test_index_growth_charts.py`.

This adds: a marker `ColumnDataSource` + scatter glyph, a `TapTool` on the dot renderer, a `CustomJS` on `source.selected` that toggles anchor→measure→restart and writes the readout, a hover `CustomJS` for "% from first point", a Clear `Button`, and a readout `Div`. Returns `(readout_div, clear_button)` so the caller can place them in the block card.

- [ ] **Step 1 — failing test** (verifies wiring, not JS behavior):
```python
def test_attach_measure_tool_adds_widgets():
    from bokeh.plotting import figure as _fig
    from bokeh.models import ColumnDataSource, Button, Div, TapTool
    fig = _fig(x_axis_type="datetime")
    src = ColumnDataSource(data={"date":[1,2,3],"value":[10.0,12.0,15.0]})
    dots = fig.scatter("date","value", source=src)
    before = len(fig.renderers)
    div, btn = igc._attach_measure_tool(fig, src, dots, money=True)
    assert isinstance(div, Div) and isinstance(btn, Button)
    assert any(isinstance(t, TapTool) for t in fig.tools)
    assert len(fig.renderers) > before          # marker glyph added
```
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** (ensure `TapTool` is imported from `bokeh.models`):
```python
def _attach_measure_tool(fig, source, dot_renderer, money=False):
    """Click-to-measure %-change tool: tap two points to read B/A-1; hover shows
    % from first point; Clear button resets. All client-side CustomJS."""
    from bokeh.models import (ColumnDataSource, CustomJS, Div, Button, TapTool,
                              HoverTool)
    unit = "$" if money else ""
    marker = ColumnDataSource(data={"date": [], "value": [], "kind": []})
    fig.scatter("date", "value", source=marker, size=14, line_width=3,
                fill_alpha=0.0, line_color="#d62728")
    seg = ColumnDataSource(data={"x0": [], "y0": [], "x1": [], "y1": []})
    fig.segment("x0", "y0", "x1", "y1", source=seg, line_color="#555",
                line_dash="dotted")
    anchor = ColumnDataSource(data={"date": [0], "value": [0], "set": [0]})
    readout = Div(text="<i>Click a point, then another, to measure % change.</i>",
                  sizing_mode="stretch_width", styles=META_STYLE)

    tap_cb = CustomJS(args=dict(source=source, marker=marker, seg=seg, anchor=anchor,
                                readout=readout, unit=unit), code="""
        const inds = source.selected.indices;
        if (!inds.length) return;
        const i = inds[0];
        const d = source.data.date[i], v = source.data.value[i];
        if (anchor.data.set[0] === 0) {
            anchor.data.date=[d]; anchor.data.value=[v]; anchor.data.set=[1];
            marker.data = {date:[d], value:[v], kind:['A']};
            seg.data = {x0:[],y0:[],x1:[],y1:[]};
            readout.text = "Anchor set — click another point.";
        } else {
            const ad = anchor.data.date[0], av = anchor.data.value[0];
            const pct = (v/av - 1)*100;
            const yrs = Math.abs(d-ad)/(365.25*86400000);
            const span = yrs >= 1 ? yrs.toFixed(1)+" yrs" : Math.round(yrs*12)+" mo";
            marker.data = {date:[ad,d], value:[av,v], kind:['A','B']};
            seg.data = {x0:[ad],y0:[av],x1:[d],y1:[v]};
            readout.text = "<b>A → B: "+(pct>=0?"+":"")+pct.toFixed(1)+
                           "%</b> &nbsp;·&nbsp; over "+span;
            anchor.data.set=[0];
        }
        marker.change.emit(); seg.change.emit(); anchor.change.emit();
        source.selected.indices = [];
    """)
    source.selected.js_on_change("indices", tap_cb)
    fig.add_tools(TapTool(renderers=[dot_renderer]))

    hover_cb = CustomJS(args=dict(source=source, anchor=anchor, readout=readout,
                                  unit=unit), code="""
        if (anchor.data.set[0] === 1) return;     // measuring; don't overwrite
        const idx = cb_data.index.indices;
        if (!idx.length) return;
        const i = idx[0];
        const first = source.data.value[0];
        const v = source.data.value[i];
        if (!first) return;
        const pct = (v/first - 1)*100;
        readout.text = "% from first point: <b>"+(pct>=0?"+":"")+pct.toFixed(1)+"%</b>";
    """)
    fig.add_tools(HoverTool(renderers=[dot_renderer], tooltips=None, callback=hover_cb))

    clear = Button(label="✕ Clear", button_type="default", width=90)
    clear.js_on_click(CustomJS(args=dict(marker=marker, seg=seg, anchor=anchor,
                                         readout=readout), code="""
        marker.data={date:[],value:[],kind:[]};
        seg.data={x0:[],y0:[],x1:[],y1:[]};
        anchor.data={date:[0],value:[0],set:[0]};
        marker.change.emit(); seg.change.emit(); anchor.change.emit();
        readout.text="<i>Click a point, then another, to measure % change.</i>";
    """))
    return readout, clear
```
- [ ] **Step 4 — run, expect PASS** (`-k attach_measure_tool`). **Step 5 — commit** `feat(charts): click-to-measure tool helper (tap %, from-first hover, clear)`.

---

## Task 4: `measure=True` path in `_build_chart_block`

**Files:** `index_growth_charts.py`; `Test/test_index_growth_charts.py`.

- [ ] **Step 1 — failing test:**
```python
def test_build_chart_block_measure_includes_widgets():
    s = pd.Series([10.0,12.0,15.0], index=pd.to_datetime(["2024-01-01","2024-06-01","2024-12-01"]))
    blk = igc._build_chart_block(s, "EPS", "EPS ($)", False, None,
                                 log_axis=True, measure=True, money=True)
    # the card layout should contain the Clear button + readout div somewhere
    from bokeh.models import Button, Div
    found = {"button": False, "div_readout": False}
    def walk(m):
        from bokeh.models import Column, Row
        if isinstance(m, Button): found["button"]=True
        if isinstance(m, Div) and "measure" in (m.text or "").lower(): found["div_readout"]=True
        for ch in getattr(m, "children", []): walk(ch)
    walk(blk.layout)
    assert found["button"] and found["div_readout"]
```
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement:** add `measure: bool = False, money: bool = False` params to `_build_chart_block`. After the `fig.line(...)` + dots (`dots = fig.circle(...)`) are created and before/around the `block_children` assembly, when `measure` is True call `_attach_measure_tool(fig, source, dots, money=money)` to get `(readout_div, clear_button)`, apply `fig.yaxis.formatter = _clean_log_formatter(money=money)` when `log_axis`, and insert `clear_button` (in a `row`) + `readout_div` into `block_children` (e.g. right after the title or under the fig). Keep non-measure behavior unchanged. (The existing dots variable is `dots = fig.circle(...)`; pass that as `dot_renderer`.)
- [ ] **Step 4 — run, expect PASS;** also run the whole chart test file (existing tests still pass — the new params default off). **Step 5 — commit** `feat(charts): measure=True block path (clean log ticks + tool widgets)`.

---

## Task 5: wire both EPS panels in `render_index_growth_charts`

**Files:** `index_growth_charts.py`; `Test/test_index_growth_charts.py`.

- [ ] **Step 1 — implement (integration):**
  1. EPS panel: in the existing EPS `blocks.append(_build_chart_block(eps_s, eps_title, "EPS ($)", False, common_range, log_axis=eps_log_axis, ...))`, add `measure=True, money=True`. (Keep the forward overlay that follows.)
  2. New indexed panel: after the EPS block + its forward overlay, build:
```python
    eps_idx = _indexed_series(eps_s)
    if not eps_idx.empty:
        idx_block = _build_chart_block(
            eps_idx, f"{tk} EPS Growth (indexed = 100)", "EPS (start=100)",
            False, common_range, log_axis=True, measure=True, money=False,
            controls=_make_controls_row())
        blocks.append(idx_block)
        if fwd_row and idx_block.fig is not None and fwd_row.get("forward_eps_index") is not None:
            base = float(eps_s.iloc[0])
            if base > 0:
                try:
                    _add_forward_eps_overlay(
                        idx_block.fig, last_date=eps_idx.index[-1],
                        last_eps=float(eps_idx.iloc[-1]),
                        forward_date=pd.Timestamp(fwd_row["horizon_date"]),
                        forward_eps_index=fwd_row["forward_eps_index"]/base*100.0)
                except Exception as exc:
                    print(f"[WARN] indexed forward overlay failed for {tk}: {exc}")
```
  Place this so `idx_block` participates in the shared range/auto-Y `chart_refs` (i.e. before the `chart_refs = [...]` line, same as other blocks).
- [ ] **Step 2 — update integration test:** extend `test_render_applies_forward_overlay_when_data_present` (or add a new test) to assert that with EPS data + measure, the render produces a block whose title contains "indexed" (patch `_build_chart_block` capture to record titles; assert an "indexed" title was requested). Keep existing assertions.
- [ ] **Step 3 — run** `python -m pytest Test/test_index_growth_charts.py -v` — all pass.
- [ ] **Step 4 — commit** `feat(charts): log EPS + indexed-growth panels with measure tool (SPY/QQQ)`.

---

## Task 6: real render verification

**Files:** none (verify).

- [ ] **Step 1 —** `python -m pytest Test/ -q` — confirm only the 4 known pre-existing failures.
- [ ] **Step 2 — render the real bundle** against a scratch DB copy and confirm artifacts build without error:
```bash
python -c "import shutil, os; shutil.copy('Stock Data.db','_r.db'); import index_growth_charts as igc; igc.DB_PATH='_r.db'; igc.render_index_growth_charts('QQQ'); print('QQQ rendered'); igc.render_index_growth_charts('SPY'); print('SPY rendered')"
```
  Confirm it prints both "rendered" with no exception, and that `charts/qqq_valuation_bundle_chart.js` / `_div.html` were updated (mtime). Delete `_r.db`(+wal/shm). Report any error.
- [ ] **Step 3 — JS behavior note:** the tap/hover/clear CustomJS can only be verified by opening the built `qqq_growth.html` in a browser (deploy or local). Note for Nick: after deploy, eyeball that clicking two EPS points shows the A→B %, hover shows % from first, and Clear resets. If a CustomJS error appears in the browser console, that's the thing to fix (the Python render won't catch JS errors).
- [ ] **Step 4 —** `git status --porcelain` clean (no `_r.db`). Report.

---

## Notes
- The CustomJS is the only part not unit-tested (it runs in the browser). Build it as written, render to confirm no Python error, then verify interactivity in-browser post-deploy.
- Deploy stays with Nick; growth pages rebuild on weekly `site-refresh.yml`/dispatch.
- If `TapTool`/`HoverTool`/`CustomJSTickFormatter` aren't already imported in `index_growth_charts.py`, add them to the `from bokeh.models import (...)` block.
