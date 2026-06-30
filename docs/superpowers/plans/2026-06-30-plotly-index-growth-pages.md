# Plotly SPY/QQQ Growth Pages — Build Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Replace the Bokeh SPY/QQQ growth charts with Plotly by finishing/wiring `generate_index_growth_pages.py`, adding a log EPS chart with the forward-EPS forecast, an indexed-to-100 chart, and a `plotly_click` measure tool; then retire the Bokeh module.

**Architecture:** All rendering moves to `generate_index_growth_pages.py` (Plotly, self-contained HTML written to `spy_growth.html`/`qqq_growth.html`). `main_remote` calls it instead of the Bokeh `render_index_growth_charts`. The data pipeline (`Index_Forward_EPS_History`, etc.) is unchanged.

**Tech:** Python, Plotly (`plotly.graph_objects`, `plotly.io.to_html`), SQLite, pandas, pytest. Run tests from repo root: `python -m pytest <p> -v`. Git: `git -C "C:/Users/ndaly/projects/sf-fix"`.

**Spec:** `docs/superpowers/specs/2026-06-30-plotly-index-growth-pages-design.md`.

**Draft facts (`generate_index_growth_pages.py`):** `generate_index_growth_pages(db_path)` → SPY/QQQ → `_load_series` (returns DataFrame indexed by date with cols `ig, ig_fwd, pe, pe_fwd, tnx, eps`) → `_build_one` → `_growth_figure`, `_eps_figure` (Plotly), `_timeframe_table_html`, `_page_html` → `_write_page`. EPS currently comes from `_load_eps_series` which only reads `Index_EPS_History` `EPS_Type='TTM_REPORTED'` (so QQQ gets none). `OUTPUT_DIR`/`OUTPUT_FILES` config near the top. Uses `to_html(..., include_plotlyjs="cdn"/False, full_html=False)`.

---

## Task 1: Get the draft running (smoke + fix breakage)

The draft was never executed; de-risk it first.

**Files:** `generate_index_growth_pages.py`

- [ ] **Step 1: Smoke-run against the real DB to a temp dir** (do NOT write into repo `charts/`):

```bash
cd "C:/Users/ndaly/projects/sf-fix" && python -c "
import shutil, os, tempfile, generate_index_growth_pages as g
shutil.copy('Stock Data.db','_g.db'); td=tempfile.mkdtemp()
g.DB_PATH='_g.db'; g.OUTPUT_DIR=td
g.generate_index_growth_pages('_g.db')
import glob; print('FILES:', [os.path.basename(p) for p in glob.glob(os.path.join(td,'*.html'))])
shutil.rmtree(td, ignore_errors=True)
" 2>&1 | tail -20; rm -f _g.db _g.db-wal _g.db-shm
```
(If `OUTPUT_DIR` is not a module global, read the file to find the output-path constant and set it; if `generate_index_growth_pages` doesn't take `db_path`, adapt.)

- [ ] **Step 2: Fix whatever breaks.** Likely issues and fixes:
  - `_resample_frames` uses `resample("M")` — pandas 3.x removed `"M"`; change to `"ME"` (and `"W-FRI"` is fine).
  - Any other deprecated/removed pandas alias → modernize.
  - If it errors that no table has implied growth, inspect with `python -c "import sqlite3;..."` — confirm `Index_Growth_History` columns; the detector uses `DATE_COLS`/`IG_TTM_COLS` etc. (defined near the top). Adjust those candidate lists if the live column names (`Implied_Growth`, `Growth_Type`, `Date`, `Ticker`) aren't matched. NOTE: `Index_Growth_History` stores TTM vs Forward in a `Growth_Type` ROW, not separate columns — if the detector expects separate columns, the draft can't read it as-is. If so, report this as a blocker for Step 2 and STOP — the data-shape mismatch needs a design decision (we'd pivot the table in `_load_series`).
- [ ] **Step 3: Re-run the smoke command until it prints `FILES: ['spy_growth.html', 'qqq_growth.html']` with no traceback.**
- [ ] **Step 4: Commit** `fix(plotly-pages): get generate_index_growth_pages running (pandas freq fix)`.

> **ALREADY CONFIRMED (controller smoke-tested 2026-06-30):** the only fix needed is
> `_resample_frames` `resample("M")` → `resample("ME")` (line ~186) — **already applied**.
> The data loader works against the live DB. After the fix, both pages render in Plotly,
> no "????". Note: `qqq_growth.html` has **no EPS chart** because QQQ has no
> `TTM_REPORTED` EPS — that's the gap Task 2 fixes (SPY's EPS renders fine). So Task 1 is
> effectively done; just commit the freq fix.

---

## Task 2: Port the full index-level EPS series

The draft's EPS (`_load_eps_series`, TTM_REPORTED only) misses QQQ. Port the current Bokeh `_series_eps` construction (TTM_REPORTED → TTM_DAILY → IMPLIED_FROM_PE×divisor).

**Files:** `generate_index_growth_pages.py`; `Test/test_plotly_index_pages.py` (new)

- [ ] **Step 1: Failing test** `Test/test_plotly_index_pages.py`:

```python
import pathlib, sqlite3, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import generate_index_growth_pages as g
import pandas as pd

def _seed_eps(conn):
    conn.execute("""CREATE TABLE Index_EPS_History (Date TEXT, Ticker TEXT, EPS_Type TEXT, EPS REAL)""")
    conn.executemany("INSERT INTO Index_EPS_History VALUES (?,?,?,?)", [
        ("2024-01-31","QQQ","IMPLIED_FROM_PE",20.0),   # ETF-level; *4 -> 80 index
        ("2024-02-29","QQQ","IMPLIED_FROM_PE",22.0),    # -> 88
        ("2024-01-31","SPY","TTM_REPORTED",230.0),
    ])
    conn.commit()

def test_index_eps_series_scales_qqq_implied():
    conn = sqlite3.connect(":memory:"); _seed_eps(conn)
    s = g._index_eps_series(conn, "QQQ")
    assert round(float(s.iloc[-1]), 2) == 88.0   # 22 * divisor(4)

def test_index_eps_series_spy_reported():
    conn = sqlite3.connect(":memory:"); _seed_eps(conn)
    s = g._index_eps_series(conn, "SPY")
    assert round(float(s.iloc[-1]), 2) == 230.0
```

- [ ] **Step 2: Run, expect FAIL** (`-k index_eps_series`).
- [ ] **Step 3: Implement.** Add to `generate_index_growth_pages.py`:

```python
_INDEX_EPS_DIVISOR = {"SPY": 10.0, "QQQ": 4.0}

def _index_eps_series(conn, ticker) -> pd.Series:
    """Index-level EPS history: TTM_REPORTED, extended with TTM_DAILY, then
    IMPLIED_FROM_PE (ETF-level, scaled by the index divisor). Mirrors the prior
    Bokeh _series_eps so SPY and QQQ both get a full series."""
    def _read(eps_type):
        df = pd.read_sql_query(
            "SELECT Date, EPS FROM Index_EPS_History WHERE Ticker=? AND EPS_Type=? ORDER BY Date",
            conn, params=(ticker, eps_type), parse_dates=["Date"])
        if df.empty:
            return pd.Series(dtype=float)
        s = pd.to_numeric(df.set_index(pd.to_datetime(df["Date"]).dt.normalize())["EPS"],
                          errors="coerce").dropna()
        return s[~s.index.duplicated(keep="last")]
    reported = _read("TTM_REPORTED")
    daily = _read("TTM_DAILY")
    implied = _read("IMPLIED_FROM_PE") * _INDEX_EPS_DIVISOR.get(ticker.upper(), 1.0)
    parts = [s for s in (reported, daily, implied) if not s.empty]
    if not parts:
        return pd.Series(dtype=float, name="eps")
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.combine_first(p)
    combined = combined.sort_index(); combined.name = "eps"
    return combined
```

- [ ] **Step 4:** Replace the `eps = _load_eps_series(conn, ticker)` call in `_load_series` with `eps = _index_eps_series(conn, ticker)` (and `if not eps.empty:` instead of `is not None`). Leave `_load_eps_series` in place (unused) or delete it.
- [ ] **Step 5: Run, expect PASS** (`-k index_eps_series`). **Step 6: Commit** `feat(plotly-pages): index-level EPS series (SPY reported + QQQ implied-scaled)`.

---

## Task 3: Forward-EPS reader

**Files:** `generate_index_growth_pages.py`; `Test/test_plotly_index_pages.py`

- [ ] **Step 1: Failing test:**

```python
def test_latest_forward_eps_reads_displayable():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE Index_Forward_EPS_History (
        date_recorded TEXT, ticker TEXT, forward_eps_index REAL, horizon_date TEXT,
        growth_this_fy REAL, growth_next_fy REAL, coverage_weight REAL, displayable INTEGER)""")
    conn.execute("INSERT INTO Index_Forward_EPS_History VALUES "
                 "('2026-06-28','QQQ',112.0,'2027-06-28',0.25,0.29,0.93,1)")
    conn.commit()
    r = g._latest_forward_eps(conn, "QQQ")
    assert r["forward_eps_index"] == 112.0 and r["growth_next_fy"] == 0.29
    conn.execute("UPDATE Index_Forward_EPS_History SET displayable=0"); conn.commit()
    assert g._latest_forward_eps(conn, "QQQ") is None
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** (port from the retired Bokeh module):

```python
def _latest_forward_eps(conn, ticker):
    try:
        row = conn.execute(
            """SELECT forward_eps_index, horizon_date, growth_this_fy, growth_next_fy,
                      coverage_weight, displayable
                 FROM Index_Forward_EPS_History
                WHERE ticker=? ORDER BY date_recorded DESC LIMIT 1""",
            (ticker.upper(),)).fetchone()
    except Exception:
        return None
    if not row or row[0] is None or not row[5]:
        return None
    return {"forward_eps_index": float(row[0]), "horizon_date": row[1],
            "growth_this_fy": row[2], "growth_next_fy": row[3],
            "coverage_weight": row[4]}
```

- [ ] **Step 4: Run, expect PASS. Step 5: Commit** `feat(plotly-pages): forward-EPS reader (displayable-gated)`.

---

## Task 4: EPS figure — log axis + forward forecast trace

**Files:** `generate_index_growth_pages.py`; `Test/test_plotly_index_pages.py`

- [ ] **Step 1: Failing test** (figure has a log y-axis and, when a forward row is passed, a forecast trace with 2 diamond markers):

```python
import plotly.graph_objects as go

def test_eps_figure_log_and_forecast():
    idx = pd.to_datetime(["2025-01-31","2025-06-30","2025-12-31"])
    df = pd.DataFrame({"eps":[80.0,85.0,90.0]}, index=idx)
    d,w,m = g._resample_frames(df)
    fwd = {"forward_eps_index":112.0,"horizon_date":"2027-06-28","growth_this_fy":0.25,
           "growth_next_fy":0.29,"coverage_weight":0.93}
    fig = g._eps_figure(d,w,m,"QQQ", fwd=fwd)
    assert fig.layout.yaxis.type == "log"
    fc = [t for t in fig.data if (t.name or "").lower().find("forecast") >= 0]
    assert fc, "expected a forecast trace"
    # forecast trace carries this-FY + next-FY points (2 markers beyond the anchor)
    assert any(len(getattr(t,"x",[]) or []) >= 2 for t in fc)
```

- [ ] **Step 2: Run, expect FAIL** (signature/log/forecast not present).
- [ ] **Step 3: Implement.** Modify `_eps_figure` to accept `fwd=None`, set `yaxis.type="log"` (keep `tickprefix="$"`), and when `fwd` is provided append a forecast trace:

```python
# inside _eps_figure, after building the EPS traces and before fig.update_layout(...):
def _eps_forecast_trace(last_dt, last_eps, fwd):
    import pandas as _pd
    h1 = _pd.Timestamp(fwd["horizon_date"]); xs=[last_dt, h1]; ys=[last_eps, fwd["forward_eps_index"]]
    g2 = fwd.get("growth_next_fy")
    if g2 is not None:
        xs.append(h1 + _pd.DateOffset(years=1)); ys.append(fwd["forward_eps_index"]*(1+g2))
    return go.Scatter(x=xs, y=ys, name="Forecast (this/next FY)", mode="lines+markers",
                      line=dict(dash="dash", color="#ff8800"),
                      marker=dict(symbol="diamond", size=11, color="#ff8800"))
```
Call it with the daily EPS series' last point (`df_d["eps"].dropna()` last index/value) when `fwd` and the series are non-empty; `fig.add_trace(...)`. Set the y-axis to log in the `update_layout`/`yaxis` dict: `yaxis=dict(title="EPS (USD)", tickprefix="$", type="log")`. Guard: if EPS has non-positive values, use linear instead.

- [ ] **Step 4:** Thread `fwd` through `_build_one`: read `fwd = _latest_forward_eps(conn, ticker)` (needs `conn`/`ticker` in scope — pass them into `_build_one`) and pass to `_eps_figure`.
- [ ] **Step 5: Run, expect PASS. Step 6: Commit** `feat(plotly-pages): log EPS axis + forward forecast (this/next FY)`.

---

## Task 5: Indexed-to-100 figure

**Files:** `generate_index_growth_pages.py`; `Test/test_plotly_index_pages.py`

- [ ] **Step 1: Failing test:**

```python
def test_eps_indexed_figure_rebases_to_100():
    idx = pd.to_datetime(["2025-01-31","2025-06-30","2025-12-31"])
    df = pd.DataFrame({"eps":[80.0,100.0,120.0]}, index=idx)
    d,w,m = g._resample_frames(df)
    fig = g._eps_indexed_figure(d,w,m,"QQQ", fwd=None)
    assert fig is not None and fig.layout.yaxis.type == "log"
    base = [t for t in fig.data if t.mode and "lines" in t.mode][0]
    assert round(float(base.y[0]),1) == 100.0 and round(float(base.y[-1]),1) == 150.0

def test_eps_indexed_figure_guards_bad_base():
    idx = pd.to_datetime(["2025-01-31","2025-06-30"])
    df = pd.DataFrame({"eps":[0.0,5.0]}, index=idx)
    d,w,m = g._resample_frames(df)
    assert g._eps_indexed_figure(d,w,m,"QQQ", fwd=None) is None
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `_eps_indexed_figure` — same shape as `_eps_figure` but rebase each frame's `eps` to 100 at the first daily point (`base=df_d["eps"].dropna().iloc[0]`; return None if base<=0), `yaxis.type="log"`, no `$` prefix, and the forecast trace scaled by `/base*100` (this-FY and next-FY). Mirror the rangeselector/slider/toggle layout from `_eps_figure`.
- [ ] **Step 4: Run, expect PASS. Step 5: Commit** `feat(plotly-pages): EPS indexed-to-100 chart`.

---

## Task 6: Measure tool (plotly_click) + page wiring

**Files:** `generate_index_growth_pages.py`; `Test/test_plotly_index_pages.py`

- [ ] **Step 1: Failing test** (page HTML contains the measured div IDs, the readout, the Clear button, and the measure script):

```python
def test_page_html_has_measure_tool():
    html = g._page_html("QQQ test", "<div>growth</div>",
                        eps_chart_html='<div id="eps-chart"></div>',
                        eps_indexed_html='<div id="eps-indexed-chart"></div>',
                        timeframe_table_html="<table></table>", callout="callout x")
    for needle in ['eps-chart','eps-indexed-chart','measure-readout','Clear',
                   'plotly_click','annualized','callout x']:
        assert needle in html, needle
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
  1. In `_build_one`, render the EPS and indexed figures with fixed div ids:
     `to_html(fig_eps, include_plotlyjs=False, full_html=False, div_id="eps-chart")` and
     `div_id="eps-indexed-chart"`. (The growth fig keeps the CDN include.)
  2. Extend `_page_html(title, growth_chart_html, eps_chart_html, eps_indexed_html, timeframe_table_html, callout)` to: render the callout line, place each measured chart with a readout `<div class="measure-readout" id="ro-eps">…</div>` + `<button>Clear</button>` beneath it, and append the measure `<script>`:

```html
<script>
(function(){
  function attach(divId, roId){
    var gd=document.getElementById(divId), ro=document.getElementById(roId);
    if(!gd||!ro) return;
    var anchor=null, first=null;
    function ms(x){ return (typeof x==='number')?x:new Date(x).getTime(); }
    gd.on('plotly_click', function(d){
      var p=d.points[0], x=ms(p.x), y=p.y;
      if(first===null) first=y;
      if(anchor===null){ anchor={x:x,y:y}; ro.innerHTML='Anchor set. Click another point.'; return; }
      var s=anchor, e={x:x,y:y};
      if(e.x < s.x){ var t=s; s=e; e=t; }
      var pct=(e.y/s.y-1)*100, yrs=(e.x-s.x)/(365.25*86400000);
      var span = yrs>=1 ? yrs.toFixed(1)+' years' : Math.round(yrs*12)+' months';
      var txt='<b>'+(pct>=0?'+':'')+pct.toFixed(1)+'%</b> over '+span;
      if(yrs>=0.02){ var ann=(Math.pow(e.y/s.y,1/yrs)-1)*100;
        if(isFinite(ann)) txt+=' ('+(ann>=0?'+':'')+ann.toFixed(1)+'% annualized)'; }
      ro.innerHTML=txt; anchor=null;
    });
    gd.on('plotly_hover', function(d){
      if(anchor!==null||first===null) return;
      var y=d.points[0].y, pct=(y/first-1)*100;
      ro.innerHTML='% from first point: <b>'+(pct>=0?'+':'')+pct.toFixed(1)+'%</b>';
    });
    gd.__clear=function(){ anchor=null; ro.innerHTML='Click a point, then another, to measure % change.'; };
  }
  function init(){ attach('eps-chart','ro-eps'); attach('eps-indexed-chart','ro-eps-indexed'); }
  if(document.readyState!=='loading') init(); else document.addEventListener('DOMContentLoaded', init);
})();
function clearMeasure(id){ var gd=document.getElementById(id); if(gd&&gd.__clear) gd.__clear(); }
</script>
```
     Each Clear button: `<button onclick="clearMeasure('eps-chart')">Clear</button>` (and `'eps-indexed-chart'`). All strings ASCII.
- [ ] **Step 4: Run, expect PASS. Step 5: Commit** `feat(plotly-pages): plotly_click measure tool + page wiring`.

---

## Task 7: Forward-growth callout

**Files:** `generate_index_growth_pages.py`; `Test/test_plotly_index_pages.py`

- [ ] **Step 1: Failing test:**

```python
def test_forward_callout_text():
    t = g._forward_callout({"growth_this_fy":0.25,"growth_next_fy":0.29,"coverage_weight":0.93})
    assert "+25.0%" in t and "+29.0%" in t and "93%" in t
    assert g._forward_callout(None) == ""
```

- [ ] **Step 2: Run, expect FAIL. Step 3: Implement:**

```python
def _forward_callout(fwd):
    if not fwd or fwd.get("growth_this_fy") is None:
        return ""
    g1=fwd["growth_this_fy"]; g2=fwd.get("growth_next_fy"); cov=fwd.get("coverage_weight") or 0
    parts=[f"Forward earnings growth (bottom-up): <b>{g1:+.1%}</b> this fiscal year"]
    if g2 is not None: parts.append(f", <b>{g2:+.1%}</b> next")
    parts.append(f". Based on {cov:.0%} of index weight.")
    return "".join(parts)
```
Thread `callout=_forward_callout(fwd)` through `_build_one` → `_page_html`.
- [ ] **Step 4: Run, expect PASS. Step 5: Commit** `feat(plotly-pages): forward-growth callout`.

---

## Task 8: Wire into main_remote, neuter the Bokeh page renderer

**Files:** `main_remote.py`; `html_generator2.py`

- [ ] **Step 1:** In `main_remote.mini_main`, replace the loop
  `for idx in ("SPY","QQQ"): render_index_growth_charts(idx)` with:

```python
        try:
            from generate_index_growth_pages import generate_index_growth_pages
            generate_index_growth_pages(DB_PATH)
        except Exception as exc:
            print(f"[WARN] Plotly index growth pages failed: {exc}")
```
  Remove the now-unused `from index_growth_charts import render_index_growth_charts` import (and `backfill_index_growth`/`render_index_growth_charts` references — check; keep `backfill_index_growth` if it's a separate concern, only remove the Bokeh render).
- [ ] **Step 2:** In `html_generator2.py`, make `render_spy_qqq_growth_pages()` a no-op (or delete it and its call) — the Plotly generator now owns `spy_growth.html`/`qqq_growth.html`. Confirm nothing else depends on the Bokeh templates.
- [ ] **Step 3:** Smoke: `python -c "import main_remote; print('ok')"` and `python -c "import html_generator2; print('ok')"`.
- [ ] **Step 4: Commit** `feat(plotly-pages): generate growth pages via Plotly in the daily run`.

---

## Task 9: Retire Bokeh

**Files:** delete `index_growth_charts.py`, `Test/test_index_growth_charts.py`, `templates/spy_growth_template.html`, `templates/qqq_growth_template.html`

- [ ] **Step 1:** Confirm nothing imports the Bokeh module:
  `git -C "C:/Users/ndaly/projects/sf-fix" grep -n "index_growth_charts\|render_index_growth_charts\|valuation_bundle_chart" -- "*.py"` — every hit must be in the file being deleted or already removed (Task 8). If a still-needed helper remains (e.g. `_INDEX_EPS_DIVISOR`), it was already ported in Task 2 — verify.
- [ ] **Step 2:** Delete the files:
```bash
git -C "C:/Users/ndaly/projects/sf-fix" rm index_growth_charts.py Test/test_index_growth_charts.py templates/spy_growth_template.html templates/qqq_growth_template.html
```
- [ ] **Step 3:** Remove deployed Bokeh bundle artifacts so they stop shipping:
```bash
git -C "C:/Users/ndaly/projects/sf-fix" rm -f charts/spy_valuation_bundle_chart.js charts/spy_valuation_bundle_chart_div.html charts/qqq_valuation_bundle_chart.js charts/qqq_valuation_bundle_chart_div.html 2>/dev/null || true
```
- [ ] **Step 4:** `python -c "import main_remote; print('ok')"` still clean; `python -m pytest Test/ -q` — the Bokeh test file is gone, the new `test_plotly_index_pages.py` passes, only the 4 known pre-existing failures remain.
- [ ] **Step 5: Commit** `chore(plotly-pages): retire Bokeh index_growth_charts + templates + bundle artifacts`.

---

## Task 10: Full suite + real render verification

**Files:** none (verify)

- [ ] **Step 1:** `python -m pytest Test/ -q` — report tally; only the 4 known pre-existing failures.
- [ ] **Step 2: Real generation** against a scratch DB, to a temp dir:
```bash
cd "C:/Users/ndaly/projects/sf-fix" && python -c "
import shutil, os, tempfile, generate_index_growth_pages as g
shutil.copy('Stock Data.db','_g.db'); td=tempfile.mkdtemp(); g.OUTPUT_DIR=td
g.generate_index_growth_pages('_g.db')
for tk in ('spy','qqq'):
    p=os.path.join(td, tk+'_growth.html'); h=open(p,encoding='utf-8').read()
    print(tk, 'size', len(h),
          '| plotly', 'Plotly' in h, '| eps-chart', 'eps-chart' in h,
          '| indexed', 'eps-indexed-chart' in h, '| measure', 'plotly_click' in h,
          '| forecast', 'Forecast' in h, '| callout', 'Forward earnings growth' in h,
          '| no-?', '????' not in h)
shutil.rmtree(td, ignore_errors=True)
"; rm -f _g.db _g.db-wal _g.db-shm
```
Expected: both `spy` and `qqq` print all flags True (`no-?` True confirms no "????"), sizes are substantial. If `forecast`/`callout` are False, that's expected when no displayable forward row exists yet (coverage gate) — note it, not a failure. Any traceback = a real bug to fix.

- [ ] **Step 3:** `git -C "C:/Users/ndaly/projects/sf-fix" status --porcelain` — confirm no `_g.db` strays and no unintended tracked-file changes beyond the pre-existing dirty `charts/*_summary.html`. Report.

- [ ] **Step 4 — JS behavior note:** the `plotly_click`/hover/Clear behavior is browser-only; eyeball post-deploy on `qqq_growth.html` (click two EPS points → `A→B %`; hover → `% from first point`; Clear resets). The Python verify can't catch JS runtime errors.

---

## Notes
- The data pipeline is untouched; this is a rendering swap. Deploy via the weekly `site-refresh.yml`/dispatch; push stays with Nick.
- Loader confirmed working against the live DB (controller smoke test 2026-06-30) — no schema adaptation needed. SPY page ~1.87 MB, QQQ ~0.48 MB (large from long history; fine, comparable to the old Bokeh bundle — resampling/optimization is a possible later polish, out of scope here).
- The in-flight Bokeh "????" fix (run #516) becomes moot once Bokeh is retired; harmless to let it land.