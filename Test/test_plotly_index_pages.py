import pathlib, sqlite3, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import generate_index_growth_pages as g
import pandas as pd

def _seed_eps(conn):
    conn.execute("""CREATE TABLE Index_EPS_History (Date TEXT, Ticker TEXT, EPS_Type TEXT, EPS REAL)""")
    conn.executemany("INSERT INTO Index_EPS_History VALUES (?,?,?,?)", [
        ("2024-01-31","QQQ","IMPLIED_FROM_PE",20.0),
        ("2024-02-29","QQQ","IMPLIED_FROM_PE",22.0),
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

def test_index_eps_series_index_is_utc_aware():
    conn = sqlite3.connect(":memory:"); _seed_eps(conn)
    s = g._index_eps_series(conn, "SPY")
    assert s.index.tz is not None   # must be tz-aware to join with _load_series

import plotly.graph_objects as go

def test_eps_figure_log_and_forecast():
    idx = pd.to_datetime(["2025-01-31","2025-06-30","2025-12-31"], utc=True)
    df = pd.DataFrame({"eps":[80.0,85.0,90.0]}, index=idx)
    d,w,m = g._resample_frames(df)
    fwd = {"forward_eps_index":112.0,"horizon_date":"2027-06-28","growth_this_fy":0.25,
           "growth_next_fy":0.29,"coverage_weight":0.93}
    fig = g._eps_figure(d,w,m,"QQQ", fwd=fwd)
    assert fig.layout.yaxis.type == "linear"  # EPS dollar chart is now linear (Change B)
    fc = [t for t in fig.data if (t.name or "").lower().find("forecast") >= 0]
    assert fc, "expected a forecast trace"
    assert any(len(getattr(t,"x",[]) or []) >= 2 for t in fc)

def test_eps_indexed_figure_rebases_to_100():
    idx = pd.to_datetime(["2025-01-31","2025-06-30","2025-12-31"], utc=True)
    df = pd.DataFrame({"eps":[80.0,100.0,120.0]}, index=idx)
    d,w,m = g._resample_frames(df)
    fig = g._eps_indexed_figure(d,w,m,"QQQ", fwd=None)
    assert fig is not None and fig.layout.yaxis.type == "log"
    base = [t for t in fig.data if t.mode and "lines" in t.mode][0]
    assert round(float(base.y[0]),1) == 100.0 and round(float(base.y[-1]),1) == 150.0

def test_eps_indexed_figure_guards_bad_base():
    idx = pd.to_datetime(["2025-01-31","2025-06-30"], utc=True)
    df = pd.DataFrame({"eps":[0.0,5.0]}, index=idx)
    d,w,m = g._resample_frames(df)
    assert g._eps_indexed_figure(d,w,m,"QQQ", fwd=None) is None

def test_page_html_has_measure_tool():
    html = g._page_html("QQQ test", "<div>growth</div>",
                        eps_chart_html='<div id="eps-chart"></div>',
                        eps_indexed_html='<div id="eps-indexed-chart"></div>',
                        timeframe_table_html="<table></table>", callout="callout x")
    for needle in ['eps-chart','eps-indexed-chart','measure-readout','Clear',
                   'plotly_click','annualized','callout x']:
        assert needle in html, needle

def test_forward_callout_text():
    t = g._forward_callout({"growth_this_fy":0.25,"growth_next_fy":0.29,"coverage_weight":0.93})
    assert "+25.0%" in t and "+29.0%" in t and "93%" in t
    assert g._forward_callout(None) == ""
