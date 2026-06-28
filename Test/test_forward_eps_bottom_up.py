import pathlib, sqlite3, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import forward_eps_bottom_up as bu

def _seed(conn):
    conn.execute("CREATE TABLE TTM_Data (Symbol TEXT, TTM_EPS REAL, Shares_Outstanding REAL)")
    conn.executemany("INSERT INTO TTM_Data VALUES (?,?,?)",
        [("AAA",10.0,100.0),("BBB",5.0,200.0)])
    conn.execute("""CREATE TABLE Forward_EPS_FY_History (date_recorded TEXT, ticker TEXT,
                    period_label TEXT, forward_eps REAL)""")
    conn.executemany("INSERT INTO Forward_EPS_FY_History VALUES (?,?,?,?)",
        [("2026-06-28","AAA","This FY",12.0),("2026-06-28","AAA","Next FY",14.0),
         ("2026-06-28","BBB","This FY",5.5),("2026-06-28","BBB","Next FY",6.0)])
    conn.commit()

def test_load_constituent_financials():
    conn = sqlite3.connect(":memory:"); _seed(conn)
    fin = bu.load_constituent_financials(conn, ["AAA","BBB","CCC"])
    assert fin["AAA"] == {"ttm_eps":10.0,"shares":100.0,"this_fy":12.0,"next_fy":14.0}
    assert "CCC" not in fin

def test_aggregate_growth_and_coverage():
    holdings = [("AAA",60.0),("BBB",30.0),("CCC",10.0)]
    fin = {"AAA":{"ttm_eps":10.0,"shares":100.0,"this_fy":12.0,"next_fy":14.0},
           "BBB":{"ttm_eps":5.0,"shares":200.0,"this_fy":5.5,"next_fy":6.0}}
    res = bu.aggregate(holdings, fin)
    assert round(res["growth_this_fy"], 4) == round(2300/2000 - 1, 4)
    assert round(res["growth_next_fy"], 4) == round(2600/2300 - 1, 4)
    assert round(res["coverage_weight"], 4) == 0.90
    assert res["fwd_earnings_this_fy"] == 2300.0
