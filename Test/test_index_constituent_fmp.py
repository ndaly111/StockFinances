"""Tests for index_constituent_fmp — FMP-backed enrichment of index constituents
so the bottom-up forward-EPS aggregation can clear the 85% coverage gate."""
import pathlib, sqlite3, sys
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import index_constituent_fmp as icf
import index_forward_eps as ife
import forward_eps_bottom_up as bu


# ---- fake HTTP session -----------------------------------------------------
class _Resp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


class _Session:
    """Routes a request to canned JSON by URL substring; records calls."""
    def __init__(self, routes):
        self.routes = routes          # {url_substring: data or callable(ticker)->data}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        for key, data in self.routes.items():
            if key in url:
                tk = url.rstrip("/").split("/")[-1]
                return _Resp(data(tk) if callable(data) else data)
        raise AssertionError(f"unexpected url: {url}")


def _estimates(this_eps, next_eps, this_end="2026-12-31", next_end="2027-12-31"):
    # Mixed past + future rows; enrichment must pick the two nearest future FYs.
    return [
        {"date": "2024-12-31", "estimatedEpsAvg": 5.0},
        {"date": "2025-12-31", "estimatedEpsAvg": 6.0},
        {"date": this_end, "estimatedEpsAvg": this_eps},
        {"date": next_end, "estimatedEpsAvg": next_eps},
    ]


def _income_q(net_incomes, shares, dates=None):
    dates = dates or ["2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30"]
    return [
        {"date": d, "netIncome": ni, "weightedAverageShsOutDil": shares}
        for d, ni in zip(dates, net_incomes)
    ]


# ---- fetch_forward_eps -----------------------------------------------------
def test_fetch_forward_eps_picks_two_nearest_future():
    sess = _Session({"analyst-estimates": _estimates(10.0, 12.0)})
    out = icf.fetch_forward_eps("NVDA", "k", today=date(2026, 7, 1), session=sess)
    assert out["this_fy"] == 10.0
    assert out["next_fy"] == 12.0
    assert out["this_end"] == "2026-12-31"
    assert out["next_end"] == "2027-12-31"


def test_fetch_forward_eps_none_when_no_future():
    sess = _Session({"analyst-estimates": [
        {"date": "2023-12-31", "estimatedEpsAvg": 4.0}]})
    assert icf.fetch_forward_eps("X", "k", today=date(2026, 7, 1), session=sess) is None


def test_fetch_forward_eps_none_on_empty():
    sess = _Session({"analyst-estimates": []})
    assert icf.fetch_forward_eps("X", "k", today=date(2026, 7, 1), session=sess) is None


# ---- fetch_ttm_eps_shares --------------------------------------------------
def test_fetch_ttm_sums_four_quarters_net_income():
    sess = _Session({"income-statement": _income_q([100, 100, 100, 100], 1000.0)})
    out = icf.fetch_ttm_eps_shares("AAA", "k", session=sess)
    assert out["ttm_net_income"] == 400.0
    assert out["shares"] == 1000.0
    # eps x shares == net income, exactly (the aggregator relies on this)
    assert abs(out["ttm_eps"] * out["shares"] - 400.0) < 1e-6


def test_fetch_ttm_none_when_fewer_than_four_quarters():
    sess = _Session({"income-statement": _income_q([100, 100], 1000.0)[:2]})
    assert icf.fetch_ttm_eps_shares("AAA", "k", session=sess) is None


# ---- enrich_constituents_fmp -----------------------------------------------
def test_enrich_writes_both_tables_and_becomes_covered(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    ife.ensure_forward_eps_table(conn)
    # minimal TTM_Data + Forward_EPS_FY_History schemas the enricher writes into
    conn.execute("""CREATE TABLE TTM_Data (Symbol TEXT PRIMARY KEY, TTM_Revenue REAL,
        TTM_Net_Income REAL, TTM_EPS REAL, Shares_Outstanding REAL, Quarter TEXT,
        Last_Updated TEXT)""")
    conn.execute("""CREATE TABLE Forward_EPS_FY_History (date_recorded TEXT NOT NULL,
        ticker TEXT NOT NULL, period_end TEXT NOT NULL, period_label TEXT,
        forward_eps REAL, eps_analysts INTEGER, source TEXT, fiscal_year INTEGER,
        forward_revenue REAL, revenue_analysts INTEGER,
        PRIMARY KEY (date_recorded, ticker, period_end))""")
    conn.commit()

    sess = _Session({
        "analyst-estimates": _estimates(10.0, 12.0),
        "income-statement": _income_q([100, 100, 100, 100], 1000.0),
    })
    n = icf.enrich_constituents_fmp(conn, ["NVDA"], today=date(2026, 7, 1),
                                    session=sess, api_key="k")
    assert n == 1

    # Both forward rows present
    rows = conn.execute(
        "SELECT period_label, forward_eps FROM Forward_EPS_FY_History "
        "WHERE ticker='NVDA' ORDER BY period_end").fetchall()
    assert ("This FY", 10.0) in rows
    assert ("Next FY", 12.0) in rows
    # TTM row present
    ttm = conn.execute(
        "SELECT TTM_EPS, Shares_Outstanding FROM TTM_Data WHERE Symbol='NVDA'").fetchone()
    assert ttm is not None and ttm[1] == 1000.0

    # And the existing aggregator now counts it as covered.
    fin = bu.load_constituent_financials(conn, ["NVDA"])
    assert "NVDA" in fin
    assert fin["NVDA"]["this_fy"] == 10.0


def test_enrich_skips_ticker_missing_estimates_no_partial_rows(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    ife.ensure_forward_eps_table(conn)
    conn.execute("""CREATE TABLE TTM_Data (Symbol TEXT PRIMARY KEY, TTM_Revenue REAL,
        TTM_Net_Income REAL, TTM_EPS REAL, Shares_Outstanding REAL, Quarter TEXT,
        Last_Updated TEXT)""")
    conn.execute("""CREATE TABLE Forward_EPS_FY_History (date_recorded TEXT NOT NULL,
        ticker TEXT NOT NULL, period_end TEXT NOT NULL, period_label TEXT,
        forward_eps REAL, eps_analysts INTEGER, source TEXT, fiscal_year INTEGER,
        forward_revenue REAL, revenue_analysts INTEGER,
        PRIMARY KEY (date_recorded, ticker, period_end))""")
    conn.commit()

    # estimates empty -> skip; income present but must NOT leave a TTM row behind
    sess = _Session({
        "analyst-estimates": [],
        "income-statement": _income_q([100, 100, 100, 100], 1000.0),
    })
    n = icf.enrich_constituents_fmp(conn, ["X"], today=date(2026, 7, 1),
                                    session=sess, api_key="k")
    assert n == 0
    assert conn.execute("SELECT COUNT(*) FROM Forward_EPS_FY_History").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM TTM_Data").fetchone()[0] == 0

