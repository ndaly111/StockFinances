"""Tests for factset_sp500_eps — parse the FactSet Earnings Insight PDF prose into
S&P 500 index-level forward EPS, and snapshot it as SPY's authoritative forecast."""
import pathlib, sqlite3, sys
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import factset_sp500_eps as fs
import index_forward_eps as ife

# Representative prose from a real Earnings Insight report (wording FactSet has
# used for years). Multi-line, with distractor Q-level and revenue figures.
SAMPLE = """
Key Metrics
- Earnings: ... For Q2 2026, the bottom-up EPS estimate is $76.78 ...
- Valuation: The forward 12-month P/E ratio for the S&P 500 is 20.4. This P/E ratio
is above the 5-year average (19.9) and above the 10-year average (19.0).
During the past two months, the Q2 2026 bottom-up EPS estimate increased by 0.7%.
From March 31 through June 30, the CY 2026 bottom-up EPS estimate increased by 6.3%
(to $340.52 from $320.39). At the sector level, nine sectors witnessed an increase.
For Q3 2026 and Q4 2026, analysts are calling for earnings growth rates of 26.8% and 24.4%.
For CY 2026, analysts are projecting earnings growth of 24.1% and revenue growth of 10.8%.
For CY 2027, analysts are projecting earnings growth of 14.0% and revenue growth of 6.5%.
"""


def test_parse_extracts_cy_eps_growth_and_pe():
    d = fs.parse_report_text(SAMPLE, today=date(2026, 7, 7))
    assert d is not None
    assert d["cy_year"] == 2026
    assert abs(d["cy_eps"] - 340.52) < 1e-6
    assert abs(d["cy_growth"] - 0.241) < 1e-9
    assert abs(d["fwd_pe"] - 20.4) < 1e-9
    assert d["next_year"] == 2027
    assert abs(d["next_growth"] - 0.140) < 1e-9


def test_parse_picks_current_calendar_year_not_a_quarter():
    # The Q2 2026 line must not be mistaken for the CY figure.
    d = fs.parse_report_text(SAMPLE, today=date(2026, 7, 7))
    assert d["cy_eps"] == 340.52          # not 76.78 (a quarterly figure)


def test_parse_returns_none_without_cy_eps():
    txt = "The forward 12-month P/E ratio for the S&P 500 is 20.4."
    assert fs.parse_report_text(txt, today=date(2026, 7, 7)) is None


def test_next_growth_optional():
    txt = ("The forward 12-month P/E ratio for the S&P 500 is 20.4.\n"
           "the CY 2026 bottom-up EPS estimate increased by 6.3% (to $340.52 from $320.39).\n"
           "For CY 2026, analysts are projecting earnings growth of 24.1% and revenue growth of 10.8%.")
    d = fs.parse_report_text(txt, today=date(2026, 7, 7))
    assert d is not None and d["next_growth"] is None


# ---- sanity gate -----------------------------------------------------------
def test_sanity_accepts_reasonable():
    assert fs.passes_sanity({"cy_eps": 340.52, "cy_growth": 0.241, "fwd_pe": 20.4,
                             "next_growth": 0.14}) is True


def test_sanity_rejects_absurd_eps():
    assert fs.passes_sanity({"cy_eps": 34.0, "cy_growth": 0.24, "fwd_pe": 20.4,
                             "next_growth": None}) is False   # ETF-scale leak
    assert fs.passes_sanity({"cy_eps": 9000.0, "cy_growth": 0.24, "fwd_pe": 20.4,
                             "next_growth": None}) is False


def test_sanity_rejects_absurd_growth_or_pe():
    assert fs.passes_sanity({"cy_eps": 340.0, "cy_growth": 0.90, "fwd_pe": 20.4,
                             "next_growth": None}) is False
    assert fs.passes_sanity({"cy_eps": 340.0, "cy_growth": 0.24, "fwd_pe": 1.5,
                             "next_growth": None}) is False


# ---- snapshot --------------------------------------------------------------
def test_snapshot_writes_displayable_spy_row(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    ife.ensure_forward_eps_table(conn)
    n = fs.snapshot_factset_spy(conn, today=date(2026, 7, 7), text=SAMPLE)
    assert n == 1
    row = conn.execute(
        """SELECT forward_eps_index, growth_this_fy, growth_next_fy, forward_pe,
                  source, displayable, coverage_weight, horizon_date
             FROM Index_Forward_EPS_History WHERE ticker='SPY'""").fetchone()
    assert row is not None
    assert abs(row[0] - 340.52) < 1e-6      # index-scale forward EPS
    assert abs(row[1] - 0.241) < 1e-9       # this-FY growth
    assert abs(row[2] - 0.140) < 1e-9       # next-FY growth
    assert abs(row[3] - 20.4) < 1e-9        # forward P/E
    assert row[4] == "factset"
    assert row[5] == 1                       # displayable
    assert row[6] == 1.0                     # full-index coverage
    assert row[7] == "2026-12-31"


def test_snapshot_does_not_overwrite_on_parse_failure(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    ife.ensure_forward_eps_table(conn)
    # pre-existing bottom-up SPY row (the fallback) must survive a bad parse
    conn.execute(
        """INSERT INTO Index_Forward_EPS_History
             (date_recorded, ticker, forward_eps_index, growth_this_fy, source, displayable)
           VALUES ('2026-07-07','SPY', 300.0, 0.30, 'bottom_up', 1)""")
    conn.commit()
    n = fs.snapshot_factset_spy(conn, today=date(2026, 7, 7), text="no usable numbers here")
    assert n == 0
    row = conn.execute(
        "SELECT forward_eps_index, source FROM Index_Forward_EPS_History WHERE ticker='SPY'").fetchone()
    assert row[0] == 300.0 and row[1] == "bottom_up"   # untouched
