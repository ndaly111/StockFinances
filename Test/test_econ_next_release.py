"""Tests for the economic-dashboard next-release resolver: FRED release-calendar
API primary, verified static calendar fallback, ASCII-only output."""
import pathlib, sys
import datetime as dt

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_economic_data as ge


def test_fred_dates_pick_first_on_or_after_today(monkeypatch):
    monkeypatch.setattr(ge, "_fred_release_dates_raw",
                        lambda rid: ["2026-06-05", "2026-07-02", "2026-08-07", "2026-09-04"])
    out = ge._next_release("empsit", today=dt.date(2026, 7, 10))
    assert out == "Aug 7, 2026"


def test_fred_same_day_release_counts(monkeypatch):
    monkeypatch.setattr(ge, "_fred_release_dates_raw", lambda rid: ["2026-07-10"])
    assert ge._next_release("empsit", today=dt.date(2026, 7, 10)) == "Jul 10, 2026"


def test_static_fallback_when_fred_fails(monkeypatch):
    monkeypatch.setattr(ge, "_fred_release_dates_raw",
                        lambda rid: (_ for _ in ()).throw(RuntimeError("no key")))
    # gdp static calendar contains 2026-07-30 (verified BEA date)
    assert ge._next_release("gdp", today=dt.date(2026, 7, 10)) == "Jul 30, 2026"


def test_tbd_when_everything_exhausted(monkeypatch):
    monkeypatch.setattr(ge, "_fred_release_dates_raw", lambda rid: [])
    assert ge._next_release("empsit", today=dt.date(2039, 1, 1)) == "TBD"


def test_fomc_static_is_correct(monkeypatch):
    # Force the static path (no network in tests): resolver must return the
    # decision day (second day) of the next meeting.
    monkeypatch.setattr(ge, "_fomc_dates_from_fed",
                        lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    assert ge._next_release("fomc", today=dt.date(2026, 7, 10)) == "Jul 29, 2026 (FOMC)"
    assert ge._next_release("fomc", today=dt.date(2026, 8, 1)) == "Sep 16, 2026 (FOMC)"


def test_output_is_pure_ascii(monkeypatch):
    # The old scrapers' em-dash fallback rendered as mojibake on the dashboard.
    monkeypatch.setattr(ge, "_fred_release_dates_raw", lambda rid: ["2026-08-07"])
    monkeypatch.setattr(ge, "_fomc_dates_from_fed", lambda: ["2026-07-29"])
    for kind in ("empsit", "cpi", "gdp", "fomc"):
        s = ge._next_release(kind, today=dt.date(2026, 7, 10))
        assert s.encode("ascii"), s   # must not raise


# ---- FOMC auto-fetch from federalreserve.gov --------------------------------
_FED_HTML = """
<div><h4>2026 FOMC Meetings</h4>
<div>January</div><div>27-28</div>
<div>March</div><div>17-18*</div>
<div>July</div><div>28-29</div>
<div>December</div><div>8-9*</div>
</div>
<div><h4>2027 FOMC Meetings</h4>
<div>January</div><div>26-27</div>
<div>October</div><div>26-27</div>
</div>
"""


def test_parse_fomc_calendar_returns_decision_days():
    out = ge._parse_fomc_calendar(_FED_HTML)
    assert "2026-01-28" in out
    assert "2026-03-18" in out          # asterisk (SEP meeting) stripped
    assert "2026-12-09" in out
    assert "2027-01-27" in out
    assert "2027-10-27" in out
    assert out == sorted(out)


def test_parse_fomc_handles_cross_month_meeting():
    html = "<h4>2026 FOMC Meetings</h4><div>October</div><div>31-November 1</div>"
    out = ge._parse_fomc_calendar(html)
    assert out == ["2026-11-01"]        # decision day lands in November


def test_fomc_uses_fed_fetch_then_static(monkeypatch):
    monkeypatch.setattr(ge, "_fomc_dates_from_fed", lambda: ["2026-08-01"])
    assert ge._next_release("fomc", today=dt.date(2026, 7, 10)) == "Aug 1, 2026 (FOMC)"
    # fetch failure -> static calendar still answers
    monkeypatch.setattr(ge, "_fomc_dates_from_fed",
                        lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert ge._next_release("fomc", today=dt.date(2026, 7, 10)) == "Jul 29, 2026 (FOMC)"


def test_static_fomc_covers_2027():
    # the static fallback must not run dry the day 2026 ends
    monkeypatch_dates = ge._STATIC_RELEASES["fomc"]
    assert any(d.startswith("2027") for d in monkeypatch_dates)


def test_parse_fomc_collapses_page_noise_to_decision_day():
    html = ("<h4>2027 FOMC Meetings</h4>"
            "<div>January</div><div>25</div>"      # stray page noise
            "<div>January</div><div>26-27</div>")  # the real meeting
    assert ge._parse_fomc_calendar(html) == ["2027-01-27"]
