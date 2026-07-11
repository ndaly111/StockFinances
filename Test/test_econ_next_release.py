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


def test_fomc_is_static_and_correct():
    # FOMC calendar is published years ahead; resolver must return the decision
    # day (second day) of the next 2026 meeting. No FRED release id for it.
    assert ge._next_release("fomc", today=dt.date(2026, 7, 10)) == "Jul 29, 2026 (FOMC)"
    assert ge._next_release("fomc", today=dt.date(2026, 8, 1)) == "Sep 16, 2026 (FOMC)"


def test_output_is_pure_ascii(monkeypatch):
    # The old scrapers' em-dash fallback rendered as mojibake on the dashboard.
    monkeypatch.setattr(ge, "_fred_release_dates_raw", lambda rid: ["2026-08-07"])
    for kind in ("empsit", "cpi", "gdp", "fomc"):
        s = ge._next_release(kind, today=dt.date(2026, 7, 10))
        assert s.encode("ascii"), s   # must not raise
