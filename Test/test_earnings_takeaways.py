import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import earnings_takeaways as et  # noqa: E402

ITEM = {"label": "Outlook", "text": "Revenue up 10%."}


@pytest.mark.parametrize("bad", [
    {"takeaways": [ITEM]},
    {"verdict": "v", "takeaways": []},
    {"verdict": "v"},
])
def test_incomplete_card_is_rejected(bad):
    with pytest.raises(RuntimeError):
        et._clean_card(bad)


def test_malformed_points_dropped_and_list_capped():
    card = et._clean_card({"verdict": "v", "takeaways": [{"label": "x"}] + [ITEM] * 9})
    assert len(card["takeaways"]) == 4


def test_transcript_read_defaults_to_false():
    assert et._clean_card({"verdict": "v", "takeaways": [ITEM]})["transcript_read"] is False


def test_prompt_formats_as_one_combined_list():
    text = et.PROMPT.format(name="Apple", ticker="AAPL", when="w", timing="t", pr_url="u",
                            prior="p", pr_text="x")
    assert '"takeaways"' in text and '"call"' not in text and '"news"' not in text


def test_recent_handles_old_and_new_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUT_DIR", tmp_path)
    today = et.dt.date.today().isoformat()
    base = {"date": today, "move_pct": 1.0, "basis": "close", "verdict": "v"}
    (tmp_path / "OLD.json").write_text(json.dumps(
        {"ticker": "OLD", "last": {**base, "reasons": [ITEM]}}), encoding="utf-8")
    (tmp_path / "NEW.json").write_text(json.dumps(
        {"ticker": "NEW", "last": {**base, "takeaways": [ITEM]}}), encoding="utf-8")
    assert et.write_recent() == 2
    rows = {r["ticker"]: r for r in json.loads((tmp_path / "recent.json").read_text(encoding="utf-8"))["reports"]}
    assert rows["OLD"]["reasons"] == [ITEM] and rows["NEW"]["takeaways"] == [ITEM]


def _setup_existing(monkeypatch, last, move):
    when = et.dt.datetime.now(et.dt.timezone.utc) - et.dt.timedelta(hours=3)
    monkeypatch.setattr(et, "_reports", lambda t: [(when, False, True)])
    monkeypatch.setattr(et, "_load", lambda t: {"ticker": t, "last": {"date": when.date().isoformat(), **last}})
    monkeypatch.setattr(et, "_price_move", lambda *a: move)
    monkeypatch.setattr(et, "_press_release", lambda *a: ("u", "x"))
    monkeypatch.setattr(et, "_sec_lookup", lambda t: (1, "Apple"))
    monkeypatch.setattr(et, "_write_with_claude", lambda p: {
        "verdict": "v", "takeaways": [ITEM], "transcript_read": True})
    saved = {}
    monkeypatch.setattr(et, "_save", lambda t, d: saved.update(d))
    return saved


def test_card_written_without_the_call_is_retried(monkeypatch):
    saved = _setup_existing(monkeypatch, {"basis": "after hours", "transcript_read": False}, (0.1, "after hours"))
    assert et.update_ticker("AAPL") == "written"
    assert saved["last"]["takeaways"] == [ITEM] and "story" not in saved


def test_complete_after_hours_card_waits_for_the_close(monkeypatch):
    _setup_existing(monkeypatch, {"basis": "after hours", "transcript_read": True}, (0.1, "after hours"))
    assert et.update_ticker("AAPL") is None


def test_finished_card_is_left_alone(monkeypatch):
    _setup_existing(monkeypatch, {"basis": "close", "transcript_read": True}, (0.1, "close"))
    assert et.update_ticker("AAPL") is None


def test_old_format_card_is_not_rewritten(monkeypatch):
    _setup_existing(monkeypatch, {"basis": "close", "reasons": [ITEM]}, (0.1, "close"))
    assert et.update_ticker("AAPL") is None


def _yahoo(monkeypatch, dates):
    import pandas as pd
    index = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize("America/New_York")
    df = pd.DataFrame({"Reported EPS": [2.0] * len(dates)}, index=index)
    monkeypatch.setattr(et.yf, "Ticker", lambda t: type("T", (), {"get_earnings_dates": lambda self, limit: df})())


def _filed(y, m, d, h=16):
    return et.dt.datetime(y, m, d, h, 30, tzinfo=et.ET)


def test_8k_newer_than_yahoo_becomes_latest_report(monkeypatch):
    _yahoo(monkeypatch, ["2026-07-30 16:00"])
    monkeypatch.setattr(et, "_latest_results_8k", lambda t: _filed(2026, 11, 2))
    when, pre, confirmed = et._reports("AAPL")[0]
    assert when.date().isoformat() == "2026-11-02" and pre is False and confirmed is True


def test_8k_same_report_as_yahoo_adds_nothing(monkeypatch):
    _yahoo(monkeypatch, ["2026-07-30 16:00"])
    monkeypatch.setattr(et, "_latest_results_8k", lambda t: _filed(2026, 7, 31))
    assert [r[0].date().isoformat() for r in et._reports("AAPL")] == ["2026-07-30"]


def test_edgar_failure_falls_back_to_yahoo(monkeypatch):
    _yahoo(monkeypatch, ["2026-07-30 16:00"])

    def boom(t):
        raise OSError("down")

    monkeypatch.setattr(et, "_latest_results_8k", boom)
    assert [r[0].date().isoformat() for r in et._reports("AAPL")] == ["2026-07-30"]


def test_8k_time_is_converted_from_utc(monkeypatch):
    monkeypatch.setattr(et, "_sec_lookup", lambda t: (1, "Apple"))
    body = {"filings": {"recent": {"form": ["8-K"], "items": ["2.02,9.01"],
                                   "acceptanceDateTime": ["2026-11-02T21:30:23.000Z"]}}}
    monkeypatch.setattr(et.requests, "get", lambda *a, **k: type("R", (), {"json": lambda self: body})())
    filed = et._latest_results_8k("AAPL")
    assert (filed.hour, filed.minute) == (16, 30)


def test_blank_latest_price_is_skipped(monkeypatch):
    import pandas as pd
    now = et.dt.datetime.now(et.ET)
    day = now.date()
    daily = pd.DataFrame({"Close": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp(day)]).tz_localize("America/New_York"))
    live = pd.DataFrame({"Close": [101.0, float("nan")]},
                        index=pd.date_range(pd.Timestamp(now), periods=2, freq="min"))

    class T:
        def history(self, **k):
            return live if "interval" in k else daily

    monkeypatch.setattr(et.yf, "Ticker", lambda t: T())
    assert et._price_move("AAPL", now, False) == (1.0, "after hours")
