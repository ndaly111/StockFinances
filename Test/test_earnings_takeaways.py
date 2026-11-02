import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import earnings_takeaways as et  # noqa: E402

ITEM = {"label": "Outlook", "text": "Revenue up 10%."}


def test_card_without_call_is_valid_until_transcript_posts():
    card = et._clean_card({"verdict": "v", "numbers": "n", "call": [], "news": [ITEM]})
    assert card["call"] == [] and card["news"] == [ITEM]


@pytest.mark.parametrize("bad", [
    {"numbers": "n", "call": [ITEM], "news": []},
    {"verdict": "v", "call": [ITEM], "news": []},
    {"verdict": "v", "numbers": "n", "call": [], "news": []},
])
def test_incomplete_card_is_rejected(bad):
    with pytest.raises(RuntimeError):
        et._clean_card(bad)


def test_malformed_items_dropped_and_lists_capped():
    card = et._clean_card({"verdict": "v", "numbers": "n", "call": [{"label": "x"}] + [ITEM] * 9,
                           "news": [ITEM] * 9})
    assert len(card["call"]) == 4 and len(card["news"]) == 3


def test_prompt_formats_and_asks_for_call_and_news():
    text = et.PROMPT.format(name="Apple", ticker="AAPL", when="w", timing="t", pr_url="u",
                            prior="p", pr_text="x")
    assert '"call"' in text and '"news"' in text and "{move}" not in text


def test_recent_handles_old_and_new_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUT_DIR", tmp_path)
    today = et.dt.date.today().isoformat()
    base = {"date": today, "move_pct": 1.0, "basis": "close", "verdict": "v"}
    (tmp_path / "OLD.json").write_text(json.dumps(
        {"ticker": "OLD", "last": {**base, "reasons": [ITEM]}}), encoding="utf-8")
    (tmp_path / "NEW.json").write_text(json.dumps(
        {"ticker": "NEW", "last": {**base, "numbers": "n", "call": [ITEM], "news": []}}), encoding="utf-8")
    assert et.write_recent() == 2
    rows = {r["ticker"]: r for r in json.loads((tmp_path / "recent.json").read_text(encoding="utf-8"))["reports"]}
    assert rows["OLD"]["reasons"] == [ITEM] and "call" not in rows["OLD"]
    assert rows["NEW"]["call"] == [ITEM] and "reasons" not in rows["NEW"]


def _setup_existing(monkeypatch, last, move):
    when = et.dt.datetime.now(et.dt.timezone.utc) - et.dt.timedelta(hours=3)
    monkeypatch.setattr(et, "_reports", lambda t: [(when, False, True)])
    monkeypatch.setattr(et, "_load", lambda t: {"ticker": t, "last": {"date": when.date().isoformat(), **last}})
    monkeypatch.setattr(et, "_price_move", lambda *a: move)
    monkeypatch.setattr(et, "_press_release", lambda *a: ("u", "x"))
    monkeypatch.setattr(et, "_sec_lookup", lambda t: (1, "Apple"))
    monkeypatch.setattr(et, "_write_with_claude", lambda p: {"verdict": "v", "numbers": "n", "call": [ITEM], "news": []})
    saved = {}
    monkeypatch.setattr(et, "_save", lambda t, d: saved.update(d))
    return saved


def test_empty_call_card_is_retried_before_the_close(monkeypatch):
    saved = _setup_existing(monkeypatch, {"basis": "after hours", "call": []}, (0.1, "after hours"))
    assert et.update_ticker("AAPL") == "written"
    assert saved["last"]["call"] == [ITEM]


def test_complete_after_hours_card_waits_for_the_close(monkeypatch):
    _setup_existing(monkeypatch, {"basis": "after hours", "call": [ITEM]}, (0.1, "after hours"))
    assert et.update_ticker("AAPL") is None


def test_finished_card_is_left_alone(monkeypatch):
    _setup_existing(monkeypatch, {"basis": "close", "call": [ITEM]}, (0.1, "close"))
    assert et.update_ticker("AAPL") is None


def test_old_format_card_is_not_rewritten(monkeypatch):
    _setup_existing(monkeypatch, {"basis": "close", "reasons": [ITEM]}, (0.1, "close"))
    assert et.update_ticker("AAPL") is None
