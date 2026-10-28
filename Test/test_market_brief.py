import datetime as dt
import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import market_brief as mb  # noqa: E402

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 28, 20, 30, tzinfo=UTC)

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Fed holds rates steady at 3.75%-4%</title><link>https://x/fed</link>
 <description>Two dissents.</description><pubDate>Wed, 28 Oct 2026 18:05:00 GMT</pubDate></item>
<item><title>Alphabet sinks 10% on capex</title><link>https://x/goog</link>
 <description><![CDATA[<ol><li><a href="https://x/a">Alphabet sinks 10%</a> - WSJ</li></ol>]]></description>
 <pubDate>Wed, 28 Oct 2026 19:00:00 GMT</pubDate></item>
<item><title>Old story from last week</title><link>https://x/old</link>
 <pubDate>Mon, 19 Oct 2026 12:00:00 GMT</pubDate></item>
<item><title>Fed Holds Rates Steady at 3.75%-4%</title><link>https://x/fed-dupe</link>
 <pubDate>Wed, 28 Oct 2026 18:30:00 GMT</pubDate></item>
</channel></rss>"""


def test_parse_feed_strips_html_and_keeps_fresh_only():
    items = mb.parse_feed(RSS, "Test", now=NOW)
    titles = [i["title"] for i in items]
    assert "Old story from last week" not in titles          # > MAX_AGE_HOURS old
    goog = next(i for i in items if i["title"].startswith("Alphabet"))
    assert "<" not in goog["summary"] and "Alphabet sinks 10%" in goog["summary"]
    assert goog["source"] == "Test" and goog["link"] == "https://x/goog"
    assert goog["published"].endswith("Z")


def test_dedupe_is_case_and_punctuation_insensitive():
    items = mb.parse_feed(RSS, "Test", now=NOW)
    deduped = mb.dedupe(items)
    fed = [i for i in deduped if "fed holds" in i["title"].lower()]
    assert len(fed) == 1


def test_change_gate_skips_only_when_similar_and_recent():
    titles = [f"story {i}" for i in range(10)]
    prev = {"updated": "2026-10-28T19:00Z", "fingerprint": titles}
    # same headlines, 90 minutes later -> skip
    assert mb.should_skip(prev, titles, NOW) is True
    # same headlines but the last brief is 4h old -> rewrite
    old = {"updated": "2026-10-28T16:00Z", "fingerprint": titles}
    assert mb.should_skip(old, titles, NOW) is False
    # mostly new headlines -> rewrite even if recent
    fresh = titles[:3] + [f"new {i}" for i in range(7)]
    assert mb.should_skip(prev, fresh, NOW) is False
    # no previous brief -> write
    assert mb.should_skip(None, titles, NOW) is False
    # --force bypasses
    assert mb.should_skip(prev, titles, NOW, force=True) is False


GOOD = json.dumps({
    "stories": [{"title": "FOMC · Oct 28 · Held at 3.75–4.00% · Vote 10–2",
                 "sections": [{"label": "Decision", "text": "No change."},
                              {"label": "Why", "text": "Inflation remains elevated."},
                              {"label": "Dissents", "text": "Cook and Hammack wanted a hike."}]}],
    "next": "Dec 8-9 meeting.",
})


def test_parse_brief_accepts_json_wrapped_in_prose():
    out = mb.parse_brief("Here you go:\n" + GOOD + "\nDone.")
    assert out["stories"][0]["sections"][1]["label"] == "Why"
    assert out["next"] == "Dec 8-9 meeting."


def test_parse_brief_rejects_empty_or_malformed():
    with pytest.raises(RuntimeError):
        mb.parse_brief("no json here")
    with pytest.raises(RuntimeError):
        mb.parse_brief(json.dumps({"stories": []}))
    with pytest.raises(RuntimeError):
        mb.parse_brief(json.dumps({"stories": [{"title": "x", "sections": []}]}))


def test_parse_brief_caps_counts_and_drops_blank_sections():
    doc = {"stories": [{"title": f"s{i}", "sections": [{"label": "A", "text": "t"},
                                                       {"label": "", "text": "dropped"}]}
                       for i in range(6)]}
    out = mb.parse_brief(json.dumps(doc))
    assert len(out["stories"]) == mb.MAX_STORIES
    assert all(len(s["sections"]) == 1 for s in out["stories"])
    assert out["next"] == ""


def test_build_prompt_contains_rules_headlines_and_articles():
    items = mb.dedupe(mb.parse_feed(RSS, "CNBC", now=NOW))
    arts = [{"title": items[0]["title"], "source": "CNBC", "text": "Body text here."}]
    p = mb.build_prompt(items, arts, NOW, next_fomc="Dec 9, 2026 (FOMC)")
    assert "Fed holds rates steady" in p and "Body text here." in p
    assert "Dec 9, 2026" in p
    assert "reasoning comes with the minutes" in p      # the anti-filler example
    assert "JSON" in p


def test_make_doc_records_fingerprint_and_sources():
    items = mb.dedupe(mb.parse_feed(RSS, "CNBC", now=NOW))
    brief = json.loads(GOOD)
    doc = mb.make_doc(brief, items, NOW)
    assert doc["updated"] == "2026-10-28T20:30Z"
    assert doc["fingerprint"] == mb.fingerprint(items)
    assert doc["stories"] == brief["stories"] and doc["next"] == brief["next"]
    assert doc["sources"][0]["link"].startswith("https://x/")


def test_next_fomc_label_skips_a_same_day_meeting(monkeypatch):
    import generate_economic_data as ge
    seen = {}
    def fake(kind, today=None):
        seen["kind"], seen["today"] = kind, today
        return "Dec 9, 2026 (FOMC)"
    monkeypatch.setattr(ge, "_next_release", fake)
    assert mb.next_fomc_label(NOW) == "Dec 9, 2026 (FOMC)"
    assert seen["kind"] == "fomc" and seen["today"] == dt.date(2026, 10, 29)
