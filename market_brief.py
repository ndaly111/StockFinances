"""Market brief: the top financial news, written for the top of the homepage.

What Nick asked for (2026-10-28): on an FOMC day, the decision, a short excerpt
of why, and anything notable in the dissents; after a CPI print, the number and
what drove it; on a quiet day, the two or three stories that matter for markets.
Built from news headlines (plus the article text where the site lets us read
it), NOT by scraping the Fed/BLS ourselves.

Runs on Nick's PC, not in GitHub Actions: the writer is `claude -p` on his
Claude subscription (no API key), same as earnings_takeaways.py. Headlines and
article bodies are fetched here and handed in as the only source material; the
model gets no tools, so it cannot wander.

Output (committed to main; the homepage fetches it straight from
raw.githubusercontent.com so it shows within minutes, and the normal site
refresh also deploys a copy):
  data/market_brief.json

Usage:
  python market_brief.py --dry-run        # print the prompt + brief, write nothing
  python market_brief.py                  # write data/market_brief.json
  python market_brief.py --push           # ...and commit + push to main
  --force rewrites even when the headlines haven't changed.

Cost control: the brief is rewritten only when the headline set has changed
meaningfully (Jaccard < SKIP_IF_SIMILAR) or the last brief is > MAX_BRIEF_AGE_H
old, so a 30-minute schedule costs a handful of calls a day.
"""
from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import html
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUT_PATH = Path("data/market_brief.json")

FEEDS = [
    ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC Economy", "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    ("WSJ Markets", "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain"),
    ("Google News Business",
     "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en"),
]
# Sites whose article bodies we can actually read (WSJ 401s, Google News links
# are JS redirects). Headlines + RSS summaries are used for everything else.
ARTICLE_HOSTS = ("cnbc.com",)

MAX_AGE_HOURS = 36          # ignore feed items older than this
MAX_HEADLINES = 45          # headlines handed to the writer
MAX_ARTICLES = 6            # article bodies handed to the writer
ARTICLE_CHARS = 2500
MAX_STORIES = 4
MAX_SECTIONS = 4
SKIP_IF_SIMILAR = 0.6       # Jaccard of headline sets at/above this = "nothing new"
MAX_BRIEF_AGE_H = 3         # ...unless the last brief is older than this

CLAUDE_MODEL = os.environ.get("MARKET_BRIEF_MODEL", "opus")
CLAUDE_TIMEOUT_S = 600
HTTP_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124 Safari/537.36"),
    "Accept-Language": "en-US",
}
UTC = dt.timezone.utc


# ───────── headlines ────────────────────────────────────────────────
def _clean(text: str) -> str:
    """RSS summaries arrive as HTML (Google News ships a whole <ol> of related
    headlines). Flatten to one line of plain text."""
    if not text:
        return ""
    soup = BeautifulSoup(html.unescape(text), "html.parser")
    out = " ".join(soup.get_text(" ", strip=True).split())
    return out[:400]


def _parse_date(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        d = email.utils.parsedate_to_datetime(s)
    except (TypeError, ValueError):
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    return d.astimezone(UTC)


def parse_feed(xml_bytes: bytes, source: str, now: dt.datetime | None = None) -> list[dict]:
    now = now or dt.datetime.now(UTC)
    cutoff = now - dt.timedelta(hours=MAX_AGE_HOURS)
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        print(f"[brief] {source}: feed unparsable ({exc})")
        return []
    items = []
    for it in root.iter("item"):
        title = " ".join((it.findtext("title") or "").split())
        if not title:
            continue
        pub = _parse_date(it.findtext("pubDate"))
        if pub is not None and pub < cutoff:
            continue
        items.append({
            "title": title,
            "link": (it.findtext("link") or "").strip(),
            "summary": _clean(it.findtext("description") or ""),
            "published": (pub or now).strftime("%Y-%m-%dT%H:%MZ"),
            "source": source,
        })
    return items


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", title.lower()).strip()


def dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        key = _norm(it["title"])
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def fetch_headlines(now: dt.datetime | None = None) -> list[dict]:
    items = []
    for source, url in FEEDS:
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=15)
            r.raise_for_status()
            got = parse_feed(r.content, source, now=now)
            print(f"[brief] {source}: {len(got)} fresh items")
            items.extend(got)
        except Exception as exc:
            print(f"[brief] {source}: fetch failed ({exc})")
    return dedupe(items)[:MAX_HEADLINES]


# ───────── article bodies (best effort) ─────────────────────────────
def fetch_article(url: str) -> str:
    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=12)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        paras = [p for p in paras if len(p) > 60]
        return " ".join(paras)[:ARTICLE_CHARS]
    except Exception:
        return ""


def fetch_articles(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        if len(out) >= MAX_ARTICLES:
            break
        if not any(h in it["link"] for h in ARTICLE_HOSTS):
            continue
        text = fetch_article(it["link"])
        if text:
            out.append({"title": it["title"], "source": it["source"], "text": text})
    print(f"[brief] read {len(out)} article bodies")
    return out


# ───────── change gate ──────────────────────────────────────────────
def fingerprint(items: list[dict]) -> list[str]:
    return sorted(_norm(i["title"]) for i in items[:25])


def _parse_updated(s: str | None) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def should_skip(prev: dict | None, titles: list[str], now: dt.datetime, force: bool = False) -> bool:
    """True when the headlines are essentially what the last brief already
    covered AND that brief is recent enough to leave alone."""
    if force or not prev:
        return False
    last = _parse_updated(prev.get("updated"))
    if last is None or now - last > dt.timedelta(hours=MAX_BRIEF_AGE_H):
        return False
    a, b = set(prev.get("fingerprint") or []), set(titles)
    if not a or not b:
        return False
    jaccard = len(a & b) / len(a | b)
    return jaccard >= SKIP_IF_SIMILAR


# ───────── the writer (claude -p on the subscription) ───────────────
PROMPT = """You write the "Top Financial News" box at the top of a personal stock website.
The reader is a well-informed investor who wants, in under a minute, what actually happened
in markets and the economy and why. Today is {today} (Eastern). Write from the material
below ONLY; you have no tools and must not use outside knowledge for facts.

SHAPE
- If one event dominates (a Fed decision, a CPI/jobs/GDP print, a major policy or market
  shock), write ONE story, 110-160 words total, with sections in this order:
  "Decision" (or "The number" / "What happened"), "Why", then "Dissents" or "Details" if
  there is something notable, then stop. Title it like a dateline, e.g.
  "FOMC · Oct 28, 2026 · Held at 3.75–4.00% · Vote 10–2".
- Otherwise write 2-3 stories, the ones that matter most for markets, 35-55 words each,
  with one or two sections each. Title = a plain headline.
- "next": one line on the next scheduled market-moving event if the material names one
  (the FOMC calendar line below counts), else "".

RULES
- Only what the material supports. If the material does not say why, say what it does say
  and stop. Never pad: no "details to come", no "markets will be watching", no sentences
  like "reasoning comes with the minutes". Every sentence must carry a fact.
- Mark speculation as speculation ("some coverage calls it a protest vote, unconfirmed").
- Short quoted phrases only, and only phrases that appear in the material.
- Numbers: include the ones that matter (rate range, vote count, the print vs expectations,
  % moves). No invented precision.
- Plain English, no first person, no investment advice, no exclamation marks.

Reply with ONLY this JSON object, nothing else:
{{"stories": [{{"title": "...", "sections": [{{"label": "...", "text": "..."}}]}}], "next": "..."}}

FOMC calendar (from the site's own calendar): next decision {next_fomc}

HEADLINES (newest first within each source; summary after the dash when there is one)
{headlines}

ARTICLE TEXT (first part of the stories we could read)
{articles}
"""


def build_prompt(items: list[dict], articles: list[dict], now: dt.datetime, next_fomc: str = "unknown") -> str:
    lines = []
    for it in items:
        line = f"- [{it['source']} {it['published'][11:16]}Z] {it['title']}"
        if it["summary"] and _norm(it["summary"]) != _norm(it["title"]):
            line += f" — {it['summary']}"
        lines.append(line)
    arts = []
    for a in articles:
        arts.append(f"### {a['title']} ({a['source']})\n{a['text']}\n")
    et = now.astimezone(dt.timezone(dt.timedelta(hours=-4)))  # label only; cron runs in ET anyway
    return PROMPT.format(today=et.strftime("%A %b %d, %Y"), next_fomc=next_fomc or "unknown",
                         headlines="\n".join(lines) or "(none)",
                         articles="\n".join(arts) or "(none readable)")


def parse_brief(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise RuntimeError(f"no JSON in writer output: {text[-300:]!r}")
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"writer JSON invalid ({exc}): {m.group(0)[-300:]!r}")
    stories = []
    for s in (raw.get("stories") or [])[:MAX_STORIES]:
        title = " ".join(str(s.get("title") or "").split())
        secs = [{"label": " ".join(str(x.get("label") or "").split()),
                 "text": " ".join(str(x.get("text") or "").split())}
                for x in (s.get("sections") or [])]
        secs = [x for x in secs if x["label"] and x["text"]][:MAX_SECTIONS]
        if title and secs:
            stories.append({"title": title, "sections": secs})
    if not stories:
        raise RuntimeError(f"writer returned no usable stories: {text[-300:]!r}")
    nxt = " ".join(str(raw.get("next") or "").split())
    return {"stories": stories, "next": nxt}


def write_with_claude(prompt: str) -> dict:
    cmd = ["claude", "-p", "--model", CLAUDE_MODEL, "--tools", "", "--output-format", "text"]
    res = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                         encoding="utf-8", timeout=CLAUDE_TIMEOUT_S, shell=(os.name == "nt"))
    if res.returncode != 0:
        raise RuntimeError(f"claude failed (rc={res.returncode}): {(res.stdout or res.stderr)[-400:]}")
    return parse_brief(res.stdout)


# ───────── output ───────────────────────────────────────────────────
def make_doc(brief: dict, items: list[dict], now: dt.datetime) -> dict:
    return {
        "updated": now.strftime("%Y-%m-%dT%H:%MZ"),
        "stories": brief["stories"],
        "next": brief.get("next", ""),
        "sources": [{"title": i["title"], "link": i["link"], "source": i["source"]}
                    for i in items[:12]],
        "fingerprint": fingerprint(items),
    }


def load_prev() -> dict | None:
    try:
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def next_fomc_label(now: dt.datetime | None = None) -> str:
    """Next FOMC decision AFTER today: on a decision day the brief covers the
    decision itself, so "next" must point at the following meeting."""
    now = now or dt.datetime.now(UTC)
    tomorrow = now.astimezone(dt.timezone(dt.timedelta(hours=-5))).date() + dt.timedelta(days=1)
    try:
        from generate_economic_data import _next_release
        return _next_release("fomc", today=tomorrow)
    except Exception as exc:
        print(f"[brief] FOMC calendar unavailable ({exc})")
        return "unknown"


def push() -> None:
    subprocess.run(["git", "add", str(OUT_PATH)], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        print("[brief] nothing to commit")
        return
    subprocess.run(["git", "commit", "-q", "-m", "Market brief: update"], check=True)
    for attempt in range(4):
        subprocess.run(["git", "pull", "-q", "--rebase", "origin", "main"], check=True)
        if subprocess.run(["git", "push", "-q", "origin", "HEAD:main"]).returncode == 0:
            print("[brief] pushed to main")
            return
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError("push failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print prompt + brief, write nothing")
    ap.add_argument("--force", action="store_true", help="rewrite even if headlines are unchanged")
    ap.add_argument("--push", action="store_true", help="commit + push data/market_brief.json to main")
    a = ap.parse_args()

    now = dt.datetime.now(UTC)
    items = fetch_headlines(now)
    if len(items) < 5:
        print(f"[brief] only {len(items)} headlines — keeping the previous brief")
        return 0
    titles = fingerprint(items)
    if not a.dry_run and should_skip(load_prev(), titles, now, force=a.force):
        print("[brief] headlines unchanged and brief is fresh — skipping")
        return 0

    articles = fetch_articles(items)
    prompt = build_prompt(items, articles, now, next_fomc=next_fomc_label(now))
    if a.dry_run:
        print(prompt)
    brief = write_with_claude(prompt)
    doc = make_doc(brief, items, now)
    words = sum(len(x["text"].split()) for s in doc["stories"] for x in s["sections"])
    print(f"[brief] {len(doc['stories'])} stories, {words} words")
    if a.dry_run:
        print(json.dumps({k: doc[k] for k in ("updated", "stories", "next")}, indent=1, ensure_ascii=False))
        return 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[brief] wrote {OUT_PATH}")
    if a.push:
        push()
    return 0


if __name__ == "__main__":
    sys.exit(main())
