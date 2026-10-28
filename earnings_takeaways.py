"""Earnings takeaways: "why did the stock move" after each report.

For every site ticker that just reported, writes a short card (the price move,
a one-line verdict, up to 3 reasons) plus a 2-3 sentence "story lately" that
ties the last few reports together.

Runs on Nick's PC, not in GitHub Actions: the writer is `claude -p` on his
Claude subscription (no API key), with WebSearch/WebFetch so it can read the
call transcript and same-day coverage. The official SEC 8-K press release and
the price move are computed here and handed in as ground truth.

Output (committed to main, deployed by the normal site refresh):
  data/earnings_takeaways/<T>.json   -- read by ticker.html
  data/earnings_takeaways/recent.json -- read by the homepage section

Usage:
  python earnings_takeaways.py                 # new reports only (last 7 days)
  python earnings_takeaways.py --tickers V GOOGL --force
  python earnings_takeaways.py --backfill 15   # up to 15 tickers with no card yet
  add --push to commit + push the JSON to main
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
import yfinance as yf
from bs4 import BeautifulSoup

from ticker_manager import read_tickers

OUT_DIR = Path("data/earnings_takeaways")
TICKERS_FILE = "tickers.csv"
NEW_REPORT_DAYS = 7       # auto mode only writes cards for reports this recent
RECENT_DAYS = 4           # homepage shows reports from the last N days
HISTORY_KEEP = 4          # past cards kept per ticker (feed the "story")
PRESS_RELEASE_CHARS = 30000
CLAUDE_MODEL = os.environ.get("TAKEAWAYS_MODEL", "sonnet")
CLAUDE_TIMEOUT_S = 900
SEC_HEADERS = {"User-Agent": "Nick's Stock Financials ndaly111@gmail.com"}


# ───────── facts we compute ourselves ─────────────────────────────

def _reports(ticker):
    """Past earnings reports, newest first: [(datetime ET, pre_market bool, confirmed)].

    Yahoo fills in "Reported EPS" hours after a release, so a report whose
    scheduled time has passed counts too -- unconfirmed ones must have a
    results 8-K on EDGAR before a card is written (see update_ticker).
    """
    df = yf.Ticker(ticker).get_earnings_dates(limit=12)
    if df is None or df.empty:
        return []
    out = []
    for ts, row in df.iterrows():
        when = ts.to_pydatetime()
        confirmed = row.get("Reported EPS") == row.get("Reported EPS")  # not NaN
        if not confirmed and when > dt.datetime.now(when.tzinfo) - dt.timedelta(minutes=20):
            continue
        out.append((when, when.hour < 12, confirmed))
    return sorted(out, reverse=True)


def _price_move(ticker, when, pre_market):
    """Reaction to the report: close before vs close of the first session after.

    Returns (pct, basis) where basis is 'close' when the reaction session has
    closed, else 'after hours' / 'intraday' from the latest print. None if no data.
    """
    day = when.date()
    hist = yf.Ticker(ticker).history(start=day - dt.timedelta(days=10),
                                     end=day + dt.timedelta(days=6), auto_adjust=False)
    if hist.empty:
        return None
    closes = hist["Close"]
    dates = [d.date() for d in closes.index]
    before = [i for i, d in enumerate(dates) if (d < day if pre_market else d <= day)]
    if not before:
        return None
    base = float(closes.iloc[before[-1]])
    after = [i for i, d in enumerate(dates) if (d >= day if pre_market else d > day)]
    today = dt.datetime.now(when.tzinfo).date()
    if after and (dates[after[0]] < today or dt.datetime.now(when.tzinfo).hour >= 16):
        return round((float(closes.iloc[after[0]]) / base - 1) * 100, 1), "close"
    # Reaction session not closed yet: use the latest print (incl. extended hours).
    live = yf.Ticker(ticker).history(period="2d", interval="1m", prepost=True)
    if live.empty:
        return None
    basis = "intraday" if after else "after hours"
    return round((float(live["Close"].iloc[-1]) / base - 1) * 100, 1), basis


_SEC_MAP = None


def _sec_lookup(ticker):
    global _SEC_MAP
    if _SEC_MAP is None:
        raw = requests.get("https://www.sec.gov/files/company_tickers.json",
                           headers=SEC_HEADERS, timeout=20).json()
        _SEC_MAP = {v["ticker"].upper(): (int(v["cik_str"]), v["title"]) for v in raw.values()}
    t = ticker.upper()
    return _SEC_MAP.get(t.replace("-", ".")) or _SEC_MAP.get(t)


def _press_release(ticker, when):
    """(url, text) of the results 8-K exhibit 99.1 filed around the report date."""
    hit = _sec_lookup(ticker)
    if not hit:
        return None, ""
    cik = hit[0]
    sub = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                       headers=SEC_HEADERS, timeout=20).json()["filings"]["recent"]
    lo = (when.date() - dt.timedelta(days=2)).isoformat()
    hi = (when.date() + dt.timedelta(days=3)).isoformat()
    for form, date, items, acc in zip(sub["form"], sub["filingDate"], sub["items"],
                                      sub["accessionNumber"]):
        if form != "8-K" or "2.02" not in items or not (lo <= date <= hi):
            continue
        folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}"
        time.sleep(0.15)
        idx = requests.get(f"{folder}/index.json", headers=SEC_HEADERS, timeout=20).json()
        names = [f["name"] for f in idx["directory"]["item"]]
        ex = [n for n in names if re.search(r"ex-?99|exhibit99|ex99", n, re.I)
              and n.lower().endswith((".htm", ".html", ".txt"))]
        if not ex:
            continue
        url = f"{folder}/{sorted(ex)[0]}"
        time.sleep(0.15)
        html = requests.get(url, headers=SEC_HEADERS, timeout=30).text
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        return url, re.sub(r"\s+", " ", text)[:PRESS_RELEASE_CHARS]
    return None, ""


# ───────── the writer (claude -p on the subscription) ─────────────

PROMPT = """You write a tiny "why did the stock move after earnings" note for a personal stock website.
The reader wants the real reason a stock jumped or dropped, in plain English, readable in 10 seconds.

Company: {name} ({ticker})
Earnings released: {when} ({timing})
Stock reaction: {move} ({basis})
Official press release (SEC 8-K exhibit): {pr_url}

Research first: use WebSearch/WebFetch to read the earnings call transcript (Motley Fool,
Investing.com, company IR, etc.) and same-day coverage of the stock's reaction. The press
release text below and the price move above are ground truth; do not contradict them.

TASK 1, the card for this report:
- "verdict": one sentence, max 12 words, that says what the market made of the report.
- "reasons": up to 3, ordered by how much they explain the move. Each has a "label"
  (2-4 words, e.g. "Slower growth ahead") and "text" (max 14 words, one concrete number where it helps).
  Only things that explain the move or that investors focused on: guidance, growth trend,
  margins, spending/capex, a key metric vs expectations, one-time items that flattered or
  hurt results, legal/regulatory news, management change. No filler metrics, no list of
  segment results, no "what wasn't the story" items.
- If the stock barely moved, say what investors were weighing.

TASK 2, "story": 2-3 short sentences, max 45 words, on what has been going on across the last
~4 earnings reports: the trend in growth and guidance, recurring worries or tailwinds,
how the stock has reacted. {prior}

Plain English, no jargon a casual investor wouldn't know (say "payments to banks to keep
them on Visa", not "client incentives"). No first person ("I", "we"), no investment advice. Every number must come from
a source you read.

Reply with ONLY this JSON object, nothing else:
{{"verdict": "...", "reasons": [{{"label": "...", "text": "..."}}], "story": "...", "sources": ["url", "..."]}}

Press release text:
{pr_text}
"""


def _write_with_claude(prompt):
    cmd = ["claude", "-p", "--model", CLAUDE_MODEL, "--allowedTools", "WebSearch,WebFetch",
           "--max-turns", "30", "--output-format", "text"]
    res = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                         encoding="utf-8", timeout=CLAUDE_TIMEOUT_S)
    out = res.stdout.strip()
    m = re.search(r"\{.*\}", out, re.S)
    if res.returncode != 0 or not m:
        raise RuntimeError(f"claude failed (rc={res.returncode}): {(out or res.stderr)[-400:]}")
    card = json.loads(m.group(0))
    reasons = [r for r in card.get("reasons", []) if r.get("label") and r.get("text")][:3]
    if not card.get("verdict") or not reasons:
        raise RuntimeError(f"claude returned an incomplete card: {out[-400:]}")
    card["reasons"] = reasons
    return card


# ───────── per-ticker update ───────────────────────────────────────

def _load(ticker):
    p = OUT_DIR / f"{ticker}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save(ticker, doc):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{ticker}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")


def update_ticker(ticker, force=False, backfill=False):
    """Returns 'written' or None."""
    reports = _reports(ticker)
    if not reports:
        return None
    when, pre, confirmed = reports[0]
    date = when.date().isoformat()
    doc = _load(ticker)
    last = (doc or {}).get("last") or {}

    if last.get("date") == date and not force:
        # Card exists. If it was written before the reaction session closed,
        # rewrite it once that close is in (transcript + full-day coverage).
        if last.get("basis") == "close":
            return None
        mv = _price_move(ticker, when, pre)
        if not mv or mv[1] != "close":
            return None
    else:
        age = (dt.datetime.now(when.tzinfo) - when).days
        if not backfill and not force and age > NEW_REPORT_DAYS:
            return None
        mv = _price_move(ticker, when, pre)
    if not mv:
        print(f"[takeaways] {ticker}: no price data, skipping")
        return None
    pr_url, pr_text = _press_release(ticker, when)
    if not confirmed and not pr_url:
        print(f"[takeaways] {ticker}: {date} not confirmed yet (no EPS, no 8-K), skipping")
        return None
    name = (_sec_lookup(ticker) or (0, ticker))[1]

    history = [h for h in (doc or {}).get("history", []) if h.get("date") != date]
    if last and last.get("date") != date:
        history.insert(0, {k: last[k] for k in ("date", "move_pct", "verdict", "reasons") if k in last})
    history = history[:HISTORY_KEEP - 1]
    if history:
        prior = "Cards we wrote for the previous reports:\n" + "\n".join(
            f"- {h['date']} ({h['move_pct']:+.1f}%): {h['verdict']} "
            + " ".join(f"{r['label']}: {r['text']}" for r in h["reasons"]) for h in history)
    else:
        prior = ("We have no notes on earlier reports; briefly research the previous ~3 "
                 "earnings reports and the stock's reactions to them.")

    prompt = PROMPT.format(
        name=name, ticker=ticker,
        when=when.strftime("%Y-%m-%d %H:%M ET"),
        timing="before the open" if pre else "after the close",
        move=f"{mv[0]:+.1f}%", basis={"close": "first session's close vs prior close",
                                      "after hours": "after-hours, session not closed yet",
                                      "intraday": "intraday, session not closed yet"}[mv[1]],
        pr_url=pr_url or "not found on EDGAR; rely on the transcript and coverage",
        pr_text=pr_text or "(none)", prior=prior)
    card = _write_with_claude(prompt)

    sources = [s for s in card.get("sources", []) if isinstance(s, str)][:6]
    if pr_url and pr_url not in sources:
        sources.insert(0, pr_url)
    doc = {
        "ticker": ticker,
        "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "last": {"date": date, "timing": "pre" if pre else "post",
                 "move_pct": mv[0], "basis": mv[1], "verdict": card["verdict"],
                 "reasons": card["reasons"], "sources": sources},
        "story": card.get("story", ""),
        "history": history,
    }
    _save(ticker, doc)
    return "written"


def write_recent():
    cutoff = (dt.date.today() - dt.timedelta(days=RECENT_DAYS)).isoformat()
    rows = []
    for p in OUT_DIR.glob("*.json"):
        if p.name == "recent.json":
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        last = d.get("last") or {}
        if last.get("date", "") >= cutoff:
            rows.append({"ticker": d["ticker"], **{k: last[k] for k in
                         ("date", "move_pct", "basis", "verdict", "reasons")}})
    rows.sort(key=lambda r: (r["date"], abs(r["move_pct"])), reverse=True)
    (OUT_DIR / "recent.json").write_text(json.dumps(
        {"updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
         "reports": rows}, indent=1), encoding="utf-8")
    return len(rows)


def _push():
    subprocess.run(["git", "add", str(OUT_DIR)], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        print("[takeaways] nothing to commit")
        return
    subprocess.run(["git", "commit", "-q", "-m", "Earnings takeaways: update cards"], check=True)
    for attempt in range(4):
        subprocess.run(["git", "pull", "-q", "--rebase", "origin", "main"], check=True)
        if subprocess.run(["git", "push", "-q", "origin", "HEAD:main"]).returncode == 0:
            print("[takeaways] pushed to main")
            return
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError("push failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--force", action="store_true", help="rewrite even if a card exists")
    ap.add_argument("--backfill", type=int, default=0,
                    help="also write cards for up to N tickers that have none yet")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    tickers = a.tickers or sorted(read_tickers(TICKERS_FILE))
    missing = [t for t in tickers if not (OUT_DIR / f"{t}.json").exists()]
    backfill = set(missing[:a.backfill])
    counts = {}
    for t in tickers:
        try:
            r = update_ticker(t, force=a.force, backfill=t in backfill)
        except Exception as exc:
            print(f"[takeaways] {t}: {exc}")
            r = "error"
        if r:
            counts[r] = counts.get(r, 0) + 1
            print(f"[takeaways] {t}: {r}")
    n = write_recent()
    print(f"[takeaways] {counts or 'no changes'}; {n} recent reports")
    if a.push:
        _push()
    return 1 if counts.get("error") and len(counts) == 1 else 0


if __name__ == "__main__":
    sys.exit(main())
