"""Homepage "What happened to my stocks today" box.

Only covers tickers on the site, and only when something happened:
  1. Big moves (|day change| >= MOVE_PCT) with the headline that explains why.
  2. New SEC 8-K filings that matter (earnings, guidance, CEO change, deals...),
     straight from EDGAR -- official, no clickbait.
Earnings beats/misses and upcoming dates are already on the homepage tables.

Output: charts/news_box.html (pulled into index.html by html_generator2).
"""

from __future__ import annotations

import datetime as dt
import html
import re
import time

import requests
import yfinance as yf

from ticker_manager import read_tickers
from utils.news import _fetch_headlines_with_dates

OUTPUT_PATH = "charts/news_box.html"
TICKERS_FILE = "tickers.csv"
MOVE_PCT = 4.0
MAX_MOVERS = 10
HEADLINE_MAX_AGE_H = 36
FILING_LOOKBACK_DAYS = 2
SEC_HEADERS = {"User-Agent": "Nick's Stock Financials ndaly111@gmail.com"}

# 8-K items worth surfacing, in plain language. 7.01/8.01/9.01 (Reg FD,
# "other events", exhibits) are mostly noise and are left out on their own.
ITEM_LABELS = {
    "1.01": "Signed a major agreement",
    "1.02": "Ended a major agreement",
    "1.03": "Bankruptcy",
    "2.01": "Completed an acquisition or sale",
    "2.02": "Reported results",
    "2.03": "Took on major debt",
    "2.05": "Restructuring / layoffs",
    "2.06": "Impairment / write-down",
    "3.01": "Delisting notice",
    "3.02": "Sold unregistered shares",
    "4.01": "Changed auditor",
    "4.02": "Prior financials can't be relied on",
    "5.01": "Change in control",
    "5.02": "Executive or board change",
}

CLICKBAIT = re.compile(
    r"\b(stocks? to buy|should you|is it too late|too late to buy|millionaire|"
    r"forever|buy now|prediction:|here'?s why|reasons? to|retire|"
    r"\d+ (stocks|reasons|things)|best .* (stocks?|play)|could soar|could make you|"
    r"no-brainer|dividend kings?|wall street (thinks|says)|motley|"
    r"\d+ [\w-]+ stocks?|concern us|imploded|respected|about to (double|soar)|"
    r"what to know|what investors|top picks?|should investors)\b",
    re.I,
)


def _day_moves(tickers):
    """Return {ticker: (pct_change, last_price)} from the last two daily bars."""
    data = yf.download(
        tickers, period="5d", interval="1d", auto_adjust=False,
        progress=False, threads=True, group_by="column",
    )
    closes = data["Close"]
    moves = {}
    for t in tickers:
        if t not in closes:
            continue
        s = closes[t].dropna()
        if len(s) < 2 or not s.iloc[-2]:
            continue
        moves[t] = ((s.iloc[-1] / s.iloc[-2] - 1) * 100, float(s.iloc[-1]))
    return moves


def _why_headline(ticker, name=""):
    """Newest recent, non-clickbait headline that names the company."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=HEADLINE_MAX_AGE_H)
    word = name.split()[0].lower() if name else ""
    for item in _fetch_headlines_with_dates(ticker, max_items=20):
        title = item["title"]
        if item["_pub_date"] < cutoff or CLICKBAIT.search(title):
            continue
        if ticker.lower() not in title.lower() and (not word or word not in title.lower()):
            continue
        return item
    return None


def _sec_ticker_map():
    """{TICKER: (cik, company name)} from SEC's public ticker file."""
    try:
        raw = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS, timeout=20,
        ).json()
    except Exception as exc:
        print(f"[WARN] news box: SEC ticker map failed: {exc}")
        return {}
    return {v["ticker"].upper(): (int(v["cik_str"]), v["title"]) for v in raw.values()}


def _lookup(sec_map, ticker):
    return sec_map.get(ticker.upper().replace("-", ".")) or sec_map.get(ticker.upper())


def _recent_filings(tickers, sec_map):
    # Last FILING_LOOKBACK_DAYS business days, so a Monday still shows Friday.
    since = dt.date.today()
    for _ in range(FILING_LOOKBACK_DAYS):
        since -= dt.timedelta(days=1)
        while since.weekday() >= 5:
            since -= dt.timedelta(days=1)
    since = since.isoformat()
    out = []
    for t in tickers:
        hit = _lookup(sec_map, t)
        if hit is None:
            continue
        cik = hit[0]
        try:
            sub = requests.get(
                f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                headers=SEC_HEADERS, timeout=20,
            ).json()["filings"]["recent"]
        except Exception as exc:
            print(f"[WARN] news box: SEC filings for {t} failed: {exc}")
            continue
        finally:
            time.sleep(0.12)  # SEC fair-access limit is 10 req/s
        for form, date, items, acc, doc in zip(
            sub["form"], sub["filingDate"], sub["items"],
            sub["accessionNumber"], sub["primaryDocument"],
        ):
            if date < since:
                break  # newest first
            if form != "8-K":
                continue
            labels = [ITEM_LABELS[i] for i in items.split(",") if i in ITEM_LABELS]
            if not labels:
                continue
            url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                   f"{acc.replace('-', '')}/{doc}")
            out.append((date, t, "; ".join(dict.fromkeys(labels)), url))
    out.sort(reverse=True)
    return out


def _render(movers, filings, stamp):
    e = html.escape
    parts = [f'<p class="news-stamp">Updated {e(stamp)} · moves of {MOVE_PCT:.0f}%+ '
             f'and new SEC filings for tickers on this site</p>']

    parts.append("<h3>Big moves</h3>")
    if movers:
        parts.append('<ul class="news-list">')
        for t, pct, price, head in movers:
            cls = "up" if pct > 0 else "down"
            line = (f'<a href="ticker.html?t={e(t)}"><b>{e(t)}</b></a> '
                    f'<span class="{cls}">{pct:+.1f}%</span> (${price:,.2f})')
            if head:
                line += f' — <a href="{e(head["link"])}" target="_blank" rel="noopener">{e(head["title"])}</a>'
            else:
                line += ' — <i>no explanation in the news yet</i>'
            parts.append(f"<li>{line}</li>")
        parts.append("</ul>")
    else:
        parts.append(f'<p class="news-quiet">Nothing moved {MOVE_PCT:.0f}% or more today.</p>')

    parts.append("<h3>Company filings (last 2 business days)</h3>")
    if filings:
        parts.append('<ul class="news-list">')
        for date, t, label, url in filings:
            parts.append(
                f'<li><a href="ticker.html?t={e(t)}"><b>{e(t)}</b></a> {e(label)} '
                f'<span class="news-date">{e(date)}</span> · '
                f'<a href="{e(url)}" target="_blank" rel="noopener">filing</a></li>'
            )
        parts.append("</ul>")
    else:
        parts.append('<p class="news-quiet">No major filings.</p>')
    return "\n".join(parts)


def generate_news_box(output_path: str = OUTPUT_PATH) -> str:
    tickers = sorted(read_tickers(TICKERS_FILE))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sec_map = _sec_ticker_map()
    movers = []
    try:
        moves = _day_moves(tickers)
        big = sorted(
            ((t, p, px) for t, (p, px) in moves.items() if abs(p) >= MOVE_PCT),
            key=lambda r: -abs(r[1]),
        )[:MAX_MOVERS]
        movers = [(t, p, px, _why_headline(t, (_lookup(sec_map, t) or (0, ""))[1]))
                  for t, p, px in big]
    except Exception as exc:
        print(f"[WARN] news box: price moves failed: {exc}")

    filings = _recent_filings(tickers, sec_map)
    out = _render(movers, filings, stamp)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[news box] {len(movers)} movers, {len(filings)} filings -> {output_path}")
    return out


if __name__ == "__main__":
    generate_news_box()
