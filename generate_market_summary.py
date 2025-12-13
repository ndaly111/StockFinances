#!/usr/bin/env python3
"""
Auto-generate a simple daily US stock market summary as HTML.

Output file: daily-market-summary.html
Place this script in the root of your GitHub Pages repo and
call it from a GitHub Actions workflow.
"""

import datetime as dt
import html
import os
import sys
from typing import List, Tuple, Dict, Optional

import requests
import yfinance as yf

# === Settings ===

OUTPUT_PATH = "daily-market-summary.html"

# Licensed market-news provider endpoints (NewsAPI).
# Requires a NewsAPI key provided via the ``NEWSAPI_API_KEY`` environment variable.
# See https://newsapi.org/docs/terms for attribution requirements (the rendered
# HTML includes a provider credit; do not remove it).
NEWSAPI_ENDPOINTS = [
    "https://newsapi.org/v2/top-headlines?country=us&category=business",
    "https://newsapi.org/v2/everything?q=stock%20market&sortBy=publishedAt&language=en",
]
NEWSAPI_API_KEY_ENV = "NEWSAPI_API_KEY"

INDEX_TICKERS = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "Nasdaq",
}

POSITIVE_WORDS = [
    "rises", "gains", "climbs", "surges", "rallies", "jumps", "soars",
    "higher", "up", "extends rally", "rebound",
]

NEGATIVE_WORDS = [
    "falls", "drops", "slides", "plunges", "tumbles", "sinks", "slumps",
    "lower", "down", "selloff", "sell-off", "slump", "hammered",
]


# === Helpers ===

def get_index_change(ticker: str) -> Optional[Tuple[float, float]]:
    """
    Return (last_close, pct_change_vs_prev_close) for a given index ticker.
    Uses last 2 daily closes. If data can't be fetched, return None.
    """
    try:
        data = yf.download(ticker, period="3d", interval="1d", progress=False)
        if len(data) < 2:
            return None
        prev_close = float(data["Close"].iloc[-2])
        last_close = float(data["Close"].iloc[-1])
        pct = (last_close - prev_close) / prev_close * 100.0
        return last_close, pct
    except Exception as e:
        print(f"Error fetching {ticker}: {e}", file=sys.stderr)
        return None


def fetch_news_items(max_items: int = 10) -> List[Dict[str, str]]:
    """Fetch market headlines from NewsAPI (licensed provider).

    A valid NewsAPI key must be provided through the ``NEWSAPI_API_KEY``
    environment variable. Requests use the official REST API instead of RSS to
    comply with provider terms and avoid the earlier unauthenticated scraping
    approach. Headlines include per-article source names to satisfy attribution
    requirements.
    """

    api_key = os.getenv(NEWSAPI_API_KEY_ENV)
    if not api_key:
        print(f"Missing {NEWSAPI_API_KEY_ENV} (required for NewsAPI headlines)", file=sys.stderr)
        return []

    items: List[Dict[str, str]] = []
    seen_titles = set()
    headers = {"X-Api-Key": api_key}

    for url in NEWSAPI_ENDPOINTS:
        if len(items) >= max_items:
            break

        try:
            resp = requests.get(url, timeout=15, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            articles = payload.get("articles") or []

            for article in articles:
                if len(items) >= max_items:
                    break

                title = (article.get("title") or "").strip()
                link = (article.get("url") or "").strip()
                source = ""
                source_obj = article.get("source") or {}
                if isinstance(source_obj, dict):
                    source = (source_obj.get("name") or "").strip()

                if title and title not in seen_titles:
                    items.append({"title": title, "link": link, "source": source})
                    seen_titles.add(title)
        except Exception as e:
            print(f"Error fetching NewsAPI feed from {url}: {e}", file=sys.stderr)

    return items


def classify_headline(title: str) -> str:
    """
    Classify a headline into 'positive', 'negative', or 'neutral'
    based on presence of simple keywords.
    """
    t = title.lower()
    for w in POSITIVE_WORDS:
        if w in t:
            return "positive"
    for w in NEGATIVE_WORDS:
        if w in t:
            return "negative"
    return "neutral"


def build_html(
    date_str: str,
    index_moves: Dict[str, Optional[Tuple[float, float]]],
    headlines: List[Dict[str, str]],
) -> str:
    """
    Construct the full HTML page for the daily market summary.
    """
    # Split headlines into sections.
    positives: List[Dict[str, str]] = []
    negatives: List[Dict[str, str]] = []
    neutrals: List[Dict[str, str]] = []

    for item in headlines:
        classification = classify_headline(item["title"])
        if classification == "positive":
            positives.append(item)
        elif classification == "negative":
            negatives.append(item)
        else:
            neutrals.append(item)

    def render_index_summary() -> str:
        parts: List[str] = []
        for ticker, name in INDEX_TICKERS.items():
            info = index_moves.get(ticker)
            if not info:
                parts.append(f"<li>{html.escape(name)}: data unavailable</li>")
            else:
                last_close, pct = info
                sign = "+" if pct >= 0 else ""
                parts.append(
                    f"<li><strong>{html.escape(name)}</strong>: "
                    f"{last_close:,.2f} ({sign}{pct:.2f}%)</li>"
                )
        return "<ul class=\"index-grid\">\n" + "\n".join(parts) + "\n</ul>"

    def render_headline_list(items: List[Dict[str, str]]) -> str:
        if not items:
            return "<p class=\"empty-state\">No headlines available (feed temporarily unavailable).</p>"
        lis = []
        for item in items:
            title = html.escape(item["title"])
            link = html.escape(item["link"] or "#")
            source = html.escape(item.get("source") or "")
            source_label = f" <span class=\"source\">({source})</span>" if source else ""
            lis.append(
                f'<li>· <a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a>{source_label}</li>'
            )
        return "<ul class=\"headline-list\">\n" + "\n".join(lis) + "\n</ul>"

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Daily US Stock Market Summary – {html.escape(date_str)}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --panel-bg: #ffffff;
      --panel-border: #b8b8ff;
      --panel-shadow: #8080ff;
      --text: #000080;
      --heading: #cc0000;
      --muted: #333366;
    }}
    body {{
      font-family: Verdana, Geneva, sans-serif;
      background: #f0f0ff;
      color: var(--text);
      margin: 0 auto;
      max-width: 980px;
      padding: 18px;
    }}
    h1, h2 {{
      color: var(--heading);
      text-shadow: 1px 1px var(--text);
      margin: 8px 0;
    }}
    h1 {{
      font-size: 1.6rem;
    }}
    h2 {{
      font-size: 1.2rem;
    }}
    .date {{
      color: var(--muted);
      margin: 2px 0 12px;
      font-size: 0.95rem;
    }}
    .panel {{
      background: var(--panel-bg);
      border: 2px solid var(--panel-border);
      box-shadow: 2px 2px 0 var(--panel-shadow);
      border-radius: 6px;
      padding: 12px 14px;
      margin-bottom: 12px;
    }}
    .panel h2 {{ margin-top: 0; }}
    .headline-list {{
      list-style: none;
      padding-left: 0;
      margin: 8px 0 0;
    }}
    .headline-list li {{ margin: 6px 0; }}
    a {{
      color: #0000cc;
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
    .source {{
      color: var(--muted);
      font-size: 0.9em;
      margin-left: 4px;
    }}
    .index-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 6px 14px;
      padding: 0;
      margin: 6px 0 4px;
      list-style: none;
    }}
    .index-grid li {{ margin: 0; }}
    .empty-state {{
      margin: 6px 0;
      color: var(--muted);
      font-style: italic;
    }}
    .disclaimer {{
      font-size: 0.8rem;
      color: var(--muted);
      margin-top: 14px;
    }}
  </style>
</head>
<body>
  <h1>Daily U.S. Stock Market Summary</h1>
  <p class="date">{html.escape(date_str)}</p>

  <section class="panel">
    <h2>Index Snapshot</h2>
    {render_index_summary()}
    <p class="empty-state">Today’s close vs. the prior trading day.</p>
  </section>

  <section class="panel">
    <h2>What’s going well</h2>
    {render_headline_list(positives)}
  </section>

  <section class="panel">
    <h2>Areas of caution</h2>
    {render_headline_list(negatives)}
  </section>

  <section class="panel">
    <h2>What to watch</h2>
    {render_headline_list(neutrals)}
  </section>

  <p class="disclaimer">
    This page is generated automatically using index data (via Yahoo Finance)
    and licensed market-news headlines delivered by NewsAPI.org. Headlines are
    shown with their original publisher names, and the page is "powered by
    NewsAPI.org" in accordance with provider attribution terms. It is for
    informational purposes only and is not investment advice.
  </p>
</body>
</html>
"""
    return html_page


def generate_market_summary(output_path: str = OUTPUT_PATH) -> str:
    """Build and write the daily market summary HTML.

    Returns the rendered HTML so callers can embed it elsewhere (for example,
    into a dashboard page) while also writing the standalone file.
    """

    # Use US Eastern date for display (simple approximation – system local time).
    today = dt.date.today()
    date_str = today.strftime("%A, %B %d, %Y")

    # 1) Fetch index moves
    index_moves: Dict[str, Optional[Tuple[float, float]]] = {}
    for ticker in INDEX_TICKERS:
        index_moves[ticker] = get_index_change(ticker)

    # 2) Fetch headlines
    headlines = fetch_news_items(max_items=12)

    # 3) Build HTML
    html_content = build_html(date_str, index_moves, headlines)

    # 4) Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Wrote {output_path}")
    return html_content


def main() -> None:
    generate_market_summary()


if __name__ == "__main__":
    main()
