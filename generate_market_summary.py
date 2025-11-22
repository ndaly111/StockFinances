#!/usr/bin/env python3
"""
Auto-generate a simple daily US stock market summary as HTML.

Output file: daily-market-summary.html
Place this script in the root of your GitHub Pages repo and
call it from a GitHub Actions workflow.
"""

import datetime as dt
import html
import sys
from typing import List, Tuple, Dict, Optional

import requests
import yfinance as yf
import xml.etree.ElementTree as ET

# === Settings ===

OUTPUT_PATH = "daily-market-summary.html"

# Yahoo Finance US markets RSS feed – used just to grab headlines.
RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US"

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


def fetch_rss_items(url: str, max_items: int = 10) -> List[Dict[str, str]]:
    """
    Fetch RSS feed and return up to max_items items with title + link.
    """
    items: List[Dict[str, str]] = []
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        for item in root.findall(".//item")[:max_items]:
            title_el = item.find("title")
            link_el = item.find("link")
            title = title_el.text if title_el is not None else ""
            link = link_el.text if link_el is not None else ""
            if title:
                items.append(
                    {
                        "title": title.strip(),
                        "link": link.strip(),
                    }
                )
    except Exception as e:
        print(f"Error fetching RSS: {e}", file=sys.stderr)

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
    positives = []
    negatives = []
    neutrals = []

    for item in headlines:
        classification = classify_headline(item["title"])
        if classification == "positive":
            positives.append(item)
        elif classification == "negative":
            negatives.append(item)
        else:
            neutrals.append(item)

    def render_index_summary() -> str:
        parts = []
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
        return "<ul>\n" + "\n".join(parts) + "\n</ul>"

    def render_headline_list(items: List[Dict[str, str]]) -> str:
        if not items:
            return "<p>No notable items from the feed.</p>"
        lis = []
        for item in items:
            title = html.escape(item["title"])
            link = html.escape(item["link"] or "#")
            lis.append(f'<li><a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a></li>')
        return "<ul>\n" + "\n".join(lis) + "\n</ul>"

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Daily US Stock Market Summary – {html.escape(date_str)}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 800px;
      margin: 2rem auto;
      padding: 0 1rem 3rem;
      line-height: 1.6;
      background-color: #f7f7f9;
      color: #222;
    }}
    h1 {{
      font-size: 1.9rem;
      margin-bottom: 0.3rem;
    }}
    h2 {{
      margin-top: 1.6rem;
      font-size: 1.3rem;
    }}
    .date {{
      color: #555;
      margin-bottom: 1.5rem;
    }}
    section {{
      background: #fff;
      border-radius: 10px;
      padding: 1rem 1.2rem;
      margin-bottom: 1rem;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    ul {{
      padding-left: 1.3rem;
    }}
    a {{
      color: #0055cc;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .disclaimer {{
      font-size: 0.8rem;
      color: #666;
      margin-top: 1.5rem;
    }}
  </style>
</head>
<body>
  <h1>Daily U.S. Stock Market Summary</h1>
  <p class="date">{html.escape(date_str)}</p>

  <section>
    <h2>Index Snapshot</h2>
    {render_index_summary()}
    <p>This compares today’s close with the prior trading day.</p>
  </section>

  <section>
    <h2>What’s going well</h2>
    {render_headline_list(positives)}
  </section>

  <section>
    <h2>Areas of caution</h2>
    {render_headline_list(negatives)}
  </section>

  <section>
    <h2>What to watch</h2>
    {render_headline_list(neutrals)}
  </section>

  <p class="disclaimer">
    This page is generated automatically using index data (via Yahoo Finance)
    and public market-news headlines. It is for informational purposes only and
    is not investment advice.
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
    headlines = fetch_rss_items(RSS_URL, max_items=12)

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
