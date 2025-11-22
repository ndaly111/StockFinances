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
            lis.append(
                f'<li>· <a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a></li>'
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
