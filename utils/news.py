"""Shared Yahoo RSS headline helpers used across the site."""

from __future__ import annotations

import datetime as dt
import email.utils
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import requests


def _parse_pub_date(value: str) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    except Exception:
        return None


def fetch_company_headlines(ticker: str, max_items: int = 6) -> List[Dict[str, str]]:
    """Return a list of headline dicts for the given ticker."""

    urls = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://finance.yahoo.com/rss/search?p={ticker}",
    ]

    items: List[Dict[str, str]] = []
    seen_titles = set()
    headers = {"User-Agent": "Mozilla/5.0 (ticker headlines fetcher)"}

    for url in urls:
        if len(items) >= max_items:
            break

        try:
            resp = requests.get(url, timeout=12, headers=headers)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item"):
                if len(items) >= max_items:
                    break
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                if not title or title in seen_titles:
                    continue
                items.append({"title": title, "link": link})
                seen_titles.add(title)
        except Exception as exc:
            print(f"[WARN] Unable to fetch headlines for {ticker} from {url}: {exc}")

    return items


def _fetch_headlines_with_dates(ticker: str, max_items: int) -> List[Dict[str, str]]:
    urls = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://finance.yahoo.com/rss/search?p={ticker}",
    ]

    headers = {"User-Agent": "Mozilla/5.0 (market headlines fetcher)"}
    items: List[Dict[str, str]] = []
    seen_titles = set()

    for url in urls:
        if len(items) >= max_items:
            break
        try:
            resp = requests.get(url, timeout=12, headers=headers)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item"):
                if len(items) >= max_items:
                    break
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                if not title or title in seen_titles:
                    continue
                pub_date = _parse_pub_date((item.findtext("pubDate") or "").strip())
                items.append(
                    {
                        "title": title,
                        "link": link,
                        "_pub_date": pub_date or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
                    }
                )
                seen_titles.add(title)
        except Exception as exc:
            print(f"[WARN] Unable to fetch headlines for {ticker} from {url}: {exc}")

    return items


def fetch_market_headlines(max_items: int = 12) -> List[Dict[str, str]]:
    """Return market headlines by merging Yahoo RSS feeds for proxy tickers."""

    tickers = ["SPY", "QQQ", "DIA", "IWM"]
    items: List[Dict[str, str]] = []
    seen_titles = set()

    for ticker in tickers:
        for item in _fetch_headlines_with_dates(ticker, max_items=max_items):
            title = item["title"]
            if title in seen_titles:
                continue
            items.append(item)
            seen_titles.add(title)

    items.sort(key=lambda entry: entry["_pub_date"], reverse=True)
    return [{"title": item["title"], "link": item["link"]} for item in items[:max_items]]
