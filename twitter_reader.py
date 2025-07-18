# twitter_reader.py
"""Fetch recent tweets for specified Twitter handles using snscrape."""

import datetime as _dt
from dataclasses import dataclass
from typing import List

import snscrape.modules.twitter as sntwitter


@dataclass
class Tweet:
    """Simple tweet container"""
    id: int
    date: _dt.datetime
    content: str


def get_recent_tweets(username: str, limit: int = 5) -> List[Tweet]:
    """Return the most recent tweets for a given user.

    Parameters
    ----------
    username : str
        Twitter handle without ``@``.
    limit : int, optional
        Maximum number of tweets to fetch, by default 5.
    """
    scraper = sntwitter.TwitterUserScraper(username)
    tweets: List[Tweet] = []
    for i, tweet in enumerate(scraper.get_items()):
        if i >= limit:
            break
        tweets.append(Tweet(id=tweet.id, date=tweet.date, content=tweet.content))
    return tweets


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch recent tweets for accounts")
    parser.add_argument("user", help="Twitter username")
    parser.add_argument("-n", type=int, default=5, help="number of tweets")
    args = parser.parse_args()
    for t in get_recent_tweets(args.user, args.n):
        print(f"{t.date:%Y-%m-%d %H:%M} - {t.content}")

