from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterator, Dict, Any, Iterable, List
import itertools


import pandas as pd
import regex as re
from tqdm import tqdm


# snscrape supports Twitter via `snscrape.modules.twitter`.
import snscrape.modules.twitter as sntwitter


from ..utils.rate_limit import polite_sleep


log = logging.getLogger(__name__)


CLEAN_WS = re.compile(r"\s+")




def _since_until(hours_back: int) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    since = (now - timedelta(hours=hours_back)).strftime("%Y-%m-%d")
    until = now.strftime("%Y-%m-%d")
    return since, until




def build_query(hashtags: List[str], hours_back: int, lang: str | None) -> str:
    since, until = _since_until(hours_back)
    tag_expr = " OR ".join([f"#{tag}" for tag in hashtags])
    q = f"({tag_expr}) since:{since} until:{until}"
    if lang:
        q += f" lang:{lang}"
    return q




def _iter_tweets(query: str, max_items: int | None = None) -> Iterator[Any]:
    scraper = sntwitter.TwitterSearchScraper(query)
    it = scraper.get_items()
    if max_items:
        it = itertools.islice(it, max_items)
        for tweet in it:
            polite_sleep(0.2, 0.15) # very light pacing
            yield tweet




def normalize_text(text: str) -> str:
    text = CLEAN_WS.sub(" ", text).strip()
    return text




def record_from_tweet(t) -> Dict[str, Any]:
    return {
        "tweet_id": t.id,
        "username": getattr(t.user, "username", None),
        "displayname": getattr(t.user, "displayname", None),
        "user_id": getattr(t.user, "id", None),
        "timestamp": pd.Timestamp(t.date).tz_convert("UTC").isoformat(),
        "content": normalize_text(getattr(t, "rawContent", "")),
        "like_count": getattr(t, "likeCount", 0),
        "retweet_count": getattr(t, "retweetCount", 0),
        "reply_count": getattr(t, "replyCount", 0),
        "quote_count": getattr(t, "quoteCount", 0),
        "hashtags": [h for h in getattr(t, "hashtags", [])] if getattr(t, "hashtags", None) else [],
        "mentions": [m.username for m in getattr(t, "mentionedUsers", [])] if getattr(t, "mentionedUsers", None) else [],
        "permalink": f"https://x.com/{getattr(t.user, 'username', 'user')}/status/{t.id}",
    }




def collect(hashtags: List[str], hours_back: int, lang: str | None, max_per_hashtag: int) -> pd.DataFrame:
    records: list[dict] = []
    for tag in hashtags:
        q = build_query([tag], hours_back, lang)
        log.info("Collecting for tag=%s | query=%s", tag, q)
        for tw in tqdm(_iter_tweets(q, max_per_hashtag), desc=f"#{tag}"):
            records.append(record_from_tweet(tw))
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
    return df