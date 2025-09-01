from __future__ import annotations
import pandas as pd
import regex as re
import emoji


URL = re.compile(r"https?://\S+")
HANDLE = re.compile(r"@[A-Za-z0-9_]{1,15}")
HASHTAG = re.compile(r"#[\p{L}0-9_]+", re.UNICODE)
MULTISPACE = re.compile(r"\s+")




def clean_content(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL.sub(" ", text)
    text = HANDLE.sub(" ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = MULTISPACE.sub(" ", text)
    return text.strip()




def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["content_clean"] = out["content"].map(clean_content)
    # strong dedupe keys: tweet_id, or fallback on user+text+timebucket
    out["timebucket_min"] = pd.to_datetime(out["timestamp"]).dt.floor("min")
    out["dedupe_key"] = (
    out["username"].astype(str)
    + "|" + out["content_clean"].astype(str)
    + "|" + out["timebucket_min"].astype(str)
    )
    out = out.drop_duplicates(subset=["tweet_id"]) # tweet_id is globally unique
    return out.reset_index(drop=True)