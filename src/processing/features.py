from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================================
# Feature Extraction Pipeline
# ============================================================

def build_vectorizers():
    """Build word- and char-level TF-IDF vectorizers."""
    word_vec = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        dtype=np.float32,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=10000,
        min_df=2,
        dtype=np.float32,
    )
    return word_vec, char_vec


def fit_transform_text(df: pd.DataFrame, text_col: str = "content_clean"):
    """Fit and transform TF-IDF vectorizers on cleaned text."""
    word_vec, char_vec = build_vectorizers()
    texts = df[text_col].fillna("").astype(str).tolist()
    Xw = word_vec.fit_transform(texts)
    Xc = char_vec.fit_transform(texts)
    return Xw, Xc, word_vec, char_vec


def sentiment_scores(texts: list[str]) -> np.ndarray:
    """Compute VADER sentiment compound score per text."""
    sid = SentimentIntensityAnalyzer()
    scores = np.zeros((len(texts),), dtype=np.float32)
    for i, t in enumerate(texts):
        s = sid.polarity_scores(t)
        scores[i] = np.float32(s.get("compound", 0.0))
    return scores


def build_feature_frame(df: pd.DataFrame, Xw, Xc, sent: np.ndarray) -> pd.DataFrame:
    """Create numeric meta features from tweet content and engagement."""
    meta = pd.DataFrame(
        {
            "len_chars": df["content_clean"].fillna("").map(len).astype(np.int32),
            "len_words": df["content_clean"]
            .fillna("")
            .map(lambda s: len(s.split()))
            .astype(np.int32),
            "likes": df["like_count"].fillna(0).astype(np.int32),
            "retweets": df["retweet_count"].fillna(0).astype(np.int32),
            "replies": df["reply_count"].fillna(0).astype(np.int32),
            "quotes": df["quote_count"].fillna(0).astype(np.int32),
            "sentiment": sent,
        }
    )
    return meta


def composite_signal(meta: pd.DataFrame) -> pd.DataFrame:
    """Combine features into a composite signal + confidence score."""
    eps = 1e-6
    z = (meta - meta.mean()) / (meta.std(ddof=0) + eps)

    # Heuristic weights: sentiment + engagement dominate
    signal = (
        0.6 * z["sentiment"]
        + 0.1 * z["len_words"]
        + 0.15 * z["likes"]
        + 0.1 * z["retweets"]
        + 0.05 * z["replies"]
    ).clip(-3, 3)

    # Confidence: based on engagement + text length variability
    conf = (z[["likes", "retweets", "replies", "len_words"]].abs().mean(axis=1)).clip(0, 3)

    out = pd.DataFrame(
        {
            "signal": signal.astype(np.float32),
            "confidence": conf.astype(np.float32),
        }
    )
    return out
