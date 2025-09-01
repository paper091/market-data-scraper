from __future__ import annotations
import numpy as np
import pandas as pd

# Aggregate signals by time, hashtag, or sample the dataset.

def by_time(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Aggregate mean signal by time bucket."""
    g = df.assign(ts=pd.to_datetime(df["timestamp"]).dt.floor(freq))
    agg = g.groupby("ts")["signal"].mean().to_frame("signal_mean").reset_index()
    return agg

def by_hashtag(df: pd.DataFrame) -> pd.DataFrame:
    """Average signal per hashtag (handles missing/empty hashtag lists)."""
    if "hashtags" not in df.columns:
        return pd.DataFrame(columns=["hashtags", "signal_mean"])
    exploded = df.explode("hashtags")
    exploded = exploded.dropna(subset=["hashtags"])
    agg = (
        exploded.groupby("hashtags")["signal"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="signal_mean")
    )
    return agg

def reservoir_sample(df: pd.DataFrame, k: int = 500, seed: int = 13) -> pd.DataFrame:
    """Memory-friendly random sample (reservoir-style via RNG choice)."""
    if len(df) <= k:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=k, replace=False)
    return df.iloc[idx].reset_index(drop=True)
