from __future__ import annotations
import logging
import os
import yaml
import pandas as pd

from .utils.logging_setup import setup_logging
from .collectors import x_scraper
try:
    from .collectors import x_scraper_playwright
except ImportError:
    x_scraper_playwright = None

from .processing.clean import apply_cleaning
from .processing.features import (
    fit_transform_text,
    sentiment_scores,
    build_feature_frame,
    composite_signal,
)
from .processing.aggregate import by_time, by_hashtag, reservoir_sample
from .storage.sink import ParquetBatchWriter


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def collect_with_fallback(cfg: dict, log: logging.Logger) -> pd.DataFrame:
    """Try snscrape first; if it fails, fall back to Playwright."""
    try:
        df = x_scraper.collect(
            hashtags=cfg["hashtags"],
            hours_back=cfg["hours_back"],
            lang=cfg.get("query_language") or None,
            max_per_hashtag=cfg.get("max_per_hashtag", 4000),
        )
        if df.empty:
            raise RuntimeError("No tweets from snscrape")
        log.info("Collected %d tweets via snscrape", len(df))
        return df

    except Exception as e:
        log.warning("snscrape failed: %s", e)
        if x_scraper_playwright is None:
            log.error("Playwright fallback not available. Install with: pip install playwright")
            return pd.DataFrame()

        import asyncio
        records = []
        for tag in cfg["hashtags"]:
            log.info("Falling back to Playwright for #%s", tag)
            res = asyncio.run(
                x_scraper_playwright.scrape_tweets(
                    f"#{tag}",
                    max_tweets=cfg.get("max_per_hashtag", 4000),
                )
            )
            records.extend(res)

        df = pd.DataFrame.from_records(records)
        if df.empty:
            log.warning("Playwright also returned no tweets.")
        else:
            log.info("Collected %d tweets via Playwright", len(df))
        return df


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    setup_logging(cfg.get("log_level", "INFO"))
    log = logging.getLogger("main")

    ensure_dirs(cfg["raw_dir"], cfg["processed_dir"])

    # 1) Collect
    df = collect_with_fallback(cfg, log)
    if df.empty:
        log.warning("No data collected, exiting.")
        return

    # 2) Persist raw in batches
    raw_writer = ParquetBatchWriter(cfg["raw_dir"], stem="tweets_raw")
    raw_writer.write_batch(df.to_dict(orient="records"))
    raw_writer.close()

    # 3) Clean & dedupe
    clean = apply_cleaning(df)
    log.info("After cleaning/dedupe: %d tweets", len(clean))

    # 4) Features
    Xw, Xc, word_vec, char_vec = fit_transform_text(clean, text_col="content_clean")
    sent = sentiment_scores(clean["content_clean"].tolist())
    meta = build_feature_frame(clean, Xw, Xc, sent)
    sig = composite_signal(meta)

    enriched = pd.concat([clean.reset_index(drop=True), sig], axis=1)

    # 5) Aggregate samples for lightweight plots / reporting
    t5 = by_time(enriched, freq="5min")
    top_tags = by_hashtag(enriched)
    sample = reservoir_sample(enriched, k=1000)

    # 6) Persist processed
    proc_writer = ParquetBatchWriter(cfg["processed_dir"], stem="tweets_processed")
    proc_writer.write_batch(enriched.to_dict(orient="records"))
    proc_writer.close()

    # Also save CSV summaries for quick manual inspection
    t5.to_csv(os.path.join(cfg["processed_dir"], "signal_by_time_5min.csv"), index=False)
    top_tags.to_csv(os.path.join(cfg["processed_dir"], "signal_by_hashtag.csv"), index=False)
    sample.to_csv(os.path.join(cfg["processed_dir"], "sampled_enriched.csv"), index=False)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
