import random
import time
from contextlib import contextmanager


# Simple, jittered sleep to mimic human pacing.
def polite_sleep(base: float = 0.8, jitter: float = 0.6):
    time.sleep(max(0.0, random.uniform(base - jitter, base + jitter)))


@contextmanager
def backoff(retries: int = 5, base: float = 1.0, factor: float = 1.7):
    try:
        yield
    except Exception:
        delay = base
        for _ in range(retries):
            time.sleep(delay)
            delay *= factor
            try:
                yield
                return
            except Exception:
                continue
        raise