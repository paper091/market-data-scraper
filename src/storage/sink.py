from __future__ import annotations
import os
from datetime import datetime
from typing import Iterable, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetBatchWriter:
    def __init__(self, out_dir: str, stem: str, schema: Optional[pa.schema] = None):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(out_dir, f"{stem}_{ts}.parquet")
        self._writer = None
        self._schema = schema


def write_batch(self, records: Iterable[Dict[str, Any]]):
    df = pd.DataFrame.from_records(list(records))
    if df.empty:
        return
    table = pa.Table.from_pandas(df, schema=self._schema, preserve_index=False)
    if self._writer is None:
        self._writer = pq.ParquetWriter(self._path, table.schema)
    self._writer.write_table(table)


def close(self):
    if self._writer is not None:
        self._writer.close()




def dedupe_on(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)