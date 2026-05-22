"""Data ingestion + cleaning. Run as ``python -m diabetes_mlops.data``."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)


def download(url: str = C.DATA_URL, dest: Path = C.RAW_CSV) -> Path:
    """Fetch the raw CSV if it isn't already on disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("Raw data already present at %s", dest)
        return dest
    log.info("Downloading %s", url)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    # The upstream file has no header — write it with the canonical column names.
    dest.write_bytes(b",".join(c.encode() for c in C.COLUMNS) + b"\n" + r.content)
    log.info("Saved raw data to %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def load_raw(path: Path = C.RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        download(dest=path)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physiologically impossible zeros with NaN.

    Imputation itself is part of the sklearn pipeline so it can fit on train
    folds only — doing it here would leak information across folds.
    """
    df = df.copy()
    for col in C.ZERO_IS_MISSING:
        df[col] = df[col].replace(0, np.nan)
    return df


def build_processed() -> Path:
    raw = load_raw()
    cleaned = clean(raw)
    C.PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(C.PROCESSED_CSV, index=False)
    log.info(
        "Wrote %s — %d rows, %d positive cases (%.1f%%)",
        C.PROCESSED_CSV,
        len(cleaned),
        int(cleaned[C.TARGET].sum()),
        100 * cleaned[C.TARGET].mean(),
    )
    return C.PROCESSED_CSV


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    build_processed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
