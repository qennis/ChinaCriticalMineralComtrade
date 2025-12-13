#!/usr/bin/env python3
"""
scripts/clean_cache.py
Deletes 'placeholder' cache files that might actually be failed API calls.
"""
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data_work/cache_partners")


def clean():
    if not CACHE_DIR.exists():
        print(f"No cache directory found at {CACHE_DIR}")
        return

    print(f"Scanning {CACHE_DIR}...")
    deleted = 0
    kept = 0

    for p in CACHE_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(p)
            # Valid data has 'primaryValue'. Placeholders do not.
            if "primaryValue" not in df.columns or df.empty:
                print(f"Deleting suspicious empty cache: {p.name}")
                p.unlink()
                deleted += 1
            else:
                kept += 1
        except Exception as e:
            print(f"Error reading {p}: {e}")

    print(f"\nSummary: Deleted {deleted} potentially corrupted files. Kept {kept} valid files.")


if __name__ == "__main__":
    clean()
