#!/usr/bin/env python3
"""
scripts/pull_data.py
Pulls GLOBAL aggregate trade data (Partner=World) for China exports.
Used as the "Total" baseline to calculate leakage.

Features:
- Reads codes dynamically from 'notes/hs_map.csv'.
- Truncates 8-digit codes to 6-digit baskets for API compatibility.
- Pulls monthly data in annual chunks.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from china_ir.comtrade import fetch_period  # noqa: E402

# --- CONFIGURATION ---
DATA_WORK = Path("data_work")
NOTES_DIR = Path("notes")
DATA_WORK.mkdir(exist_ok=True)


def _load_hs6_basket_codes(map_csv: Path):
    if not map_csv.exists():
        print(f"Error: Map file not found at {map_csv}")
        return []
    m = pd.read_csv(map_csv, dtype={"hs6": str})
    codes = set()
    for raw in m["hs6"].astype(str):
        clean = raw.strip()
        if clean.lower() == "nan" or not clean:
            continue
        codes.add(clean[:6])
    return sorted(list(codes))


def get_monthly_periods(year):
    return [f"{year}{month:02d}" for month in range(1, 13)]


def fetch_world_data(hs6_codes):
    print("--- Pulling GLOBAL Export Data (Partner=World) ---")
    print(f"Codes: {len(hs6_codes)} HS6 baskets")

    cmd_str = ",".join(hs6_codes)
    years = range(2017, 2026)

    all_dfs = []

    for year in years:
        print(f"  > Fetching {year}...", end=" ", flush=True)
        periods = get_monthly_periods(year)

        try:
            df = fetch_period(
                freq="M",
                periods=periods,
                reporter="156",  # China
                partner="0",  # World
                flow="X",  # Exports
                cmd=cmd_str,
                classification="HS",
            )

            count = len(df) if not df.empty else 0
            if count > 0:
                print(f"Got {count} rows.")
                all_dfs.append(df)
            else:
                print("(Empty)")

            time.sleep(6)

        except Exception as e:
            print(f"\n    ! Failed {year}: {e}")

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        # Normalize types
        full_df["period"] = full_df["period"].astype(str)
        full_df["cmdCode"] = full_df["cmdCode"].astype(str)

        # Save with timestamp
        t_stamp = pd.Timestamp.now().strftime("%Y%m%d")
        out_path = DATA_WORK / f"comtrade_CM_HS_China_Exports_World_{t_stamp}.parquet"
        full_df.to_parquet(out_path, index=False)
        print(f"\nSuccess! Saved {len(full_df)} rows to {out_path}")
    else:
        print("\nNo data retrieved.")


def main():
    if not os.environ.get("COMTRADE_API_KEY"):
        print("Error: COMTRADE_API_KEY not found.")
        return

    hs_codes = _load_hs6_basket_codes(NOTES_DIR / "hs_map.csv")
    if not hs_codes:
        print("No codes found in map.")
        return

    # FIXED VARIABLE NAME HERE
    fetch_world_data(hs_codes)


if __name__ == "__main__":
    main()
