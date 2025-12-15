#!/usr/bin/env python3
"""
scripts/find_missing_partners.py
Diagnostic tool to identify which countries are receiving exports
but are NOT in our current 'Adversary' or 'Intermediary' lists.

Usage:
    PYTHONPATH=src ./scripts/find_missing_partners.py
"""
import sys
from pathlib import Path

import pandas as pd

# Setup paths
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from china_ir.etl import attach_hs_map  # noqa: E402

DATA = Path("data_work")

# The list of countries we are ALREADY tracking
CURRENTLY_TRACKED = [
    "USA",
    "Japan",
    "South Korea",
    "Germany",
    "Netherlands",
    "France",
    "Belgium",
    "Slovakia",
    "United Kingdom",
    "Australia",
    "Canada",
    "Taiwan",
    "Vietnam",
    "Malaysia",
    "Thailand",
    "India",
    "Mexico",
    "Indonesia",
    "Hungary",
    "Poland",
    "Turkey",
    "Russia",
    "Hong Kong",
    "Singapore",
    "Iran",
    "Brazil",
]

EVENTS = {
    "gallium": pd.Timestamp("2023-08-01"),
    "germanium": pd.Timestamp("2023-08-01"),
    "graphite": pd.Timestamp("2023-12-01"),
}


def _read_full_data():
    # We want the files that have ALL partners, not the granular subsets
    # These usually have 'ALL' in the filename
    files = sorted(DATA.glob("comtrade_CM_HS_*_ALL_*.parquet"))
    if not files:
        print("No 'ALL' partner files found in data_work/.")
        print("Looking for files matching: comtrade_CM_HS_*_ALL_*.parquet")
        return pd.DataFrame()

    print(f"Loading {len(files)} global datasets...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Filter for Exports from China
    if "reporterCode" in df.columns:
        df = df[df["reporterCode"] == 156]
    if "flowCode" in df.columns:
        df = df[df["flowCode"] == "X"]

    # Attach Group Map
    df = attach_hs_map(df, Path("notes/hs_map.csv"))

    # Numeric cleanup
    df["primaryValue"] = pd.to_numeric(df["primaryValue"], errors="coerce").fillna(0)
    df["period_dt"] = pd.to_datetime(df["period"].astype(str), format="%Y%m")

    return df


def find_missing(df):
    for g, start_date in EVENTS.items():
        print(f"\n--- Analyzing Missing Partners for {g.upper()} ---")

        # Look at data SINCE the controls started (most critical period)
        sub = df[(df["group"] == g) & (df["period_dt"] >= start_date)].copy()

        if sub.empty:
            print("No data found for this period.")
            continue

        # Group by Partner
        # Note: We rely on 'partnerDesc' (Partner Name) if available, else partnerCode
        if "partnerDesc" in sub.columns:
            agg = sub.groupby("partnerDesc")["primaryValue"].sum().sort_values(ascending=False)
        else:
            agg = sub.groupby("partnerCode")["primaryValue"].sum().sort_values(ascending=False)

        total_val = agg.sum()

        print(f"Total Export Value (Since {start_date.date()}): ${total_val:,.0f}")
        print(f"{'Partner':<30} | {'Value ($)':<15} | {'Share':<8} | {'Status'}")
        print("-" * 70)

        count = 0
        missing_sum = 0

        for partner, val in agg.items():
            share = (val / total_val) * 100

            # Check if tracked
            is_tracked = partner in CURRENTLY_TRACKED
            status = "TRACKED" if is_tracked else "MISSING"

            # Highlight missing ones
            if not is_tracked:
                missing_sum += val
                if count < 15:  # Print top 15 missing
                    print(f"{str(partner):<30} | ${val:,.0f}      | {share:.1f}%    | **{status}**")
                    count += 1

        print("-" * 70)
        print(f"Total Missing Value: ${missing_sum:,.0f} ({(missing_sum/total_val)*100:.1f}%)")


if __name__ == "__main__":
    df = _read_full_data()
    if not df.empty:
        find_missing(df)
