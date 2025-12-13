#!/usr/bin/env python3
"""
scripts/pull_partners.py
Fetches HS6 data for specific strategic partners with CACHING.
Updated: Includes Taiwan (490), Hong Kong (344), Singapore (702) to close the 'Rest of World' gap.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from china_ir.comtrade import fetch_period

ROOT = Path(__file__).resolve().parents[1]
DATA_WORK = ROOT / "data_work"
CACHE_DIR = DATA_WORK / "cache_partners"
NOTES = ROOT / "notes"

# Define the strategic cohorts (Expanded List)
STRATEGIC_PARTNERS = {
    # --- The "Target" Block (Adversaries/Competitors) ---
    "842": "USA",
    "392": "Japan",
    "410": "South Korea",
    "276": "Germany",
    "528": "Netherlands",
    "251": "France",
    "703": "Slovakia",
    "056": "Belgium",
    "826": "United Kingdom",  # New: Five Eyes partner
    "036": "Australia",  # New: Five Eyes partner
    "124": "Canada",  # New: Five Eyes partner
    # --- The "Intermediary" Candidates (Potential Rerouting Hubs) ---
    "704": "Vietnam",
    "458": "Malaysia",
    "484": "Mexico",
    "699": "India",
    "764": "Thailand",
    "360": "Indonesia",
    "348": "Hungary",
    "616": "Poland",
    "792": "Turkey",
    "643": "Russia",
    "364": "Iran",  # New: Growing graphite market
    "076": "Brazil",  # New: Major BRICS partner
    # --- The "Hidden" Tech Hubs (Crucial for Gallium/Semiconductors) ---
    "490": "Taiwan",  # CRITICAL: Often "Other Asia, nes" in Comtrade
    "344": "Hong Kong",  # CRITICAL: Transshipment hub
    "702": "Singapore",  # CRITICAL: Logistics hub
}


def _load_hs6_list(map_csv: Path) -> List[str]:
    m = pd.read_csv(map_csv, dtype={"hs6": str})
    return [h for h in m["hs6"].astype(str).str.zfill(6).unique() if h != "nan"]


def _chunks(data: List, size: int) -> Iterable[List]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="202201-202412", help="Focus on the event window")
    ap.add_argument("--freq", default="M")
    args = ap.parse_args()

    # Ensure cache exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    hs_codes = _load_hs6_list(NOTES / "hs_map.csv")

    # Expand period range
    start, end = args.period.split("-")
    periods = []
    y, m = int(start[:4]), int(start[4:])
    ye, me = int(end[:4]), int(end[4:])
    while (y < ye) or (y == ye and m <= me):
        periods.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1

    print(f"Plan: Atomic fetch for {len(periods)} months x {len(STRATEGIC_PARTNERS)} partners.")
    print(f"Cache Directory: {CACHE_DIR}")

    hs_chunk_size = 20  # 20 codes is safe with Atomic calls
    total_tasks = len(periods) * len(STRATEGIC_PARTNERS)
    current_task = 0
    new_downloads = 0

    for month in periods:
        for partner_code, partner_name in STRATEGIC_PARTNERS.items():
            current_task += 1

            # Check Cache First
            cache_file = CACHE_DIR / f"{month}_{partner_code}.parquet"
            if cache_file.exists():
                print(f"[{current_task}/{total_tasks}] {month} | {partner_name} : SKIPPED (Cached)")
                continue

            # If not cached, fetch all HS chunks for this partner/month
            print(
                f"[{current_task}/{total_tasks}] {month} | {partner_name} : Fetching...",
                end="",
                flush=True,
            )

            month_frames = []
            failed = False

            for h_chunk in _chunks(hs_codes, hs_chunk_size):
                cmd_str = ",".join(h_chunk)
                try:
                    df = fetch_period(
                        freq=args.freq,
                        periods=[month],
                        reporter="156",
                        partner=partner_code,
                        flow="X",  # 'X' for Exports
                        cmd=cmd_str,
                    )
                    if not df.empty:
                        month_frames.append(df)

                    time.sleep(3)  # Be nice

                except Exception as e:
                    print(f" ERROR: {e}")
                    if "403" in str(e) or "Quota" in str(e):
                        print("\n!!! QUOTA EXCEEDED !!!")
                        print("Run this script again later.")
                        print("Progress saved.")
                        sys.exit(1)  # Stop immediately
                    failed = True
                    break

            if not failed:
                if month_frames:
                    combined = pd.concat(month_frames, ignore_index=True)
                    combined.to_parquet(cache_file, index=False)
                    new_downloads += 1
                    print(f" DONE ({len(combined)} rows)")
                else:
                    print(" EMPTY (Not caching)")

    # --- Combine All Cache Files ---
    print("\nCombining cache files...")
    all_files = sorted(CACHE_DIR.glob("*.parquet"))
    if not all_files:
        print("No data found.")
        return

    full_df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)

    # Tag with names
    full_df["partner_name"] = full_df["partnerCode"].astype(str).map(STRATEGIC_PARTNERS)

    if "primaryValue" in full_df.columns:
        full_df = full_df[full_df["primaryValue"].notna()]

    stamp = datetime.utcnow().strftime("%Y%m%d")
    out_path = DATA_WORK / f"partners_granular_{start}-{end}_{stamp}.parquet"
    full_df.to_parquet(out_path, index=False)
    print(f"Success! Combined {len(full_df)} rows into {out_path}")


if __name__ == "__main__":
    main()
