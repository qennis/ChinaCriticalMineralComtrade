#!/usr/bin/env python3
"""
scripts/pull_partners.py
Smart Data Puller: Fetches granular data for specific Strategic Partners.

Features:
- Verbose Logging: Reports row counts per year immediately.
- Batches: Pulls 1 year at a time to respect API limits.
- Fixes "No Data": Generates proper YYYYMM period codes.
- HS6 Proxy: Uses 6-digit basket codes (best available for public API).
"""
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

# Suppress pandas FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from china_ir.comtrade import fetch_period  # noqa: E402

# --- CONFIGURATION ---
DATA_WORK = Path("data_work")
CACHE_DIR = DATA_WORK / "cache_partners"
NOTES_DIR = Path("notes")
DATA_WORK.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# 1. STRATEGIC PARTNERS
STRATEGIC_PARTNERS = {
    # --- The "Target" Block ---
    "842": "USA",
    "392": "Japan",
    "410": "South Korea",
    "276": "Germany",
    "528": "Netherlands",
    "251": "France",
    "703": "Slovakia",
    "056": "Belgium",
    "56": "Belgium",  # Catch integer/string mismatch
    "826": "United Kingdom",
    "036": "Australia",
    "124": "Canada",
    "490": "Taiwan",
    "554": "New Zealand",  # NEW: Found via Mirror Data
    "578": "Norway",
    "040": "Austria",
    "380": "Italy",
    "724": "Spain",
    "752": "Sweden",
    "756": "Switzerland",
    "203": "Czech Republic",
    # --- Intermediaries ---
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
    "364": "Iran",
    "076": "Brazil",
    "608": "Philippines",
    "784": "UAE",
    "398": "Kazakhstan",
    "710": "South Africa",
    "834": "Tanzania",
    "508": "Mozambique",
    "818": "Egypt",
    # --- Tech Hubs ---
    "344": "Hong Kong",
    "702": "Singapore",
    # --- Black Hole Codes ---
    "156": "China (Re-import)",
    "999": "Areas NES",
    "976": "Other Asia NES",
}


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
        codes.add(clean[:6])  # Truncate to 6 digits for standard API
    return sorted(list(codes))


def get_monthly_periods(year):
    """Generates a list of YYYYMM strings for a single year (12 max)."""
    return [f"{year}{month:02d}" for month in range(1, 13)]


def fetch_partner(partner_code, hs6_codes):
    """
    Checks cache for existing years.
    Fetches ONLY the years that are missing from the cache.
    Merges and saves.
    """
    cache_path = CACHE_DIR / f"{partner_code}.parquet"
    existing_df = pd.DataFrame()
    existing_years = set()

    # 1. Load Cache
    if cache_path.exists():
        try:
            existing_df = pd.read_parquet(cache_path)
            if not existing_df.empty and "period" in existing_df.columns:
                periods = existing_df["period"].astype(str)
                existing_years = set(periods.str[:4])
        except Exception:
            existing_df = pd.DataFrame()  # Reset on corruption

    # 2. Define Target Years (2017-2024)
    target_years = [str(y) for y in range(2017, 2025)]

    cmd_str = ",".join(hs6_codes)
    new_dfs = []
    is_updated = False

    partner_name = STRATEGIC_PARTNERS.get(partner_code, partner_code)
    print(f"[{partner_name}] Checking cache...")

    # Iterate year by year
    for year in target_years:
        if year in existing_years:
            continue

        print(f"  > Fetching {year}...", end=" ", flush=True)
        periods = get_monthly_periods(year)

        try:
            df = fetch_period(
                freq="M",
                periods=periods,
                reporter="156",
                partner=partner_code,
                flow="X",
                cmd=cmd_str,
                classification="HS",
            )

            count = len(df) if not df.empty else 0

            if count > 0:
                print(f"Got {count} rows.")
                new_dfs.append(df)
                is_updated = True
            else:
                print("(Empty)")

            time.sleep(6)  # Rate limit

        except Exception as e:
            print(f"\n    ! Failed {year}: {e}")
            if "400" in str(e):
                print("    [!] API rejected query. Check limits.")
                break

    # 3. Merge and Save
    if is_updated:
        combined_list = [existing_df] + new_dfs
        valid_dfs = [d for d in combined_list if not d.empty]

        if valid_dfs:
            full_df = pd.concat(valid_dfs, ignore_index=True)
            full_df["period"] = full_df["period"].astype(str)
            full_df["cmdCode"] = full_df["cmdCode"].astype(str)

            full_df = full_df.drop_duplicates(
                subset=["period", "cmdCode", "flowCode", "partnerCode"], keep="last"
            )

            full_df = full_df.sort_values("period")
            full_df.to_parquet(cache_path, index=False)
            print(f"  -> Updated cache: {len(full_df)} total rows.")
        else:
            print("  -> No valid data retrieved.")
    else:
        print(f"  -> Up to date ({len(existing_df)} rows).")

    return True


def rebuild_master():
    print("\n--- Rebuilding Master Dataset ---")
    files = sorted(CACHE_DIR.glob("*.parquet"))
    if not files:
        return

    df_list = []
    for f in files:
        try:
            d = pd.read_parquet(f)
            if not d.empty:
                d["period"] = d["period"].astype(str)
                d["partnerCode"] = d["partnerCode"].astype(str)
                d["cmdCode"] = d["cmdCode"].astype(str)
                df_list.append(d)
        except Exception:
            pass

    if df_list:
        master = pd.concat(df_list, ignore_index=True)
        if "period" in master.columns:
            master = master.sort_values("period")

        master_out = DATA_WORK / "partners_granular_MASTER.parquet"
        master.to_parquet(master_out, index=False)
        print(f"Success! Combined {len(master)} rows into {master_out}")


def main():
    if not os.environ.get("COMTRADE_API_KEY"):
        print("Error: COMTRADE_API_KEY not found.")
        return

    hs_codes = _load_hs6_basket_codes(NOTES_DIR / "hs_map.csv")
    if not hs_codes:
        print("No codes found.")
        return
    print(f"Loaded {len(hs_codes)} HS6 basket codes.")

    print(f"--- Smart Sync: {len(STRATEGIC_PARTNERS)} Partners ---")

    for code in STRATEGIC_PARTNERS:
        fetch_partner(code, hs_codes)

    rebuild_master()


if __name__ == "__main__":
    main()
