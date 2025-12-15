#!/usr/bin/env python3
"""
scripts/diagnose_leakage.py
Diagnostic tool to find 'The Missing Link' in Gallium/Germanium trade.

Method:
1. Iterates through ALL world country codes (in batches to satisfy API limits).
2. Asks each country: "Did you import HS 811292 from China in 2023?"
3. Compares the results against your 'STRATEGIC_PARTNERS' list.
4. Outputs the TOP MISSING PARTNERS sorted by volume.
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
CODE = "811292"  # HS6 Basket: Gallium, Germanium, Indium
YEAR = "2023"  # The year controls started

# M49 Codes to check (004 to 894 covering active reporting nations)
ALL_M49_CODES = [
    "4",
    "8",
    "12",
    "16",
    "20",
    "24",
    "28",
    "31",
    "32",
    "36",
    "40",
    "44",
    "48",
    "50",
    "51",
    "52",
    "56",
    "60",
    "64",
    "68",
    "70",
    "72",
    "76",
    "84",
    "86",
    "90",
    "92",
    "96",
    "100",
    "104",
    "108",
    "112",
    "116",
    "120",
    "124",
    "132",
    "136",
    "140",
    "144",
    "148",
    "152",
    "156",
    "158",
    "162",
    "166",
    "170",
    "174",
    "178",
    "180",
    "184",
    "188",
    "191",
    "192",
    "196",
    "203",
    "204",
    "208",
    "212",
    "214",
    "218",
    "222",
    "226",
    "231",
    "232",
    "233",
    "234",
    "238",
    "239",
    "242",
    "246",
    "248",
    "250",
    "251",
    "254",
    "258",
    "260",
    "262",
    "266",
    "268",
    "270",
    "275",
    "276",
    "288",
    "292",
    "296",
    "300",
    "304",
    "308",
    "312",
    "316",
    "320",
    "324",
    "328",
    "332",
    "334",
    "336",
    "340",
    "344",
    "348",
    "352",
    "356",
    "360",
    "364",
    "368",
    "372",
    "376",
    "380",
    "384",
    "388",
    "392",
    "398",
    "400",
    "404",
    "408",
    "410",
    "414",
    "417",
    "418",
    "422",
    "426",
    "428",
    "430",
    "434",
    "438",
    "440",
    "442",
    "446",
    "450",
    "454",
    "458",
    "462",
    "466",
    "470",
    "474",
    "478",
    "480",
    "484",
    "490",
    "492",
    "496",
    "498",
    "499",
    "500",
    "504",
    "508",
    "512",
    "516",
    "520",
    "524",
    "528",
    "531",
    "533",
    "534",
    "535",
    "540",
    "548",
    "554",
    "558",
    "562",
    "566",
    "570",
    "574",
    "578",
    "580",
    "581",
    "583",
    "584",
    "585",
    "586",
    "591",
    "598",
    "600",
    "604",
    "608",
    "612",
    "616",
    "620",
    "624",
    "626",
    "630",
    "634",
    "638",
    "642",
    "643",
    "646",
    "652",
    "654",
    "659",
    "660",
    "662",
    "663",
    "666",
    "670",
    "674",
    "678",
    "682",
    "686",
    "688",
    "690",
    "694",
    "699",
    "702",
    "703",
    "704",
    "705",
    "706",
    "710",
    "716",
    "724",
    "728",
    "729",
    "732",
    "740",
    "744",
    "748",
    "752",
    "756",
    "760",
    "762",
    "764",
    "768",
    "772",
    "776",
    "780",
    "784",
    "788",
    "792",
    "795",
    "796",
    "798",
    "800",
    "804",
    "807",
    "818",
    "826",
    "834",
    "840",
    "842",
    "850",
    "854",
    "858",
    "860",
    "862",
    "876",
    "882",
    "887",
    "894",
]

# Quick Map for Readable Output
M49_MAP = {
    "233": "Estonia",
    "372": "Ireland",
    "376": "Israel",
    "440": "Lithuania",
    "428": "Latvia",
    "203": "Czechia",
    "703": "Slovakia",
    "705": "Slovenia",
    "100": "Bulgaria",
    "642": "Romania",
    "608": "Philippines",
    "764": "Thailand",
    "704": "Vietnam",
    "458": "Malaysia",
    "784": "UAE",
    "682": "Saudi Arabia",
    "818": "Egypt",
    "710": "South Africa",
    "398": "Kazakhstan",
    "792": "Turkey",
    "643": "Russia",
    "356": "India",
    "360": "Indonesia",
}

# Your EXISTING List (To exclude from output)
STRATEGIC_PARTNERS = [
    "842",
    "392",
    "410",
    "276",
    "528",
    "251",
    "703",
    "056",
    "826",
    "036",
    "124",
    "490",
    "578",
    "040",
    "380",
    "724",
    "752",
    "756",
    "203",
    "704",
    "458",
    "484",
    "699",
    "764",
    "360",
    "348",
    "616",
    "792",
    "643",
    "364",
    "076",
    "608",
    "784",
    "398",
    "710",
    "834",
    "508",
    "818",
    "344",
    "702",
    "156",
    "999",
    "976",
]


def main():
    if not os.environ.get("COMTRADE_API_KEY"):
        print("Error: COMTRADE_API_KEY not found.")
        return

    print(f"--- MIRROR DATA SEARCH: Who bought HS {CODE} from China in {YEAR}? ---")
    print(f"Scanning {len(ALL_M49_CODES)} potential reporters...\n")

    all_results = []

    # 1. Batch Query
    CHUNK_SIZE = 20
    for i in range(0, len(ALL_M49_CODES), CHUNK_SIZE):
        chunk = ALL_M49_CODES[i : i + CHUNK_SIZE]
        reporter_str = ",".join(chunk)

        try:
            df = fetch_period(
                freq="A",
                periods=[YEAR],
                reporter=reporter_str,
                partner="156",  # From China
                flow="M",  # Import
                cmd=CODE,
                classification="HS",
            )
            if not df.empty:
                print(".", end="", flush=True)
                all_results.append(df)

            time.sleep(1.5)  # Fast but safe

        except Exception:
            print("x", end="", flush=True)

    print("\n\nProcessing results...")

    if not all_results:
        print("No data found globally.")
        return

    full_df = pd.concat(all_results, ignore_index=True)
    full_df["primaryValue"] = pd.to_numeric(full_df["primaryValue"], errors="coerce").fillna(0)

    # 2. Sort & Filter
    agg = full_df.groupby("reporterCode")["primaryValue"].sum().sort_values(ascending=False)

    missing_list = []

    for r_code, val in agg.items():
        r_code = str(r_code)
        if r_code not in STRATEGIC_PARTNERS and val > 50000:  # Filter small noise
            name = M49_MAP.get(r_code, "Unknown")
            missing_list.append({"code": r_code, "name": name, "val": val})

    # 3. Print Output
    print(f"\n{'='*60}")
    print("TOP MISSING PARTNERS (By Value) - ADD THESE TO PULL LIST")
    print(f"{'='*60}")
    print(f"{'Code':<6} | {'Name':<20} | {'Value ($USD)':<15}")
    print("-" * 60)

    for item in missing_list:
        print(f"{item['code']:<6} | {item['name']:<20} | ${item['val']:,.0f}")

    print("-" * 60)
    print(f"Total Missing Volume Found: ${sum(d['val'] for d in missing_list):,.0f}")


if __name__ == "__main__":
    main()
