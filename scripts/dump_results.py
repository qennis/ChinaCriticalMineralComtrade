#!/usr/bin/env python3
"""
scripts/dump_results.py
Prints raw data tables for Track 1 analysis (Bloc Flows & Partner Shifts).
Copy-paste the output of this script into the chat.
"""
import sys
from pathlib import Path

import pandas as pd

# Setup paths (same as before)
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from china_ir.etl import attach_hs_map  # noqa: E402

# Constants
DATA = Path("data_work")
EVENTS = {
    "gallium": pd.Timestamp("2023-08-01"),
    "germanium": pd.Timestamp("2023-08-01"),
    "graphite": pd.Timestamp("2023-12-01"),
}
BLOCS = {
    "Adversary": ["USA", "Japan", "South Korea", "Germany", "Netherlands"],
    "Intermediary": ["Vietnam", "Malaysia", "Thailand", "India", "Mexico", "Indonesia"],
}


def _read_data():
    files = sorted(DATA.glob("partners_granular_*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    hs_map = Path("notes/hs_map.csv")
    df = attach_hs_map(df, hs_map)
    df["period_dt"] = pd.to_datetime(df["period"].astype(str), format="%Y%m")
    return df


def analyze_blocs(df):
    """Prints monthly export values by Bloc."""
    print("\n=== BLOC ANALYSIS (Monthly USD) ===")

    # Map partners
    bloc_map = {p: b for b, partners in BLOCS.items() for p in partners}
    df["bloc"] = df["partner_name"].map(bloc_map).fillna("Rest of World")

    for g in ["gallium", "graphite"]:
        print(f"\n--- {g.upper()} ---")
        sub = df[df["group"] == g]
        if sub.empty:
            continue

        # Pivot: Date x Bloc
        pivot = sub.pivot_table(
            index="period_dt", columns="bloc", values="primaryValue", aggfunc="sum"
        ).fillna(0)

        # Resample to quarterly to make the output shorter/readable, or keep monthly
        # Let's do monthly but only show the window around the event
        event = EVENTS[g]
        window = pivot[
            (pivot.index >= event - pd.DateOffset(months=6))
            & (pivot.index <= event + pd.DateOffset(months=6))
        ]

        print(window.to_csv(sep="\t", float_format="%.0f"))


def analyze_shifts(df):
    """Prints Pre/Post average monthly exports per partner."""
    print("\n=== PARTNER SHIFTS (Avg Monthly USD) ===")

    for g in ["gallium", "graphite"]:
        print(f"\n--- {g.upper()} Top Movers ---")
        sub = df[df["group"] == g]
        event = EVENTS[g]

        # 6-month windows
        pre = sub[
            (sub["period_dt"] < event) & (sub["period_dt"] >= event - pd.DateOffset(months=6))
        ]
        post = sub[
            (sub["period_dt"] >= event) & (sub["period_dt"] < event + pd.DateOffset(months=6))
        ]

        pre_mean = pre.groupby("partner_name")["primaryValue"].sum() / 6
        post_mean = post.groupby("partner_name")["primaryValue"].sum() / 6

        stats = pd.DataFrame({"Pre_Avg": pre_mean, "Post_Avg": post_mean}).fillna(0)
        stats["Change"] = stats["Post_Avg"] - stats["Pre_Avg"]
        stats["Growth_%"] = (stats["Change"] / stats["Pre_Avg"]) * 100

        # Sort by absolute change to see big movers
        stats = stats.sort_values("Change", ascending=False)

        # Print top winners and losers
        print(stats.head(5).to_string(float_format="%.0f"))
        print("...")
        print(stats.tail(5).to_string(float_format="%.0f"))


if __name__ == "__main__":
    df = _read_data()
    if not df.empty:
        analyze_blocs(df)
        analyze_shifts(df)
    else:
        print("No data found.")
