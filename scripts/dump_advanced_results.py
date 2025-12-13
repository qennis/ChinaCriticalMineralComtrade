#!/usr/bin/env python3
"""
scripts/dump_advanced_results.py
Prints raw data for Advanced Track 1 Analysis (Price, Placebo, Specs).
Copy-paste the output into the chat.
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
EVENTS = {
    "gallium": pd.Timestamp("2023-08-01"),
    "graphite": pd.Timestamp("2023-12-01"),
}
BLOCS = {
    "Adversary": ["USA", "Japan", "South Korea", "Germany", "Netherlands"],
    "Intermediary": ["Vietnam", "Malaysia", "Thailand", "India", "Mexico", "Indonesia"],
}
PLACEBO_PAIRS = [("gallium", "aluminum"), ("gallium", "silicon"), ("graphite", "aluminum")]


def _read_data():
    files = sorted(DATA.glob("partners_granular_*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    hs_map = Path("notes/hs_map.csv")
    df = attach_hs_map(df, hs_map)
    df["period_dt"] = pd.to_datetime(df["period"].astype(str), format="%Y%m")

    # Calculate Unit Value
    df["netWgt"] = pd.to_numeric(df["netWgt"], errors="coerce")
    df["primaryValue"] = pd.to_numeric(df["primaryValue"], errors="coerce")

    # Map Blocs
    bloc_map = {p: b for b, partners in BLOCS.items() for p in partners}
    df["bloc"] = df["partner_name"].map(bloc_map).fillna("Other")

    return df


def dump_price_divergence(df):
    print("\n=== 1. PRICE DIVERGENCE (Weighted Avg USD/kg) ===")
    for g, event_date in EVENTS.items():
        sub = df[(df["group"] == g) & (df["netWgt"] > 0)]
        if sub.empty:
            continue

        # Weighted Average Price per Bloc
        monthly = sub.groupby(["period_dt", "bloc"])[["primaryValue", "netWgt"]].sum()
        monthly["price"] = monthly["primaryValue"] / monthly["netWgt"]
        pivot = monthly["price"].unstack()

        # Filter window around event
        window = pivot[
            (pivot.index >= event_date - pd.DateOffset(months=6))
            & (pivot.index <= event_date + pd.DateOffset(months=6))
        ]

        cols = [c for c in ["Adversary", "Intermediary"] if c in window.columns]
        print(f"\n--- {g.upper()} Unit Values ---")
        print(window[cols].to_csv(sep="\t", float_format="%.2f"))


def dump_placebo_test(df):
    print("\n=== 2. PLACEBO TEST (Adversary Exports Index) ===")
    target_bloc = "Adversary"

    for treatment, control in PLACEBO_PAIRS:
        if treatment not in EVENTS:
            continue
        event = EVENTS[treatment]

        sub = df[(df["bloc"] == target_bloc) & (df["group"].isin([treatment, control]))]
        if sub.empty:
            continue

        pivot = sub.groupby(["period_dt", "group"])["primaryValue"].sum().unstack()

        # Normalize to 100 (Pre-Event Mean)
        pre_window = (sub["period_dt"] >= event - pd.DateOffset(months=6)) & (
            sub["period_dt"] < event
        )

        for col in pivot.columns:
            base = pivot.loc[pivot.index.isin(sub[pre_window]["period_dt"].unique()), col].mean()
            if base > 0:
                pivot[col] = (pivot[col] / base) * 100

        window = pivot[
            (pivot.index >= event - pd.DateOffset(months=6))
            & (pivot.index <= event + pd.DateOffset(months=6))
        ]

        print(f"\n--- {treatment.upper()} vs {control.upper()} (Index 100=Pre-Event) ---")
        print(window.to_csv(sep="\t", float_format="%.1f"))


def dump_spec_creep(df):
    print("\n=== 3. SPECIFICATION CREEP (Share of HS Codes) ===")

    for g in ["gallium", "graphite"]:
        sub = df[df["group"] == g]
        if sub.empty:
            continue

        # Top 3 codes
        top_codes = sub.groupby("hs6")["primaryValue"].sum().nlargest(3).index.tolist()
        pivot = sub.pivot_table(
            index="period_dt", columns="hs6", values="primaryValue", aggfunc="sum"
        ).fillna(0)

        # Calculate Share
        pivot_share = pivot.div(pivot.sum(axis=1), axis=0)

        # Filter window
        event = EVENTS.get(g, pd.Timestamp("2023-01-01"))
        window = pivot_share[
            (pivot_share.index >= event - pd.DateOffset(months=6))
            & (pivot_share.index <= event + pd.DateOffset(months=6))
        ]

        print(f"\n--- {g.upper()} HS Code Share ---")
        print(window[top_codes].to_csv(sep="\t", float_format="%.3f"))


if __name__ == "__main__":
    df = _read_data()
    if not df.empty:
        dump_price_divergence(df)
        dump_placebo_test(df)
        dump_spec_creep(df)
    else:
        print("No data found.")
