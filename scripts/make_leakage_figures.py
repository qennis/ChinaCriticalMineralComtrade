#!/usr/bin/env python3
"""
scripts/make_leakage_figures.py
Analyses 'The Black Hole' with explicit tracking for Bonded (156) and Unspecified (999) flows.

Updates:
- Grouping: Uses granular HS6 categories (Unwrought Ga/Ge vs Compounds vs Graphite types).
- Mapping: Truncates HS8 map codes to HS6 and assigns specific plot groups.
"""
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd

# import seaborn as sns

# Setup paths
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- CONFIGURATION ---
DATA = Path("data_work")
OUT = Path("figures/leakage")
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.family": "sans-serif", "figure.dpi": 300, "savefig.bbox": "tight"})

# Define Control Dates for specific groups
EVENTS = {
    "Unwrought Ga/Ge": pd.Timestamp("2023-08-01"),
    "Wrought Ga/Ge": pd.Timestamp("2023-08-01"),
    "Ga/Ge Compounds": pd.Timestamp("2023-08-01"),
    "Natural Graphite": pd.Timestamp("2023-12-01"),
    "Artificial Graphite": pd.Timestamp("2023-12-01"),
}

# 1. Map M49 Integer Codes to Names
CODE_TO_NAME = {
    842: "USA",
    392: "Japan",
    410: "South Korea",
    276: "Germany",
    528: "Netherlands",
    251: "France",
    703: "Slovakia",
    56: "Belgium",
    56: "Belgium",
    826: "United Kingdom",
    36: "Australia",
    124: "Canada",
    490: "Taiwan",
    578: "Norway",
    40: "Austria",
    380: "Italy",
    724: "Spain",
    752: "Sweden",
    756: "Switzerland",
    203: "Czech Republic",
    704: "Vietnam",
    458: "Malaysia",
    484: "Mexico",
    699: "India",
    764: "Thailand",
    360: "Indonesia",
    348: "Hungary",
    616: "Poland",
    792: "Turkey",
    643: "Russia",
    364: "Iran",
    76: "Brazil",
    608: "Philippines",
    784: "UAE",
    398: "Kazakhstan",
    710: "South Africa",
    834: "Tanzania",
    508: "Mozambique",
    818: "Egypt",
    344: "Hong Kong",
    702: "Singapore",
    554: "New Zealand",
    156: "China",
    999: "Areas NES",
    976: "Other Asia NES",
    977: "Africa NES",
}

# 2. Define Blocs
BLOCS = {
    "Adversary": [
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
        "Norway",
        "Austria",
        "Italy",
        "Spain",
        "Sweden",
        "Switzerland",
        "Czech Republic",
        "New Zealand",
    ],
    "Intermediary": [
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
        "Philippines",
        "UAE",
        "Kazakhstan",
        "South Africa",
        "Tanzania",
        "Mozambique",
        "Egypt",
    ],
    "Bonded": ["China", "China (Re-import)"],
    "Unspecified": ["Areas NES", "Other Asia NES", "Unspecified"],
}


def _to_dt(series):
    return pd.to_datetime(series.astype(str), format="%Y%m")


def _load_and_prep_map(map_path):
    """
    Loads HS Map, truncates to HS6, and assigns new Granular Groups.
    """
    if not map_path.exists():
        print("Error: Map file missing.")
        return pd.DataFrame()

    # Define the precise HS6 -> Group mapping based on your list
    # Note: 811292 contains BOTH Unwrought Gallium and Unwrought Germanium
    hs6_groups = {
        # --- CONTROLLED: Gallium & Germanium ---
        "811292": "Unwrought Ga/Ge",  # Basket for Raw Metal
        "811299": "Wrought Ga/Ge",  # Basket for Bars/Rods
        "285390": "Ga/Ge Compounds",  # GaAs (and others)
        "285000": "Ga/Ge Compounds",  # GaN (and others)
        "282590": "Ga/Ge Compounds",  # Gallium Oxide
        "282560": "Ga/Ge Compounds",  # Germanium Oxide
        # --- CONTROLLED: Graphite ---
        "250410": "Natural Graphite",  # Flake & Spherical
        "380110": "Artificial Graphite",
        # --- WATCHLIST (Not currently restricted by 2023 dates) ---
        "280461": "Polysilicon",
        "280530": "Rare Earths",  # Metals
        "284690": "Rare Earths",  # Oxides
        "850511": "Rare Earths",  # Magnets
    }

    # Create DataFrame from dict
    df_map = pd.DataFrame(list(hs6_groups.items()), columns=["cmdCode", "plot_group"])
    return df_map


def _read_data():
    # 1. Load Map
    hs_map = _load_and_prep_map(Path("notes/hs_map.csv"))
    if hs_map.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 2. Load Granular Data
    master_file = DATA / "partners_granular_MASTER.parquet"
    if master_file.exists():
        df_partners = pd.read_parquet(master_file)
    else:
        p_files = sorted(DATA.glob("partners_granular_*.parquet"))
        if not p_files:
            return pd.DataFrame(), pd.DataFrame()
        df_partners = pd.concat([pd.read_parquet(f) for f in p_files], ignore_index=True)

    # 3. Load World Totals
    w_files = sorted(DATA.glob("comtrade_CM_HS_*.parquet"))
    if not w_files:
        return pd.DataFrame(), pd.DataFrame()
    df_world = pd.concat([pd.read_parquet(f) for f in w_files], ignore_index=True)

    # 4. Standardize & Merge
    for df in [df_partners, df_world]:
        df["period"] = df["period"].astype(str)
        df["cmdCode"] = df["cmdCode"].astype(str)  # Ensure string match

        if "flowCode" in df.columns:
            df.drop(df[df["flowCode"] != "X"].index, inplace=True)
        if "reporterCode" in df.columns:
            df.drop(df[df["reporterCode"] != 156].index, inplace=True)

        df["partnerCode"] = pd.to_numeric(df["partnerCode"], errors="coerce").fillna(-1).astype(int)

    # --- APPLY MAP ---
    df_partners = df_partners.merge(hs_map, on="cmdCode", how="inner")
    df_world = df_world.merge(hs_map, on="cmdCode", how="inner")

    # --- MAP NAMES ---
    df_partners["partner_name"] = df_partners["partnerCode"].map(CODE_TO_NAME).fillna("Unknown")

    # Deduplicate (Summing if multiple codes map to same group? No, drop duplicates first)
    df_partners = df_partners.drop_duplicates(
        subset=["period", "partnerCode", "cmdCode"], keep="last"
    )
    df_world = df_world.drop_duplicates(subset=["period", "cmdCode"], keep="last")

    # Dates
    df_partners["period_dt"] = _to_dt(df_partners["period"])
    df_world["period_dt"] = _to_dt(df_world["period"])

    return df_partners, df_world


def plot_leakage_stack(df_p, df_w):
    print("Generating Leakage Analysis...")

    partner_to_bloc = {}
    for bloc, countries in BLOCS.items():
        for c in countries:
            partner_to_bloc[c] = bloc

    def get_bloc(row):
        code = row["partnerCode"]
        name = row["partner_name"]

        if code == 156:
            return "Bonded (China)"
        if code in [999, 976, 977]:
            return "Unspecified"
        if name in partner_to_bloc:
            return partner_to_bloc[name]
        return "Tracked_Other"

    df_p["bloc"] = df_p.apply(get_bloc, axis=1)

    # Iterate over the new granular groups
    groups = df_p["plot_group"].unique()

    for g in groups:
        w_sub = df_w[df_w["plot_group"] == g]
        p_sub = df_p[df_p["plot_group"] == g]

        if w_sub.empty:
            print(f"Skipping {g} (No World Data)")
            continue

        world_total = w_sub.groupby("period_dt")["primaryValue"].sum()
        bloc_sums = p_sub.groupby(["period_dt", "bloc"])["primaryValue"].sum().unstack().fillna(0)

        aligned = pd.DataFrame({"World_Total": world_total}).join(bloc_sums).fillna(0)

        tracked_cols = [c for c in aligned.columns if c != "World_Total"]
        aligned["True_Leakage"] = aligned["World_Total"] - aligned[tracked_cols].sum(axis=1)
        aligned["True_Leakage"] = aligned["True_Leakage"].clip(lower=0)

        # Stats
        if g in EVENTS:
            start = EVENTS[g]
            end = start + pd.DateOffset(months=4)
            win = aligned[(aligned.index >= start) & (aligned.index < end)]

            if not win.empty:
                total = win["World_Total"].sum()
                leak = win["True_Leakage"].sum()
                print(f"\n=== {g.upper()} LEAKAGE DIAGNOSTIC ===")
                print(f"Total Exports (4mo Post-Ban): ${total:,.0f}")
                print(f"Unexplained Leakage:          ${leak:,.0f} ({leak/total:.1%})")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 7))

        stack_order = [
            "Adversary",
            "Intermediary",
            "Bonded (China)",
            "Unspecified",
            "Tracked_Other",
            "True_Leakage",
        ]
        labels = [
            "Adversary",
            "Intermediary",
            "China Bonded",
            "Unspecified",
            "Other Tracked",
            "Unknown Leakage",
        ]
        colors = ["#c0392b", "#27ae60", "#f1c40f", "#8e44ad", "#2980b9", "#95a5a6"]

        final_cols = [c for c in stack_order if c in aligned.columns and aligned[c].sum() > 0]
        final_colors = [colors[stack_order.index(c)] for c in final_cols]
        final_labels = [labels[stack_order.index(c)] for c in final_cols]

        if not final_cols:
            plt.close(fig)
            continue

        plot_data = aligned[final_cols].rolling(3, min_periods=1).mean()

        ax.stackplot(
            plot_data.index, plot_data.T, labels=final_labels, colors=final_colors, alpha=0.9
        )
        ax.plot(
            aligned.index,
            aligned["World_Total"].rolling(3, min_periods=1).mean(),
            color="black",
            ls="--",
            lw=1.5,
            label="Reported World Total",
        )

        if g in EVENTS:
            ax.axvline(EVENTS[g], color="black", ls=":", lw=2)
            ax.text(
                EVENTS[g], ax.get_ylim()[1] * 1.01, " Export Control", ha="left", fontweight="bold"
            )

        ax.set_title(f"Destinations & Leakage: {g}")
        ax.set_ylabel("USD")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # Clean filename
        fname = g.replace(" ", "_").replace("/", "_").lower()
        out_file = OUT / f"leakage_{fname}.png"
        fig.savefig(out_file)
        plt.close(fig)
        print(f"Saved {out_file}")


def main():
    df_p, df_w = _read_data()
    if not df_p.empty:
        plot_leakage_stack(df_p, df_w)
    else:
        print("No partner data found. Run scripts/pull_partners.py first.")


if __name__ == "__main__":
    main()
