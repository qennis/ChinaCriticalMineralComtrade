#!/usr/bin/env python3
"""
scripts/make_leakage_figures.py
Analyses 'The Black Hole': How much export volume is going to untracked destinations?
Comparing World Total (Partner=0) vs. Sum of Tracked Partners.
"""
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Setup paths
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from china_ir.etl import attach_hs_map  # noqa: E402

DATA = Path("data_work")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

EVENTS = {
    "gallium": pd.Timestamp("2023-08-01"),
    "graphite": pd.Timestamp("2023-12-01"),
}

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
    ],
}


def _to_dt(series):
    return pd.to_datetime(series.astype(str), format="%Y%m")


def _read_data():
    # 1. Load Granular Partners
    p_files = sorted(DATA.glob("partners_granular_*.parquet"))
    if not p_files:
        print("No partner data found.")
        return pd.DataFrame(), pd.DataFrame()
    df_partners = pd.concat([pd.read_parquet(f) for f in p_files], ignore_index=True)

    # 2. Load World Totals (Partner=0) from your original pulls
    # Note: These are typically in 'comtrade_CM_HS_*.parquet' or similar
    w_files = sorted(DATA.glob("comtrade_CM_HS_*.parquet"))
    if not w_files:
        print("No world aggregate data found.")
        return pd.DataFrame(), pd.DataFrame()
    df_world = pd.concat([pd.read_parquet(f) for f in w_files], ignore_index=True)

    # Apply Mapping
    hs_map = Path("notes/hs_map.csv")
    df_partners = attach_hs_map(df_partners, hs_map)
    df_world = attach_hs_map(df_world, hs_map)

    # Normalize Dates
    df_partners["period_dt"] = _to_dt(df_partners["period"])
    df_world["period_dt"] = _to_dt(df_world["period"])

    return df_partners, df_world


def plot_leakage_stack(df_p, df_w):
    print("Generating Leakage Stack plots...")

    # Map Partners to Blocs
    bloc_map = {p: b for b, partners in BLOCS.items() for p in partners}
    df_p["bloc"] = df_p["partner_name"].map(bloc_map).fillna("Tracked_Other")

    for g in ["gallium", "graphite"]:
        if g not in EVENTS:
            continue

        # 1. Prepare World Total Series
        w_sub = df_w[df_w["group"] == g]
        world_total = w_sub.groupby("period_dt")["primaryValue"].sum()

        # 2. Prepare Bloc Sums
        p_sub = df_p[df_p["group"] == g]
        bloc_sums = p_sub.groupby(["period_dt", "bloc"])["primaryValue"].sum().unstack().fillna(0)

        # 3. Align Dataframes
        # Reindex bloc_sums to match world_total dates (filling missing months with 0)
        aligned = pd.DataFrame(index=world_total.index)
        aligned["World_Total"] = world_total
        aligned = aligned.join(bloc_sums).fillna(0)

        # 4. Calculate 'Rest of World' (Leakage)
        # Residual = World - (Adversary + Intermediary)
        tracked_sum = aligned.get("Adversary", 0) + aligned.get("Intermediary", 0)
        aligned["Rest of World"] = aligned["World_Total"] - tracked_sum

        # Handle negative residuals (data mismatches/timing issues) by clipping at 0
        aligned["Rest of World"] = aligned["Rest of World"].clip(lower=0)

        # 5. Plot Stack
        fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

        cols = ["Adversary", "Intermediary", "Rest of World"]
        colors = ["#d62728", "#2ca02c", "#7f7f7f"]  # Red, Green, Grey
        labels = ["Adversary (US/JP/EU)", "Intermediary (VN/MY/MX)", "Rest of World (Untracked)"]

        # Plot data exists
        plot_data = aligned[cols]
        if plot_data.sum().sum() == 0:
            continue

        ax.stackplot(plot_data.index, plot_data.T, labels=labels, colors=colors, alpha=0.85)

        # Add Total Line for comparison
        ax.plot(
            aligned.index,
            aligned["World_Total"],
            color="black",
            ls="--",
            lw=1.5,
            label="Reported World Total",
        )

        # Formatting
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))

        if g in EVENTS:
            ax.axvline(EVENTS[g], color="black", ls=":", lw=2)
            ax.text(EVENTS[g], ax.get_ylim()[1] * 1.02, "Control", ha="center")

        ax.set_title(f"Global Destination Structure: {g.title()} (USD)")
        ax.set_ylabel("Total Export Value (USD)")
        ax.legend(loc="upper left")

        # Save
        fig.tight_layout()
        fig.savefig(OUT / f"analysis_leakage_{g}.png")
        print(f"Saved figures/analysis_leakage_{g}.png")
        plt.close()

        # Dump Data for Chat
        print(f"\n--- {g.upper()} LEAKAGE DATA (Last 12 Months) ---")
        print(aligned[cols].tail(12).to_csv(sep="\t", float_format="%.0f"))


if __name__ == "__main__":
    df_p, df_w = _read_data()
    if not df_p.empty and not df_w.empty:
        plot_leakage_stack(df_p, df_w)
    else:
        print(
            "Missing data (Check if you have both 'partners_granular' and 'comtrade_CM_HS' files)."
        )
