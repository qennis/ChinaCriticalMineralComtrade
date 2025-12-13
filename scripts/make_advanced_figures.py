#!/usr/bin/env python3
"""
scripts/make_advanced_figures.py
Advanced Track 1 Analysis: Price Divergence, Placebo Tests, and Specification Creep.
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
# Mapping specific commodities to their 'Controls' vs 'Placebos'
# We use Aluminum (7601) or Silicon (2804) as control groups
PLACEBO_PAIRS = [("gallium", "aluminum"), ("gallium", "silicon"), ("graphite", "aluminum")]


def _read_data():
    files = sorted(DATA.glob("partners_granular_*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    hs_map = Path("notes/hs_map.csv")
    df = attach_hs_map(df, hs_map)
    df["period_dt"] = pd.to_datetime(df["period"].astype(str), format="%Y%m")

    # Calculate Unit Value (USD/kg)
    # Ensure netWgt is numeric
    df["netWgt"] = pd.to_numeric(df["netWgt"], errors="coerce")
    df["unit_val"] = df["primaryValue"] / df["netWgt"]

    # Map Blocs
    bloc_map = {p: b for b, partners in BLOCS.items() for p in partners}
    df["bloc"] = df["partner_name"].map(bloc_map).fillna("Other")

    return df


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.grid(True, alpha=0.3, ls="--")


# 1. Price Divergence Analysis
def plot_price_divergence(df):
    """Plot Unit Value (Price/kg) for Adversary vs Intermediary blocs."""
    print("Generating Price Divergence plots...")

    for g, event_date in EVENTS.items():
        sub = df[(df["group"] == g) & (df["netWgt"] > 0)]
        if sub.empty:
            continue

        # Weighted Average Price per Bloc per Month
        # Sum(Value) / Sum(Qty) is better than Mean(Unit Value)
        monthly = sub.groupby(["period_dt", "bloc"])[["primaryValue", "netWgt"]].sum()
        monthly["w_price"] = monthly["primaryValue"] / monthly["netWgt"]
        pivot = monthly["w_price"].unstack()

        # Filter for key blocs
        cols = [c for c in ["Adversary", "Intermediary"] if c in pivot.columns]
        if not cols:
            continue

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        for c in cols:
            color = "tab:red" if c == "Adversary" else "tab:green"
            ax.plot(pivot.index, pivot[c], label=c, color=color, lw=2)

        _date_axis(ax)
        ax.axvline(event_date, color="black", ls="--", label="Control Effective")
        ax.set_title(f"Price Divergence: {g.title()} (Unit Value USD/kg)")
        ax.set_ylabel("USD / kg")
        ax.legend()

        fig.tight_layout()
        fig.savefig(OUT / f"analysis_price_{g}.png")
        plt.close()


# 2. Placebo Test (Difference-in-Differences Visual)
def plot_placebo_test(df):
    """Compare Restricted Mineral trend vs Control Commodity trend for the SAME Bloc."""
    print("Generating Placebo (Control) tests...")

    # We focus on the ADVERSARY bloc, as they are the target
    target_bloc = "Adversary"

    for treatment_group, control_group in PLACEBO_PAIRS:
        if treatment_group not in EVENTS:
            continue
        event_date = EVENTS[treatment_group]

        sub = df[(df["bloc"] == target_bloc) & (df["group"].isin([treatment_group, control_group]))]

        # Normalize to Index (Pre-Event Mean = 100)
        # Define Pre-Window: 6 months before event
        pre_window = (sub["period_dt"] >= event_date - pd.DateOffset(months=6)) & (
            sub["period_dt"] < event_date
        )

        pivot = sub.groupby(["period_dt", "group"])["primaryValue"].sum().unstack()

        # Normalize
        for col in pivot.columns:
            base = pivot.loc[pivot.index.isin(sub[pre_window]["period_dt"].unique()), col].mean()
            if base > 0:
                pivot[col] = (pivot[col] / base) * 100

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        if treatment_group in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[treatment_group],
                label=f"Treated: {treatment_group.title()}",
                color="tab:red",
                lw=2.5,
            )
        if control_group in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[control_group],
                label=f"Control: {control_group.title()}",
                color="gray",
                ls="--",
                lw=1.5,
            )

        _date_axis(ax)
        ax.axvline(event_date, color="black", ls=":")
        ax.set_title(
            f"Placebo Test: {treatment_group.title()} vs "
            + f"{control_group.title()} (Exports to {target_bloc})"
        )
        ax.set_ylabel("Export Value Index (Pre-Event = 100)")
        ax.legend()

        fig.tight_layout()
        fig.savefig(OUT / f"analysis_placebo_{treatment_group}_vs_{control_group}.png")
        plt.close()


# 3. Specification Creep (Product Mix)
def plot_spec_creep(df):
    """Track share of specific HS codes within the group over time."""
    print("Generating Specification Creep analysis...")

    # Focus on Gallium (Raw vs Processed/Compounds)
    # 811292: Unwrought Gallium (Raw)
    # 284690: Compounds (Often oxides/nitrides)
    # Note: Check your map, this might be REE in some maps,
    # but for Gallium specifically look at your map.csv.
    # Let's just plot the TOP 3 HS codes share for the group.

    for g in ["gallium", "graphite"]:
        sub = df[df["group"] == g]
        if sub.empty:
            continue

        # Identify top 3 codes by total volume
        top_codes = sub.groupby("hs6")["primaryValue"].sum().nlargest(3).index.tolist()

        pivot = sub.pivot_table(
            index="period_dt", columns="hs6", values="primaryValue", aggfunc="sum"
        ).fillna(0)

        # Calculate Share
        pivot_share = pivot.div(pivot.sum(axis=1), axis=0)

        # Only plot top codes
        plot_data = pivot_share[top_codes]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        plot_data.plot(kind="area", ax=ax, alpha=0.8, stacked=True)

        _date_axis(ax)
        if g in EVENTS:
            ax.axvline(EVENTS[g], color="black", ls="--")

        ax.set_title(f"Specification Shift: {g.title()} (Share of Export Value)")
        ax.set_ylabel("Share (0-1)")
        ax.legend(title="HS Code")

        fig.tight_layout()
        fig.savefig(OUT / f"analysis_specs_{g}.png")
        plt.close()


if __name__ == "__main__":
    df = _read_data()
    if not df.empty:
        plot_price_divergence(df)
        plot_placebo_test(df)
        plot_spec_creep(df)
        print("Done. Check figures/analysis_*.png")
    else:
        print("No data found.")
