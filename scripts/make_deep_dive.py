#!/usr/bin/env python3
"""
scripts/make_deep_dive.py
Deep dive analysis for Graphite and Gallium/Germanium.

Features:
- Composition Analysis: "What specific products are key partners buying?" (Stacked Area)
- Unit Value Drilldown: "How much are they paying per HS code?" (Line + Volatility Band)
- Hierarchy: Broad Group -> Subgroup -> HS6.
"""
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

# Setup paths
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- CONFIGURATION ---
DATA = Path("data_work")
OUT = Path("figures/deep_dives")
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# --- MAPPINGS ---
EVENTS = {
    "Gallium/Germanium": pd.Timestamp("2023-08-01"),
    "Graphite": pd.Timestamp("2023-12-01"),
}

HS6_TO_SUBGROUP = {
    "811292": "Unwrought Ga/Ge",
    "811299": "Wrought Ga/Ge",
    "285390": "Ga/Ge Compounds",
    "285000": "Ga/Ge Compounds",
    "282590": "Ga/Ge Compounds",
    "282560": "Ga/Ge Compounds",
    "250410": "Natural Graphite",
    "380110": "Artificial Graphite",
    "280461": "Polysilicon",
    "280530": "Rare Earths (Metal)",
    "284690": "Rare Earths (Oxide)",
    "850511": "Rare Earths (Magnet)",
}

SUBGROUP_TO_BROAD = {
    "Unwrought Ga/Ge": "Gallium/Germanium",
    "Wrought Ga/Ge": "Gallium/Germanium",
    "Ga/Ge Compounds": "Gallium/Germanium",
    "Natural Graphite": "Graphite",
    "Artificial Graphite": "Graphite",
    "Polysilicon": "Polysilicon",
    "Rare Earths (Metal)": "Rare Earths",
    "Rare Earths (Oxide)": "Rare Earths",
    "Rare Earths (Magnet)": "Rare Earths",
}

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

# Key Partners to Drill Down Into
TARGET_COUNTRIES = [
    "USA",
    "Japan",
    "South Korea",
    "India",
    "Germany",
    "Netherlands",
    "Vietnam",
    "Poland",
]


# --- HELPERS ---
def _to_dt(yyyymm: pd.Series) -> pd.Series:
    return pd.to_datetime(yyyymm.astype(str), format="%Y%m")


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3, ls="--")


def _load_data():
    files = sorted(DATA.glob("partners_granular_*.parquet"))
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Filters
    df["period"] = df["period"].astype(str)
    df["partnerCode"] = pd.to_numeric(df["partnerCode"], errors="coerce").fillna(-1).astype(int)
    if "flowCode" in df.columns:
        df = df[df["flowCode"] == "X"]
    if "reporterCode" in df.columns:
        df = df[df["reporterCode"] == 156]

    # Map Groups
    df["hs6"] = df["cmdCode"].astype(str).str[:6]
    df["subgroup"] = df["hs6"].map(HS6_TO_SUBGROUP)
    df["broad_group"] = df["subgroup"].map(SUBGROUP_TO_BROAD)
    df = df.dropna(subset=["broad_group"])

    # Map Partners
    df["partner_name"] = df["partnerCode"].map(CODE_TO_NAME).fillna("Unknown")
    df["period_dt"] = _to_dt(df["period"])

    # Quantity & Value
    df["qty"] = np.nan
    for c in ["primaryQuantity", "netWgt", "qty"]:
        if c in df.columns:
            df["qty"] = pd.to_numeric(df[c], errors="coerce")
            break
    df["val"] = pd.to_numeric(df["primaryValue"], errors="coerce").fillna(0)

    return df


# --- 1. Composition Changes (Subgroup Stack) ---
def plot_composition_changes(df):
    """
    Shows the mix of Subgroups (e.g. Unwrought vs Compounds) for key partners.
    """
    print("Generating Composition Plots...")

    # Only keep targets
    df_sub = df[df["partner_name"].isin(TARGET_COUNTRIES)].copy()

    for bg, event_date in EVENTS.items():
        sub = df_sub[df_sub["broad_group"] == bg]
        if sub.empty:
            continue

        # Agg by Month, Partner, Subgroup
        agg = sub.groupby(["period_dt", "partner_name", "subgroup"])["val"].sum().reset_index()

        # Determine grid size
        partners = agg["partner_name"].unique()
        n = len(partners)
        cols = 2
        rows = (n // cols) + (1 if n % cols else 0)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True, sharey=True)
        axes = axes.flatten()

        # Consistent colors for subgroups
        subgroups = sorted(agg["subgroup"].unique())
        colors = plt.cm.tab10.colors
        color_map = {s: colors[i % len(colors)] for i, s in enumerate(subgroups)}

        for i, country in enumerate(partners):
            ax = axes[i]
            c_data = agg[agg["partner_name"] == country]

            pivot = c_data.pivot(index="period_dt", columns="subgroup", values="val").fillna(0)

            # Normalize to 100% Share
            pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
            pivot_smooth = pivot_pct.rolling(3, min_periods=1).mean()

            ax.stackplot(
                pivot_smooth.index,
                pivot_smooth.T,
                labels=pivot_smooth.columns,
                colors=[color_map[c] for c in pivot_smooth.columns],
                alpha=0.85,
            )

            ax.set_title(f"{country}: Import Composition", fontweight="bold")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            _date_axis(ax)
            ax.axvline(event_date, color="white", linestyle="--", linewidth=1.5)

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Unified Legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[s]) for s in subgroups]
        fig.legend(
            handles, subgroups, loc="lower center", ncol=len(subgroups), bbox_to_anchor=(0.5, 0.01)
        )

        fig.suptitle(
            f"{bg}: Product Mix Evolution (Share of Value)", y=1.01, fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)  # Make room for legend

        fname = bg.replace("/", "_").lower()
        fig.savefig(OUT / f"composition_{fname}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved composition plot for {bg}")


# --- 2. HS6 Unit Values (Line + Volatility Band) ---
def plot_hs6_unit_values(df):
    """
    Plots Price Trends ($/kg) for specific HS6 codes with Volatility Bands.
    Volatility Band = Rolling Mean +/- Rolling Std Dev.
    """
    print("Generating HS6 Unit Value Plots...")

    # Define interesting HS6 codes to drill into
    focus_codes = {
        "811292": "Unwrought Ga/Ge",
        "380110": "Artificial Graphite",
        "250410": "Natural Graphite",
    }

    palette = sns.color_palette("tab10")

    for code, desc in focus_codes.items():
        sub = df[(df["hs6"] == code) & (df["partner_name"].isin(TARGET_COUNTRIES))].copy()
        if sub.empty:
            continue

        # Filter noise
        sub = sub[sub["qty"] > 100]

        agg = (
            sub.groupby(["period_dt", "partner_name"])
            .agg({"val": "sum", "qty": "sum"})
            .reset_index()
        )
        agg["uv"] = agg["val"] / agg["qty"]
        agg = agg.sort_values("period_dt")

        # Calculate Smooth Mean & Volatility (Std Dev)
        grouped = agg.groupby("partner_name")["uv"]
        agg["uv_smooth"] = grouped.transform(lambda x: x.rolling(3, min_periods=1).mean())
        agg["uv_std"] = grouped.transform(lambda x: x.rolling(3, min_periods=1).std())

        fig, ax = plt.subplots(figsize=(12, 7))
        country_colors = {c: palette[i % len(palette)] for i, c in enumerate(TARGET_COUNTRIES)}

        # Plot countries
        plotted = False
        for country in TARGET_COUNTRIES:
            c_data = agg[agg["partner_name"] == country]
            if c_data.empty:
                continue

            plotted = True
            col = country_colors[country]

            # Mean Line
            ax.plot(
                c_data["period_dt"], c_data["uv_smooth"], label=country, color=col, linewidth=2.5
            )

            # Volatility Band (Error Bar equivalent)
            lower = c_data["uv_smooth"] - c_data["uv_std"]
            upper = c_data["uv_smooth"] + c_data["uv_std"]
            ax.fill_between(c_data["period_dt"], lower, upper, color=col, alpha=0.15)

        if not plotted:
            plt.close(fig)
            continue

        # Add Event Line based on the broad group of the code
        subgroup = HS6_TO_SUBGROUP.get(code)
        broad = SUBGROUP_TO_BROAD.get(subgroup)
        if broad in EVENTS:
            ax.axvline(EVENTS[broad], color="black", linestyle="--", alpha=0.8, label="Control")

        ax.set_title(f"Unit Value: {code} - {desc} (with Volatility Bands)")
        ax.set_ylabel("USD / kg")
        ax.set_yscale("log")  # Log scale for prices
        ax.yaxis.set_major_formatter(mtick.ScalarFormatter())

        _date_axis(ax)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.savefig(OUT / f"price_hs6_{code}.png")
        plt.close(fig)
        print(f"Saved price plot for {code}")


def main():
    df = _load_data()
    if not df.empty:
        plot_composition_changes(df)
        plot_hs6_unit_values(df)
        print(f"Done. Figures saved to {OUT}")
    else:
        print("No Data Loaded")


if __name__ == "__main__":
    main()
