#!/usr/bin/env python3
"""
scripts/make_country_figures.py
Generates Country-Level Forensic Trajectories.

Features:
- Country Trajectories: Quantity (Log), Price (Log), and Spec Concentration (HHI).
- Destination Stacks: "Who is buying?" for both Broad Sectors and HS6 codes.
- Hierarchical: Uses Broad Groups (Ga/Ge) -> Subgroups -> HS6.
"""
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
OUT = Path("figures/countries")
OUT.mkdir(parents=True, exist_ok=True)

# Styling
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


def select_top_partners(df_group, n=8):
    total = df_group.groupby("partner_name")["val"].sum()
    return total.sort_values(ascending=False).head(n).index.tolist()


def get_dominant_subgroup(df_partner):
    if df_partner.empty:
        return "N/A"
    agg = df_partner.groupby("subgroup")["val"].sum()
    if agg.empty:
        return "N/A"
    top = agg.sort_values(ascending=False)
    name = top.index[0]
    share = top.iloc[0] / top.sum()
    return f"{name} ({share:.0%})"


def calculate_spec_hhi(df_partner):
    """Calculates HHI of product mix (HS6) for a single partner."""
    monthly = df_partner.groupby(["period_dt", "hs6"])["val"].sum().reset_index()
    monthly_totals = monthly.groupby("period_dt")["val"].transform("sum")
    monthly["share"] = monthly["val"] / monthly_totals
    monthly["sq"] = monthly["share"] ** 2
    hhi = monthly.groupby("period_dt")["sq"].sum()
    return hhi.rolling(3, min_periods=1).mean()


# --- PLOT 1: Country Trajectories (Qty, Price, Spec) ---
def plot_country_trajectories(df):
    print("Generating Country Trajectories (Qty, Price, Spec)...")
    palette = sns.color_palette("tab10")

    for bg, event_date in EVENTS.items():
        sub = df[df["broad_group"] == bg].copy()
        if sub.empty:
            continue

        partners = select_top_partners(sub)
        sub = sub[sub["partner_name"].isin(partners)].copy()

        labels = {}
        for p in partners:
            p_data = sub[sub["partner_name"] == p]
            dom = get_dominant_subgroup(p_data)
            labels[p] = f"{p}\nMain: {dom}"

        # 1. Quantity (Log)
        fig, ax = plt.subplots(figsize=(12, 7))
        for i, p in enumerate(partners):
            d = sub[sub["partner_name"] == p].groupby("period_dt")["qty"].sum()
            d = d.rolling(3, min_periods=1).mean()
            ax.plot(d.index, d, label=labels[p], color=palette[i % 10], lw=2)

        _date_axis(ax)
        ax.set_yscale("log")
        ax.axvline(event_date, color="black", ls="--", label="Control")
        ax.set_title(f"{bg}: Export Quantity by Top Partners (Log Scale)")
        ax.set_ylabel("Quantity (kg/units)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.savefig(OUT / f"country_quantity_{bg.replace('/','_').lower()}.png")
        plt.close(fig)

        # 2. Unit Price (Log)
        fig, ax = plt.subplots(figsize=(12, 7))
        for i, p in enumerate(partners):
            d = (
                sub[sub["partner_name"] == p]
                .groupby("period_dt")
                .agg(val=("val", "sum"), qty=("qty", "sum"))
            )
            uv = d["val"] / d["qty"]
            uv[d["val"] < 1000] = np.nan  # Filter noise
            uv = uv.rolling(3, min_periods=1).mean()
            ax.plot(uv.index, uv, label=labels[p], color=palette[i % 10], lw=2)

        _date_axis(ax)
        ax.set_yscale("log")  # Requested Log Scale
        ax.axvline(event_date, color="black", ls="--", label="Control")
        ax.set_title(f"{bg}: Unit Price by Country (Log Scale)")
        ax.set_ylabel("Unit Value (USD/kg)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.savefig(OUT / f"country_prices_{bg.replace('/','_').lower()}.png")
        plt.close(fig)

        # 3. Specification HHI (Restored)
        fig, ax = plt.subplots(figsize=(12, 7))
        for i, p in enumerate(partners):
            p_data = sub[sub["partner_name"] == p]
            hhi = calculate_spec_hhi(p_data)
            ax.plot(hhi.index, hhi, label=labels[p], color=palette[i % 10], lw=2)

        _date_axis(ax)
        ax.axvline(event_date, color="black", ls="--", label="Control")
        ax.set_title(
            f"{bg}: Product Specification Concentration (HHI)\n"
            "(1.0 = Country buys only 1 type of HS6 code)"
        )
        ax.set_ylabel("HHI (0-1)")
        ax.set_ylim(0, 1.1)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.savefig(OUT / f"country_spec_{bg.replace('/','_').lower()}.png")
        plt.close(fig)

        print(f"Saved country plots for {bg}")


# --- PLOT 2: Destination Stacks (Broad & HS6) ---
def plot_destination_stacks(df):
    print("Generating Destination Stacks...")

    # 1. Broad Groups
    for bg in EVENTS.keys():
        sub = df[df["broad_group"] == bg].copy()
        if sub.empty:
            continue

        # Top Partners + "Rest of World"
        top = select_top_partners(sub, n=9)
        sub["display_name"] = sub["partner_name"].apply(
            lambda x: x if x in top else "Rest of World"
        )

        wide = sub.pivot_table(
            index="period_dt", columns="display_name", values="val", aggfunc="sum"
        ).fillna(0)

        # Sort cols: Top partners first, RoW last
        cols = [c for c in top if c in wide.columns] + ["Rest of World"]
        cols = [c for c in cols if c in wide.columns]
        wide = wide[cols]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.stackplot(wide.index, wide.T / 1e6, labels=wide.columns, alpha=0.85, edgecolor="white")
        _date_axis(ax)
        ax.axvline(EVENTS[bg], color="black", ls="--", lw=2)
        ax.set_title(f"{bg}: Destination Stack (Top Markets)")
        ax.set_ylabel("Value (Million USD)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.savefig(OUT / f"dest_stack_{bg.replace('/','_').lower()}.png")
        plt.close(fig)

    # 2. HS6 Codes (Drilldown)
    targets = set(EVENTS.keys())
    for code in df["hs6"].unique():
        subgroup = HS6_TO_SUBGROUP.get(code)
        broad = SUBGROUP_TO_BROAD.get(subgroup)
        if broad not in targets:
            continue  # Only plot relevant codes

        sub = df[df["hs6"] == code].copy()
        if sub["val"].sum() < 1e5:
            continue  # Skip tiny codes

        top = select_top_partners(sub, n=9)
        sub["display_name"] = sub["partner_name"].apply(
            lambda x: x if x in top else "Rest of World"
        )

        wide = sub.pivot_table(
            index="period_dt", columns="display_name", values="val", aggfunc="sum"
        ).fillna(0)
        cols = [c for c in top if c in wide.columns] + ["Rest of World"]
        cols = [c for c in cols if c in wide.columns]
        wide = wide[cols]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.stackplot(wide.index, wide.T / 1e6, labels=wide.columns, alpha=0.85, edgecolor="white")
        _date_axis(ax)
        ax.axvline(EVENTS[broad], color="black", ls="--", lw=2)
        ax.set_title(f"HS6 Destination Stack: {code}\n({subgroup})")
        ax.set_ylabel("Value (Million USD)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.savefig(OUT / f"dest_stack_hs6_{code}.png")
        plt.close(fig)
        print(f"Saved dest stack for {code}")


def main():
    df = _load_data()
    if not df.empty:
        plot_country_trajectories(df)
        plot_destination_stacks(df)
        print(f"Done. Figures saved to {OUT}")
    else:
        print("No data found.")


if __name__ == "__main__":
    main()
