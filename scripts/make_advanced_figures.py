#!/usr/bin/env python3
"""
scripts/make_advanced_figures.py
Advanced Analysis: Price Divergence, Event Studies, and Spec Creep.

Updates:
- Event Study: Window extended to +/- 12 months (2x duration).
- Divergence Overview: Split into two clear figures (Ga/Ge vs. Graphite).
- Peer Comparison: Renamed "Placebo" to "Strategic Peer" for accuracy.
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
OUT = Path("figures/advanced")
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
}

# (Treated, Control, Label)
PEER_PAIRS = [
    ("Gallium/Germanium", "Rare Earths", "Strategic Peer (Rare Earths)"),
    ("Graphite", "Polysilicon", "Industrial Peer (Polysilicon)"),
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

    # Map Partner Names & Blocs
    df["partner_name"] = df["partnerCode"].map(CODE_TO_NAME).fillna("Unknown")
    bloc_map = {p: b for b, partners in BLOCS.items() for p in partners}
    df["bloc"] = df["partner_name"].map(bloc_map).fillna("Other")

    df["period_dt"] = _to_dt(df["period"])

    # Quantity
    df["qty"] = np.nan
    for c in ["primaryQuantity", "netWgt", "qty"]:
        if c in df.columns:
            df["qty"] = pd.to_numeric(df[c], errors="coerce")
            break
    df["val"] = pd.to_numeric(df["primaryValue"], errors="coerce").fillna(0)

    return df


# --- 1. Split Divergence Overview (Adversary Premium) ---
def plot_divergence_overview(df):
    """Generates two separate plots for Ga/Ge and Graphite premiums."""
    print("Generating Price Divergence Overview (Split)...")

    families = {
        "Gallium_Germanium": {
            "broad": "Gallium/Germanium",
            "subs": ["Unwrought Ga/Ge", "Wrought Ga/Ge", "Ga/Ge Compounds"],
            "peer": "Rare Earths",
        },
        "Graphite": {
            "broad": "Graphite",
            "subs": ["Natural Graphite", "Artificial Graphite"],
            "peer": "Polysilicon",
        },
    }

    # Calculate Ratios
    all_targets = set()
    for f in families.values():
        all_targets.add(f["broad"])
        all_targets.update(f["subs"])
        all_targets.add(f["peer"])

    ratios = {}
    for t in all_targets:
        if t in SUBGROUP_TO_BROAD.values():
            col = "broad_group"
        else:
            col = "subgroup"

        sub = df[df[col] == t].copy()
        if sub["qty"].isna().all():
            continue
        sub = sub[sub["qty"] > 0]

        monthly = sub.groupby(["period_dt", "bloc"]).agg(val=("val", "sum"), qty=("qty", "sum"))
        monthly["uv"] = monthly["val"] / monthly["qty"]
        pivot = monthly["uv"].unstack()

        if "Adversary" in pivot and "Intermediary" in pivot:
            r = pivot["Adversary"] / pivot["Intermediary"]
            ratios[t] = r.rolling(3, min_periods=1).mean()

    # Generate Plots
    for fname, cfg in families.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        # Broad Group (Thick)
        if cfg["broad"] in ratios:
            ax.plot(
                ratios[cfg["broad"]].index,
                ratios[cfg["broad"]],
                label=f"SECTOR: {cfg['broad']}",
                color="#c0392b",
                lw=3,
            )

        # Peer (Dotted)
        if cfg["peer"] in ratios:
            ax.plot(
                ratios[cfg["peer"]].index,
                ratios[cfg["peer"]],
                label=f"PEER: {cfg['peer']}",
                color="#7f8c8d",
                ls=":",
                lw=2,
            )

        # Subgroups (Thin)
        colors = sns.color_palette("husl", len(cfg["subs"]))
        for i, sub in enumerate(cfg["subs"]):
            if sub in ratios:
                ax.plot(
                    ratios[sub].index, ratios[sub], label=sub, color=colors[i], lw=1.5, alpha=0.8
                )

        ax.axhline(1.0, color="black", lw=1)
        if cfg["broad"] in EVENTS:
            ax.axvline(EVENTS[cfg["broad"]], color="black", ls="--", label="Control Effective")

        ax.set_title(f"Adversary Price Premium: {cfg['broad']}")
        ax.set_ylabel("Price Ratio (Adversary / Intermediary)")
        _date_axis(ax)
        ax.legend(loc="upper left")

        out_name = f"divergence_overview_{fname.lower()}.png"
        fig.savefig(OUT / out_name)
        plt.close(fig)
        print(f"Saved {out_name}")


# --- 2. Event Study (Extended Window) ---
def plot_event_study(df):
    print("Generating Event Study Comparison...")

    fig, ax = plt.subplots(figsize=(10, 6))

    targets = ["Gallium/Germanium", "Graphite"]
    colors = {"Gallium/Germanium": "#e74c3c", "Graphite": "#2c3e50"}

    for t in targets:
        if t not in EVENTS:
            continue
        event_date = EVENTS[t]

        # Filter Data
        sub = df[df["broad_group"] == t].copy()
        monthly = sub.groupby("period_dt")["val"].sum()

        # Define 24-month window: [Event-12, Event+12]
        start_date = event_date - pd.DateOffset(months=12)
        end_date = event_date + pd.DateOffset(months=12)

        window = monthly[(monthly.index >= start_date) & (monthly.index <= end_date)]
        if window.empty:
            continue

        # Calculate Baseline from strict 6-month pre-ban window (t-6 to t-1)
        # This keeps the Index=100 anchor consistent despite the wider view.
        baseline_start = event_date - pd.DateOffset(months=6)
        baseline_data = monthly[(monthly.index >= baseline_start) & (monthly.index < event_date)]
        baseline = baseline_data.mean()

        if baseline == 0:
            continue

        indexed = (window / baseline) * 100

        # Relative Time Axis
        months_diff = []
        for d in indexed.index:
            diff = (d.year - event_date.year) * 12 + (d.month - event_date.month)
            months_diff.append(diff)

        ax.plot(
            months_diff,
            indexed.values,
            label=f"{t} (Ban={event_date.date()})",
            color=colors.get(t, "black"),
            lw=2.5,
            marker="o",
            markersize=4,
        )

    ax.axvline(0, color="black", ls="--", label="Implementation (t=0)")
    ax.axhline(100, color="gray", lw=1, alpha=0.5)

    ax.set_title("Event Study: Impact of Export Controls on Total Value")
    ax.set_xlabel("Months Relative to Implementation")
    ax.set_ylabel("Export Value Index (100 = 6mo Pre-Ban Avg)")
    ax.legend()

    # Extended X-Limits
    ax.set_xlim(-12, 12)

    fig.savefig(OUT / "event_study_comparison.png")
    plt.close(fig)
    print("Saved event_study_comparison.png")


# --- 3. Peer Comparison ---
def plot_peer_comparison(df):
    print("Generating Peer Comparisons...")
    target_bloc = "Adversary"

    for treatment, control, label in PEER_PAIRS:
        if treatment not in EVENTS:
            continue
        event_date = EVENTS[treatment]

        sub = df[
            (df["bloc"] == target_bloc) & (df["broad_group"].isin([treatment, control]))
        ].copy()
        if sub.empty:
            continue

        raw = sub.groupby(["period_dt", "broad_group"])["val"].sum().unstack()

        # Normalize
        start_win = event_date - pd.DateOffset(months=6)
        indexed = pd.DataFrame(index=raw.index)

        for col in [treatment, control]:
            if col not in raw.columns:
                continue
            baseline_slice = raw.loc[(raw.index >= start_win) & (raw.index < event_date), col]
            scalar = baseline_slice.mean()
            if scalar > 0:
                indexed[col] = (raw[col] / scalar) * 100
            else:
                indexed[col] = np.nan

        smooth = indexed.rolling(3, min_periods=1).mean()

        if treatment not in smooth.columns or control not in smooth.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            smooth.index, smooth[treatment], label=f"Target: {treatment}", color="#c0392b", lw=3
        )
        ax.plot(
            smooth.index, smooth[control], label=f"Peer: {control}", color="#95a5a6", ls="--", lw=2
        )

        _date_axis(ax)
        ax.axvline(event_date, color="black", ls=":", label="Control Effective")
        ax.axhline(100, color="black", lw=1, alpha=0.5)

        ax.set_title(f"Strategic Peer Test: Exports to {target_bloc}\n(Reference: {label})")
        ax.set_ylabel("Value Index (100 = 6mo Pre-Ban Avg)")
        ax.legend(loc="upper left")

        fname = f"peer_{treatment}_vs_{control}".replace("/", "").replace(" ", "_").lower()
        fig.savefig(OUT / f"{fname}.png")
        plt.close(fig)


# --- 4. Drilldowns ---
def plot_drilldowns(df):
    print("Generating Drilldown Plots...")
    for g, event_date in EVENTS.items():
        sub = df[df["broad_group"] == g].copy()
        if sub["qty"].isna().all():
            continue
        sub = sub[sub["qty"] > 0]
        monthly = sub.groupby(["period_dt", "bloc"]).agg(val=("val", "sum"), qty=("qty", "sum"))
        monthly["uv"] = monthly["val"] / monthly["qty"]
        pivot = monthly["uv"].unstack()
        cols = [c for c in ["Adversary", "Intermediary"] if c in pivot.columns]
        if len(cols) < 2:
            continue
        pivot = pivot[cols].rolling(3, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        for c in cols:
            ax.plot(pivot.index, pivot[c], label=c, lw=2.5)
        _date_axis(ax)
        ax.axvline(event_date, color="black", ls=":")
        ax.set_title(f"Price Divergence: {g}")
        fig.savefig(OUT / f"divergence_sector_{g.replace('/','_').lower()}.png")
        plt.close(fig)

    targets = set(EVENTS.keys())
    for code in df["hs6"].unique():
        subgroup = HS6_TO_SUBGROUP.get(code)
        broad = SUBGROUP_TO_BROAD.get(subgroup)
        if broad not in targets:
            continue

        sub = df[df["hs6"] == code].copy()
        if sub["qty"].isna().all():
            continue
        sub = sub[sub["qty"] > 10]
        monthly = sub.groupby(["period_dt", "bloc"]).agg(val=("val", "sum"), qty=("qty", "sum"))
        monthly["uv"] = monthly["val"] / monthly["qty"]
        pivot = monthly["uv"].unstack()
        cols = [c for c in ["Adversary", "Intermediary"] if c in pivot.columns]
        if len(cols) < 2:
            continue
        pivot = pivot[cols].rolling(3, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        for c in cols:
            ax.plot(pivot.index, pivot[c], label=c, lw=2.5)
        _date_axis(ax)
        ax.axvline(EVENTS[broad], color="black", ls=":")
        ax.set_title(f"HS6 Divergence: {code} ({subgroup})")
        fig.savefig(OUT / f"divergence_hs6_{code}.png")
        plt.close(fig)


def main():
    df = _load_data()
    if not df.empty:
        plot_divergence_overview(df)
        plot_event_study(df)
        plot_peer_comparison(df)
        plot_drilldowns(df)
        print(f"Done. Figures saved to {OUT}")
    else:
        print("No data found.")


if __name__ == "__main__":
    main()
