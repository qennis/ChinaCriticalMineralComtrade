#!/usr/bin/env python3
"""
scripts/make_strategic_analysis.py
Advanced Strategic Analysis of Export Controls.

Features:
- Stockpiling: Ratio with Propagated Error Bars.
- Trade Deflection: Net Volume Change with Standard Error Bars.
- Mass Outliers: Absolute kg shifts with Standard Error Bars.
- Price Outliers: Absolute ($) and Relative (%) shifts with Error Propagation.
- Friendship Premium: Weighted Average Price with Weighted IQR Bands.
- Visuals: Error bars on TOP (zorder=5), soft grid lines.
"""
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
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
OUT = Path("figures/strategic")
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,  # Soft grid lines
        "grid.color": "#cccccc",
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


# --- HELPERS ---
def _to_dt(yyyymm: pd.Series) -> pd.Series:
    return pd.to_datetime(yyyymm.astype(str), format="%Y%m")


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3, ls="--")


def _weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def _load_data():
    files = sorted(DATA.glob("partners_granular_*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["period"] = df["period"].astype(str)
    df["partnerCode"] = pd.to_numeric(df["partnerCode"], errors="coerce").fillna(-1).astype(int)
    if "flowCode" in df.columns:
        df = df[df["flowCode"] == "X"]
    if "reporterCode" in df.columns:
        df = df[df["reporterCode"] == 156]
    df["hs6"] = df["cmdCode"].astype(str).str[:6]
    df["subgroup"] = df["hs6"].map(HS6_TO_SUBGROUP)
    df["broad_group"] = df["subgroup"].map(SUBGROUP_TO_BROAD)
    df = df.dropna(subset=["broad_group"])
    df["partner_name"] = df["partnerCode"].map(CODE_TO_NAME).fillna("Unknown")
    bloc_map = {p: b for b, partners in BLOCS.items() for p in partners}
    df["bloc"] = df["partner_name"].map(bloc_map).fillna("Other")
    df["period_dt"] = _to_dt(df["period"])
    df["qty"] = np.nan
    for c in ["primaryQuantity", "netWgt", "qty"]:
        if c in df.columns:
            df["qty"] = pd.to_numeric(df[c], errors="coerce")
            break
    df["val"] = pd.to_numeric(df["primaryValue"], errors="coerce").fillna(0)
    return df


# --- 1. Stockpiling Analysis (Ratio Error) ---
def analyze_stockpiling(df):
    print("Running Stockpiling Analysis...")
    for g, ban_date in EVENTS.items():
        sub = df[df["broad_group"] == g].copy()
        if sub.empty:
            continue

        stock_start = ban_date - pd.DateOffset(months=3)
        stock_end = ban_date
        base_start = stock_start - pd.DateOffset(months=12)
        base_end = stock_start

        monthly = sub.groupby(["partner_name", "period_dt"])["qty"].sum().reset_index()

        stock_data = monthly[
            (monthly["period_dt"] >= stock_start) & (monthly["period_dt"] < stock_end)
        ]
        base_data = monthly[
            (monthly["period_dt"] >= base_start) & (monthly["period_dt"] < base_end)
        ]

        if stock_data.empty or base_data.empty:
            continue

        stock_stats = stock_data.groupby("partner_name")["qty"].agg(["mean", "sem"])
        base_stats = base_data.groupby("partner_name")["qty"].agg(["mean", "sem"])

        stats = pd.merge(
            stock_stats, base_stats, left_index=True, right_index=True, suffixes=("_stock", "_base")
        )

        # Ratio & Error Propagation
        stats["ratio"] = stats["mean_stock"] / stats["mean_base"]
        stats["ratio_err"] = stats["ratio"] * np.sqrt(
            (stats["sem_stock"] / stats["mean_stock"]) ** 2
            + (stats["sem_base"] / stats["mean_base"]) ** 2
        )

        top_vol = stats["mean_base"].sort_values(ascending=False).head(30).index
        stats = stats.loc[stats.index.intersection(top_vol)].sort_values("ratio", ascending=False)

        top_10 = stats.head(10)
        if top_10.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#e74c3c" if x >= 1.5 else "#95a5a6" for x in top_10["ratio"]]

        # ERROR BARS: zorder=5 to be visible on top
        bars = ax.barh(
            top_10.index,
            top_10["ratio"],
            xerr=top_10["ratio_err"],
            color=colors,
            capsize=4,
            ecolor="black",
            error_kw={"alpha": 0.5, "lw": 1.5, "zorder": 5},
        )
        ax.invert_yaxis()

        ax.set_title(f"{g}: Anticipatory Stockpiling (3mo Pre-Ban vs Baseline)", loc="left")
        ax.set_xlabel("Stockpiling Ratio (1.0 = Normal)")
        ax.axvline(1.0, color="black", linestyle="--")

        x_max = ax.get_xlim()[1]
        ax.set_xlim(0, x_max * 1.15)

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (x_max * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}x",
                va="center",
                fontsize=9,
                color="#555555",
                zorder=10,
            )

        fname = g.replace("/", "_").lower()
        fig.savefig(OUT / f"stockpiling_{fname}.png")
        plt.close(fig)


# --- 2. Trade Deflection (Net Change Error) ---
def analyze_trade_deflection(df):
    print("Running Trade Deflection Analysis...")
    for g, ban_date in EVENTS.items():
        sub = df[df["broad_group"] == g].copy()
        if sub.empty:
            continue

        pre_start = ban_date - pd.DateOffset(months=6)
        post_end = ban_date + pd.DateOffset(months=6)

        monthly = sub.groupby(["partner_name", "period_dt"])["qty"].sum().reset_index()

        pre = (
            monthly[(monthly["period_dt"] >= pre_start) & (monthly["period_dt"] < ban_date)]
            .groupby("partner_name")["qty"]
            .agg(["mean", "sem"])
        )
        post = (
            monthly[(monthly["period_dt"] >= ban_date) & (monthly["period_dt"] < post_end)]
            .groupby("partner_name")["qty"]
            .agg(["mean", "sem"])
        )

        stats = pd.merge(
            pre, post, left_index=True, right_index=True, suffixes=("_pre", "_post")
        ).fillna(0)
        stats["delta"] = stats["mean_post"] - stats["mean_pre"]
        stats["delta_err"] = np.sqrt(stats["sem_pre"] ** 2 + stats["sem_post"] ** 2)

        delta_sorted = stats["delta"].sort_values()
        threshold = max(abs(delta_sorted.max()), abs(delta_sorted.min())) * 0.01
        significant = stats[abs(stats["delta"]) > threshold].sort_values("delta")

        if significant.empty:
            continue
        combined = pd.concat([significant.head(5), significant.tail(5)])

        vals_million = combined["delta"] / 1e6
        errors_million = combined["delta_err"] / 1e6

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#c0392b" if x < 0 else "#27ae60" for x in vals_million]

        bars = ax.barh(
            combined.index,
            vals_million,
            xerr=errors_million,
            color=colors,
            capsize=4,
            ecolor="black",
            error_kw={"alpha": 0.5, "lw": 1.5, "zorder": 5},
        )

        ax.set_title(f"{g}: Trade Deflection (Net Change +/- Std Error)", loc="left")
        ax.set_xlabel("Change in Monthly Exports (Million kg)")
        ax.axvline(0, color="black")

        x_min, x_max = ax.get_xlim()
        pad = (x_max - x_min) * 0.15
        ax.set_xlim(x_min - pad, x_max + pad)
        offset = (x_max - x_min) * 0.02

        for bar in bars:
            width = bar.get_width()
            align = "left" if width > 0 else "right"
            label_x = width + (offset if width > 0 else -offset)
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{width:+.2f}M",
                va="center",
                ha=align,
                fontsize=9,
                fontweight="bold",
                zorder=10,
            )

        fname = g.replace("/", "_").lower()
        fig.savefig(OUT / f"deflection_{fname}.png")
        plt.close(fig)


# --- 3A. Mass Outliers (Absolute kg) ---
def analyze_hs6_outliers(df):
    print("Running Mass Outlier Analysis (Absolute kg)...")
    for g, ban_date in EVENTS.items():
        sub = df[df["broad_group"] == g].copy()
        if sub.empty:
            continue

        pre_start = ban_date - pd.DateOffset(months=6)
        post_end = ban_date + pd.DateOffset(months=6)

        monthly = sub.groupby(["partner_name", "subgroup", "period_dt"])["qty"].sum().reset_index()

        pre = (
            monthly[(monthly["period_dt"] >= pre_start) & (monthly["period_dt"] < ban_date)]
            .groupby(["partner_name", "subgroup"])["qty"]
            .agg(["mean", "sem"])
        )
        post = (
            monthly[(monthly["period_dt"] >= ban_date) & (monthly["period_dt"] < post_end)]
            .groupby(["partner_name", "subgroup"])["qty"]
            .agg(["mean", "sem"])
        )

        stats = pd.merge(
            pre, post, left_index=True, right_index=True, suffixes=("_pre", "_post")
        ).fillna(0)
        stats["delta"] = stats["mean_post"] - stats["mean_pre"]
        stats["delta_err"] = np.sqrt(stats["sem_pre"] ** 2 + stats["sem_post"] ** 2)

        # Top 12 Movers
        delta_sorted = stats.sort_values("delta")
        top_movers = pd.concat([delta_sorted.head(6), delta_sorted.tail(6)])
        if top_movers.empty:
            continue

        vals_million = top_movers["delta"] / 1e6
        errs_million = top_movers["delta_err"] / 1e6
        labels = [f"{p} - {s}" for p, s in top_movers.index]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#c0392b" if x < 0 else "#27ae60" for x in vals_million]

        bars = ax.barh(
            labels,
            vals_million,
            xerr=errs_million,
            color=colors,
            capsize=4,
            ecolor="black",
            error_kw={"alpha": 0.5, "lw": 1.5, "zorder": 5},
        )

        ax.set_title(f"{g}: Biggest Movers (Mass Volume +/- SE)", loc="left")
        ax.set_xlabel("Net Change (Million kg)")
        ax.axvline(0, color="black")

        x_min, x_max = ax.get_xlim()
        pad = (x_max - x_min) * 0.15
        ax.set_xlim(x_min - pad, x_max + pad)
        offset = (x_max - x_min) * 0.02

        for bar in bars:
            width = bar.get_width()
            align = "left" if width > 0 else "right"
            label_x = width + (offset if width > 0 else -offset)
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{width:+.2f}M",
                va="center",
                ha=align,
                fontsize=9,
                zorder=10,
            )

        fname = g.replace("/", "_").lower()
        fig.savefig(OUT / f"outliers_mass_{fname}.png")
        plt.close(fig)


# --- 3B. Percentage Outliers (Relative %) ---
def analyze_hs6_outliers_pct(df):
    print("Running Percentage Outlier Analysis...")
    for g, ban_date in EVENTS.items():
        sub = df[df["broad_group"] == g].copy()
        if sub.empty:
            continue

        pre_start = ban_date - pd.DateOffset(months=6)
        post_end = ban_date + pd.DateOffset(months=6)

        monthly = sub.groupby(["partner_name", "subgroup", "period_dt"])["qty"].sum().reset_index()

        pre = (
            monthly[(monthly["period_dt"] >= pre_start) & (monthly["period_dt"] < ban_date)]
            .groupby(["partner_name", "subgroup"])["qty"]
            .agg(["mean", "sem"])
        )
        post = (
            monthly[(monthly["period_dt"] >= ban_date) & (monthly["period_dt"] < post_end)]
            .groupby(["partner_name", "subgroup"])["qty"]
            .agg(["mean", "sem"])
        )

        # Filter baseline > 1000kg
        pre = pre[pre["mean"] > 1000]

        stats = pd.merge(
            pre, post, left_index=True, right_index=True, suffixes=("_pre", "_post")
        ).dropna()

        # Pct Change & Error
        stats["pct"] = ((stats["mean_post"] - stats["mean_pre"]) / stats["mean_pre"]) * 100
        stats["pct_err"] = (
            (stats["mean_post"] / stats["mean_pre"])
            * np.sqrt(
                (stats["sem_pre"] / stats["mean_pre"]) ** 2
                + (stats["sem_post"] / stats["mean_post"]) ** 2
            )
            * 100
        )

        top_movers = stats.sort_values("pct", ascending=False).head(12)
        if top_movers.empty:
            continue

        labels = [f"{p} - {s}" for p, s in top_movers.index]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#c0392b" if x < 0 else "#27ae60" for x in top_movers["pct"]]

        bars = ax.barh(
            labels,
            top_movers["pct"],
            xerr=top_movers["pct_err"],
            color=colors,
            capsize=4,
            ecolor="black",
            error_kw={"alpha": 0.5, "lw": 1.5, "zorder": 5},
        )

        ax.set_title(
            f"{g}: Biggest Movers (% Change +/- SE)\n(Baseline Volume > 1000kg)", loc="left"
        )
        ax.set_xlabel("Change (%)")
        ax.axvline(0, color="black")

        x_min, x_max = ax.get_xlim()
        pad = (x_max - x_min) * 0.15
        ax.set_xlim(x_min - pad, x_max + pad)
        offset = (x_max - x_min) * 0.02

        for bar in bars:
            width = bar.get_width()
            label_x = width + offset
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{width:+.0f}%",
                va="center",
                fontsize=9,
                fontweight="bold",
                zorder=10,
            )

        fname = g.replace("/", "_").lower()
        fig.savefig(OUT / f"outliers_pct_{fname}.png")
        plt.close(fig)


# --- 4. Price Analysis (Absolute Error) ---
def analyze_price_outliers(df):
    print("Running Price Outlier Analysis (Absolute $)...")
    for g, ban_date in EVENTS.items():
        sub = df[df["broad_group"] == g].copy()
        if sub.empty:
            continue

        pre_start = ban_date - pd.DateOffset(months=6)
        post_end = ban_date + pd.DateOffset(months=6)
        sub = sub[sub["qty"] > 100]

        monthly = (
            sub.groupby(["partner_name", "subgroup", "period_dt"])
            .agg({"val": "sum", "qty": "sum"})
            .reset_index()
        )
        monthly["uv"] = monthly["val"] / monthly["qty"]

        pre = (
            monthly[(monthly["period_dt"] >= pre_start) & (monthly["period_dt"] < ban_date)]
            .groupby(["partner_name", "subgroup"])["uv"]
            .agg(["mean", "sem"])
        )
        post = (
            monthly[(monthly["period_dt"] >= ban_date) & (monthly["period_dt"] < post_end)]
            .groupby(["partner_name", "subgroup"])["uv"]
            .agg(["mean", "sem"])
        )

        stats = pd.merge(
            pre, post, left_index=True, right_index=True, suffixes=("_pre", "_post")
        ).dropna()
        stats["delta"] = stats["mean_post"] - stats["mean_pre"]
        stats["delta_err"] = np.sqrt(stats["sem_pre"] ** 2 + stats["sem_post"] ** 2)

        delta_sorted = stats.sort_values("delta")
        top_movers = pd.concat([delta_sorted.head(5), delta_sorted.tail(5)])
        if top_movers.empty:
            continue

        labels = [f"{p} - {s}" for p, s in top_movers.index]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#27ae60" if x < 0 else "#c0392b" for x in top_movers["delta"]]

        bars = ax.barh(
            labels,
            top_movers["delta"],
            xerr=top_movers["delta_err"],
            color=colors,
            capsize=4,
            ecolor="black",
            error_kw={"alpha": 0.5, "lw": 1.5, "zorder": 5},
        )

        ax.set_title(f"{g}: Biggest Price Shifts ($/kg +/- SE)", loc="left")
        ax.set_xlabel("Change in Unit Price ($/kg)")
        ax.axvline(0, color="black")

        x_min, x_max = ax.get_xlim()
        pad = (x_max - x_min) * 0.15
        ax.set_xlim(x_min - pad, x_max + pad)
        offset = (x_max - x_min) * 0.02

        for bar in bars:
            width = bar.get_width()
            align = "left" if width > 0 else "right"
            label_x = width + (offset if width > 0 else -offset)
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{width:+.2f}",
                va="center",
                ha=align,
                fontsize=9,
                zorder=10,
            )

        fname = g.replace("/", "_").lower()
        fig.savefig(OUT / f"outliers_price_abs_{fname}.png")
        plt.close(fig)


# --- 5. Percentage Price Outliers (Propagated Error) ---
def analyze_price_outliers_pct(df):
    print("Running Price Outlier Analysis (%)...")
    for g, ban_date in EVENTS.items():
        sub = df[df["broad_group"] == g].copy()
        if sub.empty:
            continue

        pre_start = ban_date - pd.DateOffset(months=6)
        post_end = ban_date + pd.DateOffset(months=6)
        sub = sub[sub["qty"] > 100]
        monthly = (
            sub.groupby(["partner_name", "subgroup", "period_dt"])
            .agg({"val": "sum", "qty": "sum"})
            .reset_index()
        )
        monthly["uv"] = monthly["val"] / monthly["qty"]

        pre = (
            monthly[(monthly["period_dt"] >= pre_start) & (monthly["period_dt"] < ban_date)]
            .groupby(["partner_name", "subgroup"])["uv"]
            .agg(["mean", "sem"])
        )
        post = (
            monthly[(monthly["period_dt"] >= ban_date) & (monthly["period_dt"] < post_end)]
            .groupby(["partner_name", "subgroup"])["uv"]
            .agg(["mean", "sem"])
        )

        pre = pre[pre["mean"] > 1]  # Filter baseline > $1

        stats = pd.merge(
            pre, post, left_index=True, right_index=True, suffixes=("_pre", "_post")
        ).dropna()
        stats["pct"] = ((stats["mean_post"] - stats["mean_pre"]) / stats["mean_pre"]) * 100
        stats["pct_err"] = (
            (stats["mean_post"] / stats["mean_pre"])
            * np.sqrt(
                (stats["sem_pre"] / stats["mean_pre"]) ** 2
                + (stats["sem_post"] / stats["mean_post"]) ** 2
            )
            * 100
        )

        top_movers = stats.sort_values("pct", ascending=False).head(10)
        if top_movers.empty:
            continue

        labels = [f"{p} - {s}" for p, s in top_movers.index]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#c0392b" if x > 0 else "#27ae60" for x in top_movers["pct"]]

        bars = ax.barh(
            labels,
            top_movers["pct"],
            xerr=top_movers["pct_err"],
            color=colors,
            capsize=4,
            ecolor="black",
            error_kw={"alpha": 0.5, "lw": 1.5, "zorder": 5},
        )

        ax.set_title(f"{g}: Biggest Relative Price Hikes (% +/- SE)", loc="left")
        ax.set_xlabel("Price Change (%)")
        ax.axvline(0, color="black")

        x_min, x_max = ax.get_xlim()
        pad = (x_max - x_min) * 0.15
        ax.set_xlim(x_min - pad, x_max + pad)
        offset = (x_max - x_min) * 0.02

        for bar in bars:
            width = bar.get_width()
            label_x = width + offset
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"+{width:.0f}%",
                va="center",
                fontsize=9,
                fontweight="bold",
                zorder=10,
            )

        fname = g.replace("/", "_").lower()
        fig.savefig(OUT / f"outliers_price_pct_{fname}.png")
        plt.close(fig)


# --- 6. Friendship Premium (Weighted IQR) ---
def analyze_friendship_premium(df):
    print("Running Friendship Premium Analysis...")

    for g, ban_date in EVENTS.items():
        sub = df[df["broad_group"] == g].copy()
        if sub["qty"].isna().all():
            continue
        sub = sub[sub["qty"] > 10]

        monthly = (
            sub.groupby(["period_dt", "bloc", "partner_name"])
            .agg({"val": "sum", "qty": "sum"})
            .reset_index()
        )
        monthly["uv"] = monthly["val"] / monthly["qty"]

        results = []
        for (date, bloc), group in monthly.groupby(["period_dt", "bloc"]):
            if group.empty:
                continue
            w_mean = np.average(group["uv"], weights=group["qty"])
            try:
                w_q25 = _weighted_quantile(group["uv"], 0.25, sample_weight=group["qty"])
                w_q75 = _weighted_quantile(group["uv"], 0.75, sample_weight=group["qty"])
            except Exception:
                w_q25 = w_mean
                w_q75 = w_mean
            results.append(
                {"period_dt": date, "bloc": bloc, "mean": w_mean, "q25": w_q25, "q75": w_q75}
            )

        if not results:
            continue
        stats = pd.DataFrame(results).set_index("period_dt")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"Adversary": "#c0392b", "Intermediary": "#27ae60"}

        for bloc in ["Adversary", "Intermediary"]:
            bloc_data = stats[stats["bloc"] == bloc].sort_index()
            if bloc_data.empty:
                continue
            smooth_mean = bloc_data["mean"].rolling(3, min_periods=1).mean()
            smooth_q25 = bloc_data["q25"].rolling(3, min_periods=1).mean()
            smooth_q75 = bloc_data["q75"].rolling(3, min_periods=1).mean()

            ax.plot(smooth_mean.index, smooth_mean, color=colors[bloc], lw=2.5, label=bloc)
            ax.fill_between(
                smooth_mean.index,
                smooth_q25,
                smooth_q75,
                color=colors[bloc],
                alpha=0.15,
                label=f"{bloc} Volume-Weighted IQR",
            )

        ax.axvline(ban_date, color="black", linestyle="--", label="Control Effective")
        ax.set_title(f"{g}: The 'Friendship Premium' (Volume-Weighted Divergence)", loc="left")
        ax.set_ylabel("Unit Value (USD/kg)")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
        _date_axis(ax)
        ax.legend(loc="upper left")

        fname = g.replace("/", "_").lower()
        fig.savefig(OUT / f"premium_{fname}.png")
        plt.close(fig)


def main():
    df = _load_data()
    if not df.empty:
        analyze_stockpiling(df)
        analyze_trade_deflection(df)
        analyze_hs6_outliers(df)
        analyze_hs6_outliers_pct(df)
        analyze_price_outliers(df)
        analyze_price_outliers_pct(df)
        analyze_friendship_premium(df)
        print(f"Done. Figures saved to {OUT}")
    else:
        print("No data found.")


if __name__ == "__main__":
    main()
