#!/usr/bin/env python3
"""
scripts/make_figures.py — Comprehensive Visual Analysis.

Refactored to calculate ALL metrics on-the-fly from granular data.
Includes:
- Overview: Stacked Value, YoY Trends, Growth vs Size Matrix.
- Market: Price/Qty Panels with Volatility Bands.
- Concentration: Aggregate HHI, Component HHI, Effective Counts.
- Small Multiples: Destination HHI (Geo Risk) AND HS6 HHI (Product Risk).
- Composition: Within-group Subgroup analysis.
- Geopolitics: Bloc Quantity Trends.
- Regimes: Volatility vs Elasticity (Sector & HS6 levels).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# Setup paths
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- CONFIGURATION ---
DATA = Path("data_work")
OUT = Path("figures")

# Define Subfolders
DIRS = {
    "overview": OUT / "overview",
    "market": OUT / "market_dynamics",
    "geopolitics": OUT / "geopolitics",
    "composition": OUT / "composition",
    "concentration": OUT / "concentration",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#cccccc",
    }
)

pd.plotting.register_matplotlib_converters()

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
    "Bonded": ["China", "China (Re-import)"],
    "Unspecified": ["Areas NES", "Other Asia NES", "Unspecified", "Africa NES"],
}


# --- HELPERS ---
def _to_dt(yyyymm: pd.Series) -> pd.Series:
    return pd.to_datetime(yyyymm.astype(str), format="%Y%m")


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)


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

    # Map Partner Names
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


def _get_monthly_broad(df):
    return df.pivot_table(
        index="period_dt", columns="broad_group", values="val", aggfunc="sum"
    ).fillna(0)


# =============================================================================
# 1. OVERVIEW FIGURES
# =============================================================================


def plot_broad_overview(df):
    wide = _get_monthly_broad(df)
    cols = wide.sum().sort_values(ascending=False).index
    wide = wide[cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(wide.index, wide.T / 1e6, labels=wide.columns, alpha=0.85, edgecolor="white")
    ax.set_title("Total Export Value by Sector (Aggregated)")
    ax.set_ylabel("Value (Million USD)")
    _date_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="Sector")
    fig.savefig(DIRS["overview"] / "broad_sector_overview.png")
    plt.close(fig)


def plot_yoy_trends(df):
    wide = _get_monthly_broad(df)
    yoy = wide.pct_change(12) * 100.0
    yoy = yoy.rolling(3).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    for c in wide.columns:
        ax.plot(yoy.index, yoy[c], label=c, lw=2)

    _date_axis(ax)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_title("Year-over-Year Growth (3-Month Rolling)")
    ax.set_ylabel("YoY Change (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.savefig(DIRS["overview"] / "yoy_growth_trends.png")
    plt.close(fig)


def plot_growth_vs_size_scatter(df):
    wide = _get_monthly_broad(df)
    stats = []

    for g in wide.columns:
        s = wide[g]
        if len(s) < 12:
            continue
        start_val = s.iloc[:3].mean()
        end_val = s.iloc[-3:].mean()
        if start_val <= 0 or end_val <= 0:
            continue
        years = (s.index[-1] - s.index[0]).days / 365.25
        cagr = (end_val / start_val) ** (1 / years) - 1

        stats.append({"group": g, "cagr": cagr * 100, "size": np.log10(end_val * 12)})

    if not stats:
        return
    d = pd.DataFrame(stats)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d["size"], d["cagr"], s=100, alpha=0.7)

    for _, r in d.iterrows():
        ax.text(r["size"], r["cagr"] + 0.5, r["group"], ha="center")

    ax.set_xlabel("Log10 Annualized Export Value")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Growth Matrix: High Growth vs. High Volume")
    ax.axhline(0, color="gray", ls="--")
    fig.savefig(DIRS["overview"] / "growth_vs_size_scatter.png")
    plt.close(fig)


# =============================================================================
# 2. MARKET DYNAMICS
# =============================================================================


def plot_market_dynamics(df):
    if df["qty"].isna().all():
        return
    df_clean = df[df["qty"] > 0].copy()

    groups = df_clean["broad_group"].unique()
    for g in groups:
        sub = (
            df_clean[df_clean["broad_group"] == g]
            .groupby("period_dt")
            .agg(val=("val", "sum"), qty=("qty", "sum"))
            .reset_index()
        )

        sub["uv"] = sub["val"] / sub["qty"]
        sub["uv_smooth"] = sub["uv"].rolling(3, min_periods=1).mean()
        sub["uv_std"] = sub["uv"].rolling(3, min_periods=1).std()

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(
            sub["period_dt"], sub["val"] / 1e6, color="#bdc3c7", alpha=0.5, width=25, label="Value"
        )
        ax1.set_ylabel("Value (M USD)", color="#7f8c8d")

        ax2 = ax1.twinx()
        ax2.fill_between(
            sub["period_dt"],
            sub["uv_smooth"] - sub["uv_std"],
            sub["uv_smooth"] + sub["uv_std"],
            color="#e74c3c",
            alpha=0.15,
            zorder=0,
            label="Price Volatility (±1 SD)",
        )
        ax2.plot(sub["period_dt"], sub["uv_smooth"], color="#e74c3c", lw=2.5, label="Unit Price")
        ax2.set_ylabel("Unit Price ($/kg)", color="#c0392b")

        ax1.set_title(f"{g}: Market Dynamics (Price Volatility & Volume)")
        _date_axis(ax1)
        if g in EVENTS:
            ax1.axvline(EVENTS[g], color="black", ls=":", lw=2)

        fig.savefig(DIRS["market"] / f"market_{g.replace('/','_').lower()}.png")
        plt.close(fig)


def plot_price_regimes_scatter(df):
    """
    Elasticity Proxy with 2D Error Bars (Aggregated by Broad Group).
    Log Y-Axis.
    """
    if df["qty"].isna().all():
        return
    df_clean = df[df["qty"] > 0].copy()

    stats = []
    groups = df_clean["broad_group"].unique()

    for g in groups:
        for bloc in ["Adversary", "Intermediary"]:
            sub = df_clean[(df_clean["broad_group"] == g) & (df_clean["bloc"] == bloc)]
            if sub.empty:
                continue

            hs6_metrics = []
            for code, code_df in sub.groupby("hs6"):
                monthly = code_df.groupby("period_dt").agg(val=("val", "sum"), qty=("qty", "sum"))
                monthly["uv"] = monthly["val"] / monthly["qty"]
                monthly = monthly.dropna()
                if len(monthly) < 6:
                    continue

                vol = monthly["uv"].pct_change().std()
                corr = np.log(monthly["qty"]).corr(np.log(monthly["uv"]))

                if not np.isnan(vol) and not np.isnan(corr):
                    hs6_metrics.append({"vol": vol, "corr": corr})

            if not hs6_metrics:
                continue
            m_df = pd.DataFrame(hs6_metrics)

            stats.append(
                {
                    "group": g,
                    "bloc": bloc,
                    "vol_mean": m_df["vol"].mean(),
                    "vol_err": m_df["vol"].sem(),
                    "corr_mean": m_df["corr"].mean(),
                    "corr_err": m_df["corr"].sem(),
                }
            )

    if not stats:
        return
    d = pd.DataFrame(stats)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"Adversary": "#c0392b", "Intermediary": "#27ae60"}
    markers = {"Gallium/Germanium": "o", "Graphite": "s", "Rare Earths": "^", "Polysilicon": "D"}

    for _, r in d.iterrows():
        c = colors.get(r["bloc"], "gray")
        m = markers.get(r["group"], "o")

        ax.errorbar(
            r["corr_mean"],
            r["vol_mean"],
            xerr=r["corr_err"],
            yerr=r["vol_err"],
            fmt=m,
            color=c,
            ecolor=c,
            capsize=3,
            elinewidth=1.5,
            alpha=0.8,
            zorder=5,
        )

        label = f"{r['group']}"
        ax.text(r["corr_mean"] + 0.02, r["vol_mean"] * 1.05, label, fontsize=9, color=c)

    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Price-Quantity Correlation (Elasticity Proxy)")
    ax.set_ylabel("Price Volatility (Std Dev of Returns)")
    ax.set_title("Market Regimes by Bloc (Sector Level)")
    ax.set_yscale("log")

    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color=colors["Adversary"], lw=2),
        Line2D([0], [0], color=colors["Intermediary"], lw=2),
    ]
    ax.legend(custom_lines, ["Adversary", "Intermediary"], loc="upper left")

    fig.savefig(DIRS["market"] / "price_regimes_scatter_sector.png")
    plt.close(fig)


def plot_price_regimes_hs6(df):
    """
    Elasticity Proxy (HS6 Level).
    Points = Volume-Weighted Average of Countries.
    Error Bars = Weighted Standard Error of Countries.
    """
    if df["qty"].isna().all():
        return
    df_clean = df[df["qty"] > 0].copy()

    stats = []

    for code, code_df in df_clean.groupby("hs6"):
        subgroup = HS6_TO_SUBGROUP.get(code, code)
        broad = SUBGROUP_TO_BROAD.get(subgroup, "Other")

        for bloc in ["Adversary", "Intermediary"]:
            # Get individual country metrics in this bloc
            sub = code_df[code_df["bloc"] == bloc]
            if sub.empty:
                continue

            country_metrics = []

            for country, c_df in sub.groupby("partner_name"):
                monthly = c_df.groupby("period_dt").agg(val=("val", "sum"), qty=("qty", "sum"))
                monthly["uv"] = monthly["val"] / monthly["qty"]
                monthly = monthly.dropna()

                if len(monthly) < 6:
                    continue

                vol = monthly["uv"].pct_change().std()
                corr = np.log(monthly["qty"]).corr(np.log(monthly["uv"]))
                total_qty = monthly["qty"].sum()

                if not np.isnan(vol) and not np.isnan(corr):
                    country_metrics.append({"vol": vol, "corr": corr, "weight": total_qty})

            if not country_metrics:
                continue

            m_df = pd.DataFrame(country_metrics)

            # Weighted Mean
            w_corr = np.average(m_df["corr"], weights=m_df["weight"])
            w_vol = np.average(m_df["vol"], weights=m_df["weight"])

            # Weighted SEM (Approx)
            # SE = sqrt( Sum(w_i * (x_i - mean)^2) / (V1 - V2/V1) ) ...
            # standard weighted variance formula
            # Simplified: Weighted Std Dev / Sqrt(N)
            n = len(m_df)
            if n > 1:
                cov_vol = np.cov(m_df["vol"], aweights=m_df["weight"])
                sem_vol = np.sqrt(cov_vol / n)
                cov_corr = np.cov(m_df["corr"], aweights=m_df["weight"])
                sem_corr = np.sqrt(cov_corr / n)
            else:
                sem_vol = 0
                sem_corr = 0

            stats.append(
                {
                    "hs6": code,
                    "desc": subgroup,
                    "group": broad,
                    "bloc": bloc,
                    "vol": w_vol,
                    "vol_err": sem_vol,
                    "corr": w_corr,
                    "corr_err": sem_corr,
                }
            )

    if not stats:
        return
    d = pd.DataFrame(stats)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"Adversary": "#c0392b", "Intermediary": "#27ae60"}
    markers = {"Gallium/Germanium": "o", "Graphite": "s", "Rare Earths": "^", "Polysilicon": "D"}

    for _, r in d.iterrows():
        c = colors.get(r["bloc"], "gray")
        m = markers.get(r["group"], "o")

        # Error Bars (Country Variance)
        ax.errorbar(
            r["corr"],
            r["vol"],
            xerr=r["corr_err"],
            yerr=r["vol_err"],
            fmt=m,
            color=c,
            ecolor=c,
            capsize=3,
            elinewidth=1.5,
            alpha=0.8,
            zorder=5,
        )

        # Label all points
        ax.text(r["corr"] + 0.02, r["vol"] * 1.05, r["desc"], fontsize=8, color=c)

    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Price-Quantity Correlation")
    ax.set_ylabel("Price Volatility (Log Scale)")
    ax.set_title("Market Regimes by Bloc (HS6 Level)\n(Points = Volume-Weighted Avg of Countries)")
    ax.set_yscale("log")

    # Custom Legend
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color=colors["Adversary"], lw=2, label="Adversary"),
        Line2D([0], [0], color=colors["Intermediary"], lw=2, label="Intermediary"),
        Line2D([0], [0], marker="o", color="k", ls="", label="Ga/Ge"),
        Line2D([0], [0], marker="s", color="k", ls="", label="Graphite"),
        Line2D([0], [0], marker="^", color="k", ls="", label="Rare Earths"),
    ]
    ax.legend(handles=custom_lines, loc="upper left")

    fig.savefig(DIRS["market"] / "price_regimes_scatter_hs6.png")
    plt.close(fig)


# =============================================================================
# 3. CONCENTRATION (HHI)
# =============================================================================


def _calc_hhi(df, group_cols):
    total = df.groupby("period_dt")["val"].sum()
    comp = df.groupby(group_cols)["val"].sum().reset_index()
    comp = comp.merge(total.rename("total"), on="period_dt")
    comp["share"] = comp["val"] / comp["total"]
    comp["share_sq"] = comp["share"] ** 2
    return comp.groupby("period_dt")["share_sq"].sum()


def plot_aggregate_hhi(df):
    hhi = _calc_hhi(df, ["period_dt", "broad_group"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hhi.index, hhi, lw=2, color="#2c3e50")
    ax.set_title("Aggregate Export Concentration (HHI)")
    _date_axis(ax)
    fig.savefig(DIRS["concentration"] / "hhi_aggregate_monthly.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hhi.index, 1 / hhi, lw=2, color="#e67e22")
    ax.set_title("Effective Number of Active Sectors (1/HHI)")
    _date_axis(ax)
    fig.savefig(DIRS["concentration"] / "effective_groups.png")
    plt.close(fig)


def plot_hhi_components(df):
    total = df.groupby("period_dt")["val"].sum()
    comp = df.groupby(["period_dt", "broad_group"])["val"].sum().reset_index()
    comp = comp.merge(total.rename("total"), on="period_dt")
    comp["contrib"] = (comp["val"] / comp["total"]) ** 2
    wide = comp.pivot(index="period_dt", columns="broad_group", values="contrib").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(wide.index, wide.T, labels=wide.columns, alpha=0.85)
    ax.set_title("Components of Concentration (s² Contribution)")
    _date_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.savefig(DIRS["concentration"] / "hhi_components_stack.png")
    plt.close(fig)


def plot_dest_hhi_small_multiples(df):
    groups = df["broad_group"].unique()
    n = len(groups)
    cols = 2
    rows = (n // cols) + (1 if n % cols else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, g in enumerate(groups):
        sub = df[df["broad_group"] == g]
        monthly_partners = sub.groupby(["period_dt", "partner_name"])["val"].sum().reset_index()
        monthly_total = sub.groupby("period_dt")["val"].sum().reset_index()
        merged = monthly_partners.merge(monthly_total, on="period_dt", suffixes=("", "_tot"))
        merged["sq"] = (merged["val"] / merged["val_tot"]) ** 2
        hhi = merged.groupby("period_dt")["sq"].sum()
        ax = axes[i]
        ax.plot(hhi.index, hhi, color="#2c3e50")
        ax.fill_between(hhi.index, hhi, color="#2c3e50", alpha=0.1)
        ax.set_title(g)
        ax.set_ylim(0, 1.1)
        if g in EVENTS:
            ax.axvline(EVENTS[g], color="red", ls=":", lw=1)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Geographic Concentration (Destination HHI)", y=1.02, fontweight="bold")
    _date_axis(axes[0])
    fig.savefig(DIRS["concentration"] / "small_multiples_dest_hhi.png", bbox_inches="tight")
    plt.close(fig)


def plot_hs6_hhi_small_multiples(df):
    groups = df["broad_group"].unique()
    n = len(groups)
    cols = 2
    rows = (n // cols) + (1 if n % cols else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, g in enumerate(groups):
        sub = df[df["broad_group"] == g]
        monthly_hs6 = sub.groupby(["period_dt", "hs6"])["val"].sum().reset_index()
        monthly_total = sub.groupby("period_dt")["val"].sum().reset_index()
        merged = monthly_hs6.merge(monthly_total, on="period_dt", suffixes=("", "_tot"))
        merged["sq"] = (merged["val"] / merged["val_tot"]) ** 2
        hhi = merged.groupby("period_dt")["sq"].sum()
        ax = axes[i]
        ax.plot(hhi.index, hhi, color="#8e44ad", lw=2)
        ax.fill_between(hhi.index, hhi, color="#8e44ad", alpha=0.1)
        ax.set_title(g)
        ax.set_ylim(0, 1.1)
        if g in EVENTS:
            ax.axvline(EVENTS[g], color="red", ls=":", lw=1)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Product Concentration (Within-Sector HS6 HHI)", y=1.02, fontweight="bold")
    _date_axis(axes[0])
    fig.savefig(DIRS["concentration"] / "small_multiples_hs6_hhi.png", bbox_inches="tight")
    plt.close(fig)


def plot_hs6_diversity(df):
    groups = df["broad_group"].unique()
    fig, ax = plt.subplots(figsize=(10, 6))
    for g in groups:
        sub = df[df["broad_group"] == g]
        monthly_hs6 = sub.groupby(["period_dt", "hs6"])["val"].sum().reset_index()
        monthly_total = sub.groupby("period_dt")["val"].sum().reset_index()
        merged = monthly_hs6.merge(monthly_total, on="period_dt", suffixes=("", "_tot"))
        merged["sq"] = (merged["val"] / merged["val_tot"]) ** 2
        hhi = merged.groupby("period_dt")["sq"].sum()
        eff_n = 1 / hhi
        eff_n = eff_n.rolling(3).mean()
        ax.plot(eff_n.index, eff_n, label=g, lw=2)
    ax.set_title("Product Complexity: Effective HS6 Lines per Sector")
    _date_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.savefig(DIRS["concentration"] / "effective_hs6_lines_annual.png")
    plt.close(fig)


# =============================================================================
# 4. COMPOSITION & GEOPOLITICS
# =============================================================================


def plot_within_group_composition(df):
    broad_targets = ["Gallium/Germanium", "Graphite", "Rare Earths"]
    for bg in broad_targets:
        sub = df[df["broad_group"] == bg]
        if sub.empty:
            continue
        wide = sub.pivot_table(
            index="period_dt", columns="subgroup", values="val", aggfunc="sum"
        ).fillna(0)
        share = wide.div(wide.sum(axis=1), axis=0).fillna(0).rolling(3).mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stackplot(share.index, share.T, labels=share.columns, alpha=0.85)
        if bg in EVENTS:
            ax.axvline(EVENTS[bg], color="black", ls=":", lw=2)
        ax.set_title(f"Internal Composition: {bg}")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        _date_axis(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.savefig(DIRS["composition"] / f"composition_{bg.replace('/','_').lower()}.png")
        plt.close(fig)


def plot_geopolitics(df):
    """
    Bloc Trends: Plots Quantity (kg) per Bloc.
    """
    partner_to_bloc = {}
    for bloc, partners in BLOCS.items():
        for p in partners:
            partner_to_bloc[p] = bloc
    df["bloc"] = df["partner_name"].map(partner_to_bloc).fillna("Other")

    if "qty" not in df.columns:
        return

    groups = df["broad_group"].unique()
    colors = {"Adversary": "#c0392b", "Intermediary": "#27ae60", "Other": "#95a5a6"}

    for grp in groups:
        sub = df[df["broad_group"] == grp].copy()
        if sub.empty:
            continue

        # Aggregate Quantity by Bloc
        monthly = sub.groupby(["period_dt", "bloc"])["qty"].sum().unstack().fillna(0)

        # Smooth
        monthly = monthly.rolling(3, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 6))

        for bloc in ["Adversary", "Intermediary"]:
            if bloc in monthly.columns:
                # Convert to Tonnes for readability
                ax.plot(monthly.index, monthly[bloc] / 1000, color=colors[bloc], lw=3, label=bloc)

        if grp in EVENTS:
            ax.axvline(EVENTS[grp], color="black", ls=":", lw=2)

        ax.set_title(f"{grp}: Strategic Flows (Total Quantity)")
        ax.set_ylabel("Quantity (Metric Tonnes)")

        # Use log scale if orders of magnitude differ
        if monthly.max().max() / (monthly.min().min() + 1) > 100:
            ax.set_yscale("log")

        _date_axis(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

        fig.savefig(DIRS["geopolitics"] / f"bloc_trends_{grp.replace('/','_').lower()}.png")
        plt.close(fig)


def main():
    print("Loading Data...")
    df = _load_data()
    if df.empty:
        print("No data found! Run 'pull_partners.py' first.")
        return

    print(f"Loaded {len(df)} rows. Generating Figures...")

    plot_broad_overview(df)
    plot_yoy_trends(df)
    plot_growth_vs_size_scatter(df)
    plot_market_dynamics(df)
    plot_price_regimes_scatter(df)  # Sector
    plot_price_regimes_hs6(df)  # HS6
    plot_aggregate_hhi(df)
    plot_hhi_components(df)
    plot_dest_hhi_small_multiples(df)
    plot_hs6_hhi_small_multiples(df)
    plot_hs6_diversity(df)
    plot_within_group_composition(df)
    plot_geopolitics(df)

    print("\nDone! Check 'figures/' subfolders.")


if __name__ == "__main__":
    main()
