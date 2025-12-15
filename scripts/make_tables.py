#!/usr/bin/env python3
"""
scripts/make_tables.py
Master Table Generator for China Export Control Analysis.

Aggregates ALL analytical logic from:
- make_figures.py (Overview, Market, Concentration, Composition, Geopolitics)
- make_strategic_analysis.py (Stockpiling, Deflection, Outliers, Premium)
- make_leakage_figures.py (Leakage)
- make_advanced_figures.py (Peer comparison, HS6 divergence)
- make_country_figures.py (Country trajectories)

Outputs CSVs to data_output/tables/ for external analysis.
"""

import sys
from pathlib import Path

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
OUT = Path("data_output/tables")
OUT.mkdir(parents=True, exist_ok=True)

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

PEER_PAIRS = [
    ("Gallium/Germanium", "Rare Earths", "Strategic Peer (Rare Earths)"),
    ("Graphite", "Polysilicon", "Industrial Peer (Polysilicon)"),
]


# --- HELPER FUNCTIONS ---
def _to_dt(yyyymm: pd.Series) -> pd.Series:
    return pd.to_datetime(yyyymm.astype(str), format="%Y%m")


def _weighted_quantile(values, quantiles, sample_weight=None):
    values = np.array(values)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
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

    # Mappings
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


# =============================================================================
# 1. OVERVIEW & GROWTH
# =============================================================================


def calc_overview_monthly(df):
    print("Calculating Overview...")
    agg = (
        df.groupby(["period_dt", "broad_group"])
        .agg(total_value=("val", "sum"), total_qty=("qty", "sum"))
        .reset_index()
    )
    agg["unit_price"] = agg["total_value"] / agg["total_qty"]
    agg.to_csv(OUT / "overview_monthly.csv", index=False)


def calc_growth_metrics(df):
    print("Calculating Growth Metrics...")
    monthly = df.pivot_table(
        index="period_dt", columns="broad_group", values="val", aggfunc="sum"
    ).fillna(0)

    yoy = monthly.pct_change(12) * 100
    yoy_rolling = yoy.rolling(3).mean()
    yoy_out = yoy_rolling.reset_index().melt(
        id_vars="period_dt", var_name="broad_group", value_name="yoy_growth_pct"
    )
    yoy_out.to_csv(OUT / "growth_yoy_monthly.csv", index=False)

    stats = []
    for g in monthly.columns:
        s = monthly[g]
        if len(s) < 12:
            continue
        start_val = s.iloc[:3].mean()
        end_val = s.iloc[-3:].mean()
        if start_val <= 0 or end_val <= 0:
            continue

        years = (s.index[-1] - s.index[0]).days / 365.25
        cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
        stats.append(
            {
                "broad_group": g,
                "cagr_pct": cagr,
                "total_annualized_value": end_val * 12,
                "log10_value": np.log10(end_val * 12),
            }
        )
    pd.DataFrame(stats).to_csv(OUT / "growth_cagr_summary.csv", index=False)


# =============================================================================
# 2. MARKET DYNAMICS
# =============================================================================


def calc_market_regimes(df):
    print("Calculating Market Regimes...")
    if df["qty"].isna().all():
        return
    df = df[df["qty"] > 0].copy()

    # 1. Sector Level (Aggregated by Bloc)
    sector_stats = []
    for g in df["broad_group"].unique():
        for bloc in ["Adversary", "Intermediary"]:
            sub = df[(df["broad_group"] == g) & (df["bloc"] == bloc)]
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

            if hs6_metrics:
                m_df = pd.DataFrame(hs6_metrics)
                sector_stats.append(
                    {
                        "group": g,
                        "bloc": bloc,
                        "volatility_mean": m_df["vol"].mean(),
                        "volatility_se": m_df["vol"].sem(),
                        "elasticity_corr_mean": m_df["corr"].mean(),
                        "elasticity_corr_se": m_df["corr"].sem(),
                    }
                )
    pd.DataFrame(sector_stats).to_csv(OUT / "market_regimes_sector.csv", index=False)

    # 2. HS6 Level (Weighted Avg of Countries)
    hs6_stats = []
    for code, code_df in df.groupby("hs6"):
        subgroup = HS6_TO_SUBGROUP.get(code, code)
        broad = SUBGROUP_TO_BROAD.get(subgroup, "Other")

        for bloc in ["Adversary", "Intermediary"]:
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
                qty_weight = monthly["qty"].sum()

                if not np.isnan(vol) and not np.isnan(corr):
                    country_metrics.append({"vol": vol, "corr": corr, "weight": qty_weight})

            if country_metrics:
                m_df = pd.DataFrame(country_metrics)
                w_vol = np.average(m_df["vol"], weights=m_df["weight"])
                w_corr = np.average(m_df["corr"], weights=m_df["weight"])
                n = len(m_df)
                sem_vol = np.sqrt(np.cov(m_df["vol"], aweights=m_df["weight"]) / n) if n > 1 else 0
                sem_corr = (
                    np.sqrt(np.cov(m_df["corr"], aweights=m_df["weight"]) / n) if n > 1 else 0
                )

                hs6_stats.append(
                    {
                        "hs6": code,
                        "subgroup": subgroup,
                        "broad_group": broad,
                        "bloc": bloc,
                        "volatility_weighted": w_vol,
                        "volatility_se": sem_vol,
                        "correlation_weighted": w_corr,
                        "correlation_se": sem_corr,
                        "num_countries": n,
                    }
                )
    pd.DataFrame(hs6_stats).to_csv(OUT / "market_regimes_hs6.csv", index=False)


# =============================================================================
# 3. CONCENTRATION & COMPOSITION
# =============================================================================


def calc_concentration(df):
    print("Calculating Concentration Indices...")
    # 1. Aggregate
    total = df.groupby("period_dt")["val"].sum()
    comp = df.groupby(["period_dt", "broad_group"])["val"].sum().reset_index()
    comp = comp.merge(total.rename("total"), on="period_dt")
    comp["share_sq"] = (comp["val"] / comp["total"]) ** 2
    agg_hhi = comp.groupby("period_dt")["share_sq"].sum().reset_index()
    agg_hhi.columns = ["period_dt", "aggregate_hhi"]
    agg_hhi.to_csv(OUT / "concentration_aggregate_hhi.csv", index=False)

    # 2. Destination
    dest_rows = []
    for g in df["broad_group"].unique():
        sub = df[df["broad_group"] == g]
        monthly_partners = sub.groupby(["period_dt", "partner_name"])["val"].sum().reset_index()
        monthly_total = sub.groupby("period_dt")["val"].sum().reset_index()
        merged = monthly_partners.merge(monthly_total, on="period_dt", suffixes=("", "_tot"))
        merged["sq"] = (merged["val"] / merged["val_tot"]) ** 2
        hhi = merged.groupby("period_dt")["sq"].sum().reset_index()
        hhi["broad_group"] = g
        dest_rows.append(hhi)
    pd.concat(dest_rows).to_csv(OUT / "concentration_destination_hhi.csv", index=False)

    # 3. Product HS6
    hs6_rows = []
    for g in df["broad_group"].unique():
        sub = df[df["broad_group"] == g]
        monthly_hs6 = sub.groupby(["period_dt", "hs6"])["val"].sum().reset_index()
        monthly_total = sub.groupby("period_dt")["val"].sum().reset_index()
        merged = monthly_hs6.merge(monthly_total, on="period_dt", suffixes=("", "_tot"))
        merged["sq"] = (merged["val"] / merged["val_tot"]) ** 2
        hhi = merged.groupby("period_dt")["sq"].sum().reset_index()
        hhi["broad_group"] = g
        hs6_rows.append(hhi)
    pd.concat(hs6_rows).to_csv(OUT / "concentration_product_hhi.csv", index=False)


def calc_composition(df):
    print("Calculating Subgroup Composition...")
    rows = []
    for g in df["broad_group"].unique():
        sub = df[df["broad_group"] == g]
        monthly = sub.groupby(["period_dt", "subgroup"])["val"].sum().reset_index()
        monthly_total = sub.groupby("period_dt")["val"].sum().reset_index()
        merged = monthly.merge(monthly_total, on="period_dt", suffixes=("", "_tot"))
        merged["share"] = merged["val"] / merged["val_tot"]
        merged["broad_group"] = g
        rows.append(merged)
    if rows:
        pd.concat(rows).to_csv(OUT / "composition_subgroup.csv", index=False)


def calc_geopolitics_qty(df):
    print("Calculating Geopolitics (Quantity)...")
    if "qty" not in df.columns:
        return
    agg = df.groupby(["period_dt", "broad_group", "bloc"])["qty"].sum().reset_index()
    agg.to_csv(OUT / "geopolitics_qty.csv", index=False)


# =============================================================================
# 4. STRATEGIC ANALYSIS
# =============================================================================


def calc_stockpiling(df):
    print("Calculating Stockpiling...")
    results = []
    for g, ban_date in EVENTS.items():
        sub = df[df["broad_group"] == g].copy()
        if sub.empty:
            continue

        stock_start = ban_date - pd.DateOffset(months=3)
        base_start = stock_start - pd.DateOffset(months=12)

        monthly = sub.groupby(["partner_name", "period_dt"])["qty"].sum().reset_index()
        stock_data = monthly[
            (monthly["period_dt"] >= stock_start) & (monthly["period_dt"] < ban_date)
        ]
        base_data = monthly[
            (monthly["period_dt"] >= base_start) & (monthly["period_dt"] < stock_start)
        ]

        stock_stats = stock_data.groupby("partner_name")["qty"].agg(["mean", "sem"])
        base_stats = base_data.groupby("partner_name")["qty"].agg(["mean", "sem"])

        stats = pd.merge(
            stock_stats, base_stats, left_index=True, right_index=True, suffixes=("_stock", "_base")
        )
        stats["ratio"] = stats["mean_stock"] / stats["mean_base"]
        stats["ratio_error"] = stats["ratio"] * np.sqrt(
            (stats["sem_stock"] / stats["mean_stock"]) ** 2
            + (stats["sem_base"] / stats["mean_base"]) ** 2
        )
        stats = stats.reset_index()
        stats["broad_group"] = g
        results.append(stats)
    if results:
        pd.concat(results).to_csv(OUT / "strategic_stockpiling.csv", index=False)


def calc_deflection_and_outliers(df):
    print("Calculating Deflection & Outliers (Price & Qty)...")
    deflection_rows = []
    qty_outliers = []
    price_outliers = []

    for g, ban_date in EVENTS.items():
        sub = df[df["broad_group"] == g].copy()
        if sub.empty:
            continue

        pre_start = ban_date - pd.DateOffset(months=6)
        post_end = ban_date + pd.DateOffset(months=6)

        # 1. Partner Deflection (Qty)
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
        stats["net_change_kg"] = stats["mean_post"] - stats["mean_pre"]
        stats["net_change_se"] = np.sqrt(stats["sem_pre"] ** 2 + stats["sem_post"] ** 2)
        stats = stats.reset_index()
        stats["broad_group"] = g
        deflection_rows.append(stats)

        # 2. HS6 Outliers (Qty)
        monthly_sub = (
            sub.groupby(["partner_name", "subgroup", "period_dt"])["qty"].sum().reset_index()
        )
        pre_sub = (
            monthly_sub[
                (monthly_sub["period_dt"] >= pre_start) & (monthly_sub["period_dt"] < ban_date)
            ]
            .groupby(["partner_name", "subgroup"])["qty"]
            .agg(["mean", "sem"])
        )
        post_sub = (
            monthly_sub[
                (monthly_sub["period_dt"] >= ban_date) & (monthly_sub["period_dt"] < post_end)
            ]
            .groupby(["partner_name", "subgroup"])["qty"]
            .agg(["mean", "sem"])
        )

        diff = pd.merge(
            pre_sub, post_sub, left_index=True, right_index=True, suffixes=("_pre", "_post")
        ).fillna(0)
        diff["net_change_kg"] = diff["mean_post"] - diff["mean_pre"]
        diff["pct_change"] = (diff["net_change_kg"] / diff["mean_pre"]) * 100
        diff = diff.reset_index()
        diff["broad_group"] = g
        qty_outliers.append(diff)

        # 3. Price Outliers
        monthly_price = (
            sub[sub["qty"] > 100]
            .groupby(["partner_name", "subgroup", "period_dt"])
            .agg({"val": "sum", "qty": "sum"})
            .reset_index()
        )
        monthly_price["uv"] = monthly_price["val"] / monthly_price["qty"]

        pre_p = (
            monthly_price[
                (monthly_price["period_dt"] >= pre_start) & (monthly_price["period_dt"] < ban_date)
            ]
            .groupby(["partner_name", "subgroup"])["uv"]
            .agg(["mean", "sem"])
        )
        post_p = (
            monthly_price[
                (monthly_price["period_dt"] >= ban_date) & (monthly_price["period_dt"] < post_end)
            ]
            .groupby(["partner_name", "subgroup"])["uv"]
            .agg(["mean", "sem"])
        )

        diff_p = pd.merge(
            pre_p, post_p, left_index=True, right_index=True, suffixes=("_pre", "_post")
        ).dropna()
        diff_p["abs_change"] = diff_p["mean_post"] - diff_p["mean_pre"]
        diff_p["pct_change"] = (diff_p["abs_change"] / diff_p["mean_pre"]) * 100
        diff_p["se_diff"] = np.sqrt(diff_p["sem_pre"] ** 2 + diff_p["sem_post"] ** 2)
        diff_p = diff_p.reset_index()
        diff_p["broad_group"] = g
        price_outliers.append(diff_p)

    if deflection_rows:
        pd.concat(deflection_rows).to_csv(OUT / "strategic_deflection.csv", index=False)
    if qty_outliers:
        pd.concat(qty_outliers).to_csv(OUT / "strategic_outliers_qty_hs6.csv", index=False)
    if price_outliers:
        pd.concat(price_outliers).to_csv(OUT / "strategic_outliers_price.csv", index=False)


def calc_premium(df):
    """Sector-Level and HS6-Level Price Divergence."""
    print("Calculating Friendship Premium...")
    if df["qty"].isna().all():
        return
    df = df[df["qty"] > 10]

    # 1. Sector Level (Existing)
    sector_results = []
    for g in df["broad_group"].unique():
        sub = df[df["broad_group"] == g]
        monthly = (
            sub.groupby(["period_dt", "bloc", "partner_name"])
            .agg({"val": "sum", "qty": "sum"})
            .reset_index()
        )
        monthly["uv"] = monthly["val"] / monthly["qty"]
        for (date, bloc), group in monthly.groupby(["period_dt", "bloc"]):
            if bloc not in ["Adversary", "Intermediary"]:
                continue
            w_mean = np.average(group["uv"], weights=group["qty"])
            try:
                w_q25 = _weighted_quantile(group["uv"], 0.25, sample_weight=group["qty"])
                w_q75 = _weighted_quantile(group["uv"], 0.75, sample_weight=group["qty"])
            except Exception:
                w_q25 = w_mean
                w_q75 = w_mean
            sector_results.append(
                {
                    "broad_group": g,
                    "period_dt": date,
                    "bloc": bloc,
                    "weighted_price": w_mean,
                    "q25": w_q25,
                    "q75": w_q75,
                }
            )
    pd.DataFrame(sector_results).to_csv(OUT / "strategic_premium_sector.csv", index=False)

    # 2. HS6 Level (New - for advanced divergence plots)
    hs6_results = []
    for code, code_df in df.groupby("hs6"):
        subgroup = HS6_TO_SUBGROUP.get(code, code)
        broad = SUBGROUP_TO_BROAD.get(subgroup, "Other")
        monthly = (
            code_df.groupby(["period_dt", "bloc"]).agg({"val": "sum", "qty": "sum"}).reset_index()
        )
        monthly["uv"] = monthly["val"] / monthly["qty"]
        monthly["hs6"] = code
        monthly["broad_group"] = broad
        hs6_results.append(monthly)
    pd.concat(hs6_results).to_csv(OUT / "strategic_premium_hs6.csv", index=False)


# =============================================================================
# 5. ADVANCED & COUNTRIES & LEAKAGE
# =============================================================================


def calc_peer_comparison(df):
    """Comparison of Treatment vs Control sectors (Indexed)."""
    print("Calculating Peer Comparisons...")
    rows = []
    for treatment, control, label in PEER_PAIRS:
        if treatment not in EVENTS:
            continue
        event_date = EVENTS[treatment]
        start_win = event_date - pd.DateOffset(months=6)

        sub = df[df["broad_group"].isin([treatment, control]) & (df["bloc"] == "Adversary")]
        if sub.empty:
            continue

        monthly = sub.groupby(["period_dt", "broad_group"])["val"].sum().unstack()
        for col in monthly.columns:
            base = monthly.loc[
                (monthly.index >= start_win) & (monthly.index < event_date), col
            ].mean()
            if base > 0:
                idx_series = (monthly[col] / base) * 100
                for date, val in idx_series.items():
                    rows.append(
                        {
                            "treatment_group": treatment,
                            "control_group": control,
                            "group_shown": col,
                            "date": date,
                            "value_index": val,
                        }
                    )
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "peer_comparison.csv", index=False)


def calc_country_series(df):
    """Raw Monthly Time Series for Top Partners (Qty, Price, HHI)."""
    print("Calculating Country Time Series...")
    # Identify top partners per group
    top_partners = []
    for g in df["broad_group"].unique():
        sub = df[df["broad_group"] == g]
        top = sub.groupby("partner_name")["val"].sum().nlargest(8).index.tolist()
        top_partners.extend([(g, p) for p in top])

    rows = []
    for g, p in top_partners:
        sub = df[(df["broad_group"] == g) & (df["partner_name"] == p)]

        # Spec HHI
        monthly_hhi = sub.groupby(["period_dt", "hs6"])["val"].sum().reset_index()
        monthly_total = sub.groupby("period_dt")["val"].sum().reset_index()
        merged = monthly_hhi.merge(monthly_total, on="period_dt")
        merged["sq"] = (merged["val_x"] / merged["val_y"]) ** 2
        hhi_series = merged.groupby("period_dt")["sq"].sum()

        # Qty & Price
        agg = sub.groupby("period_dt").agg({"val": "sum", "qty": "sum"})
        agg["uv"] = agg["val"] / agg["qty"]
        agg["hhi"] = hhi_series

        agg = agg.reset_index()
        agg["broad_group"] = g
        agg["partner_name"] = p
        rows.append(agg)

    if rows:
        pd.concat(rows).to_csv(OUT / "country_monthly_stats.csv", index=False)


def calc_deep_dive_hs6(df):
    """Detailed stats for Deep Dive HS6 codes."""
    print("Calculating Deep Dive HS6...")
    target_codes = ["811292", "380110", "250410"]
    sub = df[df["hs6"].isin(target_codes)]
    if sub.empty:
        return

    agg = (
        sub.groupby(["period_dt", "partner_name", "hs6"])
        .agg(val=("val", "sum"), qty=("qty", "sum"))
        .reset_index()
    )
    agg["uv"] = agg["val"] / agg["qty"]
    agg.to_csv(OUT / "deep_dive_hs6_monthly.csv", index=False)


def calc_leakage(df):
    print("Calculating Leakage...")
    agg = df.groupby(["period_dt", "broad_group", "bloc"])["val"].sum().reset_index()
    pivot = (
        agg.pivot_table(index=["period_dt", "broad_group"], columns="bloc", values="val")
        .fillna(0)
        .reset_index()
    )
    pivot.to_csv(OUT / "leakage_analysis.csv", index=False)


def calc_event_studies(df):
    print("Calculating Event Studies...")
    rows = []
    for g, ban_date in EVENTS.items():
        sub = df[df["broad_group"] == g].copy()
        if sub.empty:
            continue
        monthly = sub.groupby("period_dt")["val"].sum()
        base_start = ban_date - pd.DateOffset(months=6)
        baseline = monthly[(monthly.index >= base_start) & (monthly.index < ban_date)].mean()
        if baseline == 0:
            continue

        win_start = ban_date - pd.DateOffset(months=12)
        win_end = ban_date + pd.DateOffset(months=12)
        window = monthly[(monthly.index >= win_start) & (monthly.index <= win_end)]

        for date, val in window.items():
            t = (date.year - ban_date.year) * 12 + (date.month - ban_date.month)
            rows.append(
                {
                    "broad_group": g,
                    "event_date": ban_date,
                    "date": date,
                    "t_month": t,
                    "value_index": (val / baseline) * 100,
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "event_studies.csv", index=False)


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("Loading Data...")
    df = _load_data()
    if df.empty:
        print("No data found! Run 'pull_partners.py' first.")
        return

    print(f"Loaded {len(df)} rows. Calculating tables...")

    calc_overview_monthly(df)
    calc_growth_metrics(df)
    calc_market_regimes(df)
    calc_concentration(df)
    calc_composition(df)
    calc_geopolitics_qty(df)
    calc_stockpiling(df)
    calc_deflection_and_outliers(df)
    calc_premium(df)
    calc_peer_comparison(df)
    calc_country_series(df)
    calc_deep_dive_hs6(df)
    calc_leakage(df)
    calc_event_studies(df)

    print(f"\nDone! Tables saved to {OUT}")


if __name__ == "__main__":
    main()
