#!/usr/bin/env python3
"""
make_figures.py — generate paper figures from data_work/*.parquet

Outputs (figures/):
  - monthly_stack.png
  - monthly_share_stack.png
  - hhi_monthly.png
  - hhi_components.png
  - effective_groups.png
  - yoy_by_group.png
  - dest_hhi_small_multiples.png
  - hs6_hhi_small_multiples.png
  - unit_value_<group>.png and quantity_<group>.png   (top groups)
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # repo root (one level up from scripts/)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from china_ir.etl import _to_int64_series, _value_qty_cols, attach_hs_map  # noqa: E402

pd.plotting.register_matplotlib_converters()

DATA = Path("data_work")
OUT = Path("figures")


# ------------- helpers -------------
def _ensure_outdir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def _to_dt(yyyymm: pd.Series) -> pd.Series:
    return pd.to_datetime(yyyymm.astype(str), format="%Y%m")


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.margins(x=0)
    ax.get_figure().autofmt_xdate()


def _monthly_wide() -> pd.DataFrame:
    m = pd.read_parquet(DATA / "materials_monthly.parquet")
    m = m[m["group"].notna() & (m["group"] != "TOTAL")].copy()
    m["period_dt"] = _to_dt(m["period"])
    wide = (
        pd.pivot_table(m, index="period_dt", columns="group", values="value", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )
    return wide


def _top_groups(n: int = 8) -> list[str]:
    m = pd.read_parquet(DATA / "materials_monthly.parquet")
    m = m[m["group"].notna() & (m["group"] != "TOTAL")].copy()
    top = m.groupby("group")["value"].sum().sort_values(ascending=False).head(n).index.tolist()
    return top


# ------------- core figures already validated -------------
def plot_monthly_stack():
    wide = _monthly_wide()
    x = mdates.date2num(wide.index.to_pydatetime())
    y = (wide / 1e9).T.values

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    ax.stackplot(x, y, labels=list(wide.columns))
    ax.set_xlim(x.min(), x.max())
    ax.set_title("Monthly export value by group (USD)")
    ax.set_ylabel("Value (billions, current USD)")
    ax.set_xlabel("Month")
    _date_axis(ax)
    ax.legend(ncol=3, fontsize=8, title="Group")
    fig.tight_layout()
    out = OUT / "monthly_stack.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_monthly_share_stack():
    wide = _monthly_wide()
    share = wide.div(wide.sum(axis=1), axis=0).fillna(0.0)
    x = mdates.date2num(share.index.to_pydatetime())
    y = share.T.values

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    ax.stackplot(x, y, labels=list(share.columns))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(x.min(), x.max())
    ax.set_title("Export shares by material group (monthly)")
    ax.set_ylabel("Share of tracked basket")
    ax.set_xlabel("Month")
    _date_axis(ax)
    ax.legend(ncol=3, fontsize=8, title="Group")
    fig.tight_layout()
    out = OUT / "monthly_share_stack.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_hhi_monthly() -> Path:
    h = pd.read_parquet(DATA / "materials_hhi.parquet").copy()
    h["period_dt"] = _to_dt(h["period"])

    fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
    ax.plot(h["period_dt"], h["hhi"], lw=1.6)
    _date_axis(ax)
    ax.set_title("Monthly market concentration (HHI) across groups")
    ax.set_xlabel("Month")
    ax.set_ylabel("HHI (0–1)")
    ymin = max(0.0, float(h["hhi"].min()) * 0.9)
    ymax = min(1.0, float(h["hhi"].max()) * 1.1)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    out = OUT / "hhi_monthly.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def plot_hhi_components():
    m = _monthly_wide()
    shares = m.div(m.sum(axis=1), axis=0).fillna(0.0)
    x = mdates.date2num(shares.index.to_pydatetime())
    y = shares.pow(2).T.values
    fig, ax = plt.subplots(figsize=(18, 7), dpi=150)
    ax.stackplot(x, y, labels=list(shares.columns))
    _date_axis(ax)
    ax.set_title("HHI components by group (squared shares, monthly)")
    ax.set_ylabel("sᵢ² contribution")
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=3, fontsize=8, title="Group")
    out = OUT / "hhi_components.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_effective_groups():
    h = pd.read_parquet(DATA / "materials_hhi.parquet").copy()
    h["period_dt"] = _to_dt(h["period"])
    fig, ax = plt.subplots(figsize=(18, 5), dpi=150)
    (1.0 / h["hhi"]).plot(ax=ax)
    _date_axis(ax)
    ax.set_title("Effective number of groups (1 / HHI)")
    ax.set_ylabel("Count-equivalent groups")
    out = OUT / "effective_groups.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_yoy_by_group() -> Path:
    m = pd.read_parquet(DATA / "materials_monthly.parquet")
    m = m[m["group"].notna() & (m["group"] != "TOTAL")].copy()
    m["period_dt"] = _to_dt(m["period"])
    wide = (
        pd.pivot_table(m, index="period_dt", columns="group", values="value", aggfunc="sum")
        .sort_index()
        .astype(float)
    )
    yoy = wide.pct_change(12) * 100.0
    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    yoy.plot(ax=ax, linewidth=1.2)
    _date_axis(ax)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_title("Year-over-Year change in export value by group (monthly, %)")
    ax.set_xlabel("Month")
    ax.set_ylabel("YoY (%)")
    ax.legend(ncol=3, fontsize=8, title="Group", frameon=False)
    fig.tight_layout()
    out = OUT / "yoy_by_group.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ------------- new: destination HHI & HS6 HHI -------------
def plot_dest_hhi_small_multiples(k: int = 6) -> None:
    """
    Small-multiples of destination-market HHI for the top k groups.
    Expects data_work/dest_hhi.parquet with columns: period (YYYYMM Int64), group, hhi.
    """
    dpath = DATA / "dest_hhi.parquet"
    if not dpath.exists():
        print(f"skip dest HHI: {dpath} not found")
        return

    d = pd.read_parquet(dpath)
    if d.empty:
        print(f"skip dest HHI: {dpath} is empty")
        return

    top = _top_groups(k)
    d = d[d["group"].isin(top)].copy()
    if d.empty:
        print("skip dest HHI: no rows for top groups")
        return

    d["period_dt"] = _to_dt(d["period"])

    n = len(top)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 4.5 * rows),
        dpi=150,
        sharex=True,
        sharey=True,
    )
    axes = np.ravel(axes)

    for i, g in enumerate(top):
        ax = axes[i]
        sub = d[d["group"] == g].sort_values("period_dt")
        ax.plot(sub["period_dt"], sub["hhi"], lw=1.5)
        _date_axis(ax)
        ax.set_title(g)
        ax.set_ylabel("Destination HHI")

    # drop unused axes if k not a multiple of cols
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    fig.suptitle("Destination concentration by group (HHI across partners)", y=0.98)
    fig.tight_layout()
    out = OUT / "dest_hhi_small_multiples.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_hs6_hhi_small_multiples(k: int = 6) -> None:
    """
    Small-multiples of within-group HS6-line HHI for the top k groups.
    Expects data_work/hs6_hhi.parquet with columns: period (YYYYMM Int64), group, hs6_hhi.
    """
    hpath = DATA / "hs6_hhi.parquet"
    if not hpath.exists():
        print(f"skip HS6 HHI: {hpath} not found")
        return

    d = pd.read_parquet(hpath)
    if d.empty:
        print(f"skip HS6 HHI: {hpath} is empty")
        return

    top = _top_groups(k)
    d = d[d["group"].isin(top)].copy()
    if d.empty:
        print("skip HS6 HHI: no rows for top groups")
        return

    d["period_dt"] = _to_dt(d["period"])

    n = len(top)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 4.5 * rows),
        dpi=150,
        sharex=True,
        sharey=True,
    )
    axes = np.ravel(axes)

    for i, g in enumerate(top):
        ax = axes[i]
        sub = d[d["group"] == g].sort_values("period_dt")
        ax.plot(sub["period_dt"], sub["hs6_hhi"], lw=1.5)
        _date_axis(ax)
        ax.set_title(g)
        ax.set_ylabel("Within-group HS6 HHI")

    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    fig.suptitle("Within-group concentration (HHI across HS6 lines)", y=0.98)
    fig.tight_layout()
    out = OUT / "hs6_hhi_small_multiples.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------- new: unit value & quantity panels -------------
def plot_unit_value_and_quantity_for_top(k: int = 6):
    m = pd.read_parquet(DATA / "materials_monthly.parquet")

    # normalize groups and drop TOTAL/NaNs
    m = m[m["group"].notna() & (m["group"] != "TOTAL")].copy()

    # derive a canonical quantity column if at all possible
    if "quantity_kg" in m.columns:
        qcol = "quantity_kg"
    elif "qty" in m.columns:
        qcol = "qty"
    else:
        qcol = None

    if qcol is not None:
        m["quantity_kg"] = pd.to_numeric(m[qcol], errors="coerce")
        uv = m["value"] / m["quantity_kg"]
        # replace infs with NaN explicitly
        uv = uv.replace([float("inf"), float("-inf")], pd.NA)
        m["unit_value"] = uv
    else:
        m["quantity_kg"] = pd.NA
        m["unit_value"] = pd.NA

    m["period_dt"] = _to_dt(m["period"])
    top = _top_groups(k)

    for g in top:
        sub = m[m["group"] == g].sort_values("period_dt")

        # value + unit value panel
        fig, ax1 = plt.subplots(figsize=(12, 5), dpi=150)
        ax1.plot(sub["period_dt"], sub["value"] / 1e9, lw=1.6, label="Value (USD bn)")
        ax1.set_ylabel("Value (USD bn)")
        _date_axis(ax1)

        ax2 = ax1.twinx()
        uv = pd.to_numeric(sub["unit_value"], errors="coerce").astype(float)
        ax2.plot(sub["period_dt"], uv, lw=1.2, alpha=0.8, label="Unit value (USD/kg)")
        ax2.set_ylabel("Unit value (USD/kg)")

        ax1.set_title(f"{g}: value vs unit value")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=8)

        out = OUT / f"unit_value_{g}.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")

        # quantity panel – only really informative if we had a quantity column
        fig2, axq = plt.subplots(figsize=(12, 4), dpi=150)
        q = pd.to_numeric(sub["quantity_kg"], errors="coerce") / 1e6
        axq.plot(sub["period_dt"], q, lw=1.6)
        _date_axis(axq)
        axq.set_title(f"{g}: quantity")
        axq.set_ylabel("Quantity (million kg)")
        out2 = OUT / f"quantity_{g}.png"
        fig2.tight_layout()
        fig2.savefig(out2)
        plt.close(fig2)
        print(f"wrote {out2}")


def plot_group_growth_scatter():
    """
    Scatter of value CAGR vs final export level (log) per group, 2018–last year.
    """
    m = pd.read_parquet(DATA / "materials_monthly.parquet").copy()
    m["period_dt"] = pd.to_datetime(m["period"].astype(str), format="%Y%m")
    m["year"] = m["period_dt"].dt.year

    g_year = m.groupby(["year", "group"], as_index=False)[["value", "qty"]].sum()

    def cagr(series: pd.Series) -> float:
        s = series.dropna()
        if len(s) < 2:
            return np.nan
        start = s.iloc[0]
        end = s.iloc[-1]
        n = len(s) - 1
        if start <= 0 or end <= 0:
            return np.nan
        return (end / start) ** (1.0 / n) - 1.0

    rows = []
    for g, sub in g_year.groupby("group"):
        sub = sub.sort_values("year")
        v_cagr = cagr(sub["value"])
        year_last = int(sub["year"].iloc[-1])
        val_last = float(sub["value"].iloc[-1])
        rows.append(
            {
                "group": g,
                "year_last": year_last,
                "value_last": val_last,
                "value_cagr": v_cagr,
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["value_last", "value_cagr"])

    # Scatter: log final value vs CAGR
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    x = np.log10(df["value_last"])
    y = df["value_cagr"] * 100.0  # percent

    ax.scatter(x, y)

    for _, r in df.iterrows():
        ax.annotate(
            r["group"],
            (np.log10(r["value_last"]), r["value_cagr"] * 100.0),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Log₁₀ final annual export value (USD)")
    ax.set_ylabel("Value CAGR, 2018–{} (%)".format(int(df["year_last"].max())))
    ax.set_title("Growth vs size: export value by group")

    fig.tight_layout()
    out = OUT / "group_growth_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_price_quantity_regimes():
    """
    For each group, summarize price–volume behavior:
      x: corr(log qty, log unit value)
      y: avg |monthly % change in unit value|
    """
    m = pd.read_parquet(DATA / "materials_monthly.parquet").copy()
    m["period_dt"] = pd.to_datetime(m["period"].astype(str), format="%Y%m")

    # Build unit value
    m["unit_value"] = m["value"] / m["qty"]
    m = m.replace([np.inf, -np.inf], np.nan)

    rows = []
    for g, sub in m.groupby("group"):
        sub = sub.sort_values("period_dt").copy()
        # avoid log(0)
        sub = sub[(sub["qty"] > 0) & (sub["unit_value"] > 0)].copy()
        sub["log_q"] = np.log(sub["qty"])
        sub["log_uv"] = np.log(sub["unit_value"])
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_q", "log_uv"])
        if len(sub) < 6:
            corr = np.nan
            uv_vol = np.nan
        else:
            corr = sub["log_q"].corr(sub["log_uv"])
            uv_vol = sub["unit_value"].pct_change().abs().mean()
        rows.append(
            {
                "group": g,
                "N": len(sub),
                "corr_log_q_log_uv": corr,
                "avg_abs_pct_change_uv": uv_vol,
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["corr_log_q_log_uv"])
    df = df.sort_values("corr_log_q_log_uv")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.axvline(0.0, color="0.7", lw=1, ls="--")

    x = df["corr_log_q_log_uv"]
    y = df["avg_abs_pct_change_uv"] * 100.0  # %

    ax.scatter(x, y)

    for _, r in df.iterrows():
        ax.annotate(
            r["group"],
            (r["corr_log_q_log_uv"], r["avg_abs_pct_change_uv"] * 100.0),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Corr(log quantity, log unit value)")
    ax.set_ylabel("Avg |monthly % change in unit value|")
    ax.set_title("Price–volume regimes by group")

    fig.tight_layout()
    out = OUT / "price_quantity_regimes.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_eff_hs6_over_time(groups: list[str] | None = None):
    """
    Plot effective number of HS6 lines (1/HHI) over time for each group.
    If groups is None, use all groups.
    """
    hpath = DATA / "hs6_hhi.parquet"
    if not hpath.exists():
        print(f"skip eff HS6: {hpath} not found")
        return

    d = pd.read_parquet(hpath).copy()
    if d.empty:
        print(f"skip eff HS6: {hpath} is empty")
        return

    d["period_dt"] = pd.to_datetime(d["period"].astype(str), format="%Y%m")
    d["year"] = d["period_dt"].dt.year

    ann = d.groupby(["year", "group"], as_index=False)["hs6_hhi"].mean()
    ann["eff_hs6"] = 1.0 / ann["hs6_hhi"]

    if groups is not None:
        ann = ann[ann["group"].isin(groups)].copy()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for g, sub in ann.groupby("group"):
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub["eff_hs6"], marker="o", label=g)

    ax.set_xlabel("Year")
    ax.set_ylabel("Effective number of HS6 lines (1/HHI)")
    ax.set_title("Within-group product-line diversification")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out = OUT / "eff_hs6_over_time.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_hs6_composition_for_group(group: str, year: int, top_n: int = 8):
    """
    For a given material group and year, plot the HS6 composition
    (top N HS6 lines by export value, plus 'other').
    """
    # load raw COMTRADE + map
    paths = sorted(glob.glob(str(DATA / "comtrade_CM_HS_*_MAP_*.parquet")))
    if not paths:
        print("skip hs6 composition: no COMTRADE parquet files found")
        return

    raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if raw.empty:
        print("skip hs6 composition: raw is empty")
        return

    vcol, _ = _value_qty_cols(raw)
    if vcol is None:
        print("skip hs6 composition: no value column found")
        return

    raw = raw.rename(columns={vcol: "value"}).copy()
    raw["period"] = _to_int64_series(raw["period"])
    raw["year"] = (raw["period"] // 100).astype(int)

    # attach HS map
    hs_map_path = Path("notes/hs_map.csv")
    raw = attach_hs_map(raw, hs_map_path)
    raw = raw[raw["group"].notna() & (raw["group"] != "TOTAL")].copy()

    df = raw[(raw["group"] == group) & (raw["year"] == year)].copy()
    if df.empty:
        print(f"skip hs6 composition: no data for group={group}, year={year}")
        return

    # aggregate by HS6
    if "cmdCode" not in df.columns:
        print("skip hs6 composition: cmdCode column not found")
        return

    desc_col = "cmdDesc" if "cmdDesc" in df.columns else None

    gcols = ["cmdCode"]
    if desc_col is not None:
        gcols.append(desc_col)

    g = df.groupby(gcols, as_index=False)["value"].sum().sort_values("value", ascending=False)

    total = g["value"].sum()
    g["share"] = g["value"] / total

    top = g.head(top_n).copy()
    if len(g) > top_n:
        other_val = g["value"].iloc[top_n:].sum()
        top.loc[len(top)] = {
            "cmdCode": "OTHER",
            desc_col: "Other HS6 lines" if desc_col is not None else None,
            "value": other_val,
            "share": other_val / total,
        }

    # plotting
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    labels = top["cmdCode"].astype(str).tolist()
    if desc_col is not None:
        # optional: add a shortened desc to labels
        descs = top[desc_col].fillna("").astype(str).str.slice(0, 20)
        labels = [f"{c}\n{d}" if d else c for c, d in zip(top["cmdCode"], descs)]

    ax.bar(labels, top["share"] * 100.0)
    ax.set_ylabel("Share of group export value (%)")
    ax.set_title(f"{group} HS6 composition, {year}")

    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()

    out = OUT / f"hs6_composition_{group}_{year}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------- driver -------------
def main():
    _ensure_outdir()
    plot_monthly_stack()
    plot_monthly_share_stack()
    plot_hhi_monthly()
    plot_hhi_components()
    plot_effective_groups()
    plot_yoy_by_group()

    dpath = DATA / "dest_hhi.parquet"
    if dpath.exists():
        plot_dest_hhi_small_multiples(k=6)
    else:
        print(f"skip dest HHI: {dpath} not found")

    hpath = DATA / "hs6_hhi.parquet"
    if hpath.exists():
        plot_hs6_hhi_small_multiples(k=6)
        # new: effective HS6 figure
        focus_eff_groups = [
            "aluminum",
            "cobalt",
            "lithium",
            "graphite",
            "rare_earths",
            "gallium",
            "germanium",
        ]
        plot_eff_hs6_over_time(groups=focus_eff_groups)
    else:
        print(f"skip HS6 HHI: {hpath} not found")

    # unit value & quantity for key groups (you can swap to a custom list if you like)
    plot_unit_value_and_quantity_for_top(k=12)

    # new: growth vs size scatter
    plot_group_growth_scatter()

    # new: price–volume regimes
    plot_price_quantity_regimes()

    for g in ["lithium", "graphite", "rare_earths", "gallium", "germanium"]:
        plot_hs6_composition_for_group(g, year=2024, top_n=6)


if __name__ == "__main__":
    main()
