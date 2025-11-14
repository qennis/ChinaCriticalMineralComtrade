#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator

DATA = "data_work"
OUT = "figures"


def load_monthly_only(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    m = df[df["period"].astype(str).str.len() == 6].copy()  # keep YYYYMM only
    m["period_dt"] = pd.to_datetime(m["period"].astype(str), format="%Y%m")
    m.sort_values("period_dt", inplace=True)
    return m


def plot_monthly_stack(outdir: str = OUT):
    m = load_monthly_only(os.path.join(DATA, "materials_monthly.parquet"))
    m = m[m["group"].ne("TOTAL")]
    # pivot to [time x group]
    piv = m.pivot_table(index="period_dt", columns="group", values="value", aggfunc="sum").fillna(
        0.0
    )
    # order layers by total size over the window (largest on bottom)
    order = piv.sum(axis=0).sort_values(ascending=False).index.tolist()
    X = piv.index
    Y = (piv[order] / 1e9).T.values  # USD billions

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.stackplot(X, Y, labels=order, linewidth=0)
    ax.set_title("Exports by material group (monthly)")
    ax.set_ylabel("USD (billions)")
    ax.set_xlabel("Month")
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(MonthLocator(bymonth=(1, 4, 7, 10)))  # quarter ticks
    ax.legend(title="group", loc="upper left", fontsize=9, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "monthly_stack.png"), dpi=200)


def plot_hhi_monthly(outdir: str = OUT):
    h = pd.read_parquet(os.path.join(DATA, "materials_hhi.parquet"))
    h = h[h["period"].astype(str).str.len() == 6].copy()
    h["period_dt"] = pd.to_datetime(h["period"].astype(str), format="%Y%m")
    h.sort_values("period_dt", inplace=True)

    # equal-share baseline (1/k), where k = #groups with positive value in month
    m = load_monthly_only(os.path.join(DATA, "materials_monthly.parquet"))
    k = (
        m[m["value"] > 0]
        .groupby("period_dt")["group"]
        .nunique()
        .reindex(h["period_dt"])
        .interpolate()
        .ffill()
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(h["period_dt"], h["hhi"], marker="o", ms=2, lw=1.5, label="HHI")
    ax.plot(h["period_dt"], 1 / k, ls="--", lw=1, alpha=0.65, label="equal-share 1/k")
    ax.set_title("Concentration (HHI) by month")
    ax.set_ylabel("HHI (sum of share²)")
    ax.set_xlabel("Month")
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, min(1.0, h["hhi"].max() * 1.25))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "hhi_monthly.png"), dpi=200)


def main(outdir: str = OUT):
    os.makedirs(outdir, exist_ok=True)
    plot_monthly_stack(outdir)
    plot_hhi_monthly(outdir)


if __name__ == "__main__":
    main(OUT)
