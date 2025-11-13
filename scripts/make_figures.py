#!/usr/bin/env python3
"""
Make figures from ETL parquet outputs.

Inputs:
- data_work/materials_annual.parquet   (period: year-ish; group; value)
- data_work/materials_hhi.parquet      (period: YYYYMM; hhi)

Outputs:
- figures/annual_stack.png
- figures/hhi_monthly.png
"""

from pathlib import Path


def main(outdir: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import io

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import pandas as pd

    def fig_to_png_bytes(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return buf.read()

    data_dir = Path("data_work")
    outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Annual stacked area (billions)
    # ----------------------------
    ann_path = data_dir / "materials_annual.parquet"
    ann = pd.read_parquet(ann_path)

    # Robustly coerce to year: pick the first 4 consecutive digits
    year_str = ann["period"].astype(str).str.extract(r"(\d{4})", expand=False)
    ann = ann.loc[year_str.notna()].copy()
    ann["period"] = pd.to_datetime(year_str, format="%Y")

    # Drop TOTAL if present
    if "group" in ann.columns:
        ann = ann[ann["group"].ne("TOTAL")]

    # Order groups by most recent year’s totals
    latest_year = ann["period"].dt.year.max()
    order = (
        ann.loc[ann["period"].dt.year.eq(latest_year)]
        .groupby("group", as_index=True)["value"]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    pivot = (
        ann.pivot_table(index="period", columns="group", values="value", aggfunc="sum")
        .fillna(0.0)
        .reindex(columns=order)
        .sort_index()
    )
    pivot_b = pivot / 1e9  # billions

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.stackplot(pivot_b.index, pivot_b.T.values, labels=pivot_b.columns)
    ax.set_title("Exports by material group (annual)")
    ax.set_ylabel("USD (billions)")
    ax.set_xlabel("Year")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(7))
    ax.legend(title="group", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    (outdir / "annual_stack.png").write_bytes(fig_to_png_bytes(fig))
    plt.close(fig)

    # ----------------------------
    # Monthly HHI
    # ----------------------------
    hhi_path = data_dir / "materials_hhi.parquet"
    if hhi_path.exists():
        hhi = pd.read_parquet(hhi_path)
        mm_str = hhi["period"].astype(str).str.extract(r"(\d{6})", expand=False)
        hhi = hhi.loc[mm_str.notna()].copy()
        hhi["period"] = pd.to_datetime(mm_str, format="%Y%m", errors="coerce")
        hhi = hhi.dropna(subset=["period"]).sort_values("period")

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(hhi["period"], hhi["hhi"], marker="o", linewidth=1.5)
        ax.set_title("Concentration (HHI) by month")
        ax.set_xlabel("Month")
        ax.set_ylabel("HHI (sum of share^2)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b"))
        fig.autofmt_xdate()

        upper = float(hhi["hhi"].max() * 1.2) if len(hhi) else 0.2
        upper = min(max(upper, 0.2), 1.0)
        ax.set_ylim(0.0, upper)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        fig.tight_layout()
        (outdir / "hhi_monthly.png").write_bytes(fig_to_png_bytes(fig))
        plt.close(fig)
    else:
        print(f"[warn] {hhi_path} not found; skipping HHI plot")

    print(f"figures written to {outdir.resolve()}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="figures", type=Path)
    args = p.parse_args()
    main(args.outdir)
