#!/usr/bin/env python3
from __future__ import annotations

import matplotlib as mpl  # safe to import at top
import pandas as pd

from china_ir.paths import DATA_WORK, FIGURES, ensure_dirs


def _read_ok(p):
    try:
        df = pd.read_parquet(p)
        return df if (df is not None and not df.empty and df.columns.size) else None
    except Exception:
        return None


def main() -> None:
    # Set backend *before* importing pyplot, but inside a function to avoid E402
    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433 (local import by design)

    ensure_dirs()

    ann = _read_ok(DATA_WORK / "materials_annual.parquet")
    if ann is not None:
        pivot = (
            ann.pivot_table(index="period", columns="group", values="value", aggfunc="sum")
            .fillna(0.0)
            .sort_index()
        )
        ax = pivot.plot.area(figsize=(8, 5))
        ax.set_title("Exports by material group (annual)")
        ax.set_xlabel("Year")
        ax.set_ylabel("USD")
        plt.tight_layout()
        plt.savefig(FIGURES / "annual_stack.png", dpi=160)
        plt.close()

    hhi = _read_ok(DATA_WORK / "materials_hhi.parquet")
    if hhi is not None:
        hhi = hhi.sort_values("period")
        ax = hhi.plot(x="period", y="hhi", legend=False, figsize=(8, 4))
        ax.set_ylim(0, 1)
        ax.set_title("Concentration (HHI) by month")
        ax.set_xlabel("Period")
        ax.set_ylabel("HHI (sum of share^2)")
        plt.tight_layout()
        plt.savefig(FIGURES / "hhi_monthly.png", dpi=160)
        plt.close()

    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
