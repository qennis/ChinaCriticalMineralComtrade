#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from china_ir.paths import DATA_WORK

# ----------------------------- helpers ---------------------------------


def attach_hs_map(df: pd.DataFrame, map_csv: Path) -> pd.DataFrame:
    """Attach material/group from notes/hs_map.csv by HS6; preserve TOTAL rows."""
    out = df.copy()

    # Normalize cmdCode
    out["cmdCode"] = out["cmdCode"].astype(str).str.strip()
    is_total = out["cmdCode"].str.upper().eq("TOTAL")

    # Build hs6 only for non-TOTAL rows
    out["hs6"] = pd.NA
    out.loc[~is_total, "hs6"] = (
        out.loc[~is_total, "cmdCode"].str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )

    # Load mapping (may be empty); dedupe on hs6 to avoid many-to-one blowup
    m = pd.read_csv(map_csv, dtype={"hs6": str})
    if not m.empty:
        m["hs6"] = m["hs6"].astype(str).str.zfill(6)
        m = m.drop_duplicates(subset=["hs6"], keep="first")
        out = out.merge(m, how="left", on="hs6")

    # Ensure columns exist even if merge added none
    if "material" not in out.columns:
        out["material"] = pd.NA
    if "group" not in out.columns:
        out["group"] = pd.NA

    # Recompute TOTAL mask *after* merge (so length matches), then set explicitly
    is_total = out["cmdCode"].str.upper().eq("TOTAL")
    out.loc[is_total, "material"] = "TOTAL"
    out.loc[is_total, "group"] = "TOTAL"

    return out


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to period/material/group with value from primaryValue."""
    need = {"period", "material", "group", "primaryValue"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    g = (
        df.groupby(["period", "material", "group"], as_index=False, sort=True)["primaryValue"]
        .sum()
        .rename(columns={"primaryValue": "value"})
    )
    return g


def add_shares(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add per-period share; TOTAL has share 1; others in (0,1]."""
    base = monthly.copy()

    # Prefer explicit TOTAL rows if present
    tot = (
        base[base["material"].eq("TOTAL")][["period", "value"]]
        .rename(columns={"value": "total_value"})
        .drop_duplicates(subset=["period"])
    )
    if tot.empty:
        tot = (
            base.groupby("period", as_index=False)["value"]
            .sum()
            .rename(columns={"value": "total_value"})
        )

    base = base.merge(tot, on="period", how="left")
    base["share"] = (base["value"] / base["total_value"]).where(base["total_value"] > 0, 0.0)
    base.loc[base["material"].eq("TOTAL"), "share"] = 1.0
    return base.drop(columns=["total_value"])


def compute_hhi(monthly_with_shares: pd.DataFrame) -> pd.DataFrame:
    """HHI by period over non-TOTAL materials: sum(share^2)."""
    s = monthly_with_shares[~monthly_with_shares["material"].eq("TOTAL")]
    hhi = (
        s.assign(w=lambda x: x["share"] ** 2)
        .groupby("period", as_index=False)["w"]
        .sum()
        .rename(columns={"w": "hhi"})
    )
    return hhi


# --------------------------- annual aggregation ------------------------


def _period_as_year(period: pd.Series) -> pd.Series:
    """
    Normalize Comtrade 'period' to YYYY.
    Accepts YYYYMM (e.g., 202403) or YYYY (e.g., 2024).
    """
    s = pd.to_numeric(period, errors="coerce").astype("Int64")
    mask = s > 9999  # YYYYMM
    s = s.where(~mask, s // 100)
    return s.astype("int64", errors="ignore")


def aggregate_annual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to annual totals per (period=YYYY, material, group).

    Accepts either raw Comtrade frames (with 'primaryValue') or frames already
    carrying 'value'. Requires that 'material' and 'group' exist (i.e., call
    attach_hs_map(...) beforehand if starting from raw Comtrade).
    """
    if "material" not in df.columns or "group" not in df.columns:
        raise ValueError(
            "aggregate_annual expects 'material' and 'group' columns; "
            "run attach_hs_map(...) first."
        )

    value_col = (
        "primaryValue"
        if "primaryValue" in df.columns
        else ("value" if "value" in df.columns else None)
    )
    if value_col is None:
        raise ValueError("DataFrame must contain either 'primaryValue' or 'value'.")

    out = (
        df.assign(period=_period_as_year(df["period"]))
        .groupby(["period", "material", "group"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "value"})
        .sort_values(["period", "group", "material"])
        .reset_index(drop=True)
    )
    return out


# ----------------------------- CLI runners ------------------------------


def _run_monthly(in_glob: str, hs_map: Path) -> None:
    files = sorted(Path(DATA_WORK).glob(in_glob))
    parts: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f)
        df = df.dropna(axis="columns", how="all")
        if len(df):
            parts.append(df)

    parts = [d for d in parts if not d.empty and d.dropna(axis=1, how="all").shape[1] > 0]
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    raw = raw[raw["period"].astype(str).str.len() == 6].copy()
    if raw.empty:
        # Keep pipeline stable even if raw slices are missing
        (DATA_WORK / "materials_monthly.parquet").write_bytes(b"")
        (DATA_WORK / "materials_hhi.parquet").write_bytes(b"")
        print("no monthly raw inputs found; wrote empty outputs")
        return

    with_map = attach_hs_map(raw, hs_map)
    monthly = _aggregate(with_map)
    monthly = add_shares(monthly)
    monthly["period"] = pd.to_numeric(monthly["period"], errors="coerce").astype("Int64")
    monthly.to_parquet(DATA_WORK / "materials_monthly.parquet", index=False)

    hhi = compute_hhi(monthly)
    hhi.to_parquet(DATA_WORK / "materials_hhi.parquet", index=False)

    out_month = DATA_WORK / "materials_monthly.parquet"
    out_hhi = DATA_WORK / "materials_hhi.parquet"
    print("wrote " f"{out_month} rows: {len(monthly)}; " f"{out_hhi} rows: {len(hhi)}")


def _run_annual(in_glob: str, hs_map: Path) -> None:
    files = sorted(Path(DATA_WORK).glob(in_glob))
    parts: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f)
        df = df.dropna(axis="columns", how="all")
        if len(df):
            parts.append(df)

    parts = [d for d in parts if not d.empty and d.dropna(axis=1, how="all").shape[1] > 0]
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if raw.empty:
        (DATA_WORK / "materials_annual.parquet").write_bytes(b"")
        print("no annual raw inputs found; wrote empty annual output")
        return

    with_map = attach_hs_map(raw, hs_map)
    annual = _aggregate(with_map)
    annual["period"] = pd.to_numeric(annual["period"], errors="coerce").astype("Int64")
    annual.to_parquet(DATA_WORK / "materials_annual.parquet", index=False)
    print(f"wrote {DATA_WORK/'materials_annual.parquet'} rows: {len(annual)}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["monthly", "annual"], required=True)
    ap.add_argument(
        "--in-glob",
        default="comtrade_*_X_*.parquet",
        help="glob in data_work for raw comtrade slices",
    )
    ap.add_argument("--hs-map", default="notes/hs_map.csv")
    args = ap.parse_args()

    if args.mode == "monthly":
        _run_monthly(args.in_glob, Path(args.hs_map))
    else:
        _run_annual(args.in_glob, Path(args.hs_map))


if __name__ == "__main__":
    _main()
