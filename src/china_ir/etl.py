#!/usr/bin/env python3
"""
china_ir.etl — build monthly/annual tables + concentration metrics from Comtrade MAP parquet

Inputs (glob inside data_work/):
  comtrade_CM_HS_*_MAP_*.parquet  (monthly pulls; columns include: period, cmdCode, primaryValue, partnerISO, ...)   # noqa: E501
  (optionally) notes/hs_map.csv    (hs6 -> material,group mapping)

Outputs (in data_work/):
  materials_monthly.parquet  columns: period(YYYYMM, Int64), group, material, value(float), share(float)   # noqa: E501
  materials_hhi.parquet      columns: period(YYYYMM, Int64), hhi(float in [0,1])
  materials_annual.parquet   columns: period(YYYY,   Int64), group, material, value(float)
  dest_hhi.parquet (optional) columns: period(YYYYMM, Int64), group, hhi(float)  [destination-market concentration]   # noqa: E501
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

DATA = Path("data_work")
DEFAULT_MAP = Path("notes/hs_map.csv")  # can be missing; we’ll still run


# ---------- IO helpers ----------


def _pick_partner_col(df):
    for c in ("partnerISO", "partnerCode", "partnerDesc"):
        if c in df.columns:
            return c
    return None


def _read_glob(pattern: str) -> pd.DataFrame:
    """Robustly read a glob of parquet files under DATA, skipping empties/bad files."""
    files = sorted((DATA / pattern).parent.glob((DATA / pattern).name))
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df is None or df.empty or len(df.columns) == 0:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _value_qty_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect the value and (optional) quantity columns in Comtrade MAP parquet.
    Returns (value_col, qty_col). Either can be None; caller handles renaming.
    """
    if df is None or df.empty:
        return None, None

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    nmap = {c: norm(c) for c in df.columns}

    # Value variants observed in MAP outputs
    val_keys = {
        "value",
        "tradevalue",
        "tradevalueusd",
        "tradevalueus",
        "primaryvalue",
        "primaryvalueusd",
        "primaryvalueus",
        # extra safety:
        "customsvalue",
        "fobvalue",
        "cifvalue",
    }

    # Common qty/weight variants
    qty_keys = {
        "qty",
        "quantity",
        "netweight",
        "netweightkg",
        "quantitykg",
        "grossweight",
        "grossweightkg",
        "weightkg",
    }

    vcol = next((c for c in df.columns if nmap[c] in val_keys), None)
    qcol = next((c for c in df.columns if nmap[c] in qty_keys), None)

    # Heuristic fallback for value: largest-sum numeric column
    if vcol is None:
        num = df.select_dtypes(include="number")
        if not num.empty:
            vcol = num.sum(numeric_only=True).sort_values(ascending=False).index[0]

    return vcol, qcol


# ---------- mapping & normalization ----------


def attach_hs_map(df: pd.DataFrame, map_csv: Path) -> pd.DataFrame:
    """
    Attach hs6→(material, group). If map_csv missing, create empty material/group columns.
    Ensure TOTAL lines are labeled accordingly and skip hs6 on TOTAL.
    """
    out = df.copy()

    # 1. Normalize cmdCode and generate hs6 (using pre-merge mask)
    if "cmdCode" in out.columns:
        out["cmdCode"] = out["cmdCode"].astype(str).str.strip()
        # Temporary mask for generating hs6 only on non-total rows
        _is_tot_pre = out["cmdCode"].str.upper().eq("TOTAL")
    else:
        _is_tot_pre = pd.Series(False, index=out.index)

    out["hs6"] = pd.NA
    if "cmdCode" in out.columns:
        out.loc[~_is_tot_pre, "hs6"] = (
            out.loc[~_is_tot_pre, "cmdCode"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(6)
        )

    # 2. Merge mapping (This effectively re-indexes 'out')
    if map_csv.exists():
        m = pd.read_csv(map_csv, dtype={"hs6": str})
        m["hs6"] = m["hs6"].astype(str).str.zfill(6)
        out = out.merge(m, how="left", on="hs6")
    else:
        if "material" not in out.columns:
            out["material"] = pd.NA
        if "group" not in out.columns:
            out["group"] = pd.NA

    # 3. Label TOTAL lines (Calculate mask on the NEW post-merge dataframe)
    if "cmdCode" in out.columns:
        is_total = out["cmdCode"].astype(str).str.upper().eq("TOTAL")
        if is_total.any():
            out.loc[is_total, "material"] = "TOTAL"
            out.loc[is_total, "group"] = "TOTAL"

    return out


def _to_int64_series(x: pd.Series) -> pd.Series:
    """Coerce to pandas nullable Int64 dtype."""
    return pd.to_numeric(x, errors="coerce").astype("Int64")


# ---------- core builders ----------


def _build_monthly(
    raw: pd.DataFrame,
    hs_map: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    From raw MAP rows, produce:
      monthly_long: period(YYYYMM Int64), material, group, value, qty, share
      hhi_monthly:  period(YYYYMM Int64), hhi
    """
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Detect and normalize value/qty columns
    vcol, qcol = _value_qty_cols(raw)
    if vcol is None:
        raise ValueError(f"No value-like column found; columns={list(raw.columns)}")

    rename_map: dict[str, str] = {vcol: "value"}
    if qcol is not None and qcol != vcol:
        rename_map[qcol] = "qty"

    raw = raw.rename(columns=rename_map)

    if "qty" not in raw.columns:
        raw["qty"] = pd.NA

    # period → Int64, keep only monthly-looking rows (YYYYMM)
    raw["period"] = _to_int64_series(raw["period"])
    s = raw["period"].astype(str).str.len()
    raw = raw[s == 6].copy()

    # attach mapping; allow missing map
    raw = attach_hs_map(raw, hs_map)

    # aggregate to dedupe: (period, group, material), summing value and qty
    agg_cols = ["value", "qty"]
    gsum = raw.groupby(["period", "group", "material"], dropna=False)[agg_cols].sum().reset_index()

    # compute TOTAL per period (prefer explicit TOTAL rows; else sum of groups)
    tot_explicit = (
        gsum[gsum["group"] == "TOTAL"]
        .groupby("period", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "total"})
    )
    tot_from_groups = (
        gsum[gsum["group"].notna() & (gsum["group"] != "TOTAL")]
        .groupby("period", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "total"})
    )

    total = (
        tot_explicit.set_index("period")
        .combine_first(tot_from_groups.set_index("period"))
        .rename_axis("period")
        .reset_index()
    )

    # shares by group within period (exclude TOTAL)
    groups_only = gsum[gsum["group"].notna()].copy()
    groups_only = groups_only.merge(total, on="period", how="left")
    groups_only["share"] = groups_only["value"] / groups_only["total"]

    # HHI across groups per period (exclude TOTAL)
    hhi = (
        groups_only[groups_only["group"] != "TOTAL"]
        .groupby("period")["share"]
        .apply(lambda s: float((s.fillna(0.0) ** 2).sum()))
        .rename("hhi")
        .reset_index()
        .astype({"period": "Int64"})
    )

    monthly_long = groups_only.copy()
    monthly_long["period"] = monthly_long["period"].astype("Int64")

    return monthly_long, hhi


def _build_annual_from_monthly(monthly_long: pd.DataFrame) -> pd.DataFrame:
    """Sum monthly values to year totals by (group, material)."""
    if monthly_long.empty:
        return pd.DataFrame()
    y = monthly_long.copy()
    # year = floor(period / 100)
    y["period"] = (y["period"].astype("Int64") // 100).astype("Int64")
    ann = y.groupby(["period", "group", "material"], dropna=False, as_index=False)["value"].sum()
    return ann


def _build_dest_hhi(raw: pd.DataFrame, hs_map: Path) -> pd.DataFrame:
    """
    Destination-market concentration (HHI across partners).

    For each (period, group), compute the HHI of export value across partner markets.
    Returns DataFrame with columns: period (YYYYMM Int64), group, hhi.
    """
    if raw.empty:
        return pd.DataFrame()

    vcol, _ = _value_qty_cols(raw)
    if vcol is None:
        return pd.DataFrame()

    df = raw.rename(columns={vcol: "value"}).copy()

    # normalize period and keep monthly
    df["period"] = _to_int64_series(df["period"])
    s = df["period"].astype(str).str.len()
    df = df[s == 6].copy()
    if df.empty:
        return pd.DataFrame()

    # attach HS6→group mapping
    df = attach_hs_map(df, hs_map)

    # keep only mapped, non-TOTAL groups
    df = df[df["group"].notna() & (df["group"] != "TOTAL")].copy()
    if df.empty:
        return pd.DataFrame()

    # pick a partner dimension that actually has variation
    candidate_partner_cols = [
        "partner2Code",
        "partnerCode",
        "partnerISO",
        "partner2ISO",
        "partnerDesc",
        "partner2Desc",
    ]
    partner_col = None
    for c in candidate_partner_cols:
        if c in df.columns:
            nunique = df[c].dropna().nunique()
            if nunique > 1:
                partner_col = c
                break

    if partner_col is None:
        # no usable partner dimension
        return pd.DataFrame()

    # drop rows with missing partner value, but DON'T over-filter by 0/WLD etc.
    df = df[df[partner_col].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    # aggregate value to (period, group, partner)
    g = df.groupby(["period", "group", partner_col], dropna=False)["value"].sum().reset_index()

    if g.empty:
        return pd.DataFrame()

    # totals by (period, group)
    totals = (
        g.groupby(["period", "group"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "tot"})
    )

    g = g.merge(totals, on=["period", "group"], how="left")
    g["share"] = g["value"] / g["tot"]

    dest_hhi = (
        g.groupby(["period", "group"])["share"]
        .apply(lambda s: float((s.fillna(0.0) ** 2).sum()))
        .rename("hhi")
        .reset_index()
    )
    dest_hhi["period"] = dest_hhi["period"].astype("Int64")
    return dest_hhi


def _build_hs6_hhi(raw: pd.DataFrame, hs_map: Path) -> pd.DataFrame:
    """
    Within-group HS6-line concentration.

    For each (period, group), compute HHI over HS6 lines that belong to that group.
    Returns DataFrame with columns: period (YYYYMM Int64), group, hs6_hhi.
    """
    if raw.empty:
        return pd.DataFrame()

    vcol, _ = _value_qty_cols(raw)
    if vcol is None:
        return pd.DataFrame()

    df = raw.rename(columns={vcol: "value"}).copy()
    df["period"] = _to_int64_series(df["period"])
    s = df["period"].astype(str).str.len()
    df = df[s == 6].copy()

    # attach HS6 + group mapping
    df = attach_hs_map(df, hs_map)

    # keep only mapped, non-TOTAL groups
    df = df[df["group"].notna() & (df["group"] != "TOTAL")].copy()
    if df.empty:
        return pd.DataFrame()

    # aggregate value to (period, group, hs6)
    gph = df.groupby(["period", "group", "hs6"], dropna=False, as_index=False)["value"].sum()

    # total by (period, group)
    totals = (
        gph.groupby(["period", "group"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "tot"})
    )
    gph = gph.merge(totals, on=["period", "group"], how="left")
    gph["share"] = gph["value"] / gph["tot"]

    hs6_hhi = (
        gph.groupby(["period", "group"])["share"]
        .apply(lambda s: float((s.fillna(0.0) ** 2).sum()))
        .rename("hs6_hhi")
        .reset_index()
    )
    hs6_hhi["period"] = hs6_hhi["period"].astype("Int64")
    return hs6_hhi


# ---------- CLI ----------


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-glob",
        default="comtrade_CM_HS_*_MAP_*.parquet",
        help="glob under data_work/ for monthly MAP parquet files",
    )
    ap.add_argument(
        "--hs-map",
        default=str(DEFAULT_MAP),
        help="CSV mapping hs6→material,group (optional)",
    )
    ap.add_argument(
        "--outdir",
        default=str(DATA),
        help="output directory (default: data_work/)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hs_map = Path(args.hs_map)

    raw = _read_glob(args.in_glob)
    if raw.empty:
        print("No MAP parquet rows found; nothing to do.")
        return

    # Build monthly + HHI over groups
    monthly_long, hhi = _build_monthly(raw, hs_map)

    # Save monthly (keep qty + any other useful columns)
    if not monthly_long.empty:
        monthly_path = outdir / "materials_monthly.parquet"
        monthly_long.to_parquet(monthly_path, index=False)
        print(f"wrote {monthly_path} rows: {len(monthly_long)}")

    # Save HHI over groups
    if not hhi.empty:
        hhi_path = outdir / "materials_hhi.parquet"
        hhi.to_parquet(hhi_path, index=False)
        print(f"wrote {hhi_path} rows: {len(hhi)}")

    # Annual from monthly
    annual = _build_annual_from_monthly(monthly_long)
    if not annual.empty:
        annual_path = outdir / "materials_annual.parquet"
        annual.to_parquet(annual_path, index=False)
        print(f"wrote {annual_path} rows: {len(annual)}")

    # Destination HHI (optional)
    dest_hhi = _build_dest_hhi(raw, hs_map)
    if not dest_hhi.empty:
        dest_path = outdir / "dest_hhi.parquet"
        dest_hhi.to_parquet(dest_path, index=False)
        print(f"wrote {dest_path} rows: {len(dest_hhi)}")
    else:
        print("skip dest_hhi.parquet (no partner columns found or no data)")

    # Within-group HS6 HHI (optional)
    hs6_hhi = _build_hs6_hhi(raw, hs_map)
    if not hs6_hhi.empty:
        hs6_path = outdir / "hs6_hhi.parquet"
        hs6_hhi.to_parquet(hs6_path, index=False)
        print(f"wrote {hs6_path} rows: {len(hs6_hhi)}")
    else:
        print("skip hs6_hhi.parquet (no hs6/group data)")


if __name__ == "__main__":
    _main()
