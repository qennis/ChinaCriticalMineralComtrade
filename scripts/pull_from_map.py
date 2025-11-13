#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Iterable, List

import pandas as pd

from china_ir.comtrade import fetch_period
from china_ir.paths import DATA_WORK


def _chunks(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _codes_from_map(map_csv: str) -> List[str]:
    """Load hs6 codes from a CSV with columns: hs6, material, group, notes."""
    m = pd.read_csv(map_csv, dtype=str)
    if "hs6" not in m.columns:
        return []
    hs6 = m["hs6"].dropna().astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    # Keep only 6-digit numeric entries
    hs6 = hs6[hs6.str.fullmatch(r"\d{6}")]
    return sorted(hs6.unique().tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull Comtrade slices defined by hs_map.csv.")
    ap.add_argument("--freq", choices=["A", "M"], required=True, help="A (annual) or M (monthly)")
    ap.add_argument(
        "--period",
        required=True,
        help="Comma-separated periods, e.g. 2018,2019 or 202401,202402",
    )
    ap.add_argument("--reporter", default="156", help="Reporter code (default: 156 China)")
    ap.add_argument("--partner", default="0", help="Partner code (default: 0 World)")
    ap.add_argument("--flow", choices=["X", "M"], required=True, help="X (exports) or M (imports)")
    ap.add_argument("--map", default="notes/hs_map.csv", help="Path to hs_map.csv")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="How many HS6 codes to query per request (default: 20)",
    )
    args = ap.parse_args()

    # Load HS codes from map
    hs6 = _codes_from_map(args.map)
    if not hs6:
        print("No HS6 codes found in map.")
        # Still emit an empty artifact for downstream pipelines
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        tag = (
            f"comtrade_C{args.freq}_HS_{args.period.replace(',', '-')}_"
            f"{args.reporter}_{args.partner}_{args.flow}_MAP_{stamp}.parquet"
        )
        out_path = DATA_WORK / tag
        pd.DataFrame().to_parquet(out_path, index=False)
        print(f"wrote {out_path} rows: 0")
        return

    periods = [p.strip() for p in args.period.split(",") if p.strip()]
    parts: List[pd.DataFrame] = []

    # IMPORTANT: match the working fetch_period signature (6 positional args)
    # fetch_period(freq, periods, reporter, partner, flow, cmd)
    for batch in _chunks(hs6, args.batch_size):
        cmd = ",".join(batch)
        df = fetch_period(args.freq, periods, args.reporter, args.partner, args.flow, cmd)
        if df is not None and len(df):
            parts.append(df)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tag = (
        f"comtrade_C{args.freq}_HS_{'-'.join(periods)}_{args.reporter}_"
        f"{args.partner}_{args.flow}_MAP_{stamp}.parquet"
    )
    out_path = DATA_WORK / tag
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path} rows: {len(out)}")


if __name__ == "__main__":
    main()
