#!/usr/bin/env python3
"""
Download Comtrade data for all HS6 codes listed in notes/hs_map.csv.

Examples
--------
Annual 2018–2024, world partner:
    PYTHONPATH=src ./scripts/pull_from_map.py --freq A --period 2018-2024 --flow X

Monthly 2018–2024, world partner:
    PYTHONPATH=src ./scripts/pull_from_map.py --freq M --period 201801-202412 --flow X

Monthly by partner (United States = 842):
    PYTHONPATH=src ./scripts/pull_from_map.py --freq M --period 201801-202412 --flow X --partner 842
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from china_ir.comtrade import fetch_period

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"
DATA_WORK = ROOT / "data_work"


def _chunks(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _expand_periods(arg: str, freq: str) -> List[str]:
    """Accept 'a,b,c' or 'start-end' and return list of period strings."""
    arg = arg.strip()
    if "," in arg:
        return [p.strip() for p in arg.split(",") if p.strip()]

    start, end = arg.split("-")
    start, end = start.strip(), end.strip()

    if freq.upper() == "A":
        ys = range(int(start), int(end) + 1)
        return [f"{y:d}" for y in ys]
    if freq.upper() == "M":
        ys, ms = int(start[:4]), int(start[4:])
        ye, me = int(end[:4]), int(end[4:])
        out = []
        y, m = ys, ms
        while (y < ye) or (y == ye and m <= me):
            out.append(f"{y}{m:02d}")
            m += 1
            if m == 13:
                m, y = 1, y + 1
        return out

    raise ValueError(f"Unsupported freq={freq!r}")


def _load_hs6_from_map(map_csv: Path) -> List[str]:
    m = pd.read_csv(map_csv, dtype={"hs6": str})
    hs = m["hs6"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6).unique()
    return [h for h in hs if h and h != "nan"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", choices=["A", "M"], required=True)
    ap.add_argument("--period", required=True, help="Comma list or start-end")
    ap.add_argument("--flow", required=True, help="X, M, or T")
    ap.add_argument("--reporter", default="156", help="Reporter numeric code (default China=156)")
    ap.add_argument("--partner", default="0", help="Partner code (0=World)")
    ap.add_argument("--map", default=str(NOTES / "hs_map.csv"))
    ap.add_argument("--chunk", type=int, default=20, help="HS6 per API call")
    args = ap.parse_args()

    periods = _expand_periods(args.period, args.freq)
    hs6 = _load_hs6_from_map(Path(args.map))

    frames: List[pd.DataFrame] = []
    for block in _chunks(hs6, args.chunk):
        df = fetch_period(
            args.freq, periods, args.reporter, args.partner, args.flow, ",".join(block)
        )
        if not df.empty:
            frames.append(df)

    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "period",
                "cmdCode",
                "primaryValue",
                "reporterCode",
                "partnerCode",
                "flowCode",
            ]
        )
    )

    DATA_WORK.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    period_token = "-".join([periods[0], periods[-1]]) if len(periods) > 3 else "-".join(periods)
    outname = (
        f"comtrade_C{args.freq}_HS_{period_token}_{args.reporter}_{args.partner}_"
        f"{args.flow}_MAP_{stamp}.parquet"
    )
    outpath = DATA_WORK / outname
    out.to_parquet(outpath, index=False)
    print(f"wrote {outpath} rows: {len(out)}")


if __name__ == "__main__":
    main()
