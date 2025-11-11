#!/usr/bin/env python3
"""
UN Comtrade data puller (data/v1 API).

Usage (examples):
  # China → World, monthly TOTAL exports for Jan–Mar 2024
  PYTHONPATH=src ./scripts/pull_comtrade.py \
    --type C --freq M --class HS \
    --period 202401,202402,202403 \
    --reporter 156 --partner 0 \
    --flow X --cmd TOTAL

  # Single month
  PYTHONPATH=src ./scripts/pull_comtrade.py \
    --period 202405 --reporter 156 --partner 0 --flow X --cmd TOTAL

Environment:
  COMTRADE_API_KEY  (required)  Your subscription key
  COMTRADE_API_BASE (optional)  Defaults to https://comtradeapi.un.org/data/v1/get
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Iterable, List

import pandas as pd
import requests

from china_ir.paths import DATA_WORK, ensure_dirs

DEFAULT_BASE = os.getenv("COMTRADE_API_BASE", "https://comtradeapi.un.org/data/v1/get")
API_KEY = os.getenv("COMTRADE_API_KEY")

# Treat these as transient and retry with backoff
RETRY_STATUSES = {429, 500, 502, 503, 504}


def _json_or_raise(resp: requests.Response) -> Dict:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "json" not in ctype:
        preview = resp.text[:300].replace("\n", " ")
        raise RuntimeError(f"Non-JSON response (status {resp.status_code}): {preview}")
    return resp.json()


def _sleep_backoff(attempt: int, resp: requests.Response) -> None:
    # Respect Retry-After if provided, else exponential backoff with cap
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            delay = max(1, int(retry_after))
            time.sleep(delay)
            return
        except Exception:
            pass
    time.sleep(min(60, 2**attempt))


def _fetch_once(
    type_code: str,
    freq_code: str,
    cl_code: str,
    period: str,
    reporter: str,
    partner: str,
    flow_code: str,
    cmd_code: str,
    include_desc: bool = True,
) -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError("COMTRADE_API_KEY not set. Export it before running this script.")

    url = f"{DEFAULT_BASE}/{type_code}/{freq_code}/{cl_code}"
    params = {
        "reporterCode": reporter,
        "partnerCode": partner,
        "period": period,  # 'YYYY' for A, or 'YYYYMM' for M
        "flowCode": flow_code,  # 'X' exports, 'M' imports
        "cmdCode": cmd_code,  # 'TOTAL' or comma-separated HS codes
        "includeDesc": "TRUE" if include_desc else "FALSE",
        # Other optional params you might use later:
        # "customsCode": "C00",
        # "motCode": "0",
        # "breakdownMode": "classic",
        # "maxRecords": 250000,
    }
    headers = {
        "Accept": "application/json",
        "User-Agent": "china-ir-paper/0.1 (+github.com/qennis/china-ir-paper)",
        "Ocp-Apim-Subscription-Key": API_KEY,
    }

    for attempt in range(6):
        resp = requests.get(url, params=params, headers=headers, timeout=90)
        if resp.status_code in RETRY_STATUSES:
            _sleep_backoff(attempt, resp)
            continue
        resp.raise_for_status()
        js = _json_or_raise(resp)

        # New API returns an object with a 'data' list (or an empty list)
        data = js.get("data") if isinstance(js, dict) else None
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    raise RuntimeError("Comtrade API kept returning transient errors.")


def _chunk(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def fetch_frame_batched(
    type_code: str,
    freq_code: str,
    cl_code: str,
    periods: List[str],
    reporter: str,
    partner: str,
    flow_code: str,
    cmd_code: str,
    include_desc: bool = True,
    batch_size: int = 12,
) -> pd.DataFrame:
    """
    Fetch multiple periods in batches to avoid very long query strings.
    For monthly pulls, a batch_size of 12 keeps requests modest in size.
    """
    frames: List[pd.DataFrame] = []
    for group in _chunk(periods, batch_size):
        period_str = ",".join(group)
        df = _fetch_once(
            type_code=type_code,
            freq_code=freq_code,
            cl_code=cl_code,
            period=period_str,
            reporter=reporter,
            partner=partner,
            flow_code=flow_code,
            cmd_code=cmd_code,
            include_desc=include_desc,
        )
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _parse_periods(period_arg: str) -> List[str]:
    """
    Accepts comma-separated periods. For now we do not expand ranges; pass YYYY or YYYYMM.
    Examples:
      "2024"                 (annual)
      "202401"               (single month)
      "202401,202402,202403" (multiple months)
    """
    items = [p.strip() for p in period_arg.split(",") if p.strip()]
    if not items:
        raise ValueError("No valid periods provided.")
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="C", help="Trade type, e.g. 'C' (goods)")
    ap.add_argument("--freq", default="M", help="'M' monthly or 'A' annual")
    ap.add_argument("--class", dest="cl", default="HS", help="Classification, e.g. 'HS'")
    ap.add_argument("--period", default="202401", help="YYYY for A, or YYYYMM[,YYYYMM...] for M")
    ap.add_argument("--reporter", default="156", help="Reporter code (156 = China)")
    ap.add_argument("--partner", default="0", help="Partner code (0 = World)")
    ap.add_argument("--flow", default="X", help="'X' exports, 'M' imports")
    ap.add_argument(
        "--cmd",
        default="TOTAL",
        help="'TOTAL' or comma-separated HS codes (e.g., '760120,260300')",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=12,
        help="Max periods per request (to keep URLs small).",
    )
    args = ap.parse_args()

    ensure_dirs()
    periods = _parse_periods(args.period)

    df = fetch_frame_batched(
        type_code=args.type,
        freq_code=args.freq,
        cl_code=args.cl,
        periods=periods,
        reporter=args.reporter,
        partner=args.partner,
        flow_code=args.flow,
        cmd_code=args.cmd,
        include_desc=True,
        batch_size=args.batch,
    )
    df["pulled_at"] = pd.Timestamp.utcnow()

    tag = (
        f"{args.type}{args.freq}_{args.cl}_"
        f"{'-'.join(periods)}_{args.reporter}_{args.partner}_{args.flow}_{args.cmd}"
    )
    out = DATA_WORK / f"comtrade_{tag}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} with {len(df):,} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
