#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

import pandas as pd
import requests

from china_ir.paths import DATA_WORK, ensure_dirs

# Classic v1 endpoint; we request JSON explicitly
DEFAULT_BASE = os.getenv("COMTRADE_API_BASE", "https://comtrade.un.org/api/get")
API_TOKEN = os.getenv("COMTRADE_API_TOKEN", None)

RETRY_STATUSES = {429, 500, 502, 503, 504}


def _json_or_raise(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        txt = resp.text[:300].replace("\n", " ")
        raise RuntimeError(f"Non-JSON response (status {resp.status_code}): {txt}")


def fetch_frame(px: str, years: str, reporter: str, partner: str, flow: str, hs: str | None):
    params = {
        "max": 50000,  # page size; v1 caps at 50k
        "type": "C",  # trade data
        "freq": "M",
        "px": px,  # "HS"
        "ps": years,  # e.g. "2022,2023,2024,2025"
        "r": reporter,  # 156 = China
        "p": partner,  # 0 = World (use partner codes later)
        "rg": flow,  # 2 = exports, 1 = imports
        "cc": hs or "ALL",
        "fmt": "json",  # force JSON
    }
    headers = {}
    if API_TOKEN:
        headers["X-API-Key"] = API_TOKEN

    # very light retry loop
    for attempt in range(5):
        resp = requests.get(DEFAULT_BASE, params=params, headers=headers, timeout=60)
        if resp.status_code in RETRY_STATUSES:
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        js = _json_or_raise(resp)
        if "dataset" not in js:
            raise RuntimeError(f"No 'dataset' in response keys: {list(js.keys())}")
        return pd.DataFrame(js["dataset"])

    raise RuntimeError("Failed after retries; server kept returning transient errors.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", default="HS")
    ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--reporter", default="156")  # China
    ap.add_argument("--partner", default="0")  # World aggregate
    ap.add_argument("--flow", default="2")  # 2 = exports
    ap.add_argument("--hs", default=None)  # e.g., "760120" or "ALL"
    args = ap.parse_args()

    ensure_dirs()
    df = fetch_frame(args.px, args.years, args.reporter, args.partner, args.flow, args.hs)
    df["pulled_at"] = pd.Timestamp.utcnow()
    tag = f"{args.px}_{args.years.replace(',', '-')}_{args.reporter}_{args.partner}_{args.flow}"
    out = DATA_WORK / f"comtrade_{tag}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} with {len(df):,} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
