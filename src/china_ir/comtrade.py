# src/china_ir/comtrade.py
from __future__ import annotations

import os
from typing import List  # Iterable

import pandas as pd
import requests

__all__ = ["fetch_period"]

_BASE = "https://comtradeapi.un.org/data/v1/get"
_TYP = "C"  # commodity trade

_STD_COLS = (
    "period",
    "cmdCode",
    "primaryValue",
    "reporterCode",
    "partnerCode",
    "flowCode",
)


def _http_get(url: str, params: dict) -> dict | None:
    headers = {}
    api_key = os.environ.get("COMTRADE_API_KEY")
    if api_key:
        headers["Ocp-Apim-Subscription-Key"] = api_key
        params.setdefault("subscription-key", api_key)

    debug = os.environ.get("COMTRADE_DEBUG") == "1"
    try:
        if debug:
            print(f"[comtrade] GET {url} PARAMS: {params}")

        r = requests.get(url, params=params, headers=headers, timeout=60)

        if debug:
            print(f"[comtrade] STATUS: {r.status_code}")
            if r.status_code != 200:
                print(f"[comtrade] ERROR: {r.text}")

        if r.status_code == 404:
            return None

        r.raise_for_status()
        j = r.json()
        return j if isinstance(j, dict) else None

    except Exception as exc:
        if debug:
            print(f"[comtrade] request failed: {exc}")
        return None


def _to_df(payload: dict | None) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows)

    # Normalize Columns
    ren = {
        "rtCode": "reporterCode",
        "ptCode": "partnerCode",
        "cmdCode": "cmdCode",
        "primaryValue": "primaryValue",
    }
    df.rename(columns=ren, inplace=True)

    # Ensure specific columns exist
    for c in _STD_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    return df


def fetch_period(
    freq: str,
    periods: List[str],
    reporter: str,
    partner: str,
    flow: str,
    cmd: str,
    classification: str = "HS",  # Unused in URL logic below, but kept for signature compat
) -> pd.DataFrame:

    url = f"{_BASE}/{_TYP}/{freq}/HS"  # Always force HS for stability

    params = {
        "reporterCode": reporter,
        "partnerCode": partner if partner else 0,
        "flowCode": flow,
        "cmdCode": cmd,
        "period": ",".join(periods),
    }

    return _to_df(_http_get(url, params))
