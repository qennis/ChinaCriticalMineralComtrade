# src/china_ir/comtrade.py
from __future__ import annotations

import os
from typing import Iterable, List

import pandas as pd
import requests

__all__ = ["fetch_period"]

_BASE = "https://comtradeapi.un.org/data/v1/get"
_TYP = "C"  # commodity trade
_CL = "HS"  # HS classification

_STD_COLS = (
    "period",
    "cmdCode",
    "primaryValue",
    "reporterCode",
    "partnerCode",
    "flowCode",
)


def _empty_std() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in _STD_COLS})


def _http_get(url: str, params: dict) -> dict | None:
    headers = {}
    api_key = os.environ.get("COMTRADE_API_KEY")
    if api_key:
        headers["Ocp-Apim-Subscription-Key"] = api_key
        params.setdefault("subscription-key", api_key)

    debug = os.environ.get("COMTRADE_DEBUG") == "1"
    try:
        r = requests.get(url, params=params, headers=headers, timeout=60)
        if debug:
            print("[comtrade] GET", r.url)
        if r.status_code == 404:
            if debug:
                print("[comtrade] 404")
            return None
        r.raise_for_status()
        j = r.json()
        if debug and isinstance(j, dict):
            v = j.get("validation") or {}
            print("[comtrade] validation:", {k: v.get(k) for k in ("count", "message", "status")})
            print("[comtrade] keys:", list(j.keys()))
        return j if isinstance(j, dict) else None
    except Exception as exc:  # noqa: BLE001
        if debug:
            print(f"[comtrade] request failed: {exc}")
        return None


def _extract_rows(payload: dict | None) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    for key in ("data", "dataset", "Data"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    return []


def _to_df(payload: dict | None) -> pd.DataFrame:
    rows = _extract_rows(payload)
    if not rows:
        return _empty_std()

    df = pd.json_normalize(rows)

    ren = {
        "rtCode": "reporterCode",
        "ptCode": "partnerCode",
        "Reporter.Code": "reporterCode",
        "Partner.Code": "partnerCode",
        "Trade.Value": "primaryValue",
    }
    for k, v in ren.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    for c in _STD_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    with pd.option_context("mode.copy_on_write", False):
        df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
        df["primaryValue"] = pd.to_numeric(df["primaryValue"], errors="coerce")

    cols = [c for c in _STD_COLS if c in df.columns]
    return df[cols + [c for c in df.columns if c not in cols]]


def fetch_period(
    freq: str,
    periods: List[str] | Iterable[str],
    reporter: str,
    partner: str,
    flow: str,
    cmd: str,
) -> pd.DataFrame:
    """
    Fetch a Comtrade slice.

    freq     : 'A' or 'M'
    periods  : iterable of 'YYYY' (A) or 'YYYYMM' (M)
    reporter : e.g., '156' (China)
    partner  : e.g., '0' (World)
    flow     : 'X' or 'M'
    cmd      : comma-separated HS6 (e.g. '380110,250410') or 'TOTAL'
    """
    url = f"{_BASE}/{_TYP}/{freq}/{_CL}"
    per = ",".join(str(p).strip() for p in periods if str(p).strip())
    params = {
        "reporterCode": reporter,
        "partnerCode": partner,
        "flowCode": flow,
        "cmdCode": cmd,
        "period": per,  # <-- period is a QUERY PARAM in v1
    }

    payload = _http_get(url, params)
    if payload is None:
        # Belt-and-suspenders fallback (some older proxies use timePeriod)
        params2 = dict(params)
        params2.pop("period", None)
        params2["timePeriod"] = per
        payload = _http_get(url, params2)

    return _to_df(payload)
