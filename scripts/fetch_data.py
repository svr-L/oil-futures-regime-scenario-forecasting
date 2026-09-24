#!/usr/bin/env python
"""Download/cache all public data used by the project.

Main forecasting sample::

    python scripts/fetch_data.py

The script also keeps a longer WTI/Brent history *only* for structural-stability
work and downloads the EIA WTI C1--C4 curve plus Cushing inventories used by the
commodity-state extension.  Use ``--skip-eia`` when the EIA workbook endpoint is
unavailable; the core v0.9/v1.0 forecast study can still run from the Yahoo cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_futures_regime import (  # noqa: E402
    load_eia_cushing,
    load_eia_wti_curve,
    load_long_oil_history,
    load_or_download,
    roll_gap_diagnostics,
)

OIL_TICKERS = {"WTI": "CL=F", "BRENT": "BZ=F"}
STATE_TICKERS = {"VIX": "^VIX", "DXY": "DX-Y.NYB", "TNX": "^TNX"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2025-03-08", help="main-study end; yfinance end is exclusive")
    ap.add_argument("--stability-start", default="2008-01-01")
    ap.add_argument("--latest-end", default="2026-09-25",
                    help="separate latest snapshot for the frozen forward holdout")
    ap.add_argument("--anchor", default="W-FRI")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip-eia", action="store_true")
    args = ap.parse_args()

    oil = load_or_download("oil", OIL_TICKERS, args.start, args.end, args.anchor,
                           cache_dir=args.cache_dir, refresh=args.refresh, dropna="any")
    state = load_or_download("state", STATE_TICKERS, args.start, args.end, args.anchor,
                             cache_dir=args.cache_dir, refresh=args.refresh, dropna="all")
    long_oil = load_long_oil_history(args.stability_start, args.end, args.anchor,
                                     args.cache_dir, args.refresh)
    latest_oil = load_or_download("oil_latest", OIL_TICKERS, args.start, args.latest_end, args.anchor,
                                  cache_dir=args.cache_dir, refresh=args.refresh, dropna="any")
    latest_state = load_or_download("state_latest", STATE_TICKERS, args.start, args.latest_end, args.anchor,
                                    cache_dir=args.cache_dir, refresh=args.refresh, dropna="all")

    print(f"oil main      : {len(oil)} weeks  {oil.index.min().date()} -> {oil.index.max().date()}")
    print(f"state main    : {len(state)} weeks  {state.index.min().date()} -> {state.index.max().date()}")
    print(f"oil stability : {len(long_oil)} weeks  {long_oil.index.min().date()} -> {long_oil.index.max().date()}")
    print(f"oil latest    : {len(latest_oil)} weeks  {latest_oil.index.min().date()} -> {latest_oil.index.max().date()}")
    print(f"state latest  : {len(latest_state)} weeks  {latest_state.index.min().date()} -> {latest_state.index.max().date()}")

    if not args.skip_eia:
        curve = load_eia_wti_curve(args.cache_dir, refresh=args.refresh)
        cushing = load_eia_cushing(args.cache_dir, refresh=args.refresh)
        print(f"EIA WTI C1-C4 : {len(curve)} weeks  {curve.index.min().date()} -> {curve.index.max().date()}")
        print(f"EIA Cushing   : {len(cushing)} weeks  {cushing.index.min().date()} -> {cushing.index.max().date()}")

    flags = roll_gap_diagnostics(oil)
    print(f"\nroll/crisis diagnostic: {len(flags)} weekly moves beyond 4 robust sigma")
    if not flags.empty:
        print(flags.head(10).to_string(index=False))
    print(f"\ncache: {args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
