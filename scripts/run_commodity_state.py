#!/usr/bin/env python
"""Run the commodity-specific state-dependent WTI/Brent spread extension.

Requires caches created by ``scripts/fetch_data.py``.  The EIA WTI futures curve
ends in April 2024, so this is a historical mechanism test on the common sample,
not a live-signal claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_futures_regime import (  # noqa: E402
    DEFAULT_STATE_FEATURES,
    build_commodity_state,
    conditional_reversion_table,
    fit_state_dependent_spread,
    holm_adjust,
    interaction_wald,
    load_eia_cushing,
    load_eia_wti_curve,
    load_or_download,
    paired_score_table,
    rolling_state_dependent_oos,
)

OIL_TICKERS = {"WTI": "CL=F", "BRENT": "BZ=F"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "commodity_state"))
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2025-03-08")
    ap.add_argument("--oos-start", default="2018-01-01")
    ap.add_argument("--n-sim", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--synthetic", action="store_true", help="offline DGP check of the extension")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        rng = np.random.default_rng(20260924)
        idx = pd.date_range("2013-01-04", periods=620, freq="W-FRI")
        slope = 0.8 * np.sin(np.arange(len(idx)) / 17.0) + 0.25 * rng.normal(size=len(idx))
        curvature = 0.5 * np.sin(np.arange(len(idx)) / 9.0) + 0.20 * rng.normal(size=len(idx))
        inv_z = 0.7 * np.sin(np.arange(len(idx)) / 31.0) + 0.30 * rng.normal(size=len(idx))
        dinv = np.r_[0.0, np.diff(inv_z)]
        state = pd.DataFrame({
            "curve_slope_14": slope, "curve_curvature": curvature,
            "inventory_z_52": inv_z, "dlog_inventory": dinv,
        }, index=idx)
        x = np.zeros(len(idx))
        for t in range(1, len(idx)):
            phi = np.clip(0.84 + 0.10 * slope[t-1] - 0.05 * inv_z[t-1], 0.55, 0.98)
            x[t] = phi * x[t-1] + rng.normal(0, 0.018)
        spread = pd.Series(x, index=idx, name="SPREAD")
    else:
        oil = load_or_download("oil", OIL_TICKERS, args.start, args.end, "W-FRI",
                               cache_dir=args.cache_dir, refresh=False, dropna="any")
        curve = load_eia_wti_curve(args.cache_dir, refresh=False)
        cushing = load_eia_cushing(args.cache_dir, refresh=False)
        state = build_commodity_state(curve, cushing, inventory_release_lag_weeks=1)
        spread = (np.log(oil["WTI"]) - np.log(oil["BRENT"])).rename("SPREAD")

    common_end = min(state.dropna().index.max(), spread.index.max())
    full = fit_state_dependent_spread(spread.loc[:common_end], state.loc[:common_end],
                                      features=DEFAULT_STATE_FEATURES)
    wald = interaction_wald(full)
    rev = conditional_reversion_table(full)

    rows = rolling_state_dependent_oos(
        spread, state, features=DEFAULT_STATE_FEATURES,
        horizons=(1, 4, 12, 20), oos_start=args.oos_start,
        stride=args.stride, n_sim=args.n_sim,
    )
    if rows.empty:
        raise RuntimeError("commodity-state OOS produced no balanced origins")
    sig = paired_score_table(rows, baseline="SPREAD_AR1", stride=args.stride, n_boot=2000)
    challenger = sig[sig["model"] == "SD_EC"].copy()
    fam = holm_adjust({f"h{int(r.horizon_weeks)}": r.dm_pvalue for r in challenger.itertuples()})

    full_summary = pd.DataFrame([{
        "nobs": full.nobs,
        "interaction_wald": wald["wald_stat"],
        "interaction_df": wald["df"],
        "interaction_pvalue_hac": wald["p_value"],
        **{name: val for name, val in zip(full.param_names, full.params)},
    }])
    rows.to_csv(out / "forecast_rows.csv.gz", index=False, compression="gzip")
    sig.to_csv(out / "significance.csv", index=False)
    fam.to_csv(out / "horizon_family_holm.csv", index=False)
    rev.to_csv(out / "conditional_reversion.csv", index=False)
    full_summary.to_csv(out / "full_sample_fit.csv", index=False)
    state.to_csv(out / "commodity_state.csv")

    print("Commodity-state mechanism test")
    print(f"common sample through {common_end.date()} | OOS origins {rows['origin'].nunique()}")
    print(f"joint HAC Wald on spread×state interactions: p={wald['p_value']:.4f}")
    print(challenger[["horizon_weeks", "improvement_pct", "dm_pvalue"]].round(4).to_string(index=False))
    print("\nHolm across horizons")
    print(fam.to_string(index=False))
    print("\nConditional persistence")
    print(rev.round(4).to_string(index=False))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
