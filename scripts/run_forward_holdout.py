#!/usr/bin/env python
"""Evaluate the frozen v1.0 procedure on observations after 7 March 2025.

No model/rule selection should be changed after inspecting this output.  This is
an external forward validation of the *procedure*: at each post-freeze origin the
same model grid is re-estimated using information then available, exactly as it
would have been in live use.
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
    DEFAULT_MODEL_SPECS, DEFAULT_RULES, aggregate_oos, baseline_in_mcs,
    build_macro_features, evaluate_rules, load_or_download, mcs_by_cell,
    mcs_summary, paired_score_table, rolling_oos_validate,
)

OIL_TICKERS = {"WTI": "CL=F", "BRENT": "BZ=F"}
STATE_TICKERS = {"VIX": "^VIX", "DXY": "DX-Y.NYB", "TNX": "^TNX"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "forward_holdout"))
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2026-09-25")
    ap.add_argument("--freeze-date", default="2025-03-07")
    ap.add_argument("--n-sim", type=int, default=5000)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    oil = load_or_download("oil_latest", OIL_TICKERS, args.start, args.end, "W-FRI",
                           cache_dir=args.cache_dir, refresh=False, dropna="any")
    state_raw = load_or_download("state_latest", STATE_TICKERS, args.start, args.end, "W-FRI",
                                 cache_dir=args.cache_dir, refresh=False, dropna="all")
    logp = np.log(oil[["WTI", "BRENT"]])
    macro = build_macro_features(state_raw)
    vo = rolling_oos_validate(
        logp, macro, horizons=(1, 4, 12, 20),
        oos_start=str((pd.Timestamp(args.freeze_date) + pd.Timedelta(days=7)).date()),
        stride=4, n_sim=args.n_sim, model_specs=DEFAULT_MODEL_SPECS, verbose=True,
    )
    if vo.results.empty:
        raise RuntimeError("no forward-holdout origins available; refresh latest caches first")
    summary = aggregate_oos(vo.results)
    sig = paired_score_table(vo.results, baseline="RW_equal", stride=4, n_boot=2000)
    vo.results.to_csv(out / "forecast_rows.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "summary.csv", index=False)
    sig.to_csv(out / "significance.csv", index=False)
    vo.failures.to_csv(out / "failures.csv", index=False)

    # Apply the same frozen selection-free / rule-level inference when the forward
    # block is large enough.  With too few origins these procedures correctly refuse
    # to manufacture precision; the scored panel is still written for later updates.
    try:
        mcs = mcs_by_cell(vo.results, stride=4, n_boot=2000)
        mcs.to_csv(out / "mcs.csv", index=False)
        mcs_summary(mcs).to_csv(out / "mcs_summary.csv", index=False)
        baseline = baseline_in_mcs(mcs, baseline="RW_equal")
        baseline.to_csv(out / "baseline_vs_mcs.csv", index=False)
    except ValueError as exc:
        baseline = None
        print(f"MCS not reported: {exc}")

    try:
        rule_summary, rule_detail = evaluate_rules(
            vo.results, rules=DEFAULT_RULES, stride=4, n_boot=2000, seed=2026
        )
        rule_summary.to_csv(out / "decision_rule_summary.csv", index=False)
        rule_detail.to_csv(out / "decision_rule_cells.csv", index=False)
    except ValueError as exc:
        print(f"rule-level inference not reported: {exc}")

    print(f"forward origins: {vo.results['origin'].nunique()} | balanced={vo.is_balanced}")
    focus = sig[(sig.variable == 'SPREAD') &
                (sig.model.isin(['CTSF_equal','VECM_equal','VARL_equal']))]
    print(focus[["horizon_weeks","model","improvement_pct","dm_pvalue"]]
          .round(4).to_string(index=False))
    if baseline is not None:
        print("\nRW excluded from 90% MCS:")
        print(baseline[baseline["something_beats_baseline"]]
              [["variable","horizon_weeks"]].to_string(index=False))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
