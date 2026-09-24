#!/usr/bin/env python
"""Run the rolling pseudo-OOS ablation and write every report artefact to disk.

Real data (requires ``scripts/fetch_data.py`` to have been run once)::

    python scripts/run_validation.py --out reports/oos

Offline reproducibility check on simulated data with a known data-generating
process (no network needed, used by CI and by the committed demo)::

    python scripts/run_validation.py --synthetic --fast --out reports/synthetic_demo

Artefacts written to ``--out``:

======================  ====================================================
``forecast_rows.csv``   one row per origin x model x horizon x variable
``summary.csv``         mean CRPS, coverage, width, pinball by cell
``significance.csv``    DM tests + block-bootstrap CIs vs the baseline
``best_models.csv``     winner per variable/horizon and whether it is significant
``pit.csv``             descriptive PIT uniformity summary
``var_es.csv``          Kupiec / Christoffersen / ES Z2 backtests
``failures.csv``        every dropped origin with the reason
``config.json``         full run configuration
======================  ====================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oil_futures_regime import (  # noqa: E402
    DEFAULT_MODEL_SPECS,
    mcs_by_cell,
    mcs_summary,
    FAST_MODEL_SPECS,
    aggregate_oos,
    best_model_summary,
    build_macro_features,
    load_or_download,
    paired_score_table,
    pit_summary,
    rolling_oos_validate,
    var_es_report,
)

OIL_TICKERS = {"WTI": "CL=F", "BRENT": "BZ=F"}
STATE_TICKERS = {"VIX": "^VIX", "DXY": "DX-Y.NYB", "TNX": "^TNX"}


def synthetic_panel(n: int = 620, seed: int = 11, break_date: str | None = None):
    """Cointegrated WTI/Brent with GARCH volatility plus a correlated macro state.

    ``break_date`` plants a structural change in the spread shaped like the one
    the export-ban repeal should have produced: the mean WTI discount narrows
    (-0.08 to -0.03) and deviations revert faster (persistence 0.95 to 0.80).
    With the default ``None`` the generator is exactly the original one, so seeds
    reproduce previously committed results.

    The DGP is deliberately one where the *right* answers are known: there is a
    stationary spread (so VECM should help the spread), volatility clusters (so
    FHS should help short horizons) and the macro state is genuinely informative
    about volatility (so macro conditioning should help a little).  If the
    harness cannot recover that ordering, the harness is broken.
    """
    rng = np.random.default_rng(seed)
    sigma2 = np.zeros(n)
    sigma2[0] = 0.03 ** 2
    omega, a, b = 0.03 ** 2 * 0.05, 0.10, 0.85
    shock = np.zeros(n)
    for t in range(1, n):
        sigma2[t] = omega + a * shock[t - 1] ** 2 + b * sigma2[t - 1]
        shock[t] = np.sqrt(sigma2[t]) * rng.standard_t(6) / np.sqrt(6 / 4)

    common = np.cumsum(0.0008 + shock) + 4.2
    idx = pd.date_range("2013-01-04", periods=n, freq="W-FRI")
    spread = np.zeros(n)
    if break_date is None:
        for t in range(1, n):
            spread[t] = 0.86 * spread[t - 1] + rng.normal(0.0, 0.018)
    else:
        b = int(np.searchsorted(idx, pd.Timestamp(break_date)))
        mean = np.where(np.arange(n) < b, -0.08, -0.03)
        rho = np.where(np.arange(n) < b, 0.95, 0.80)
        spread[0] = mean[0]
        for t in range(1, n):
            spread[t] = mean[t] + rho[t] * (spread[t - 1] - mean[t]) + rng.normal(0.0, 0.018)

    oil = pd.DataFrame(
        {"WTI": np.exp(common + 0.5 * spread), "BRENT": np.exp(common - 0.5 * spread)},
        index=idx,
    )

    # VIX-like state driven by the same volatility process, plus noise.
    vix = 12.0 + 700.0 * np.sqrt(sigma2) + rng.normal(0, 1.2, n).cumsum() * 0.0
    vix = np.clip(vix + rng.normal(0, 1.0, n), 8.0, None)
    dxy = 90.0 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    tnx = np.clip(20.0 + np.cumsum(rng.normal(0, 0.35, n)), 3.0, None)
    state = pd.DataFrame({"VIX": vix, "DXY": dxy, "TNX": tnx}, index=idx)
    return oil, state


def write_markdown_report(out: Path, vo, summary, sig, best, risk, baseline,
                          sig_vs_vecm=None, mcs_view=None, label: str = "") -> None:
    """Compact human-readable results file so the repo shows conclusions, not just code."""
    lines = [f"# Rolling pseudo-OOS results ({label})", ""]
    cfg = vo.config
    lines += [
        f"- origins used: **{cfg['n_origins_used']}** (dropped: {cfg['n_origins_dropped']})",
        f"- horizons: {cfg['horizons']} weeks, stride {cfg['stride']} weeks, "
        f"{cfg['n_sim']} scenarios per forecast",
        f"- models: {', '.join(cfg['models'])}",
        f"- baseline for all comparisons: `{baseline}`",
        f"- balanced panel: **{vo.is_balanced}**",
        "",
        "All models are scored on identical origins with common random numbers. "
        "p-values are Diebold-Mariano with Newey-West HAC variance at the overlap "
        "lag implied by horizon and stride, plus the HLN small-sample correction.",
        "",
        "## Mean CRPS (lower is better)",
        "",
    ]
    for variable in sorted(summary["variable"].unique()):
        piv = (summary[summary["variable"] == variable]
               .pivot(index="model", columns="horizon_weeks", values="mean_crps")
               .round(5))
        lines += [f"### {variable}", "", piv.to_markdown(), ""]

    if mcs_view is not None and not mcs_view.empty:
        lines += ["## Model Confidence Set (90%)", "",
                  "How many models survive per cell. Where most of them survive, the",
                  "data cannot separate the models and no winner should be claimed.", "",
                  mcs_view.to_markdown(index=False), ""]

    lines += ["## Winner per cell", "",
              best.round(4).to_markdown(index=False), "",
              "## Statistically significant improvements over the baseline", ""]
    winners = sig[(sig["significant_5pct"]) & (sig["improvement_pct"] > 0)]
    if winners.empty:
        lines += ["None: no model beats the baseline beyond sampling noise.", ""]
    else:
        cols = ["variable", "horizon_weeks", "model", "improvement_pct",
                "dm_stat", "dm_pvalue", "eff_independent_n"]
        lines += [winners[cols].round(4).to_markdown(index=False), ""]

    if sig_vs_vecm is not None and not sig_vs_vecm.empty:
        lines += ["## State space vs VECM on the spread (baseline `VECM_time`)", "",
                  "The common-trend model forecasts from the *filtered* relative",
                  "component; the VECM error-correction term is a function of the raw",
                  "observed prices and therefore inherits their measurement noise.", ""]
        sel = sig_vs_vecm[(sig_vs_vecm["variable"] == "SPREAD") &
                          (sig_vs_vecm["model"].str.startswith("CTS_"))]
        cols = ["horizon_weeks", "model", "mean_crps", "improvement_pct",
                "dm_stat", "dm_pvalue"]
        lines += [sel[cols].round(4).to_markdown(index=False), ""]

    lines += ["## 90% interval calibration", ""]
    cov = (summary.pivot_table(index="model", columns=["variable", "horizon_weeks"],
                               values="coverage_90").round(3))
    lines += [cov.to_markdown(), "",
              "## VaR / ES backtests (5% level, overlap-thinned)", ""]
    r5 = risk[(risk["alpha"] == 0.05) & (~risk["underpowered"])]
    if r5.empty:
        lines += ["All cells underpowered at this sample size.", ""]
    else:
        cols = ["variable", "horizon_weeks", "model", "n_used", "exception_rate",
                "kupiec_p", "christoffersen_cc_p", "es_z2", "es_z2_p"]
        lines += [r5[cols].round(4).to_markdown(index=False), ""]

    (out / "RESULTS.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "reports" / "oos"))
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--end", default="2025-03-08")
    ap.add_argument("--anchor", default="W-FRI")
    ap.add_argument("--oos-start", default="2018-01-01")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--horizons", default="1,4,12,20")
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--half-life", type=int, default=52)
    ap.add_argument("--min-macro-ess-fraction", type=float, default=0.35)
    ap.add_argument("--baseline", default="RW_equal")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--fast", action="store_true", help="reduced model grid")
    ap.add_argument("--synthetic", action="store_true", help="simulated data, no network")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    horizons = [int(h) for h in args.horizons.split(",")]
    specs = FAST_MODEL_SPECS if args.fast else DEFAULT_MODEL_SPECS

    if args.synthetic:
        oil, state = synthetic_panel()
        oos_start = oil.index[260]
        print(f"Synthetic panel: {oil.shape[0]} weeks, OOS from {oos_start.date()}")
    else:
        oil = load_or_download("oil", OIL_TICKERS, args.start, args.end, args.anchor,
                               cache_dir=args.cache_dir, dropna="any")
        state = load_or_download("state", STATE_TICKERS, args.start, args.end, args.anchor,
                                 cache_dir=args.cache_dir, dropna="all")
        oos_start = args.oos_start
        print(f"Cached panel: {oil.shape[0]} weeks "
              f"{oil.index.min().date()} -> {oil.index.max().date()}")

    logp = np.log(oil[["WTI", "BRENT"]]).dropna()
    macro = build_macro_features(state)

    print(f"Models: {[s.name for s in specs]}")
    print("Running rolling validation ...")
    vo = rolling_oos_validate(
        log_prices=logp,
        macro_features=macro,
        horizons=horizons,
        oos_start=oos_start,
        stride=args.stride,
        n_sim=args.n_sim,
        half_life=args.half_life,
        min_macro_ess_fraction=args.min_macro_ess_fraction,
        model_specs=specs,
        seed=args.seed,
    )

    print(f"\nOrigins kept: {len(vo.origins_used)}   dropped: {len(vo.origins_dropped)}")
    print(f"Balanced panel: {vo.is_balanced}")
    if vo.results.empty:
        print("No results produced; see failures.csv")
        vo.failures.to_csv(out / "failures.csv", index=False)
        return 1

    summary = aggregate_oos(vo.results)
    sig = paired_score_table(vo.results, baseline=args.baseline, stride=args.stride)
    best = best_model_summary(sig)
    pit = pit_summary(vo.results)
    risk = var_es_report(vo.results, stride=args.stride)
    mcs_table = mcs_by_cell(vo.results, stride=args.stride, n_boot=min(args.n_boot, 2000))
    mcs_view = mcs_summary(mcs_table)
    mcs_table.to_csv(out / "mcs.csv", index=False)
    mcs_view.to_csv(out / "mcs_summary.csv", index=False)

    # Head-to-head: does the filtered state space beat the raw error-correction
    # model on the relative component it was built to describe?
    sig_vs_vecm = None
    models_present = set(vo.results["model"].unique())
    if {"CTS_time", "VECM_time"}.issubset(models_present):
        sig_vs_vecm = paired_score_table(
            vo.results[vo.results["model"].str.startswith(("CTS_", "VECM_"))],
            baseline="VECM_time", stride=args.stride,
        )
        sig_vs_vecm.to_csv(out / "significance_cts_vs_vecm.csv", index=False)

    vo.results.to_csv(out / "forecast_rows.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "summary.csv", index=False)
    sig.to_csv(out / "significance.csv", index=False)
    best.to_csv(out / "best_models.csv", index=False)
    pit.to_csv(out / "pit.csv", index=False)
    risk.to_csv(out / "var_es.csv", index=False)
    vo.failures.to_csv(out / "failures.csv", index=False)
    (out / "config.json").write_text(json.dumps(vo.config, indent=2, default=str))
    write_markdown_report(
        out, vo, summary, sig, best, risk, args.baseline, sig_vs_vecm, mcs_view,
        label="simulated data" if args.synthetic else "WTI/Brent weekly, cached Yahoo data",
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print("\n=== Mean CRPS (lower is better) ===")
    for variable in sorted(summary["variable"].unique()):
        piv = (summary[summary["variable"] == variable]
               .pivot(index="model", columns="horizon_weeks", values="mean_crps"))
        print(f"\n{variable}")
        print(piv.round(5).to_string())

    print("\n=== Model Confidence Set (90%): models that cannot be separated ===")
    print(mcs_view[["variable", "horizon_weeks", "n_in_mcs_90", "n_models",
                    "best_model"]].to_string(index=False))

    print(f"\n=== Winner per cell (vs baseline {args.baseline}) ===")
    print(best.round(4).to_string(index=False))

    print("\n=== Significant improvements vs baseline (DM p < 0.05) ===")
    winners = sig[(sig["significant_5pct"]) & (sig["improvement_pct"] > 0)]
    if winners.empty:
        print("None. No model beats the baseline beyond sampling noise.")
    else:
        print(winners[["variable", "horizon_weeks", "model", "improvement_pct",
                       "dm_stat", "dm_pvalue", "eff_independent_n"]]
              .round(4).to_string(index=False))

    print(f"\nArtefacts written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
