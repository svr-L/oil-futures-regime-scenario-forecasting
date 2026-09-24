"""Rolling pseudo-out-of-sample validation harness.

Three fixes over v0.3, all of which change the conclusions rather than just the
plumbing:

1. **Balanced panel.**  v0.3 wrapped each model in ``try/except: continue``, so a
   model that failed at some origins was scored on a different (and easier)
   subset of dates than its competitors -- and the origins that break a VECM are
   exactly the turbulent ones.  Here an origin is committed only if *every*
   model produced a forecast; otherwise the whole origin is dropped and the
   reason is recorded in a failure log instead of being swallowed silently.

2. **Common random numbers.**  One ``(horizon, n_sim)`` uniform array is drawn
   per origin and shared by all models, so paired score differences reflect
   model differences rather than independent Monte Carlo noise.

3. **External baselines.**  The grid now includes a driftless random walk and a
   filtered-historical-simulation AR-GARCH-t model, plus the missing
   ``VECM_equal`` cell and a no-recentering variant, so the ablation is a
   balanced design with a benchmark that can actually win.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kstest

from .common_trend import fit_cts_model  # noqa: F401  (registers the CTS simulator)
from .regimes import prepare_residual_pool
from .scenario_models import (
    common_uniforms,
    fit_fhs_model,
    fit_joint_ar_model,
    fit_rw_model,
    fit_var_levels_model,
    fit_vecm_scenario_model,
    simulate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def sample_crps(samples: np.ndarray, y: float) -> float:
    """CRPS of an empirical predictive distribution, O(n log n)."""
    x = np.sort(np.asarray(samples, dtype=float).ravel())
    n = len(x)
    if n == 0:
        return np.nan
    term1 = np.mean(np.abs(x - float(y)))
    coeff = 2.0 * np.arange(1, n + 1) - n - 1.0
    half_pairwise = float(np.sum(coeff * x)) / (n * n)
    return float(term1 - half_pairwise)


def quantile_loss(y: float, qhat: float, q: float) -> float:
    """Pinball / quantile loss."""
    e = float(y) - float(qhat)
    return float((q - (e < 0.0)) * e)


def empirical_pit(samples: np.ndarray, y: float) -> float:
    """Mid-rank PIT of an observation against Monte Carlo draws."""
    x = np.asarray(samples, dtype=float)
    less = np.sum(x < y)
    equal = np.sum(x == y)
    return float((less + 0.5 * equal) / len(x))


def _expected_shortfall(x: np.ndarray, q: float) -> float:
    tail = x[x <= q]
    return float(tail.mean()) if tail.size else float(q)


def evaluate_samples(samples: np.ndarray, y: float) -> dict:
    """Distributional diagnostics for one predictive sample against one outcome."""
    x = np.asarray(samples, dtype=float)
    q01, q05, q50, q95, q99 = np.quantile(x, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "actual": float(y),
        "crps": sample_crps(x, y),
        "pit": empirical_pit(x, y),
        "q01": float(q01),
        "q05": float(q05),
        "q50": float(q50),
        "q95": float(q95),
        "q99": float(q99),
        "es05": _expected_shortfall(x, q05),
        "es01": _expected_shortfall(x, q01),
        "covered_90": float(q05 <= y <= q95),
        "width_90": float(q95 - q05),
        "lower_tail_hit_05": float(y < q05),
        "qloss_05": quantile_loss(y, q05, 0.05),
        "qloss_50": quantile_loss(y, q50, 0.50),
        "qloss_95": quantile_loss(y, q95, 0.95),
    }


# ---------------------------------------------------------------------------
# Model grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One cell of the ablation: joint dynamics x probability scheme x sample.

    ``train_start`` discards everything before a fixed date (e.g. a structural
    break); ``window`` keeps only the most recent observations.  Both default to
    ``None``: an expanding window from the start of the sample, which is what
    every model in the main grid uses.
    """

    name: str
    family: str          # RW | AR | VECM | FHS | CTS | CTSF | VARL
    weight_mode: str     # equal | time | macro
    recenter: bool = True
    train_start: str | None = None
    window: int | None = None

    @property
    def sample_key(self) -> tuple:
        return (self.family, self.train_start, self.window)


DEFAULT_MODEL_SPECS: tuple = (
    # External benchmark: no dynamics at all.
    ModelSpec("RW_equal", "RW", "equal"),
    # Marginal AR + joint residual rows.
    ModelSpec("AR_equal", "AR", "equal"),
    ModelSpec("AR_time", "AR", "time"),
    ModelSpec("AR_macro", "AR", "macro"),
    # Ablates the re-centering convention rather than assuming it.
    ModelSpec("AR_macro_nc", "AR", "macro", recenter=False),
    # Cointegration-aware dynamics; full 2x3 grid including the missing equal cell.
    ModelSpec("VECM_equal", "VECM", "equal"),
    ModelSpec("VECM_time", "VECM", "time"),
    ModelSpec("VECM_macro", "VECM", "macro"),
    # External benchmark with conditional volatility.
    ModelSpec("FHS_equal", "FHS", "equal"),
    ModelSpec("FHS_time", "FHS", "time"),
    ModelSpec("FHS_macro", "FHS", "macro"),
    # Cointegration-consistent state space with the symmetric restriction
    # beta = [1, -1] imposed.
    ModelSpec("CTS_equal", "CTS", "equal"),
    ModelSpec("CTS_time", "CTS", "time"),
    ModelSpec("CTS_macro", "CTS", "macro"),
    # Same state space with the trend loading on Brent estimated freely, so the
    # cointegrating vector is [a, -1] rather than [1, -1].  The restriction is
    # rejected on WTI/Brent data (LR p ~ 2e-4) and imposing it leaves a slice of
    # the common trend inside the "spread", which inflates its persistence.
    ModelSpec("CTSF_equal", "CTSF", "equal"),
    ModelSpec("CTSF_time", "CTSF", "time"),
    ModelSpec("CTSF_macro", "CTSF", "macro"),
    # Sensitivity to the cointegration rank: the model Johansen implies when it
    # rejects r <= 1 as well as r = 0.
    ModelSpec("VARL_equal", "VARL", "equal"),
    ModelSpec("VARL_time", "VARL", "time"),
)

def stability_model_specs(break_date: str = "2015-12-18", window: int = 156) -> tuple:
    """Grid for the forecasting consequence of an unstable cointegrating relation.

    Kept separate from the main 19-model grid because this is a distinct,
    pre-specified hypothesis family about parameter instability rather than a new
    set of candidates for the primary forecast-selection exercise.  Its evidence
    is therefore reported with its own multiplicity control.

    * ``*_equal``  expanding window from the start of the sample (incumbent);
    * ``*_post``   estimated only on data after ``break_date``;
    * ``VECM_roll`` estimated on the most recent ``window`` weeks.

    If the relation genuinely changed at the break, the post-break models should
    win despite using less data; if it did not, discarding three years of data
    should cost accuracy.  Both outcomes are informative.
    """
    return (
        ModelSpec("RW_equal", "RW", "equal"),
        ModelSpec("VECM_equal", "VECM", "equal"),
        ModelSpec("VECM_post", "VECM", "equal", train_start=break_date),
        ModelSpec("VECM_roll", "VECM", "equal", window=window),
        ModelSpec("CTSF_equal", "CTSF", "equal"),
        ModelSpec("CTSF_post", "CTSF", "equal", train_start=break_date),
    )


FAST_MODEL_SPECS: tuple = tuple(
    s for s in DEFAULT_MODEL_SPECS if s.name in
    {"RW_equal", "AR_equal", "AR_macro", "VECM_equal", "CTS_equal", "CTSF_equal"}
)


@dataclass
class ValidationOutput:
    """Everything the harness produced, including what it could not produce."""

    results: pd.DataFrame
    failures: pd.DataFrame
    origins_used: list = field(default_factory=list)
    origins_dropped: list = field(default_factory=list)
    config: dict = field(default_factory=dict)

    @property
    def is_balanced(self) -> bool:
        if self.results.empty:
            return False
        counts = self.results.groupby("model", observed=True)["origin"].nunique()
        return int(counts.nunique()) == 1


def _choose_origins(index: pd.DatetimeIndex, oos_start, max_horizon: int, stride: int) -> list:
    index = pd.DatetimeIndex(index)
    eligible = np.where(index >= pd.Timestamp(oos_start))[0]
    eligible = eligible[eligible <= len(index) - 1 - max_horizon]
    return [index[i] for i in eligible[::stride]]


def _fit_families(train: pd.DataFrame, needed: set, cfg: dict) -> dict:
    """Fit one model object per family; raises so the caller can drop the origin."""
    fitted = {}
    if "RW" in needed:
        fitted["RW"] = fit_rw_model(train)
    if "AR" in needed or "FHS" in needed:
        ar = fit_joint_ar_model(train, p_max=cfg["p_max"])
        if "AR" in needed:
            fitted["AR"] = ar
    if "FHS" in needed:
        fitted["FHS"] = fit_fhs_model(train, p_max=cfg["p_max"])
    if "CTS" in needed:
        fitted["CTS"] = fit_cts_model(train, restrict_trend=True)
    if "CTSF" in needed:
        fitted["CTSF"] = fit_cts_model(train, restrict_trend=False)
    if "VARL" in needed:
        fitted["VARL"] = fit_var_levels_model(train, lags=cfg.get("var_lags", 1))
    if "VECM" in needed:
        fitted["VECM"] = fit_vecm_scenario_model(
            train,
            coint_rank=cfg["coint_rank"],
            k_ar_diff=cfg["k_ar_diff"],
            deterministic=cfg["deterministic"],
        )
    return fitted


def _spec_train(train: pd.DataFrame, spec: ModelSpec, min_window_obs: int) -> pd.DataFrame:
    """Apply a spec's estimation window to the expanding training sample."""
    out = train
    if spec.train_start is not None:
        out = out.loc[pd.Timestamp(spec.train_start):]
    if spec.window is not None:
        out = out.iloc[-int(spec.window):]
    if (spec.train_start is not None or spec.window is not None) and len(out) < min_window_obs:
        raise ValueError(f"{spec.name}: only {len(out)} observations in its estimation window")
    return out


def rolling_oos_validate(
    log_prices: pd.DataFrame,
    macro_features: pd.DataFrame,
    horizons=(1, 4, 12, 20),
    oos_start="2018-01-01",
    stride: int = 4,
    min_train_obs: int = 180,
    min_window_obs: int = 100,
    n_sim: int = 2_000,
    half_life: int = 52,
    p_max: int = 3,
    coint_rank: int = 1,
    k_ar_diff: int = 1,
    deterministic: str = "ci",
    restrict_trend: bool = True,
    var_lags: int = 1,
    min_macro_ess_fraction: float = 0.35,
    model_specs=DEFAULT_MODEL_SPECS,
    variables=("WTI", "BRENT", "SPREAD"),
    seed: int = 1234,
    verbose: bool = True,
) -> ValidationOutput:
    """Rolling pseudo-OOS ablation producing a *balanced* score panel.

    All estimation -- AR order selection, VECM, GARCH filter, macro
    standardization and kernel bandwidth -- uses only observations at or before
    each forecast origin.
    """
    Y = log_prices.dropna(how="any").copy()
    if not {"WTI", "BRENT"}.issubset(Y.columns):
        raise ValueError("log_prices must contain WTI and BRENT")

    horizons = sorted({int(h) for h in horizons})
    max_h = max(horizons)
    origins = _choose_origins(Y.index, oos_start, max_h, stride)
    cfg = {
        "p_max": p_max, "coint_rank": coint_rank,
        "k_ar_diff": k_ar_diff, "deterministic": deterministic,
        "restrict_trend": restrict_trend, "var_lags": var_lags,
    }

    rows: list = []
    failures: list = []
    used: list = []
    dropped: list = []

    for origin_no, origin in enumerate(origins):
        train = Y.loc[:origin]
        if len(train) < min_train_obs:
            dropped.append(origin)
            failures.append({"origin": origin, "model": "<all>", "stage": "min_train_obs",
                             "error": f"only {len(train)} training observations"})
            continue

        try:
            fitted = {}
            # Models sharing a family and an estimation sample are fitted once.
            for key in dict.fromkeys(spec.sample_key for spec in model_specs):
                family = key[0]
                rep = next(sp for sp in model_specs if sp.sample_key == key)
                sub = _spec_train(train, rep, min_window_obs)
                fitted[key] = _fit_families(sub, {family}, cfg)[family]
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            dropped.append(origin)
            failures.append({"origin": origin, "model": "<all>", "stage": "fit",
                             "error": f"{type(exc).__name__}: {exc}"})
            logger.warning("origin %s dropped during fit: %s", origin.date(), exc)
            continue

        # Common random numbers: identical uniforms for every model at this origin.
        u = common_uniforms(max_h, n_sim, seed + origin_no)

        pos = Y.index.get_loc(origin)
        origin_rows: list = []
        origin_failed = False

        for spec in model_specs:
            model = fitted[spec.sample_key]
            try:
                pool, prob, wdiag = prepare_residual_pool(
                    model.residuals,
                    mode=spec.weight_mode,
                    half_life=half_life,
                    macro_features=macro_features,
                    origin=origin,
                    min_macro_ess_fraction=min_macro_ess_fraction,
                    recenter=spec.recenter,
                )
                paths = simulate(model, pool, prob, horizon=max_h, n_sim=n_sim, u=u)
            except Exception as exc:  # noqa: BLE001
                origin_failed = True
                failures.append({"origin": origin, "model": spec.name, "stage": "simulate",
                                 "error": f"{type(exc).__name__}: {exc}"})
                logger.warning("origin %s model %s failed: %s", origin.date(), spec.name, exc)
                break

            col_idx = {c: j for j, c in enumerate(model.columns)}
            for h in horizons:
                target_date = Y.index[pos + h]
                actual = Y.iloc[pos + h]
                sim = {
                    "WTI": paths[h, :, col_idx["WTI"]],
                    "BRENT": paths[h, :, col_idx["BRENT"]],
                }
                sim["SPREAD"] = sim["WTI"] - sim["BRENT"]
                realized = {
                    "WTI": float(actual["WTI"]),
                    "BRENT": float(actual["BRENT"]),
                }
                realized["SPREAD"] = realized["WTI"] - realized["BRENT"]

                for variable in variables:
                    row = {
                        "origin": pd.Timestamp(origin),
                        "target_date": pd.Timestamp(target_date),
                        "horizon_weeks": h,
                        "model": spec.name,
                        "family": spec.family,
                        "variable": variable,
                        "weight_mode": spec.weight_mode,
                        "recentered": spec.recenter,
                        "train_start": spec.train_start,
                        "window": spec.window,
                        "weight_ess": wdiag.ess,
                        "weight_ess_fraction": wdiag.ess_fraction,
                        "bandwidth": wdiag.bandwidth,
                        "bandwidth_binding": wdiag.bandwidth_binding,
                        "n_pool": len(pool),
                    }
                    row.update(evaluate_samples(sim[variable], realized[variable]))
                    origin_rows.append(row)

        if origin_failed:
            # Drop the entire origin so every model is scored on the same dates.
            dropped.append(origin)
            continue

        rows.extend(origin_rows)
        used.append(origin)
        if verbose and (origin_no % 10 == 0):
            print(f"  origin {origin.date()}  ({len(used)} kept / {len(dropped)} dropped)", flush=True)

    results = pd.DataFrame(rows)
    out = ValidationOutput(
        results=results,
        failures=pd.DataFrame(failures),
        origins_used=used,
        origins_dropped=dropped,
        config={
            "horizons": horizons, "oos_start": str(oos_start), "stride": stride,
            "n_sim": n_sim, "half_life": half_life, "p_max": p_max,
            "coint_rank": coint_rank, "k_ar_diff": k_ar_diff,
            "deterministic": deterministic, "restrict_trend": restrict_trend,
            "var_lags": var_lags,
            "min_macro_ess_fraction": min_macro_ess_fraction,
            "models": [s.name for s in model_specs], "seed": seed,
            "min_window_obs": min_window_obs,
            "windows": {s.name: {"train_start": s.train_start, "window": s.window}
                        for s in model_specs if s.train_start or s.window},
            "n_origins_used": len(used), "n_origins_dropped": len(dropped),
        },
    )
    if not results.empty and not out.is_balanced:  # pragma: no cover - defensive
        raise RuntimeError("panel is unbalanced; this should be impossible")
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_oos(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CRPS, calibration and sharpness by model / variable / horizon."""
    if results.empty:
        return results.copy()
    grp = results.groupby(["model", "variable", "horizon_weeks"], observed=True)
    return grp.agg(
        n_forecasts=("crps", "size"),
        mean_crps=("crps", "mean"),
        median_crps=("crps", "median"),
        coverage_90=("covered_90", "mean"),
        avg_width_90=("width_90", "mean"),
        lower_tail_rate_05=("lower_tail_hit_05", "mean"),
        mean_qloss_05=("qloss_05", "mean"),
        mean_qloss_50=("qloss_50", "mean"),
        mean_qloss_95=("qloss_95", "mean"),
        mean_ess_fraction=("weight_ess_fraction", "mean"),
    ).reset_index()


def pit_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Descriptive PIT-uniformity summary.

    The KS p-values are *not* valid inference: PIT values from overlapping
    forecast windows are serially dependent, which makes the test far too
    liberal.  Read the statistic and the PIT mean/variance as descriptive
    calibration evidence and use the DM tests in
    :mod:`oil_futures_regime.significance` for actual model comparison.
    """
    if results.empty:
        return results.copy()
    rows = []
    for keys, g in results.groupby(["model", "variable", "horizon_weeks"], observed=True):
        pit = g["pit"].dropna().values
        if len(pit) == 0:
            continue
        ks = kstest(pit, "uniform")
        rows.append({
            "model": keys[0], "variable": keys[1], "horizon_weeks": keys[2],
            "n": len(pit),
            "pit_mean": float(np.mean(pit)),
            "pit_var": float(np.var(pit, ddof=1)) if len(pit) > 1 else np.nan,
            "ks_uniform_stat": float(ks.statistic),
            "ks_uniform_pvalue_descriptive": float(ks.pvalue),
        })
    return pd.DataFrame(rows)
