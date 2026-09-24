"""Statistical significance for forecast-score comparisons.

With weekly origins spaced ``stride`` apart and horizons up to 20 weeks, the
forecast targets overlap heavily: at h=20 and stride=4 the effective number of
independent evaluations is roughly ``n_origins * stride / h``.  Raw mean-CRPS
leaderboards therefore rank models on differences that are frequently inside the
noise band.

This module supplies the two standard corrections:

* :func:`diebold_mariano` -- paired test on score differentials with a
  Newey-West HAC variance and the Harvey-Leybourne-Newbold small-sample
  correction, referred to a t distribution;
* :func:`block_bootstrap_mean_ci` -- circular moving-block bootstrap confidence
  interval for the mean differential, which makes no asymptotic-normality claim.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def overlap_lag(horizon: int, stride: int) -> int:
    """Number of *origin steps* over which forecast errors overlap.

    Two origins separated by fewer than ``horizon`` calendar periods share part
    of their target window.  In origin-index units that is
    ``ceil(horizon / stride) - 1``.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    return max(int(math.ceil(horizon / stride)) - 1, 0)


def newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Long-run variance of a series with Bartlett (Newey-West) weights."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    dc = d - d.mean()
    gamma0 = float(dc @ dc) / n
    lrv = gamma0
    for j in range(1, min(lag, n - 1) + 1):
        gamma_j = float(dc[j:] @ dc[:-j]) / n
        lrv += 2.0 * (1.0 - j / (lag + 1.0)) * gamma_j
    return lrv


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
    stride: int = 1,
    lag: int | None = None,
    hln_correction: bool = True,
) -> dict:
    """Paired Diebold-Mariano test on ``loss_a - loss_b``.

    A negative statistic means model A has the lower average loss (A is better).

    The HAC truncation lag defaults to the origin-space overlap implied by
    ``horizon`` and ``stride``.  ``hln_correction`` applies the Harvey-
    Leybourne-Newbold finite-sample adjustment and refers the statistic to a
    t distribution with ``n - 1`` degrees of freedom.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must be aligned and equally long")
    d = a - b
    mask = np.isfinite(d)
    d = d[mask]
    n = len(d)
    if n < 8:
        return {"n": n, "mean_diff": float(np.mean(d)) if n else np.nan,
                "dm_stat": np.nan, "p_value": np.nan, "hac_lag": np.nan}

    L = overlap_lag(horizon, stride) if lag is None else int(lag)
    lrv = newey_west_lrv(d, L)
    if not np.isfinite(lrv) or lrv <= 0:
        return {"n": n, "mean_diff": float(d.mean()), "dm_stat": np.nan,
                "p_value": np.nan, "hac_lag": L}

    dm = d.mean() / math.sqrt(lrv / n)
    h_steps = L + 1
    if hln_correction:
        adj = (n + 1 - 2 * h_steps + h_steps * (h_steps - 1) / n) / n
        adj = max(adj, 1e-8)
        dm *= math.sqrt(adj)
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    return {"n": n, "mean_diff": float(d.mean()), "dm_stat": float(dm),
            "p_value": float(p), "hac_lag": L}


def block_bootstrap_mean_ci(
    d: np.ndarray,
    block_length: int | None = None,
    horizon: int = 1,
    stride: int = 1,
    n_boot: int = 5_000,
    alpha: float = 0.05,
    seed: int = 20260101,
) -> dict:
    """Circular moving-block bootstrap CI for the mean of a dependent series."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return {"mean": float(np.mean(d)) if n else np.nan, "lo": np.nan,
                "hi": np.nan, "block_length": np.nan, "p_two_sided": np.nan}

    if block_length is None:
        block_length = max(overlap_lag(horizon, stride) + 1, 2)
    block_length = int(min(block_length, n))
    n_blocks = int(math.ceil(n / block_length))

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    means = d[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)

    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    centered = means - means.mean()
    p = float(np.mean(np.abs(centered) >= abs(d.mean())))
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "block_length": block_length, "p_two_sided": p}


def paired_score_table(
    results: pd.DataFrame,
    baseline: str = "RW_equal",
    score: str = "crps",
    stride: int = 4,
    n_boot: int = 2_000,
    seed: int = 20260101,
) -> pd.DataFrame:
    """Leaderboard with DM tests and bootstrap CIs against a single baseline.

    ``results`` must be the balanced panel returned by
    :func:`oil_futures_regime.validation.rolling_oos_validate`, i.e. every model
    evaluated on exactly the same origins.
    """
    required = {"model", "variable", "horizon_weeks", "origin", score}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results is missing columns: {sorted(missing)}")

    rows = []
    for (variable, horizon), g in results.groupby(["variable", "horizon_weeks"], observed=True):
        wide = g.pivot(index="origin", columns="model", values=score).dropna(how="any")
        if baseline not in wide.columns:
            raise ValueError(f"baseline {baseline!r} not present for {variable} h={horizon}")
        base = wide[baseline].values
        for model in sorted(wide.columns):
            d = wide[model].values - base
            dm = diebold_mariano(wide[model].values, base, horizon=int(horizon), stride=stride)
            bs = block_bootstrap_mean_ci(
                d, horizon=int(horizon), stride=stride, n_boot=n_boot, seed=seed
            )
            base_mean = float(np.mean(base))
            rows.append({
                "variable": variable,
                "horizon_weeks": int(horizon),
                "model": model,
                "n_origins": int(len(wide)),
                "eff_independent_n": round(
                    min(len(wide), len(wide) * stride / max(int(horizon), 1)), 1
                ),
                f"mean_{score}": float(np.mean(wide[model].values)),
                f"delta_vs_{baseline}": float(np.mean(d)),
                "improvement_pct": 100.0 * (base_mean - float(np.mean(wide[model].values))) / base_mean
                if base_mean else np.nan,
                "dm_stat": dm["dm_stat"],
                "dm_pvalue": dm["p_value"],
                "hac_lag": dm["hac_lag"],
                "boot_lo": bs["lo"],
                "boot_hi": bs["hi"],
                "significant_5pct": bool(np.isfinite(dm["p_value"]) and dm["p_value"] < 0.05),
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["variable", "horizon_weeks", f"mean_{score}"]).reset_index(drop=True)


def best_model_summary(table: pd.DataFrame, score: str = "crps") -> pd.DataFrame:
    """Per variable/horizon: best model and whether it beats the baseline significantly."""
    col = f"mean_{score}"
    rows = []
    for (variable, horizon), g in table.groupby(["variable", "horizon_weeks"], observed=True):
        g = g.sort_values(col)
        best = g.iloc[0]
        rows.append({
            "variable": variable,
            "horizon_weeks": int(horizon),
            "best_model": best["model"],
            col: best[col],
            "improvement_pct": best["improvement_pct"],
            "dm_pvalue": best["dm_pvalue"],
            "beats_baseline_at_5pct": bool(best["significant_5pct"] and best["improvement_pct"] > 0),
        })
    return pd.DataFrame(rows).sort_values(["variable", "horizon_weeks"]).reset_index(drop=True)
