"""Flexible probabilities and macro-state conditioning.

This module owns *all* probability logic used by the scenario engine:
exponential half-life weights, effective sample size, the Gaussian
state-similarity kernel and the bandwidth selection rule.

Everything here is written so that it can be called inside a rolling
pseudo-out-of-sample loop without look-ahead: standardization moments are
estimated only from the observations passed in by the caller, and the
bandwidth rule uses no realized forecast outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeightDiagnostics:
    """Book-keeping for the probability vector attached to a residual pool."""

    mode: str
    ess: float
    n_scenarios: int
    ess_fraction: float
    bandwidth: float | None = None
    bandwidth_binding: bool = False
    recentered: bool = True


def exp_half_life_weights(index: pd.DatetimeIndex, half_life: int) -> pd.Series:
    """Exponentially decaying probabilities with the latest observation at age zero."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    index = pd.DatetimeIndex(index)
    ages = np.arange(len(index) - 1, -1, -1, dtype=float)
    w = np.power(0.5, ages / float(half_life))
    w /= w.sum()
    return pd.Series(w, index=index, name="prob")


def effective_sample_size(w: np.ndarray | pd.Series) -> float:
    """Inverse-Herfindahl effective sample size."""
    w = np.asarray(w, dtype=float).ravel()
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("weights must have a positive finite sum")
    w = w / s
    return float(1.0 / np.sum(w * w))


def weighted_mean_cov(x, w):
    """Weighted mean and covariance under probabilities that sum to one."""
    values = np.asarray(x, dtype=float)
    wv = np.asarray(w, dtype=float).ravel()
    if len(values) != len(wv):
        raise ValueError("x and w must have the same number of observations")
    wv = wv / wv.sum()
    mu = np.sum(values * wv[:, None], axis=0)
    xc = values - mu
    cov = (xc * wv[:, None]).T @ xc
    return mu, cov


def build_macro_features(state_levels: pd.DataFrame) -> pd.DataFrame:
    """Construct stationary-ish macro/risk conditioning features.

    Expected columns are ``VIX``, ``DXY`` and ``TNX``.  VIX enters in log-level
    form because it is itself a state variable; DXY and the US 10Y yield proxy
    enter as weekly changes so that the kernel does not condition on persistent
    price/yield levels.
    """
    required = {"VIX", "DXY", "TNX"}
    missing = required.difference(state_levels.columns)
    if missing:
        raise ValueError(f"state_levels is missing columns: {sorted(missing)}")

    out = pd.DataFrame(index=state_levels.index)
    out["log_vix"] = np.log(state_levels["VIX"].astype(float))
    out["dlog_dxy"] = np.log(state_levels["DXY"].astype(float)).diff()
    # Yahoo's ^TNX is roughly 10x the percentage yield; scaling is immaterial
    # after z-scoring but keeps the feature interpretable.
    out["d10y_pctpt"] = (state_levels["TNX"].astype(float) / 10.0).diff()
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def macro_kernel_weights(
    Z: pd.DataFrame,
    z0: pd.Series,
    base_w,
    bw: float,
) -> np.ndarray:
    """Gaussian state-similarity kernel multiplied by base probabilities."""
    if bw <= 0:
        raise ValueError("bw must be positive")
    values = np.asarray(Z, dtype=float)
    w = np.asarray(base_w, dtype=float).ravel()
    if len(values) != len(w):
        raise ValueError("Z and base_w must have the same number of rows")
    w = w / w.sum()

    mu = np.sum(values * w[:, None], axis=0)
    var = np.sum(((values - mu) ** 2) * w[:, None], axis=0)
    sig = np.sqrt(var)
    sig = np.where(sig < 1e-12, 1.0, sig)

    Zs = (values - mu) / sig
    z0s = (np.asarray(z0, dtype=float) - mu) / sig
    dist2 = np.sum((Zs - z0s) ** 2, axis=1)
    kernel = np.exp(-0.5 * dist2 / (bw * bw))
    total = w * kernel
    if not np.isfinite(total).all() or total.sum() <= 0:
        raise ValueError("macro kernel produced invalid weights")
    return total / total.sum()


DEFAULT_BANDWIDTH_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


def select_macro_bandwidth(
    Z: pd.DataFrame,
    z0: pd.Series,
    base_w: pd.Series,
    bandwidth_grid: Iterable[float] = DEFAULT_BANDWIDTH_GRID,
    min_ess_fraction: float = 0.35,
):
    """Choose the tightest kernel whose ESS is at least a *fraction* of the pool.

    v0.3 used an absolute ESS floor (75 observations).  Inside a rolling loop the
    pool grows from ~250 to ~600 weeks, so an absolute floor silently makes the
    conditioning tighter over time and the model is not comparable across
    origins.  A fraction of the pool keeps conditioning intensity stable.

    The rule uses no realized forecast outcome, so it is safe to repeat at every
    forecast origin.
    """
    grid = sorted(float(x) for x in bandwidth_grid)
    if not grid:
        raise ValueError("bandwidth_grid cannot be empty")
    n = len(Z)
    min_ess = float(min_ess_fraction) * n

    last_w, last_bw = None, None
    for bw in grid:
        candidate = macro_kernel_weights(Z, z0, base_w, bw)
        last_w, last_bw = candidate, bw
        ess = effective_sample_size(candidate)
        if ess >= min_ess:
            return candidate, WeightDiagnostics(
                mode="macro",
                ess=ess,
                n_scenarios=n,
                ess_fraction=ess / n,
                bandwidth=bw,
                bandwidth_binding=False,
            )

    # Grid exhausted: report the widest kernel and flag that the floor binds.
    assert last_w is not None and last_bw is not None
    ess = effective_sample_size(last_w)
    return last_w, WeightDiagnostics(
        mode="macro",
        ess=ess,
        n_scenarios=n,
        ess_fraction=ess / n,
        bandwidth=last_bw,
        bandwidth_binding=True,
    )


def prepare_residual_pool(
    residuals: pd.DataFrame,
    mode: str,
    half_life: int = 52,
    macro_features: pd.DataFrame | None = None,
    origin=None,
    bandwidth_grid: Iterable[float] = DEFAULT_BANDWIDTH_GRID,
    min_macro_ess_fraction: float = 0.35,
    recenter: bool = True,
):
    """Align a residual matrix with equal/time/macro flexible probabilities.

    Parameters
    ----------
    recenter:
        If True the residual matrix is centered under the *final* scenario
        probabilities, so regime conditioning changes dispersion and tails but
        not the conditional mean.  This is the conservative default for a risk
        scenario generator, but it is a modelling choice rather than a neutral
        one: ``recenter=False`` lets the macro kernel also shift the drift, and
        v0.4 ablates both settings explicitly.

    Returns
    -------
    (pool, prob, diagnostics)
    """
    residuals = residuals.dropna(how="any").copy()
    if residuals.empty:
        raise ValueError("residual pool is empty")

    mode = mode.lower()
    if mode not in {"equal", "time", "macro"}:
        raise ValueError("mode must be one of: equal, time, macro")

    if mode == "equal":
        n = len(residuals)
        prob = np.full(n, 1.0 / n)
        diag = WeightDiagnostics("equal", float(n), n, 1.0, None, False, recenter)
    elif mode == "time":
        prob = exp_half_life_weights(residuals.index, half_life).values
        ess = effective_sample_size(prob)
        diag = WeightDiagnostics("time", ess, len(prob), ess / len(prob), None, False, recenter)
    else:
        if macro_features is None or origin is None:
            raise ValueError("macro_features and origin are required for macro mode")
        z_hist = macro_features.reindex(residuals.index).dropna()
        idx = residuals.index.intersection(z_hist.index)
        residuals = residuals.loc[idx]
        if residuals.empty:
            raise ValueError("no overlap between residuals and macro features")
        z_hist = z_hist.loc[residuals.index]
        base = exp_half_life_weights(residuals.index, half_life)
        current = macro_features.loc[: pd.Timestamp(origin)]
        if current.empty:
            raise ValueError("no macro feature observation is available at or before origin")
        z0 = current.iloc[-1]
        prob, kdiag = select_macro_bandwidth(
            z_hist,
            z0,
            base,
            bandwidth_grid=bandwidth_grid,
            min_ess_fraction=min_macro_ess_fraction,
        )
        diag = WeightDiagnostics(
            kdiag.mode, kdiag.ess, kdiag.n_scenarios, kdiag.ess_fraction,
            kdiag.bandwidth, kdiag.bandwidth_binding, recenter,
        )

    prob = np.array(prob, dtype=float, copy=True)
    prob /= prob.sum()
    if recenter:
        mu = np.sum(residuals.values * prob[:, None], axis=0)
        residuals = residuals - mu

    return residuals, prob, diag
