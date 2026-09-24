"""Marginal AR modelling utilities.

Probability logic lives in :mod:`oil_futures_regime.regimes`; v0.3 duplicated
``exp_half_life_weights``, ``effective_sample_size``, ``weighted_mean_cov`` and
``macro_kernel_weights`` here as well, which meant two copies that could drift
apart.  This module now only handles AR order selection and the conversion of
statsmodels ARIMA output into direct recursion form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA


def fit_ar_orders(x: pd.Series, p_max: int = 3, lb_lags: int = 10) -> pd.DataFrame:
    """Fit AR(p) models to returns and rank by BIC."""
    x = x.dropna()
    out = []
    for p in range(p_max + 1):
        res = ARIMA(x, order=(p, 0, 0), trend="c").fit()
        resid = pd.Series(np.asarray(res.resid)).dropna()
        lb_p = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)["lb_pvalue"].iloc[0]
        out.append({"p": p, "AIC": res.aic, "BIC": res.bic, "LB_pvalue_resid": float(lb_p)})
    return pd.DataFrame(out).sort_values("BIC")


def choose_p_bic_with_lb(table: pd.DataFrame, lb_threshold: float = 0.05) -> tuple[int, bool]:
    """Lowest-BIC AR order among models with acceptable residual autocorrelation."""
    ok = table[table["LB_pvalue_resid"] > lb_threshold]
    if len(ok):
        return int(ok.iloc[0]["p"]), True
    return int(table.iloc[0]["p"]), False


def fit_final_ar(x: pd.Series, p: int):
    """Fit final AR(p) = ARIMA(p, 0, 0) with constant mean."""
    return ARIMA(x.dropna(), order=(p, 0, 0), trend="c").fit()


def arima_recursion_params(res) -> tuple[float, float, np.ndarray]:
    """Convert ``statsmodels`` ARIMA ``trend='c'`` output to AR recursion form.

    ``statsmodels`` reports ``const`` as the *mean level* of the stationary AR
    process, not the recursion intercept.  For

        r_t = a + sum_i phi_i r_{t-i} + eps_t,

    the direct intercept is ``a = mean * (1 - sum(phi))``.  Using ``const``
    directly as ``a`` (the v0.1 behaviour) biases every simulated drift.

    Returns
    -------
    (mean, intercept, phi)
    """
    params = res.params
    mean = float(params.get("const", params.get("intercept", 0.0)))
    phi = np.asarray(getattr(res, "arparams", []), dtype=float)
    intercept = mean * (1.0 - float(phi.sum()))
    return mean, intercept, phi


# Backwards-compatible alias used by notebook 01.
def extract_ar_params(res) -> tuple[float, np.ndarray, int]:
    """Return (recursion intercept, phi, order)."""
    _, intercept, phi = arima_recursion_params(res)
    return intercept, phi, len(phi)


def simulate_ar_bootstrap(
    x0: float,
    r_init: np.ndarray,
    c: float,
    phi: np.ndarray,
    eps_pool: np.ndarray,
    prob: np.ndarray,
    horizon: int = 20,
    n_sim: int = 10_000,
    seed: int = 42,
):
    """Legacy single-series AR residual bootstrap (kept for notebook 01).

    ``c`` must be the *recursion intercept* from :func:`arima_recursion_params`,
    not the ARIMA ``const``.  New work should use the joint simulators in
    :mod:`oil_futures_regime.scenario_models`.
    """
    rng = np.random.default_rng(seed)
    eps_pool = np.asarray(eps_pool, dtype=float)
    prob = np.asarray(prob, dtype=float)
    prob = prob / prob.sum()
    phi = np.asarray(phi, dtype=float)
    p = len(phi)

    x_paths = np.zeros((horizon + 1, n_sim))
    r_paths = np.zeros((horizon + 1, n_sim))
    x_paths[0, :] = x0

    if p == 0:
        for h in range(1, horizon + 1):
            r_h = c + rng.choice(eps_pool, size=n_sim, replace=True, p=prob)
            r_paths[h, :] = r_h
            x_paths[h, :] = x_paths[h - 1, :] + r_h
        return x_paths, r_paths

    if len(r_init) != p:
        raise ValueError(f"r_init must have length {p}; got {len(r_init)}.")

    state = np.tile(np.asarray(r_init, dtype=float).reshape(-1, 1), (1, n_sim))
    for h in range(1, horizon + 1):
        eps_h = rng.choice(eps_pool, size=n_sim, replace=True, p=prob)
        r_h = c + phi @ state + eps_h
        r_paths[h, :] = r_h
        x_paths[h, :] = x_paths[h - 1, :] + r_h
        state = np.vstack([r_h.reshape(1, -1), state[:-1, :]])

    return x_paths, r_paths
