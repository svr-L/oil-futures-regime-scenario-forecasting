"""Cointegration-consistent state-space model for WTI/Brent.

The v0.1-v0.3 state-space block was a generic one-factor dynamic factor model:
useful as an interpretability aside, but it told a different story from the VECM
sitting next to it.  This module replaces it with the common-trends
representation that *is* the cointegration result, so the two agree by
construction.

Model
-----
For ``y_t = [log WTI_t, log Brent_t]'`` with one common stochastic trend
``tau_t`` and one stationary relative component ``s_t``::

    tau_t = tau_{t-1} + mu + eta_t          eta_t ~ (0, sigma_eta^2)
    s_t   = phi s_{t-1} + nu_t              nu_t  ~ (0, sigma_nu^2),  |phi| < 1

    log WTI_t   = tau_t + 0.5 s_t + eps_1t
    log Brent_t = a tau_t - 0.5 s_t + eps_2t

With ``a = 1`` (the default, ``restrict_trend=True``) the cointegrating vector is
exactly ``beta = [1, -1]'`` and ``s_t`` is the latent WTI-Brent log spread
stripped of measurement noise.  With ``a`` free, the implied cointegrating
vector is ``beta = [a, -1]'`` up to scale, which can be compared directly with
the Johansen estimate -- a genuine specification check rather than a
side-by-side.

Why this matters for the forecasting engine
-------------------------------------------
The VECM error-correction term is a *contemporaneous* function of observed
prices, so it inherits every roll artefact and quote error in the front-month
series.  The Kalman filter instead delivers ``E[s_t | y_1..t]``: the same
dislocation, filtered.  Scenarios generated from the filtered state are
therefore driven by the estimated signal rather than by the raw spread.

Simulation
----------
Forecasting uses the innovations (steady-state) form::

    a_{t+1|t} = T a_{t|t-1} + c + K v_t
    y_t       = Z a_{t|t-1} + d + v_t

Standardized one-step prediction errors ``F^{-1/2} v_t`` are bootstrapped with
the project's flexible probabilities and re-inflated by ``F^{1/2}``, which makes
this the state-space analogue of filtered historical simulation and lets the
model slot into the existing ablation with equal / time / macro weighting and
common random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate,
    unconstrain_stationary_univariate,
)

from .scenario_models import (
    SIMULATORS,
    _prepare_pool,
    _resolve_uniforms,
    indices_from_uniforms,
)


class CommonTrendSpread(MLEModel):
    """Common stochastic trend + stationary relative component, estimated by MLE."""

    def __init__(self, endog, restrict_trend: bool = True):
        super().__init__(endog, k_states=2, k_posdef=2,
                         initialization="approximate_diffuse",
                         loglikelihood_burn=2)
        self.restrict_trend = bool(restrict_trend)

        # Transition: [tau, s]
        self["transition"] = np.array([[1.0, 0.0], [0.0, 0.0]])
        self["selection"] = np.eye(2)
        self["state_intercept"] = np.zeros((2, 1))
        # Design filled in update(); the spread loadings are fixed at +/-0.5 so
        # that s_t is on the scale of the WTI-Brent log spread.
        self["design"] = np.array([[1.0, 0.5], [1.0, -0.5]])
        self["obs_cov"] = np.eye(2) * 1e-4

    @property
    def param_names(self):
        names = ["mu", "phi", "log_var_eta", "log_var_nu", "log_var_eps1", "log_var_eps2"]
        if not self.restrict_trend:
            names.append("trend_loading_brent")
        return names

    @property
    def start_params(self):
        y = np.asarray(self.endog, dtype=float)
        common = y.mean(axis=1)
        spread = y[:, 0] - y[:, 1]
        d_common = np.diff(common)
        mu = float(np.mean(d_common))
        var_c = float(np.var(d_common)) if np.var(d_common) > 0 else 1e-4
        var_s = float(np.var(np.diff(spread))) if np.var(np.diff(spread)) > 0 else 1e-4
        params = [mu, 0.85,
                  np.log(max(var_c * 0.8, 1e-10)),
                  np.log(max(var_s * 0.5, 1e-10)),
                  np.log(max(var_c * 0.05, 1e-10)),
                  np.log(max(var_c * 0.05, 1e-10))]
        if not self.restrict_trend:
            params.append(1.0)
        return np.array(params, dtype=float)

    def transform_params(self, unconstrained):
        p = np.array(unconstrained, dtype=float)
        p[1] = constrain_stationary_univariate(p[1:2])[0]
        return p

    def untransform_params(self, constrained):
        p = np.array(constrained, dtype=float)
        p[1] = unconstrain_stationary_univariate(np.array([p[1]]))[0]
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        # Values are kept in their incoming dtype: statsmodels differentiates the
        # likelihood by complex step, and casting to float here would silently
        # discard the imaginary part and corrupt the score and Hessian.
        mu, phi = params[0], params[1]
        var_eta, var_nu = np.exp(params[2]), np.exp(params[3])
        var_e1, var_e2 = np.exp(params[4]), np.exp(params[5])
        a = 1.0 if self.restrict_trend else params[6]

        self["state_intercept", 0, 0] = mu
        self["transition", 1, 1] = phi
        self["design", 1, 0] = a
        self["state_cov"] = np.diag([var_eta, var_nu])
        self["obs_cov"] = np.diag([var_e1, var_e2])


@dataclass
class CTSScenarioModel:
    """Fitted common-trend/spread model in innovations form, ready to simulate."""

    columns: list
    result: object
    residuals: pd.DataFrame        # standardized one-step prediction errors
    last_log_price: np.ndarray
    a_next: np.ndarray             # predicted state at the origin, a_{T+1|T}
    T: np.ndarray
    Z: np.ndarray
    c: np.ndarray
    K: np.ndarray                  # steady-state Kalman gain
    F_chol: np.ndarray             # Cholesky of the steady-state error covariance
    params: dict
    family: str = "CTS"
    restricted: bool = True


def fit_cts_model(
    log_prices: pd.DataFrame,
    restrict_trend: bool = True,
    burn: int = 10,
    maxiter: int = 200,
) -> CTSScenarioModel:
    """Fit the common-trend/spread state space and expose innovations-form pieces."""
    Y = log_prices.dropna(how="any").copy()
    mod = CommonTrendSpread(Y.values, restrict_trend=restrict_trend)
    res = mod.fit(disp=False, maxiter=maxiter)

    fr = res.filter_results
    v = np.asarray(fr.forecasts_error).T                    # nobs x 2
    F = np.asarray(fr.forecasts_error_cov)                  # 2 x 2 x nobs
    K = np.asarray(fr.kalman_gain)                          # k_states x k_endog x nobs

    # Steady-state quantities: the filter has converged well before the end.
    F_last = F[:, :, -1]
    F_last = 0.5 * (F_last + F_last.T) + 1e-12 * np.eye(2)
    F_chol = np.linalg.cholesky(F_last)
    F_chol_inv = np.linalg.inv(F_chol)

    z = (F_chol_inv @ v[burn:].T).T                         # standardized innovations
    residuals = pd.DataFrame(z, index=Y.index[burn:], columns=Y.columns)

    predicted = np.asarray(fr.predicted_state)              # k_states x (nobs + 1)
    a_next = predicted[:, -1].astype(float)

    p = res.params
    params = {
        "mu": float(p[0]),
        "phi": float(p[1]),
        "sigma_eta": float(np.sqrt(np.exp(p[2]))),
        "sigma_nu": float(np.sqrt(np.exp(p[3]))),
        "sigma_eps_wti": float(np.sqrt(np.exp(p[4]))),
        "sigma_eps_brent": float(np.sqrt(np.exp(p[5]))),
        "trend_loading_brent": 1.0 if restrict_trend else float(p[6]),
        "loglike": float(res.llf),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }

    return CTSScenarioModel(
        columns=list(Y.columns),
        result=res,
        residuals=residuals,
        last_log_price=Y.iloc[-1].values.astype(float),
        a_next=a_next,
        T=np.asarray(mod["transition"], dtype=float).copy(),
        Z=np.asarray(mod["design"], dtype=float).copy(),
        c=np.asarray(mod["state_intercept"], dtype=float).reshape(-1).copy(),
        K=K[:, :, -1].astype(float).copy(),
        F_chol=F_chol,
        params=params,
        family="CTS" if restrict_trend else "CTSF",
        restricted=bool(restrict_trend),
    )


def simulate_cts_paths(model, residual_pool, prob, horizon=20, n_sim=10_000, u=None, seed=None):
    """Simulate log-price paths from the innovations form with bootstrapped shocks."""
    pool, cum_prob = _prepare_pool(model, residual_pool, prob)
    u = _resolve_uniforms(u, horizon, n_sim, seed)

    k = len(model.columns)
    paths = np.zeros((horizon + 1, n_sim, k), dtype=float)
    paths[0, :, :] = model.last_log_price[None, :]

    a = np.tile(model.a_next[:, None], (1, n_sim))          # k_states x n_sim
    Z, T, c, K, L = model.Z, model.T, model.c, model.K, model.F_chol

    for h in range(1, horizon + 1):
        z_h = pool[indices_from_uniforms(u[h - 1], cum_prob), :]    # n_sim x 2
        v_h = z_h @ L.T                                             # re-inflate
        paths[h] = (Z @ a).T + v_h
        a = T @ a + c[:, None] + K @ v_h.T

    return paths


def cts_state_summary(model: CTSScenarioModel) -> dict:
    """Interpretable summary comparable with the Johansen / VECM output."""
    phi = model.params["phi"]
    half_life = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.nan
    a = model.params["trend_loading_brent"]
    var_s = model.params["sigma_nu"] ** 2 / (1 - phi ** 2) if abs(phi) < 1 else np.nan
    return {
        "beta_implied": np.array([a, -1.0]),
        "spread_ar_coefficient": phi,
        "spread_half_life_weeks": float(half_life),
        "spread_uncond_sd": float(np.sqrt(var_s)) if np.isfinite(var_s) else np.nan,
        "trend_drift_weekly": model.params["mu"],
        "trend_sd": model.params["sigma_eta"],
        "measurement_sd_wti": model.params["sigma_eps_wti"],
        "measurement_sd_brent": model.params["sigma_eps_brent"],
        "signal_to_noise_spread": (
            float(np.sqrt(var_s) / max(model.params["sigma_eps_wti"], 1e-12))
            if np.isfinite(var_s) else np.nan
        ),
    }


def filtered_states(model: CTSScenarioModel, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Filtered and smoothed common trend and relative component."""
    res = model.result
    filt = np.asarray(res.filtered_state).T
    try:
        smooth = np.asarray(res.smoothed_state).T
    except Exception:  # pragma: no cover - smoothing unavailable
        smooth = np.full_like(filt, np.nan)
    idx = index if index is not None else pd.RangeIndex(len(filt))
    return pd.DataFrame(
        {
            "trend_filtered": filt[:, 0],
            "spread_filtered": filt[:, 1],
            "trend_smoothed": smooth[:, 0],
            "spread_smoothed": smooth[:, 1],
        },
        index=idx[: len(filt)],
    )


# Registered here rather than in scenario_models to avoid a circular import;
# `oil_futures_regime/__init__` imports this module, so the dispatch table is
# always complete for anyone using the package.
SIMULATORS["CTS"] = simulate_cts_paths
SIMULATORS["CTSF"] = simulate_cts_paths   # same dynamics, free trend loading
