"""Joint scenario generators for WTI/Brent log prices.

Four families share one interface so that the validation harness can treat them
interchangeably:

``RW``
    Driftless random walk with bootstrapped joint return vectors.  This is the
    benchmark any multi-horizon price forecast has to beat and its absence was
    the main gap in the v0.3 ablation.
``AR``
    Separate low-order AR models per series, joint residual *rows* resampled so
    contemporaneous WTI/Brent shock dependence is preserved.
``VECM``
    Cointegration-aware joint dynamics with the error-correction term recomputed
    path-by-path.
``FHS``
    Filtered historical simulation: AR mean + GARCH(1,1)-t volatility filter,
    bootstrap of *standardized* residuals, conditional variance evolving along
    each simulated path.  This is the standard market-risk answer to the
    volatility clustering documented in the v0.1 diagnostics, and it is the
    strongest competitor to the macro kernel.

Common random numbers
---------------------
Every simulator consumes a ``(horizon, n_sim)`` array of uniforms rather than a
seed.  The harness draws one such array per forecast origin and reuses it for
all models, so paired score differences are not polluted by Monte Carlo noise
from independent draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.vecm import VECM

from statsmodels.tsa.api import VAR

from .models import arima_recursion_params, choose_p_bic_with_lb, fit_ar_orders

# ---------------------------------------------------------------------------
# Common random numbers
# ---------------------------------------------------------------------------


def common_uniforms(horizon: int, n_sim: int, seed: int) -> np.ndarray:
    """Uniform draws shared by every model at a given forecast origin."""
    rng = np.random.default_rng(seed)
    return rng.random((horizon, n_sim))


def _resolve_uniforms(u, horizon: int, n_sim: int, seed: int | None) -> np.ndarray:
    if u is None:
        if seed is None:
            raise ValueError("either u or seed must be provided")
        return common_uniforms(horizon, n_sim, seed)
    u = np.asarray(u, dtype=float)
    if u.shape != (horizon, n_sim):
        raise ValueError(f"u must have shape {(horizon, n_sim)}; got {u.shape}")
    return u


def indices_from_uniforms(u_row: np.ndarray, cum_prob: np.ndarray) -> np.ndarray:
    """Inverse-CDF sampling of pool rows: same uniforms -> comparable draws."""
    return np.searchsorted(cum_prob, u_row, side="right").clip(0, len(cum_prob) - 1)


def _prepare_pool(model, residual_pool: pd.DataFrame, prob: np.ndarray):
    pool = residual_pool.loc[:, model.columns].values.astype(float)
    prob = np.asarray(prob, dtype=float)
    prob = prob / prob.sum()
    if len(pool) != len(prob):
        raise ValueError("residual_pool and prob lengths differ")
    return pool, np.cumsum(prob)


# ---------------------------------------------------------------------------
# Model containers
# ---------------------------------------------------------------------------


@dataclass
class ARSeriesSpec:
    name: str
    order: int
    mean: float
    intercept: float
    phi: np.ndarray
    result: object


@dataclass
class RWScenarioModel:
    """Driftless random walk: Delta y_t = eps_t, eps bootstrapped jointly."""

    columns: list
    residuals: pd.DataFrame
    last_log_price: np.ndarray
    family: str = "RW"


@dataclass
class JointARScenarioModel:
    columns: list
    specs: dict
    residuals: pd.DataFrame
    last_log_price: np.ndarray
    return_history: dict
    family: str = "AR"


@dataclass
class VECMScenarioModel:
    columns: list
    result: object
    residuals: pd.DataFrame
    last_log_price: np.ndarray
    delta_history: np.ndarray
    deterministic: str
    family: str = "VECM"


@dataclass
class VARLevelsScenarioModel:
    """Unrestricted VAR in log levels: the model implied by Johansen rank = n."""

    columns: list
    intercept: np.ndarray
    coefs: np.ndarray            # (lags, k, k)
    residuals: pd.DataFrame
    last_log_price: np.ndarray
    level_history: np.ndarray    # (lags, k), most recent first
    family: str = "VARL"


@dataclass
class FHSScenarioModel:
    """AR mean + GARCH(1,1)-t filter with standardized-residual bootstrap."""

    columns: list
    specs: dict
    residuals: pd.DataFrame          # standardized residuals z_t
    last_log_price: np.ndarray
    return_history: dict
    garch: dict = field(default_factory=dict)   # col -> (omega, alpha, beta, sigma2_next)
    scale: float = 100.0
    family: str = "FHS"


# ---------------------------------------------------------------------------
# Random walk
# ---------------------------------------------------------------------------


def fit_rw_model(log_prices: pd.DataFrame) -> RWScenarioModel:
    """Driftless random walk benchmark; the 'residual' pool is the return matrix."""
    Y = log_prices.dropna(how="any").copy()
    rets = Y.diff().dropna()
    return RWScenarioModel(
        columns=list(Y.columns),
        residuals=rets,
        last_log_price=Y.iloc[-1].values.astype(float),
    )


def simulate_rw_paths(model, residual_pool, prob, horizon=20, n_sim=10_000, u=None, seed=None):
    pool, cum_prob = _prepare_pool(model, residual_pool, prob)
    u = _resolve_uniforms(u, horizon, n_sim, seed)

    k = len(model.columns)
    paths = np.zeros((horizon + 1, n_sim, k), dtype=float)
    paths[0, :, :] = model.last_log_price[None, :]
    for h in range(1, horizon + 1):
        idx = indices_from_uniforms(u[h - 1], cum_prob)
        paths[h] = paths[h - 1] + pool[idx, :]
    return paths


# ---------------------------------------------------------------------------
# Joint AR
# ---------------------------------------------------------------------------


def fit_joint_ar_model(
    log_prices: pd.DataFrame,
    p_max: int = 3,
    lb_lags: int = 10,
    lb_threshold: float = 0.05,
) -> JointARScenarioModel:
    """Fit per-series AR models to log returns and align joint residual vectors."""
    Y = log_prices.dropna(how="any").copy()
    rets = Y.diff().dropna()
    specs = {}
    residuals = pd.DataFrame(index=rets.index)
    return_history = {}

    max_order = 0
    for col in Y.columns:
        table = fit_ar_orders(rets[col], p_max=p_max, lb_lags=lb_lags)
        p, _ = choose_p_bic_with_lb(table, lb_threshold=lb_threshold)
        res = ARIMA(rets[col], order=(p, 0, 0), trend="c").fit()
        mean, intercept, phi = arima_recursion_params(res)
        specs[col] = ARSeriesSpec(col, p, mean, intercept, phi, res)
        residuals[col] = pd.Series(np.asarray(res.resid), index=rets.index)
        return_history[col] = rets[col].iloc[-p:].values[::-1] if p > 0 else np.empty(0)
        max_order = max(max_order, p)

    burn = max(1, max_order)
    residuals = residuals.iloc[burn:].dropna(how="any")

    return JointARScenarioModel(
        columns=list(Y.columns),
        specs=specs,
        residuals=residuals,
        last_log_price=Y.iloc[-1].values.astype(float),
        return_history=return_history,
    )


def _init_ar_states(model, n_sim):
    states = {}
    for col in model.columns:
        spec = model.specs[col]
        if spec.order > 0:
            states[col] = np.tile(model.return_history[col][:, None], (1, n_sim))
        else:
            states[col] = np.empty((0, n_sim))
    return states


def simulate_joint_ar_paths(model, residual_pool, prob, horizon=20, n_sim=10_000, u=None, seed=None):
    """Simulate joint log-price paths: (horizon + 1, n_sim, n_assets)."""
    pool, cum_prob = _prepare_pool(model, residual_pool, prob)
    u = _resolve_uniforms(u, horizon, n_sim, seed)

    k = len(model.columns)
    paths = np.zeros((horizon + 1, n_sim, k), dtype=float)
    paths[0, :, :] = model.last_log_price[None, :]
    states = _init_ar_states(model, n_sim)

    for h in range(1, horizon + 1):
        eps_h = pool[indices_from_uniforms(u[h - 1], cum_prob), :]
        r_h = np.empty((n_sim, k), dtype=float)
        for j, col in enumerate(model.columns):
            spec = model.specs[col]
            if spec.order == 0:
                r = spec.mean + eps_h[:, j]
            else:
                r = spec.intercept + spec.phi @ states[col] + eps_h[:, j]
                states[col] = np.vstack([r[None, :], states[col][:-1, :]])
            r_h[:, j] = r
        paths[h] = paths[h - 1] + r_h
    return paths


# ---------------------------------------------------------------------------
# VECM
# ---------------------------------------------------------------------------


def fit_vecm_scenario_model(
    log_prices: pd.DataFrame,
    coint_rank: int = 1,
    k_ar_diff: int = 1,
    deterministic: str = "ci",
) -> VECMScenarioModel:
    """Fit a VECM and expose aligned joint innovations for bootstrap simulation."""
    Y = log_prices.dropna(how="any").copy()
    res = VECM(Y, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic).fit()

    resid_index = Y.index[res.k_ar :]
    residuals = pd.DataFrame(np.asarray(res.resid), index=resid_index, columns=Y.columns)

    if k_ar_diff > 0:
        delta_history = Y.diff().dropna().iloc[-k_ar_diff:].values[::-1].astype(float)
    else:
        delta_history = np.empty((0, Y.shape[1]), dtype=float)

    return VECMScenarioModel(
        columns=list(Y.columns),
        result=res,
        residuals=residuals,
        last_log_price=Y.iloc[-1].values.astype(float),
        delta_history=delta_history,
        deterministic=deterministic,
    )


def simulate_vecm_paths(model, residual_pool, prob, horizon=20, n_sim=10_000, u=None, seed=None):
    """VECM simulation with the error-correction term recomputed path-by-path."""
    pool, cum_prob = _prepare_pool(model, residual_pool, prob)
    u = _resolve_uniforms(u, horizon, n_sim, seed)

    res = model.result
    k = len(model.columns)
    q = model.delta_history.shape[0]

    if any(flag in model.deterministic for flag in ("li", "lo")):
        raise NotImplementedError("linear deterministic trends are not supported by the simulator")

    paths = np.zeros((horizon + 1, n_sim, k), dtype=float)
    paths[0, :, :] = model.last_log_price[None, :]

    if q > 0:
        dy_states = np.repeat(model.delta_history[:, None, :], n_sim, axis=1)
    else:
        dy_states = np.empty((0, n_sim, k), dtype=float)

    gamma = np.asarray(res.gamma, dtype=float)
    alpha = np.asarray(res.alpha, dtype=float)
    beta = np.asarray(res.beta, dtype=float)
    coint_const = np.asarray(getattr(res, "det_coef_coint", np.empty((0, 0))), dtype=float)
    outside_det = np.asarray(getattr(res, "det_coef", np.empty((k, 0))), dtype=float)

    for h in range(1, horizon + 1):
        y_prev = paths[h - 1, :, :]
        ect = y_prev @ beta
        if "ci" in model.deterministic and coint_const.size:
            ect = ect + coint_const.reshape(1, -1)
        dy = ect @ alpha.T
        if "co" in model.deterministic and outside_det.size:
            dy = dy + outside_det[:, 0][None, :]
        for lag in range(q):
            G = gamma[:, lag * k : (lag + 1) * k]
            dy = dy + dy_states[lag, :, :] @ G.T

        dy = dy + pool[indices_from_uniforms(u[h - 1], cum_prob), :]
        paths[h] = y_prev + dy
        if q > 0:
            dy_states = np.concatenate([dy[None, :, :], dy_states[:-1, :, :]], axis=0)

    return paths


# ---------------------------------------------------------------------------
# VAR in levels
# ---------------------------------------------------------------------------


def fit_var_levels_model(log_prices: pd.DataFrame, lags: int = 1) -> VARLevelsScenarioModel:
    """Fit an unrestricted VAR on log levels.

    Johansen on the WTI/Brent system can marginally reject ``r <= 1`` as well as
    ``r = 0``, i.e. point at a *stationary* system rather than one common trend.
    That conclusion is implausible economically and typical of local-to-unity
    samples, but it should be tested rather than argued away: this model is
    exactly what such a rank decision implies, so putting it in the ablation lets
    the forecast comparison settle the question.
    """
    Y = log_prices.dropna(how="any").copy()
    res = VAR(Y).fit(lags)
    coefs = np.asarray(res.coefs, dtype=float)                 # (lags, k, k)
    intercept = np.asarray(res.intercept, dtype=float).ravel()
    resid = pd.DataFrame(np.asarray(res.resid), index=Y.index[lags:], columns=Y.columns)
    history = Y.iloc[-lags:].values[::-1].astype(float)        # most recent first
    return VARLevelsScenarioModel(
        columns=list(Y.columns),
        intercept=intercept,
        coefs=coefs,
        residuals=resid,
        last_log_price=Y.iloc[-1].values.astype(float),
        level_history=history,
    )


def simulate_var_levels_paths(model, residual_pool, prob, horizon=20, n_sim=10_000,
                              u=None, seed=None):
    """Simulate log-price paths from a VAR in levels with bootstrapped innovations."""
    pool, cum_prob = _prepare_pool(model, residual_pool, prob)
    u = _resolve_uniforms(u, horizon, n_sim, seed)

    k = len(model.columns)
    lags = model.coefs.shape[0]
    paths = np.zeros((horizon + 1, n_sim, k), dtype=float)
    paths[0, :, :] = model.last_log_price[None, :]

    states = np.repeat(model.level_history[:, None, :], n_sim, axis=1)   # (lags, n_sim, k)
    for h in range(1, horizon + 1):
        y = np.tile(model.intercept[None, :], (n_sim, 1))
        for lag in range(lags):
            y = y + states[lag] @ model.coefs[lag].T
        y = y + pool[indices_from_uniforms(u[h - 1], cum_prob), :]
        paths[h] = y
        states = np.concatenate([y[None, :, :], states[:-1, :, :]], axis=0)
    return paths


# ---------------------------------------------------------------------------
# Filtered historical simulation (AR + GARCH(1,1)-t)
# ---------------------------------------------------------------------------


def _fit_garch11_t(resid: pd.Series, scale: float = 100.0):
    """Zero-mean GARCH(1,1) filter on AR residuals.

    The preferred implementation uses :mod:`arch` with Student-t innovations.
    A small Gaussian-QMLE fallback is included so the core package and tests stay
    usable in offline/minimal environments where ``arch`` cannot be installed.
    The fallback is *only* a dependency fallback: when ``arch`` is available the
    original Student-t specification is used, so the published real-data results
    remain reproducible.

    Returns ``(omega, alpha, beta, sigma_series, sigma2_next)`` in the *scaled*
    units (residuals are multiplied by ``scale`` for numerical stability, which
    leaves standardized residuals unchanged).
    """
    y = np.asarray(resid, dtype=float) * scale
    try:
        from arch import arch_model
    except ImportError:  # deterministic offline fallback
        from scipy.optimize import minimize

        var0 = float(np.var(y, ddof=1)) if len(y) > 1 else float(y[0] ** 2)
        var0 = max(var0, 1e-8)

        # Parameterisation enforces omega>0, alpha>=0, beta>=0 and alpha+beta<1.
        def unpack(theta):
            omega = np.exp(theta[0])
            ea, eb = np.exp(theta[1]), np.exp(theta[2])
            den = 1.0 + ea + eb
            alpha = 0.999 * ea / den
            beta = 0.999 * eb / den
            return omega, alpha, beta

        def variance_path(theta):
            omega, alpha, beta = unpack(theta)
            h = np.empty_like(y, dtype=float)
            h[0] = var0
            for t in range(1, len(y)):
                h[t] = omega + alpha * y[t - 1] ** 2 + beta * h[t - 1]
                h[t] = max(h[t], 1e-10)
            return h

        def nll(theta):
            h = variance_path(theta)
            return 0.5 * float(np.sum(np.log(h) + y ** 2 / h))

        # alpha~0.08, beta~0.88, unconditional variance ~ sample variance.
        a0, b0 = 0.08, 0.88
        o0 = max(var0 * (1.0 - a0 - b0), 1e-8)
        # Invert the softmax-style transform approximately.
        rest = max(0.999 - a0 - b0, 1e-4)
        theta0 = np.array([np.log(o0), np.log(a0 / rest), np.log(b0 / rest)])
        opt = minimize(nll, theta0, method="L-BFGS-B")
        theta = opt.x if opt.success else theta0
        omega, alpha, beta = unpack(theta)
        h = variance_path(theta)
        sigma2_next = omega + alpha * y[-1] ** 2 + beta * h[-1]
        return omega, alpha, beta, pd.Series(np.sqrt(h), index=resid.index), float(sigma2_next)

    am = arch_model(y, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])
    sigma = np.asarray(res.conditional_volatility, dtype=float)
    sigma2_next = omega + alpha * y[-1] ** 2 + beta * sigma[-1] ** 2
    return omega, alpha, beta, pd.Series(sigma, index=resid.index), float(sigma2_next)


def fit_fhs_model(
    log_prices: pd.DataFrame,
    p_max: int = 3,
    lb_lags: int = 10,
    lb_threshold: float = 0.05,
    scale: float = 100.0,
) -> FHSScenarioModel:
    """AR mean dynamics + GARCH(1,1)-t variance filter; pool = standardized shocks."""
    ar = fit_joint_ar_model(log_prices, p_max=p_max, lb_lags=lb_lags, lb_threshold=lb_threshold)

    z = pd.DataFrame(index=ar.residuals.index)
    garch = {}
    for col in ar.columns:
        omega, alpha, beta, sigma, sigma2_next = _fit_garch11_t(ar.residuals[col], scale=scale)
        garch[col] = (omega, alpha, beta, sigma2_next)
        z[col] = (ar.residuals[col] * scale) / sigma

    z = z.dropna(how="any")
    return FHSScenarioModel(
        columns=list(ar.columns),
        specs=ar.specs,
        residuals=z,
        last_log_price=ar.last_log_price,
        return_history=ar.return_history,
        garch=garch,
        scale=scale,
    )


def simulate_fhs_paths(model, residual_pool, prob, horizon=20, n_sim=10_000, u=None, seed=None):
    """FHS simulation: conditional variance evolves along every simulated path."""
    pool, cum_prob = _prepare_pool(model, residual_pool, prob)
    u = _resolve_uniforms(u, horizon, n_sim, seed)

    k = len(model.columns)
    s = model.scale
    paths = np.zeros((horizon + 1, n_sim, k), dtype=float)
    paths[0, :, :] = model.last_log_price[None, :]
    states = _init_ar_states(model, n_sim)

    sigma2 = {col: np.full(n_sim, model.garch[col][3], dtype=float) for col in model.columns}

    for h in range(1, horizon + 1):
        z_h = pool[indices_from_uniforms(u[h - 1], cum_prob), :]
        r_h = np.empty((n_sim, k), dtype=float)
        for j, col in enumerate(model.columns):
            omega, alpha, beta, _ = model.garch[col]
            e_scaled = np.sqrt(sigma2[col]) * z_h[:, j]      # shock in scaled units
            eps = e_scaled / s                               # back to log-return units
            spec = model.specs[col]
            if spec.order == 0:
                r = spec.mean + eps
            else:
                r = spec.intercept + spec.phi @ states[col] + eps
                states[col] = np.vstack([r[None, :], states[col][:-1, :]])
            r_h[:, j] = r
            sigma2[col] = omega + alpha * e_scaled ** 2 + beta * sigma2[col]
        paths[h] = paths[h - 1] + r_h
    return paths


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SIMULATORS = {
    "RW": simulate_rw_paths,
    "AR": simulate_joint_ar_paths,
    "VECM": simulate_vecm_paths,
    "VARL": simulate_var_levels_paths,
    "FHS": simulate_fhs_paths,
}


def simulate(model, residual_pool, prob, horizon=20, n_sim=10_000, u=None, seed=None):
    """Dispatch to the simulator matching ``model.family``."""
    try:
        fn = SIMULATORS[model.family]
    except KeyError as exc:
        raise ValueError(f"unknown model family: {model.family!r}") from exc
    return fn(model, residual_pool, prob, horizon=horizon, n_sim=n_sim, u=u, seed=seed)


def path_frame(paths: np.ndarray, columns: list, step: int) -> pd.DataFrame:
    """Convenience conversion of one simulation horizon to a DataFrame."""
    return pd.DataFrame(paths[step, :, :], columns=columns)
