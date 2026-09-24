"""Is the WTI/Brent cointegrating relation stable?

The full-sample estimate of the cointegrating vector (``a = 1.016``, rejecting
``beta = [1, -1]`` at p = 2e-4) averages over a period that contains a known
structural change: the repeal of the US crude-oil export ban, signed on
18 December 2015.  Before the repeal, US crude was largely landlocked and WTI
traded at a variable, often wide discount to Brent; afterwards, export arbitrage
should have tied WTI to Brent minus transport costs.  If that is right, three
things should have moved, and this module tests each separately rather than
bundling them into one "instability" verdict:

1. **the level of the relation** (the intercept of the cointegrating regression:
   the mean discount should narrow);
2. **the slope** (the cointegrating coefficient itself);
3. **the dynamics of the spread** (tighter arbitrage should make deviations
   revert faster, and possibly be less volatile).

Tools
-----
* :func:`dols` -- dynamic OLS (Stock and Watson, 1993).  With I(1) regressors,
  plain OLS on levels is superconsistent but its t-statistics are not usable;
  adding leads and lags of the differenced regressor and scaling by the long-run
  residual variance restores asymptotically normal inference on the
  cointegrating coefficients.
* :func:`dols_break_test` -- known-date break at the repeal, testing intercept and
  slope jointly and separately.  Wald statistics are asymptotically chi-square
  under DOLS.
* :func:`sup_wald_break` -- the same statistic over every admissible date
  (15% trimming, Andrews 1993), so the data can say *where* the break is rather
  than only whether one exists at the date we chose.  The sup statistic has a
  non-standard distribution with I(1) regressors, so its p-value comes from a
  fixed-regressor block bootstrap under the no-break null, in the spirit of
  Hansen (2000).
* :func:`fit_tvp_beta` -- a time-varying-parameter regression estimated by
  Kalman filter, with both intercept and slope following random walks.  A
  likelihood-ratio test of a constant slope sits on the boundary of the
  parameter space, so its p-value uses the 0.5 chi2(0) + 0.5 chi2(1) mixture
  (Self and Liang, 1987) instead of the naive chi2(1).
* :func:`spread_dynamics_break` -- pre/post mean, mean-reversion speed and
  volatility of the spread.
* :func:`rolling_dols_beta` -- rolling-window DOLS as a model-free picture.

Calibration: why every p-value here is bootstrapped
---------------------------------------------------
The first version of this module used asymptotic chi-square p-values for the
known-date test.  Simulated under **no break at all**, with a spread persistence
of 0.9 (about the level observed on WTI/Brent), that test rejected in 41-43% of
samples at a nominal 5%; at persistence 0.95, 54%.  Newey-West with a
rule-of-thumb bandwidth captures only a fraction of the long-run variance of
residuals this persistent, and the pre-break sample is about three years.  On
real data it would have "confirmed" a break at the export-ban repeal almost
regardless of the truth.

Every test is therefore calibrated by a fixed-regressor **AR-sieve bootstrap**
under its own null.  A moving-block bootstrap was tried first and discarded: its
short blocks cut the long excursions of the spread and understated location
uncertainty badly enough to reject a correctly specified break date.  Measured
over 100 simulated samples per row:

=====================  ==============  ================
test                   size, asympt.   size, bootstrap
=====================  ==============  ================
level + slope (joint)  41%             6%
level                  27%             7%
slope                  28%             4%
spread speed           8%              3%
=====================  ==============  ================

Power against a planted break (mean discount narrowing from -0.08 to -0.03,
persistence falling from 0.95 to 0.80): 79% for the joint test, 71% for the speed
change, 54% for the level alone.

Evidence hierarchy, in the order results should be read:

1. known-date test at the repeal, bootstrap p-values (primary, economically
   motivated);
2. spread dynamics before and after, bootstrap p-values (the mechanism);
3. sup-Wald for existence at an unknown date; location only as a range;
4. Kalman and rolling DOLS paths (descriptive).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter
from statsmodels.tsa.statespace.mlemodel import MLEModel

from .significance import newey_west_lrv

EXPORT_BAN_REPEAL = "2015-12-18"


# ---------------------------------------------------------------------------
# DOLS machinery
# ---------------------------------------------------------------------------


def _default_hac_lag(n: int) -> int:
    """Newey-West rule of thumb, floor(4 (n/100)^(2/9))."""
    return max(1, int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def _dols_design(y: pd.Series, x: pd.Series, leads: int, lags: int):
    """Align y, x and leads/lags of dx; return (index, y, base regressors, dx block)."""
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    dx = df["x"].diff()
    cols = {}
    for j in range(-leads, lags + 1):
        cols[f"dx_{j:+d}"] = dx.shift(j)
    block = pd.DataFrame(cols, index=df.index)
    full = pd.concat([df, block], axis=1).dropna()
    return full.index, full["y"].values, full["x"].values, full[list(cols)].values


def _ols_lrv(Y: np.ndarray, X: np.ndarray, hac_lag: int):
    """OLS coefficients and LRV-scaled covariance (the DOLS inference recipe)."""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    lrv = newey_west_lrv(resid, hac_lag)
    XtX_inv = np.linalg.pinv(X.T @ X)
    cov = XtX_inv * lrv
    return beta, cov, resid, lrv


def _wald(beta: np.ndarray, cov: np.ndarray, idx) -> float:
    b = beta[idx]
    V = cov[np.ix_(idx, idx)]
    try:
        return float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        return float(b @ np.linalg.pinv(V) @ b)


class _ARSieve:
    """AR-sieve bootstrap for persistent residuals.

    Cointegrating residuals here are, up to scale, deviations of the spread from
    its equilibrium, with persistence around 0.9-0.95.  A moving-block bootstrap
    with blocks of a few weeks cuts precisely the long excursions that make break
    dates hard to locate, so it *understates* location uncertainty (in
    simulation it rejected a correctly specified break date).  The sieve fits an
    AR(p) to the residuals, resamples its innovations i.i.d., and rebuilds series
    with the same persistence.
    """

    def __init__(self, u: np.ndarray, p_max: int = 4, burn: int = 200):
        u = np.asarray(u, dtype=float) - float(np.mean(u))
        n = len(u)
        best = None
        for p in range(1, p_max + 1):
            X = np.column_stack([u[p - j - 1:n - j - 1] for j in range(p)])
            yv = u[p:]
            phi, *_ = np.linalg.lstsq(X, yv, rcond=None)
            e = yv - X @ phi
            aic = len(yv) * np.log(e @ e / len(yv)) + 2 * p
            if best is None or aic < best[0]:
                best = (aic, phi, e)
        _, phi, e = best
        # Keep the recursion stable: shrink towards the unit circle's interior.
        roots = np.roots(np.r_[1.0, -phi])
        if np.any(np.abs(roots) >= 0.999):
            phi = phi * 0.995 / max(np.abs(roots).max(), 1.0)
        self.phi = phi
        self.innov = e - e.mean()
        self.n = n
        self.burn = burn

    def draw(self, rng) -> np.ndarray:
        total = self.n + self.burn
        eps = rng.choice(self.innov, size=total, replace=True)
        out = lfilter([1.0], np.r_[1.0, -self.phi], eps)
        return out[self.burn:]


@dataclass
class DOLSResult:
    alpha: float
    beta: float
    se_alpha: float
    se_beta: float
    n: int
    leads: int
    lags: int
    hac_lag: int
    resid: pd.Series


def dols(y: pd.Series, x: pd.Series, leads: int = 2, lags: int = 2,
         hac_lag: int | None = None) -> DOLSResult:
    """Dynamic OLS estimate of y_t = alpha + beta x_t + sum gamma_j dx_{t-j} + u_t."""
    idx, Y, X_lvl, DX = _dols_design(y, x, leads, lags)
    n = len(Y)
    L = _default_hac_lag(n) if hac_lag is None else int(hac_lag)
    X = np.column_stack([np.ones(n), X_lvl, DX])
    b, cov, resid, _ = _ols_lrv(Y, X, L)
    return DOLSResult(
        alpha=float(b[0]), beta=float(b[1]),
        se_alpha=float(np.sqrt(cov[0, 0])), se_beta=float(np.sqrt(cov[1, 1])),
        n=n, leads=leads, lags=lags, hac_lag=L,
        resid=pd.Series(resid, index=idx, name="dols_resid"),
    )


# ---------------------------------------------------------------------------
# Known-date break
# ---------------------------------------------------------------------------


def _break_design(Y, X_lvl, DX, d):
    n = len(Y)
    return np.column_stack([np.ones(n), X_lvl, d, d * X_lvl, DX])


def dols_break_test(
    y: pd.Series,
    x: pd.Series,
    break_date: str = EXPORT_BAN_REPEAL,
    leads: int = 2,
    lags: int = 2,
    hac_lag: int | None = None,
    n_boot: int = 499,
    seed: int = 20151219,
) -> dict:
    """Known-date break in the intercept and slope of the cointegrating regression.

    Model: y_t = a + b x_t + da D_t + db D_t x_t + leads/lags of dx + u_t,
    with D_t = 1{t >= break_date}.  Returns pre/post coefficients with standard
    errors and Wald tests of (da, db) jointly and individually.

    The level regressor is **centred at the break-date value of x** before
    interacting, so ``da`` measures the jump in the relation *at the break* rather
    than an extrapolated intercept at x = 0.  Without centring, a small slope
    change mechanically produces a large, meaningless intercept change in
    log-price space, where x sits around 4.

    **Use the bootstrap p-values.**  The asymptotic chi-square p-values are
    reported for reference only.  In simulation with no break and a spread
    persistence of 0.9 -- about the level observed on WTI/Brent -- the asymptotic
    joint test rejected in 43% of samples at a nominal 5%: the Newey-West
    long-run variance captures only a fraction of the true one for residuals this
    persistent, and the pre-break sample is short.  The ``p_*_boot`` values come
    from a fixed-regressor AR-sieve bootstrap under the no-break null and are the
    ones to report.
    """
    idx, Y, X_lvl, DX = _dols_design(y, x, leads, lags)
    n = len(Y)
    L = _default_hac_lag(n) if hac_lag is None else int(hac_lag)
    bd = pd.Timestamp(break_date)
    d = (idx >= bd).astype(float)
    if d.sum() < 20 or (1 - d).sum() < 20:
        raise ValueError("break date leaves fewer than 20 observations on one side")

    x_at_break = float(pd.Series(X_lvl, index=idx).loc[:bd].iloc[-1])
    Xc = X_lvl - x_at_break

    Z = _break_design(Y, Xc, DX, d)
    b, cov, resid, _ = _ols_lrv(Y, Z, L)

    w_joint = _wald(b, cov, [2, 3])
    w_int = _wald(b, cov, [2])
    w_slope = _wald(b, cov, [3])

    # --- bootstrap under the no-break null -------------------------------
    X0 = np.column_stack([np.ones(n), Xc, DX])
    b0, *_ = np.linalg.lstsq(X0, Y, rcond=None)
    fitted0 = X0 @ b0
    sieve = _ARSieve(Y - fitted0)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, 3))
    for i in range(n_boot):
        Ystar = fitted0 + sieve.draw(rng)
        bs, covs, _, _ = _ols_lrv(Ystar, Z, L)
        boot[i] = (_wald(bs, covs, [2, 3]), _wald(bs, covs, [2]), _wald(bs, covs, [3]))

    def _pb(obs, col):
        return float((1 + np.sum(boot[:, col] >= obs)) / (n_boot + 1))

    pre_level, post_level = b[0], b[0] + b[2]
    pre_slope, post_slope = b[1], b[1] + b[3]
    se_post_slope = math.sqrt(max(cov[1, 1] + cov[3, 3] + 2 * cov[1, 3], 0.0))
    se_post_level = math.sqrt(max(cov[0, 0] + cov[2, 2] + 2 * cov[0, 2], 0.0))

    return {
        "break_date": bd,
        "n": n,
        "n_pre": int((1 - d).sum()),
        "n_post": int(d.sum()),
        "hac_lag": L,
        "x_at_break": x_at_break,
        "level_pre": float(pre_level), "level_pre_se": float(math.sqrt(cov[0, 0])),
        "level_post": float(post_level), "level_post_se": se_post_level,
        "slope_pre": float(pre_slope), "slope_pre_se": float(math.sqrt(cov[1, 1])),
        "slope_post": float(post_slope), "slope_post_se": se_post_slope,
        "wald_joint": w_joint, "p_joint": float(stats.chi2.sf(w_joint, 2)),
        "wald_level": w_int, "p_level": float(stats.chi2.sf(w_int, 1)),
        "wald_slope": w_slope, "p_slope": float(stats.chi2.sf(w_slope, 1)),
        "p_joint_boot": _pb(w_joint, 0),
        "p_level_boot": _pb(w_int, 1),
        "p_slope_boot": _pb(w_slope, 2),
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Unknown-date break: sup-Wald with fixed-regressor block bootstrap
# ---------------------------------------------------------------------------


def _sup_wald_path(Y, X_lvl, DX, positions, L):
    """Wald statistic for (da, db) and the break-model SSR at each candidate date."""
    n = len(Y)
    wald = np.empty(len(positions))
    ssr = np.empty(len(positions))
    for k, pos in enumerate(positions):
        d = np.zeros(n)
        d[pos:] = 1.0
        Xc = X_lvl - X_lvl[pos - 1]
        Z = _break_design(Y, Xc, DX, d)
        b, cov, resid, _ = _ols_lrv(Y, Z, L)
        wald[k] = _wald(b, cov, [2, 3])
        ssr[k] = float(resid @ resid)
    return wald, ssr


def sup_wald_break(
    y: pd.Series,
    x: pd.Series,
    trim: float = 0.15,
    step: int = 2,
    leads: int = 2,
    lags: int = 2,
    hac_lag: int | None = None,
    n_boot: int = 199,
    hypothesised_date: str | None = EXPORT_BAN_REPEAL,
    seed: int = 20151218,
) -> dict:
    """Is there a break at an unknown date, and where could it be?

    Two different questions, answered by two different statistics.

    **Existence.**  The sup-Wald statistic over all admissible dates (15%
    trimming).  Its null distribution is non-standard with I(1) regressors, so the
    p-value comes from a fixed-regressor bootstrap under the no-break null, in the
    spirit of Hansen (2000), with residuals regenerated by an AR sieve so their
    persistence is preserved.

    **Location.**  The least-squares break-date estimator of Bai (1997), i.e. the
    date minimising the break model's sum of squared residuals, with a bootstrap
    confidence set obtained by resampling under the estimated break model.

    Why the two are kept apart: in simulation, a shift in the mean of a spread with
    persistence around 0.95 is *detected* reliably but *located* poorly -- a
    median error of about six months over 20 replications.  The shift is smaller
    than one stationary standard deviation of the spread, so long excursions of
    the process compete with the true break for the argmax.  Reporting a point
    date would overstate what the data can say; the confidence set does not.  A
    practical corollary on WTI/Brent: the search will be drawn towards extreme
    episodes such as April 2020, which is exactly why the economically motivated
    known-date test is the primary one.
    """
    idx, Y, X_lvl, DX = _dols_design(y, x, leads, lags)
    n = len(Y)
    L = _default_hac_lag(n) if hac_lag is None else int(hac_lag)
    lo, hi = int(math.ceil(trim * n)), int(math.floor((1 - trim) * n))
    positions = np.arange(max(lo, 1), hi, max(1, int(step)))
    rng = np.random.default_rng(seed)

    wald, ssr = _sup_wald_path(Y, X_lvl, DX, positions, L)
    sup_obs = float(np.max(wald))
    k_ls = int(np.argmin(ssr))
    date_ls = pd.Timestamp(idx[positions[k_ls]])

    # --- existence: bootstrap under the no-break null ---------------------
    X0 = np.column_stack([np.ones(n), X_lvl, DX])
    b0, _, u0, _ = _ols_lrv(Y, X0, L)
    fitted0 = X0 @ b0
    sieve0 = _ARSieve(u0)
    sups = np.empty(n_boot)
    for i in range(n_boot):
        Ystar = fitted0 + sieve0.draw(rng)
        w_star, _ = _sup_wald_path(Ystar, X_lvl, DX, positions, L)
        sups[i] = float(np.max(w_star))
    p_exist = float((1 + np.sum(sups >= sup_obs)) / (n_boot + 1))

    # --- location: how would the estimator scatter if the hypothesis held? --
    # A bootstrap around the *estimated* date is centred on that estimate and
    # undercovers precisely when localisation fails.  The informative question is
    # instead: if the break really were at the hypothesised date, how far from it
    # would the least-squares estimator typically land?  That gives both a
    # p-value for "the break is at the hypothesised date" and a direct picture of
    # how imprecise localisation is even when the date is known.
    loc = {"location_p_value": np.nan, "location_range90_under_h0": (pd.NaT, pd.NaT),
           "estimate_minus_hypothesis_weeks": np.nan}
    if hypothesised_date is not None:
        h = pd.Timestamp(hypothesised_date)
        pos_h = int(np.searchsorted(idx, h))
        if lo < pos_h < hi:
            d1 = np.zeros(n)
            d1[pos_h:] = 1.0
            Z1 = _break_design(Y, X_lvl - X_lvl[pos_h - 1], DX, d1)
            b1, *_ = np.linalg.lstsq(Z1, Y, rcond=None)
            fitted1 = Z1 @ b1
            sieve1 = _ARSieve(Y - fitted1)
            est_pos = []
            for _ in range(n_boot):
                Ystar = fitted1 + sieve1.draw(rng)
                _, ssr_star = _sup_wald_path(Ystar, X_lvl, DX, positions, L)
                est_pos.append(int(positions[int(np.argmin(ssr_star))]))
            est_pos = np.array(est_pos)
            dist_boot = np.abs(est_pos - pos_h)
            dist_obs = abs(int(positions[k_ls]) - pos_h)
            q05, q95 = np.quantile(est_pos, [0.05, 0.95]).astype(int)
            loc = {
                "location_p_value": float((1 + np.sum(dist_boot >= dist_obs)) / (n_boot + 1)),
                "location_range90_under_h0": (pd.Timestamp(idx[q05]), pd.Timestamp(idx[q95])),
                "estimate_minus_hypothesis_weeks": float((date_ls - h).days / 7.0),
            }

    return {
        "sup_wald": sup_obs,
        "p_value_bootstrap": p_exist,
        "boot_95": float(np.quantile(sups, 0.95)),
        "break_date_hat": date_ls,
        **loc,
        "wald_argmax_date": pd.Timestamp(idx[positions[int(np.argmax(wald))]]),
        "path": pd.Series(wald, index=idx[positions], name="wald"),
        "ssr_path": pd.Series(ssr, index=idx[positions], name="ssr"),
        "n_boot": n_boot,
        "trim": trim,
    }


# ---------------------------------------------------------------------------
# Time-varying parameters by Kalman filter
# ---------------------------------------------------------------------------


class TVPRegression(MLEModel):
    """y_t = a_t + b_t x_t + e_t with a_t, b_t random walks.

    ``fix_slope=True`` gives the restricted model with a constant slope (its
    state variance is set to zero), used for the likelihood-ratio test.
    """

    def __init__(self, y, x, fix_slope: bool = False):
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        super().__init__(y, k_states=2, k_posdef=2,
                         initialization="approximate_diffuse",
                         loglikelihood_burn=10)
        self.fix_slope = bool(fix_slope)
        design = np.zeros((1, 2, len(y)))
        design[0, 0, :] = 1.0
        design[0, 1, :] = x
        self["design"] = design
        self["transition"] = np.eye(2)
        self["selection"] = np.eye(2)
        self._y_var = float(np.var(y)) if np.var(y) > 0 else 1e-4

    @property
    def param_names(self):
        names = ["log_q_level", "log_r"]
        if not self.fix_slope:
            names.insert(1, "log_q_slope")
        return names

    @property
    def start_params(self):
        v = self._y_var
        if self.fix_slope:
            return np.array([np.log(v * 1e-3), np.log(v * 1e-2)])
        return np.array([np.log(v * 1e-3), np.log(v * 1e-5), np.log(v * 1e-2)])

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        if self.fix_slope:
            q_a, r = np.exp(params[0]), np.exp(params[1])
            q_b = 0.0
        else:
            q_a, q_b, r = np.exp(params[0]), np.exp(params[1]), np.exp(params[2])
        self["state_cov"] = np.diag([q_a, q_b])
        self["obs_cov"] = np.array([[r]])


def fit_tvp_beta(
    y: pd.Series,
    x: pd.Series,
    center: bool = True,
    maxiter: int = 500,
) -> dict:
    """Time-varying intercept and slope; LR test of a constant slope.

    **Read this as a picture, not a test.**  In simulation the boundary-corrected
    LR test had a size of about 3% (slightly conservative: no false alarms) but
    only about 15% power against a genuine random-walk drift in the slope.  With
    the price moving slowly, the random-walk level absorbs most slope variation.
    A non-rejection therefore says little about stability; the bootstrap-calibrated
    DOLS tests carry the inferential weight.

    The test also fails in the other direction under **heteroskedasticity**: the
    model assumes homoskedastic Gaussian noise, and volatility clustering can be
    read as slope variation.  On a synthetic panel with GARCH volatility, t(6)
    shocks and no break at all it returned p = 0.026.  Oil returns cluster
    strongly, so a Kalman rejection on real data is not evidence of a changing
    slope unless the bootstrap-calibrated tests agree.

    ``x`` is centred at its sample mean by default.  With log prices around 4,
    an uncentred intercept and slope are nearly collinear and the filter trades
    one off against the other; centring makes ``a_t`` the relation's level at the
    average price and decouples the two states.  Both filtered (real-time) and
    smoothed (full-sample) paths are returned: use the smoothed path to describe
    history and the filtered one for anything that must be known at time t.
    """
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    xm = float(df["x"].mean()) if center else 0.0
    xc = df["x"].values - xm

    full = TVPRegression(df["y"].values, xc, fix_slope=False).fit(disp=False, maxiter=maxiter)
    restr = TVPRegression(df["y"].values, xc, fix_slope=True).fit(disp=False, maxiter=maxiter)

    lr = max(2.0 * (full.llf - restr.llf), 0.0)
    p_boundary = 0.5 * float(stats.chi2.sf(lr, 1)) if lr > 0 else 1.0

    fs = np.asarray(full.filtered_state)
    fcov = np.asarray(full.filtered_state_cov)
    ss = np.asarray(full.smoothed_state)
    scov = np.asarray(full.smoothed_state_cov)

    paths = pd.DataFrame({
        "level_filtered": fs[0], "slope_filtered": fs[1],
        "slope_filtered_se": np.sqrt(np.maximum(fcov[1, 1], 0)),
        "level_smoothed": ss[0], "slope_smoothed": ss[1],
        "slope_smoothed_se": np.sqrt(np.maximum(scov[1, 1], 0)),
        "level_smoothed_se": np.sqrt(np.maximum(scov[0, 0], 0)),
    }, index=df.index)

    p = full.params
    return {
        "paths": paths,
        "x_center": xm,
        "q_level": float(np.exp(p[0])),
        "q_slope": float(np.exp(p[1])),
        "r": float(np.exp(p[2])),
        "loglike_full": float(full.llf),
        "loglike_constant_slope": float(restr.llf),
        "lr_constant_slope": float(lr),
        "p_constant_slope": p_boundary,
        "result": full,
    }


# ---------------------------------------------------------------------------
# Spread dynamics before and after
# ---------------------------------------------------------------------------


def _spread_stats(sv: np.ndarray, d: np.ndarray, L: int):
    """Regression pieces and test statistics for one spread path."""
    ds = np.diff(sv)
    s_lag = sv[:-1]
    n = len(ds)
    X = np.column_stack([np.ones(n), s_lag, d, d * s_lag])
    b, *_ = np.linalg.lstsq(X, ds, rcond=None)
    resid = ds - X @ b
    Xe = X * resid[:, None]
    S = Xe.T @ Xe / n
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        G = Xe[j:].T @ Xe[:-j] / n
        S += w * (G + G.T)
    XtX_inv = np.linalg.pinv(X.T @ X / n)
    cov = XtX_inv @ S @ XtX_inv / n
    pre, post = ds[d == 0], ds[d == 1]
    lev = stats.levene(pre, post, center="median").statistic
    return b, cov, _wald(b, cov, [2, 3]), _wald(b, cov, [3]), float(lev)


def spread_dynamics_break(
    spread: pd.Series,
    break_date: str = EXPORT_BAN_REPEAL,
    hac_lag: int | None = None,
    n_boot: int = 499,
    seed: int = 20151220,
) -> dict:
    """Mean, mean-reversion speed and volatility of the spread, pre vs post.

    Estimates  ds_t = c + k s_{t-1} + dc D_t + dk D_t s_{t-1} + e_t  with HAC
    inference.  Under cointegration the spread is stationary, so standard
    asymptotics apply to this regression (unlike the levels regression).
    ``rho = 1 + k`` and the half-life is log(0.5) / log(rho).
    """
    s = spread.dropna().astype(float)
    bd = pd.Timestamp(break_date)
    sv = s.values
    idx = s.index[1:]
    d = (idx >= bd).astype(float)
    n = len(idx)
    L = _default_hac_lag(n) if hac_lag is None else int(hac_lag)

    b, cov, w_joint, w_speed, lev_obs = _spread_stats(sv, d, L)

    # --- bootstrap under constant mean and dynamics -----------------------
    # Fit the no-change AR by sieve on the demeaned spread, regenerate paths of
    # the same length, recompute every statistic.  The HAC Wald tests on this
    # regression were oversized in simulation (10-14% at a nominal 5%), so the
    # bootstrap p-values are the ones to report here as well.
    mu = float(np.mean(sv))
    sieve = _ARSieve(sv - mu)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, 3))
    for i in range(n_boot):
        sv_star = mu + sieve.draw(rng)
        _, _, wj, wk, lv = _spread_stats(sv_star, d, L)
        boot[i] = (wj, wk, lv)

    def _pb(obs, col):
        return float((1 + np.sum(boot[:, col] >= obs)) / (n_boot + 1))

    def _hl(rho):
        return float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else np.nan

    k_pre, k_post = b[1], b[1] + b[3]
    rho_pre, rho_post = 1 + k_pre, 1 + k_post
    mean_pre = -b[0] / k_pre if k_pre != 0 else np.nan
    mean_post = -(b[0] + b[2]) / k_post if k_post != 0 else np.nan

    pre = s.loc[s.index < bd]
    post = s.loc[s.index >= bd]

    return {
        "break_date": bd,
        "n_pre": int((1 - d).sum()), "n_post": int(d.sum()),
        "rho_pre": float(rho_pre), "rho_post": float(rho_post),
        "half_life_pre": _hl(rho_pre), "half_life_post": _hl(rho_post),
        "mean_pre": float(mean_pre), "mean_post": float(mean_post),
        "sample_mean_pre": float(pre.mean()), "sample_mean_post": float(post.mean()),
        "vol_pre": float(pre.diff().std()), "vol_post": float(post.diff().std()),
        "wald_joint": w_joint, "p_joint": float(stats.chi2.sf(w_joint, 2)),
        "wald_speed": w_speed, "p_speed": float(stats.chi2.sf(w_speed, 1)),
        "levene_stat": lev_obs,
        "p_joint_boot": _pb(w_joint, 0),
        "p_speed_boot": _pb(w_speed, 1),
        "p_vol_boot": _pb(lev_obs, 2),
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Rolling picture
# ---------------------------------------------------------------------------


def rolling_dols_beta(
    y: pd.Series,
    x: pd.Series,
    window: int = 156,
    step: int = 4,
    leads: int = 2,
    lags: int = 2,
) -> pd.DataFrame:
    """Rolling-window DOLS slope with approximate 95% bands."""
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    rows = []
    for end in range(window, len(df) + 1, step):
        sub = df.iloc[end - window:end]
        try:
            r = dols(sub["y"], sub["x"], leads=leads, lags=lags)
        except Exception:  # noqa: BLE001
            continue
        rows.append({"date": sub.index[-1], "beta": r.beta,
                     "lo": r.beta - 1.96 * r.se_beta, "hi": r.beta + 1.96 * r.se_beta,
                     "alpha": r.alpha})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def holm_adjust(pvalues: dict[str, float]) -> pd.DataFrame:
    """Holm step-down familywise correction for a named p-value family.

    The stability section mixes bootstrap tests from different null-generating
    mechanisms (DOLS level/slope and spread dynamics), so a single shared max-T
    resample is not available without rebuilding the whole bootstrap jointly.
    Holm controls FWER under arbitrary dependence and is therefore the conservative
    default for the primary known-date family.
    """
    items = [(k, float(v)) for k, v in pvalues.items() if np.isfinite(v)]
    if not items:
        return pd.DataFrame(columns=["test", "p_raw", "p_holm", "reject_5pct"])
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adj_sorted = []
    running = 0.0
    for i, (name, pval) in enumerate(items):
        adj = min(1.0, (m - i) * pval)
        running = max(running, adj)
        adj_sorted.append((name, pval, running))
    out = pd.DataFrame(adj_sorted, columns=["test", "p_raw", "p_holm"])
    out["reject_5pct"] = out["p_holm"] < 0.05
    return out.sort_values("test").reset_index(drop=True)


def stability_report(
    log_prices: pd.DataFrame,
    break_date: str = EXPORT_BAN_REPEAL,
    dependent: str = "BRENT",
    regressor: str = "WTI",
    n_boot: int = 199,
    rolling_window: int = 156,
) -> dict:
    """Run every stability diagnostic on one pair and collect the results."""
    y = log_prices[dependent]
    x = log_prices[regressor]
    spread = (log_prices[regressor] - log_prices[dependent]).rename("spread")
    kb = dols_break_test(y, x, break_date=break_date, n_boot=max(n_boot, 199))
    sd = spread_dynamics_break(spread, break_date=break_date, n_boot=max(n_boot, 199))
    family = holm_adjust({
        "level": kb["p_level_boot"],
        "slope": kb["p_slope_boot"],
        "level+slope_joint": kb["p_joint_boot"],
        "spread_speed": sd["p_speed_boot"],
        "spread_volatility": sd["p_vol_boot"],
    })
    return {
        "dols_full": dols(y, x),
        "known_break": kb,
        "known_break_family": family,
        "sup_wald": sup_wald_break(y, x, n_boot=n_boot, hypothesised_date=break_date),
        "tvp": fit_tvp_beta(y, x),
        "spread_dynamics": sd,
        "rolling": rolling_dols_beta(y, x, window=rolling_window),
        "break_date": pd.Timestamp(break_date),
    }


def format_stability(rep: dict) -> str:
    """Plain-text summary of :func:`stability_report`."""
    kb, sw, tvp, sd, dl = (rep["known_break"], rep["sup_wald"], rep["tvp"],
                           rep["spread_dynamics"], rep["dols_full"])
    bd = rep["break_date"].date()
    lines = [
        "=" * 80,
        f"STABILITY OF THE COINTEGRATING RELATION (known date: {bd})",
        "=" * 80,
        f"Full-sample DOLS slope        {dl.beta:.4f}  (se {dl.se_beta:.4f})",
        "",
        "p-values are bootstrap-calibrated; asymptotic values in brackets are for",
        "reference only (they over-reject heavily with residuals this persistent)",
        "",
        "1. Level of the relation (jump at the break, centred at the break-date price)",
        f"   pre {kb['level_pre']:+.4f}   post {kb['level_post']:+.4f}   "
        f"jump {kb['level_post'] - kb['level_pre']:+.4f}   "
        f"p = {kb['p_level_boot']:.4f}  [{kb['p_level']:.4f}]",
        "2. Slope of the relation",
        f"   pre {kb['slope_pre']:.4f} ({kb['slope_pre_se']:.4f})   "
        f"post {kb['slope_post']:.4f} ({kb['slope_post_se']:.4f})   "
        f"p = {kb['p_slope_boot']:.4f}  [{kb['p_slope']:.4f}]",
        f"   joint level + slope      p = {kb['p_joint_boot']:.4f}  [{kb['p_joint']:.4f}]",
        "3. Spread dynamics",
        f"   mean       pre {sd['mean_pre']:+.4f}   post {sd['mean_post']:+.4f}   "
        f"p (mean+speed) = {sd['p_joint_boot']:.4f}",
        f"   rho        pre {sd['rho_pre']:.4f}   post {sd['rho_post']:.4f}   "
        f"p (speed) = {sd['p_speed_boot']:.4f}  [{sd['p_speed']:.4f}]",
        f"   half-life  pre {sd['half_life_pre']:.2f}w   post {sd['half_life_post']:.2f}w",
        f"   weekly vol pre {sd['vol_pre']:.4f}   post {sd['vol_post']:.4f}   "
        f"p (vol) = {sd['p_vol_boot']:.4f}",
        "",
        "Familywise correction across the five primary known-date tests (Holm):",
    ]
    fam = rep.get("known_break_family")
    if fam is not None and not fam.empty:
        for row in fam.sort_values("p_raw").itertuples():
            lines.append(f"   {row.test:20s} raw {row.p_raw:.4f}   Holm {row.p_holm:.4f}   "
                         f"{'REJECT' if row.reject_5pct else 'do not reject'}")
    lines += [
        "",
        "Is there a break at an unknown date? (sup-Wald, 15% trimming)",
        f"   sup-Wald {sw['sup_wald']:.2f}   bootstrap p = {sw['p_value_bootstrap']:.4f}"
        f"  (n_boot {sw['n_boot']})",
        "Where? (least-squares date, Bai 1997)",
        f"   estimate {sw['break_date_hat'].date()}   "
        f"({sw['estimate_minus_hypothesis_weeks']:+.0f} weeks from {bd})",
        f"   if the break really were at {bd}, the estimator would land in",
        f"   {sw['location_range90_under_h0'][0].date()} -> "
        f"{sw['location_range90_under_h0'][1].date()} (90% of the time)",
        f"   p-value that the break is at {bd}: {sw['location_p_value']:.4f}",
        "",
        "Time-varying parameters (Kalman) -- descriptive; low power, see docstring",
        f"   LR test of a constant slope {tvp['lr_constant_slope']:.2f}   "
        f"boundary-corrected p = {tvp['p_constant_slope']:.4f}",
    ]
    return "\n".join(lines)
