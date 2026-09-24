"""Commodity-specific state variables and state-dependent WTI/Brent error correction.

The original project conditions residual probabilities on generic macro variables.
This module asks a more economic question: *does the speed of WTI/Brent relative-
price convergence depend on the physical / carry state of the crude market?*

Two public EIA inputs are used:

* WTI NYMEX contracts 1--4 (history ends 5 April 2024), from which curve slope,
  curvature and contango/backwardation are constructed;
* weekly Cushing crude inventories (available from 2004), lagged one week in the
  forecasting state because the Friday stock observation is released after the
  Friday forecast origin.

The forecasting model is deliberately small and interpretable::

    ds_{t+1} = c + lambda s_t + theta' z_t + delta' (s_t * z_t) + eps_{t+1}

where ``s`` is the WTI-Brent log spread and ``z`` is the standardized commodity
state.  Therefore ``phi(z) = 1 + lambda + delta'z`` is the conditional AR
coefficient of the spread.  The state is frozen at its origin value during a
multi-week simulation; this is a conditional forecast, not a hidden forecast of
future inventories or future curve shape.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .validation import evaluate_samples


DEFAULT_STATE_FEATURES = (
    "curve_slope_14",
    "curve_curvature",
    "inventory_z_52",
    "dlog_inventory",
)


def build_wti_curve_features(curve: pd.DataFrame) -> pd.DataFrame:
    """Create economically interpretable WTI curve-state variables from C1--C4.

    All features are in log-price space, so they are scale free.  ``curve_slope_14``
    is the average log carry per contract step from C1 to C4: positive is contango,
    negative is backwardation.  ``backwardation_score`` reverses that sign so
    positive means a tighter front end.  ``curve_curvature`` is a front-end second
    difference using C1/C2/C3.
    """
    need = ["C1", "C2", "C3", "C4"]
    missing = set(need).difference(curve.columns)
    if missing:
        raise ValueError(f"curve is missing columns: {sorted(missing)}")
    c = curve[need].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if (c <= 0).any().any():
        raise ValueError("WTI futures prices must be positive to build log-curve features")
    l = np.log(c)
    out = pd.DataFrame(index=c.index)
    out["curve_slope_12"] = l["C2"] - l["C1"]
    out["curve_slope_14"] = (l["C4"] - l["C1"]) / 3.0
    out["curve_curvature"] = l["C1"] - 2.0 * l["C2"] + l["C3"]
    out["backwardation_score"] = -out["curve_slope_14"]
    out["contango"] = (out["curve_slope_14"] > 0).astype(float)
    return out.sort_index()


def build_inventory_features(
    cushing: pd.Series | pd.DataFrame,
    z_window: int = 52,
    min_periods: int = 26,
) -> pd.DataFrame:
    """Cushing inventory level, change and trailing z-score without future data."""
    if isinstance(cushing, pd.DataFrame):
        if cushing.shape[1] != 1:
            raise ValueError("cushing DataFrame must have exactly one column")
        x = cushing.iloc[:, 0]
    else:
        x = cushing
    x = pd.Series(x, copy=True).astype(float).sort_index().rename("CUSHING")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if (x <= 0).any():
        raise ValueError("Cushing inventories must be positive")
    lx = np.log(x)
    mean = lx.rolling(z_window, min_periods=min_periods).mean()
    sd = lx.rolling(z_window, min_periods=min_periods).std(ddof=1)
    out = pd.DataFrame(index=x.index)
    out["inventory_kbbl"] = x
    out["dlog_inventory"] = lx.diff()
    out["inventory_z_52"] = (lx - mean) / sd.replace(0.0, np.nan)
    return out


def build_commodity_state(
    curve: pd.DataFrame,
    cushing: pd.Series | pd.DataFrame,
    inventory_release_lag_weeks: int = 1,
) -> pd.DataFrame:
    """Join WTI curve and physical inventory state on the curve's weekly dates.

    The EIA stock number for a week ending Friday is released afterwards.  By
    default the inventory block is shifted one weekly observation before joining,
    preventing a subtle look-ahead at Friday forecast origins.
    """
    cf = build_wti_curve_features(curve)
    inv = build_inventory_features(cushing)
    if inventory_release_lag_weeks < 0:
        raise ValueError("inventory_release_lag_weeks must be non-negative")
    inv = inv.shift(int(inventory_release_lag_weeks))
    # EIA series are weekly but occasional calendar labels differ by a day.  A
    # time reindex with a seven-day tolerance is explicit and avoids broad ffill.
    inv = inv.reindex(cf.index, method="ffill", tolerance=pd.Timedelta(days=7))
    return cf.join(inv, how="left").replace([np.inf, -np.inf], np.nan)


@dataclass
class StateDependentSpreadModel:
    feature_columns: tuple[str, ...]
    param_names: tuple[str, ...]
    params: np.ndarray
    cov: np.ndarray
    residuals: pd.Series
    state_mean: pd.Series
    state_std: pd.Series
    last_spread: float
    last_state_z: np.ndarray
    nobs: int
    hac_lag: int

    def state_to_z(self, state_row: pd.Series | np.ndarray) -> np.ndarray:
        if len(self.feature_columns) == 0:
            return np.empty(0, dtype=float)
        if isinstance(state_row, pd.Series):
            x = state_row.loc[list(self.feature_columns)].astype(float)
        else:
            x = pd.Series(np.asarray(state_row, dtype=float), index=self.feature_columns)
        return ((x - self.state_mean) / self.state_std).to_numpy(dtype=float)

    @property
    def lam(self) -> float:
        return float(self.params[1])

    @property
    def delta(self) -> np.ndarray:
        k = len(self.feature_columns)
        return self.params[2 + k: 2 + 2 * k] if k else np.empty(0)

    def conditional_phi(self, state_z: np.ndarray | None = None) -> float:
        z = self.last_state_z if state_z is None else np.asarray(state_z, dtype=float)
        return float(1.0 + self.lam + (self.delta @ z if len(z) else 0.0))

    def conditional_half_life(self, state_z: np.ndarray | None = None) -> float:
        phi = self.conditional_phi(state_z)
        if not (0.0 < phi < 1.0):
            return np.nan
        return float(math.log(0.5) / math.log(phi))


def _design_frame(spread: pd.Series, state: pd.DataFrame, features: tuple[str, ...]):
    s = pd.Series(spread, copy=True).astype(float).sort_index().rename("spread")
    if features:
        missing = set(features).difference(state.columns)
        if missing:
            raise ValueError(f"state is missing features: {sorted(missing)}")
        z = state.loc[:, list(features)].astype(float).sort_index()
    else:
        z = pd.DataFrame(index=s.index)
    df = pd.concat([s, z], axis=1).sort_index()
    df["ds"] = df["spread"].diff()
    df["s_lag"] = df["spread"].shift(1)
    for col in features:
        df[f"{col}_lag"] = df[col].shift(1)
    return df.dropna()


def fit_state_dependent_spread(
    spread: pd.Series,
    state: pd.DataFrame,
    features=DEFAULT_STATE_FEATURES,
    hac_lag: int = 4,
) -> StateDependentSpreadModel:
    """Estimate state-dependent error correction using only lagged states."""
    features = tuple(features)
    df = _design_frame(spread, state, features)
    if len(df) < max(60, 8 + 4 * len(features)):
        raise ValueError(f"too few aligned observations for state-dependent fit: {len(df)}")

    lag_cols = [f"{c}_lag" for c in features]
    if features:
        mean = df[lag_cols].mean()
        std = df[lag_cols].std(ddof=1).replace(0.0, np.nan)
        if std.isna().any():
            bad = list(std[std.isna()].index)
            raise ValueError(f"zero-variance state features: {bad}")
        Z = (df[lag_cols] - mean) / std
        Z.columns = list(features)
    else:
        mean = pd.Series(dtype=float)
        std = pd.Series(dtype=float)
        Z = pd.DataFrame(index=df.index)

    Xparts = [pd.Series(1.0, index=df.index, name="const"), df["s_lag"]]
    names = ["const", "lambda"]
    for c in features:
        Xparts.append(Z[c].rename(c))
        names.append(c)
    for c in features:
        Xparts.append((df["s_lag"] * Z[c]).rename(f"spread_x_{c}"))
        names.append(f"spread_x_{c}")
    X = pd.concat(Xparts, axis=1)
    y = df["ds"]
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lag)})

    # State known at the final origin; the state frame has already been release-lagged.
    last_date = df.index[-1]
    current = state.loc[:last_date, list(features)].dropna().iloc[-1] if features else pd.Series(dtype=float)
    if features:
        raw_mean = pd.Series(mean.to_numpy(), index=features)
        raw_std = pd.Series(std.to_numpy(), index=features)
        last_z = ((current - raw_mean) / raw_std).to_numpy(dtype=float)
    else:
        raw_mean, raw_std, last_z = mean, std, np.empty(0)

    return StateDependentSpreadModel(
        feature_columns=features,
        param_names=tuple(names),
        params=np.asarray(fit.params, dtype=float),
        cov=np.asarray(fit.cov_params(), dtype=float),
        residuals=pd.Series(fit.resid, index=df.index, name="spread_innovation"),
        state_mean=raw_mean,
        state_std=raw_std,
        last_spread=float(spread.loc[:last_date].iloc[-1]),
        last_state_z=last_z,
        nobs=int(fit.nobs),
        hac_lag=int(hac_lag),
    )


def interaction_wald(model: StateDependentSpreadModel) -> dict:
    """HAC Wald test that all spread×state interaction coefficients are zero."""
    k = len(model.feature_columns)
    if k == 0:
        return {"wald_stat": np.nan, "df": 0, "p_value": np.nan}
    idx = np.arange(2 + k, 2 + 2 * k)
    b = model.params[idx]
    V = model.cov[np.ix_(idx, idx)]
    try:
        stat = float(b @ np.linalg.pinv(V) @ b)
    except np.linalg.LinAlgError:
        stat = np.nan
    p = float(stats.chi2.sf(stat, k)) if np.isfinite(stat) else np.nan
    return {"wald_stat": stat, "df": k, "p_value": p}


def conditional_reversion_table(
    model: StateDependentSpreadModel,
    states: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Conditional persistence/half-life at representative standardized states."""
    k = len(model.feature_columns)
    if states is None:
        states = {
            "mean_state": np.zeros(k),
            "current_state": model.last_state_z,
        }
        if k:
            # One-standard-deviation move in the WTI curve slope only, holding the
            # other states at their training means.
            j = list(model.feature_columns).index("curve_slope_14") if "curve_slope_14" in model.feature_columns else 0
            lo = np.zeros(k); lo[j] = -1.0
            hi = np.zeros(k); hi[j] = +1.0
            states["backwardation_1sd"] = lo
            states["contango_1sd"] = hi
    rows = []
    for label, z in states.items():
        phi = model.conditional_phi(z)
        rows.append({"state": label, "phi": phi,
                     "half_life_weeks": model.conditional_half_life(z)})
    return pd.DataFrame(rows)


def simulate_spread_paths(
    model: StateDependentSpreadModel,
    horizon: int,
    n_sim: int = 5000,
    state_z: np.ndarray | None = None,
    u: np.ndarray | None = None,
    seed: int = 1234,
) -> np.ndarray:
    """Conditional spread paths with origin state frozen over the horizon."""
    z = model.last_state_z if state_z is None else np.asarray(state_z, dtype=float)
    k = len(model.feature_columns)
    if len(z) != k:
        raise ValueError("state_z has the wrong dimension")
    resid = np.asarray(model.residuals, dtype=float)
    resid = resid[np.isfinite(resid)]
    resid = resid - resid.mean()
    if len(resid) < 20:
        raise ValueError("too few residuals to bootstrap")
    if u is None:
        u = np.random.default_rng(seed).random((horizon, n_sim))
    u = np.asarray(u, dtype=float)
    if u.shape != (horizon, n_sim):
        raise ValueError(f"u must have shape {(horizon, n_sim)}")
    pool = np.sort(resid)
    idx = np.minimum((u * len(pool)).astype(int), len(pool) - 1)

    c = model.params[0]
    lam = model.params[1]
    theta = model.params[2:2 + k] if k else np.empty(0)
    delta = model.delta
    s = np.full(n_sim, model.last_spread, dtype=float)
    paths = np.empty((horizon + 1, n_sim), dtype=float)
    paths[0] = s
    direct = float(theta @ z) if k else 0.0
    interaction = float(delta @ z) if k else 0.0
    for h in range(1, horizon + 1):
        ds = c + (lam + interaction) * s + direct + pool[idx[h - 1]]
        s = s + ds
        paths[h] = s
    return paths


def rolling_state_dependent_oos(
    spread: pd.Series,
    state: pd.DataFrame,
    features=DEFAULT_STATE_FEATURES,
    horizons=(1, 4, 12, 20),
    oos_start="2018-01-01",
    stride: int = 4,
    min_train_obs: int = 180,
    n_sim: int = 3000,
    hac_lag: int = 4,
    seed: int = 20260924,
) -> pd.DataFrame:
    """Balanced OOS comparison: unconditional spread AR(1) vs state-dependent EC.

    Both models use the same residual uniforms at each origin.  State variables are
    standardized inside each training window and only values observable by the
    origin enter the forecast.  The future state is frozen at its origin value.
    """
    s = pd.Series(spread, copy=True).astype(float).sort_index().rename("SPREAD")
    features = tuple(features)
    aligned = pd.concat([s, state.loc[:, list(features)]], axis=1).dropna()
    horizons = sorted({int(h) for h in horizons})
    max_h = max(horizons)
    # Target observations come from the full spread series, not only dates where
    # EIA state is available.  Origins, however, require a complete current state.
    origins = [d for d in aligned.index if d >= pd.Timestamp(oos_start) and d in s.index]
    origins = origins[::stride]
    rows = []
    rng = np.random.default_rng(seed)
    for origin_no, origin in enumerate(origins):
        pos = s.index.get_indexer([origin])[0]
        if pos < 0 or pos + max_h >= len(s):
            continue
        train_s = s.loc[:origin]
        train_state = state.loc[:origin]
        if len(train_s.dropna()) < min_train_obs:
            continue
        try:
            base = fit_state_dependent_spread(train_s, train_state, features=(), hac_lag=hac_lag)
            sd = fit_state_dependent_spread(train_s, train_state, features=features, hac_lag=hac_lag)
        except Exception:
            continue
        u = rng.random((max_h, n_sim))
        pb = simulate_spread_paths(base, max_h, n_sim=n_sim, u=u)
        ps = simulate_spread_paths(sd, max_h, n_sim=n_sim, u=u)
        for h in horizons:
            target_date = s.index[pos + h]
            y = float(s.iloc[pos + h])
            for name, paths in (("SPREAD_AR1", pb), ("SD_EC", ps)):
                row = {
                    "origin": pd.Timestamp(origin),
                    "target_date": pd.Timestamp(target_date),
                    "horizon_weeks": int(h),
                    "model": name,
                    "family": "SPREAD_STATE",
                    "variable": "SPREAD",
                }
                row.update(evaluate_samples(paths[h], y))
                rows.append(row)
    return pd.DataFrame(rows)
