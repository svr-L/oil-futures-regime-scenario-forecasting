import numpy as np
import pandas as pd

from oil_futures_regime import (
    build_commodity_state,
    build_inventory_features,
    build_wti_curve_features,
    conditional_reversion_table,
    fit_state_dependent_spread,
    interaction_wald,
    rolling_state_dependent_oos,
    simulate_spread_paths,
)


def _synthetic_state_dependent(n=420, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2013-01-04", periods=n, freq="W-FRI")
    z = np.sin(np.arange(n) / 17.0) + 0.25 * rng.normal(size=n)
    state = pd.DataFrame({"curve_slope_14": z}, index=idx)
    s = np.zeros(n)
    for t in range(1, n):
        # Persistence changes materially with the lagged curve state.
        phi = np.clip(0.82 + 0.10 * z[t - 1], 0.55, 0.97)
        s[t] = phi * s[t - 1] + rng.normal(0, 0.018)
    return pd.Series(s, index=idx, name="SPREAD"), state


def test_curve_feature_signs_and_curvature():
    idx = pd.date_range("2020-01-03", periods=3, freq="W-FRI")
    curve = pd.DataFrame({
        "C1": [50, 50, 50], "C2": [51, 49, 50],
        "C3": [52, 48, 50], "C4": [53, 47, 50],
    }, index=idx)
    f = build_wti_curve_features(curve)
    assert f.loc[idx[0], "curve_slope_14"] > 0
    assert f.loc[idx[1], "curve_slope_14"] < 0
    assert f.loc[idx[0], "contango"] == 1.0
    assert f.loc[idx[1], "backwardation_score"] > 0
    assert abs(f.loc[idx[2], "curve_curvature"]) < 1e-12


def test_inventory_features_and_release_lag():
    idx = pd.date_range("2020-01-03", periods=80, freq="W-FRI")
    cushing = pd.Series(np.linspace(30_000, 40_000, len(idx)), index=idx)
    curve = pd.DataFrame({
        "C1": 50.0, "C2": 50.5, "C3": 51.0, "C4": 51.5,
    }, index=idx)
    inv = build_inventory_features(cushing)
    st = build_commodity_state(curve, cushing, inventory_release_lag_weeks=1)
    # The level observed for week t becomes usable at the next weekly origin.
    assert np.isclose(st.loc[idx[40], "inventory_kbbl"], cushing.loc[idx[39]])
    assert np.isfinite(inv["inventory_z_52"].iloc[-1])


def test_state_dependent_fit_finds_interaction_and_simulates():
    spread, state = _synthetic_state_dependent()
    fit = fit_state_dependent_spread(spread, state, features=("curve_slope_14",), hac_lag=4)
    w = interaction_wald(fit)
    assert fit.nobs > 300
    assert w["p_value"] < 0.05
    tab = conditional_reversion_table(fit)
    assert {"mean_state", "current_state"}.issubset(set(tab["state"]))
    u = np.full((4, 200), 0.5)
    paths = simulate_spread_paths(fit, 4, n_sim=200, u=u)
    assert paths.shape == (5, 200)
    assert np.all(np.isfinite(paths))


def test_state_dependent_oos_is_balanced():
    spread, state = _synthetic_state_dependent(n=360)
    out = rolling_state_dependent_oos(
        spread, state, features=("curve_slope_14",), horizons=(1, 4),
        oos_start="2018-01-01", stride=8, min_train_obs=160, n_sim=250, seed=4,
    )
    assert not out.empty
    counts = out.groupby("model")["origin"].nunique()
    assert counts.nunique() == 1
    assert set(out["model"]) == {"SPREAD_AR1", "SD_EC"}
    assert set(out["horizon_weeks"]) == {1, 4}
