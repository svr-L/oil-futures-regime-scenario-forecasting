import numpy as np
import pandas as pd
import pytest

from oil_futures_regime.stability import (
    _ARSieve,
    dols,
    dols_break_test,
    fit_tvp_beta,
    rolling_dols_beta,
    spread_dynamics_break,
    sup_wald_break,
)
from oil_futures_regime.validation import ModelSpec, rolling_oos_validate
from oil_futures_regime.robustness import relative_gain_by_period

BREAK = "2015-12-18"


def pair(n=636, seed=5, planted_break=False, rho0=0.90):
    """Cointegrated pair; optionally a repeal-shaped break in the spread."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2013-01-04", periods=n, freq="W-FRI")
    tau = np.cumsum(0.0005 + rng.normal(0, 0.045, n)) + 4.3
    s = np.zeros(n)
    m = np.zeros(n)
    rho = np.full(n, rho0)
    if planted_break:
        b = int(np.searchsorted(idx, pd.Timestamp(BREAK)))
        m[:b], m[b:] = -0.08, -0.03
        rho[:b], rho[b:] = 0.95, 0.80
    s[0] = m[0]
    for t in range(1, n):
        s[t] = m[t] + rho[t] * (s[t - 1] - m[t]) + rng.normal(0, 0.018)
    return pd.DataFrame({"WTI": tau + 0.5 * s, "BRENT": tau - 0.5 * s}, index=idx)


def test_dols_recovers_unit_slope():
    Y = pair()
    r = dols(Y["BRENT"], Y["WTI"])
    assert r.beta == pytest.approx(1.0, abs=0.05)
    assert r.se_beta > 0


def test_known_date_test_finds_a_planted_level_break_and_not_a_slope_break():
    Y = pair(planted_break=True)
    r = dols_break_test(Y["BRENT"], Y["WTI"], break_date=BREAK, n_boot=199)
    # Brent rises relative to WTI by the narrowing of the WTI discount (0.05).
    assert (r["level_post"] - r["level_pre"]) == pytest.approx(-0.05, abs=0.02)
    assert r["p_level_boot"] < 0.05
    assert r["p_slope_boot"] > 0.05


def test_bootstrap_fixes_the_oversized_asymptotic_test():
    """The v0.9 calibration finding, frozen as a regression test.

    Under no break with persistent residuals the asymptotic chi-square test
    rejects far too often; the sieve-bootstrap p-values must not.
    """
    asy, boot = [], []
    for seed in range(40):
        Y = pair(seed=1000 + seed, rho0=0.90)
        r = dols_break_test(Y["BRENT"], Y["WTI"], break_date=BREAK, n_boot=99)
        asy.append(r["p_joint"] < 0.05)
        boot.append(r["p_joint_boot"] < 0.05)
    assert np.mean(asy) > 0.20          # the problem is real
    assert np.mean(boot) <= 0.15        # and the bootstrap removes it


def test_spread_dynamics_detects_faster_mean_reversion():
    Y = pair(planted_break=True)
    d = spread_dynamics_break(Y["WTI"] - Y["BRENT"], break_date=BREAK, n_boot=199)
    assert d["rho_pre"] > d["rho_post"]
    assert d["half_life_pre"] > d["half_life_post"]
    assert d["mean_post"] > d["mean_pre"]          # discount narrows
    assert d["p_speed_boot"] < 0.05
    assert d["p_vol_boot"] > 0.05                   # volatility was not changed


def test_spread_dynamics_quiet_under_the_null():
    Y = pair(seed=77)
    d = spread_dynamics_break(Y["WTI"] - Y["BRENT"], break_date=BREAK, n_boot=199)
    assert d["p_joint_boot"] > 0.05


def test_sup_wald_existence_and_location_logic():
    # The unknown-date test has only moderate power (~50% in simulation against
    # this break), so asserting a rejection on one seed would test luck, not
    # code.  Compare the same seed with and without the break instead.
    Y = pair(planted_break=True)
    Y0 = pair(planted_break=False)
    r = sup_wald_break(Y["BRENT"], Y["WTI"], n_boot=49, step=3, hypothesised_date=BREAK)
    r0 = sup_wald_break(Y0["BRENT"], Y0["WTI"], n_boot=19, step=3, hypothesised_date=None)
    assert r["sup_wald"] > r0["sup_wald"]
    assert 0.0 < r["p_value_bootstrap"] <= 1.0
    lo, hi = r["location_range90_under_h0"]
    assert lo <= pd.Timestamp(BREAK) <= hi
    # The planted break is at the hypothesised date: location must not be rejected.
    assert r["location_p_value"] > 0.05


def test_ar_sieve_preserves_persistence():
    rng = np.random.default_rng(3)
    u = np.zeros(3000)
    for t in range(1, 3000):
        u[t] = 0.9 * u[t - 1] + rng.normal()
    draw = _ARSieve(u).draw(np.random.default_rng(4))
    lag1 = np.corrcoef(draw[1:], draw[:-1])[0, 1]
    assert lag1 == pytest.approx(0.9, abs=0.05)


def test_tvp_is_quiet_when_the_slope_is_constant():
    Y = pair(seed=9)
    r = fit_tvp_beta(Y["BRENT"], Y["WTI"])
    assert r["p_constant_slope"] > 0.05
    assert {"slope_smoothed", "slope_filtered", "level_smoothed"}.issubset(r["paths"].columns)


def test_rolling_dols_returns_bands():
    Y = pair()
    roll = rolling_dols_beta(Y["BRENT"], Y["WTI"], window=156, step=26)
    assert len(roll) > 5
    assert (roll["lo"] < roll["beta"]).all() and (roll["beta"] < roll["hi"]).all()


# --------------------------------------------------------------------------
# Windowed estimation inside the harness
# --------------------------------------------------------------------------


def _macro(index, seed=4):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"log_vix": np.log(np.clip(18 + rng.normal(0, 4, len(index)), 8, None)),
                         "dlog_dxy": rng.normal(0, 0.005, len(index)),
                         "d10y_pctpt": rng.normal(0, 0.08, len(index))}, index=index)


def test_harness_respects_estimation_windows_and_stays_balanced():
    Y = pair(n=500, seed=21)
    specs = (
        ModelSpec("VECM_equal", "VECM", "equal"),
        ModelSpec("VECM_post", "VECM", "equal", train_start=BREAK),
        ModelSpec("VECM_roll", "VECM", "equal", window=120),
    )
    vo = rolling_oos_validate(Y, _macro(Y.index), horizons=(4,), oos_start=Y.index[300],
                              stride=16, n_sim=200, model_specs=specs,
                              variables=("SPREAD",), seed=1, verbose=False)
    assert vo.is_balanced
    pools = vo.results.groupby("model")["n_pool"].max()
    assert pools["VECM_roll"] <= 120
    assert pools["VECM_post"] < pools["VECM_equal"]
    assert set(vo.results["train_start"].dropna()) == {BREAK}


def test_window_too_short_drops_the_origin_for_everyone():
    Y = pair(n=420, seed=22)
    specs = (
        ModelSpec("VECM_equal", "VECM", "equal"),
        ModelSpec("VECM_late", "VECM", "equal", train_start="2019-06-07"),
    )
    vo = rolling_oos_validate(Y, _macro(Y.index), horizons=(4,), oos_start=Y.index[300],
                              stride=16, n_sim=100, model_specs=specs,
                              variables=("SPREAD",), min_window_obs=100, seed=1, verbose=False)
    assert vo.is_balanced or vo.results.empty
    assert (vo.failures["stage"] == "fit").any()


def test_relative_gain_by_period_shapes():
    rng = np.random.default_rng(5)
    origins = pd.date_range("2018-01-05", periods=60, freq="4W-FRI")
    rows = []
    for o in origins:
        for h in (12, 20):
            base = rng.gamma(2, 0.01)
            rows += [{"origin": o, "horizon_weeks": h, "variable": "SPREAD",
                      "model": "A", "crps": base},
                     {"origin": o, "horizon_weeks": h, "variable": "SPREAD",
                      "model": "B", "crps": base * 0.9}]
    t = relative_gain_by_period(pd.DataFrame(rows), challenger="B", incumbent="A")
    assert "n_origins" in t.columns
    assert np.allclose(t[[12, 20]].values, 10.0)


def test_holm_adjust_controls_familywise_monotonically():
    from oil_futures_regime import holm_adjust
    out = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.20, "d": 0.9})
    assert (out["p_holm"] >= out["p_raw"]).all()
    by_raw = out.sort_values("p_raw")
    assert (np.diff(by_raw["p_holm"].to_numpy()) >= -1e-12).all()
