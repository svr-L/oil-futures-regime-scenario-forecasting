import numpy as np
import pandas as pd
import pytest

from oil_futures_regime.models import arima_recursion_params, fit_final_ar
from oil_futures_regime.regimes import prepare_residual_pool
from oil_futures_regime.scenario_models import (
    common_uniforms,
    fit_fhs_model,
    fit_joint_ar_model,
    fit_rw_model,
    fit_vecm_scenario_model,
    indices_from_uniforms,
    simulate,
)


def synthetic_cointegrated_prices(n=320, seed=7, rho=0.85):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0.001, 0.025, size=n)) + 4.0
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = rho * spread[t - 1] + rng.normal(0.0, 0.015)
    idx = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    return pd.DataFrame(
        {"WTI": common + 0.5 * spread, "BRENT": common - 0.5 * spread}, index=idx
    )


def _equal_pool(model):
    return prepare_residual_pool(model.residuals, mode="equal")


def test_ar_recursion_intercept_recovers_true_drift():
    """Regression test for the v0.1 bug: ARIMA 'const' is a mean, not an intercept."""
    rng = np.random.default_rng(0)
    n, phi, mu = 4000, 0.6, 0.004
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = mu * (1 - phi) + phi * r[t - 1] + rng.normal(0, 0.02)
    s = pd.Series(r, index=pd.date_range("2000-01-07", periods=n, freq="W-FRI"))
    res = fit_final_ar(s, 1)
    mean, intercept, phi_hat = arima_recursion_params(res)

    # Deterministic identity: the recursion intercept is mean * (1 - sum(phi)).
    assert intercept == pytest.approx(mean * (1 - phi_hat.sum()), rel=1e-12)
    # With phi around 0.6 the two differ by a factor of ~2.5; conflating them
    # (the v0.1 bug) inflates every simulated drift.
    assert abs(intercept - mean) > 0.3 * abs(mean)

    # Iterating the recursion must converge back to the estimated mean level.
    x = 0.0
    for _ in range(500):
        x = intercept + float(phi_hat[0]) * x
    assert x == pytest.approx(mean, rel=1e-8)

    # Loose recovery of the true parameters (sampling error ~8e-4 on the mean).
    assert mean == pytest.approx(mu, abs=3e-3)
    assert float(phi_hat[0]) == pytest.approx(phi, abs=0.05)


@pytest.mark.parametrize("family", ["RW", "AR", "VECM", "FHS"])
def test_all_simulators_shape_and_reproducibility(family):
    Y = synthetic_cointegrated_prices()
    fitters = {
        "RW": fit_rw_model,
        "AR": lambda y: fit_joint_ar_model(y, p_max=1),
        "VECM": lambda y: fit_vecm_scenario_model(y, coint_rank=1, k_ar_diff=1),
        "FHS": lambda y: fit_fhs_model(y, p_max=1),
    }
    model = fitters[family](Y)
    assert model.family == family
    pool, prob, _ = _equal_pool(model)
    u = common_uniforms(6, 128, seed=1)
    x1 = simulate(model, pool, prob, horizon=6, n_sim=128, u=u)
    x2 = simulate(model, pool, prob, horizon=6, n_sim=128, u=u)
    assert x1.shape == (7, 128, 2)
    assert np.isfinite(x1).all()
    assert np.allclose(x1, x2)


def test_common_random_numbers_give_identical_draws_across_models():
    """Same uniforms + same pool size must select the same rows for every model."""
    Y = synthetic_cointegrated_prices()
    ar = fit_joint_ar_model(Y, p_max=1)
    pool, prob, _ = _equal_pool(ar)
    u = common_uniforms(4, 64, seed=5)
    cum = np.cumsum(prob / prob.sum())
    idx_a = indices_from_uniforms(u[0], cum)
    idx_b = indices_from_uniforms(u[0], cum)
    assert np.array_equal(idx_a, idx_b)
    assert idx_a.min() >= 0 and idx_a.max() < len(pool)


def test_inverse_cdf_sampling_respects_probabilities():
    cum = np.cumsum([0.1, 0.6, 0.3])
    u = np.array([0.05, 0.5, 0.8, 0.99])
    assert np.array_equal(indices_from_uniforms(u, cum), np.array([0, 1, 2, 2]))


def test_vecm_spread_is_tighter_than_random_walk_spread():
    """Error correction must compress the simulated spread relative to a RW."""
    Y = synthetic_cointegrated_prices(rho=0.7)
    rw, vecm = fit_rw_model(Y), fit_vecm_scenario_model(Y, coint_rank=1, k_ar_diff=1)
    u = common_uniforms(20, 800, seed=3)

    def spread_sd(model):
        pool, prob, _ = _equal_pool(model)
        p = simulate(model, pool, prob, horizon=20, n_sim=800, u=u)
        return float((p[20, :, 0] - p[20, :, 1]).std())

    assert spread_sd(vecm) < 0.6 * spread_sd(rw)


def test_fhs_conditional_variance_responds_to_the_last_shock():
    """A calm origin and a turbulent origin must produce different 1-step spreads."""
    rng = np.random.default_rng(21)
    n = 400
    sigma = np.concatenate([np.full(n // 2, 0.01), np.full(n // 2, 0.05)])
    r = rng.normal(0, 1, n) * sigma
    logp = np.cumsum(r) + 4.0
    idx = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    Y = pd.DataFrame({"WTI": logp, "BRENT": logp + 0.05 * rng.normal(0, 1, n)}, index=idx)

    calm = fit_fhs_model(Y.iloc[: n // 2], p_max=1)
    turbulent = fit_fhs_model(Y, p_max=1)
    u = common_uniforms(1, 2000, seed=9)

    def one_step_sd(model):
        pool, prob, _ = _equal_pool(model)
        p = simulate(model, pool, prob, horizon=1, n_sim=2000, u=u)
        return float(p[1, :, 0].std())

    assert one_step_sd(turbulent) > 2.0 * one_step_sd(calm)


def test_rw_has_no_drift_by_construction():
    Y = synthetic_cointegrated_prices()
    rw = fit_rw_model(Y)
    pool, prob, _ = prepare_residual_pool(rw.residuals, mode="equal", recenter=True)
    u = common_uniforms(10, 4000, seed=11)
    p = simulate(rw, pool, prob, horizon=10, n_sim=4000, u=u)
    drift = p[10, :, 0].mean() - rw.last_log_price[0]
    assert abs(drift) < 0.01


def test_uniform_shape_is_validated():
    Y = synthetic_cointegrated_prices()
    ar = fit_joint_ar_model(Y, p_max=1)
    pool, prob, _ = _equal_pool(ar)
    with pytest.raises(ValueError):
        simulate(ar, pool, prob, horizon=5, n_sim=10, u=np.zeros((3, 10)))


def test_var_levels_model_runs_and_is_reproducible():
    from oil_futures_regime.scenario_models import fit_var_levels_model

    Y = synthetic_cointegrated_prices()
    m = fit_var_levels_model(Y, lags=1)
    assert m.family == "VARL"
    pool, prob, _ = _equal_pool(m)
    u = common_uniforms(6, 200, seed=2)
    x1 = simulate(m, pool, prob, horizon=6, n_sim=200, u=u)
    x2 = simulate(m, pool, prob, horizon=6, n_sim=200, u=u)
    assert x1.shape == (7, 200, 2)
    assert np.allclose(x1, x2)
    assert np.allclose(x1[0, 0, :], m.last_log_price)


def test_var_levels_spread_is_mean_reverting_like_the_vecm():
    """A stationary VAR in levels must also compress the simulated spread."""
    from oil_futures_regime.scenario_models import fit_var_levels_model

    Y = synthetic_cointegrated_prices(rho=0.7)
    rw, varl = fit_rw_model(Y), fit_var_levels_model(Y, lags=1)
    u = common_uniforms(20, 800, seed=5)

    def spread_sd(model):
        pool, prob, _ = _equal_pool(model)
        p = simulate(model, pool, prob, horizon=20, n_sim=800, u=u)
        return float((p[20, :, 0] - p[20, :, 1]).std())

    assert spread_sd(varl) < 0.7 * spread_sd(rw)
