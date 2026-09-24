import numpy as np
import pandas as pd
import pytest

from oil_futures_regime.regimes import (
    build_macro_features,
    effective_sample_size,
    exp_half_life_weights,
    macro_kernel_weights,
    prepare_residual_pool,
    select_macro_bandwidth,
)


def _resid(n=300, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    return pd.DataFrame(rng.normal(0.001, 0.02, size=(n, 2)), index=idx, columns=["WTI", "BRENT"])


def _macro(index, seed=4):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "log_vix": np.log(np.clip(18 + rng.normal(0, 4, len(index)), 8, None)),
            "dlog_dxy": rng.normal(0, 0.005, len(index)),
            "d10y_pctpt": rng.normal(0, 0.08, len(index)),
        },
        index=index,
    )


def test_half_life_weights_normalize_and_recent_is_largest():
    idx = pd.date_range("2020-01-03", periods=104, freq="W-FRI")
    w = exp_half_life_weights(idx, half_life=52)
    assert np.isclose(w.sum(), 1.0)
    assert w.iloc[-1] > w.iloc[0]
    assert np.isclose(w.iloc[-1 - 52] / w.iloc[-1], 0.5, rtol=1e-10)


def test_ess_bounds():
    assert np.isclose(effective_sample_size(np.full(100, 0.01)), 100.0)
    peaked = np.zeros(100)
    peaked[0] = 1.0
    assert np.isclose(effective_sample_size(peaked), 1.0)


def test_macro_features_are_changes_except_vix_state():
    idx = pd.date_range("2019-01-04", periods=50, freq="W-FRI")
    levels = pd.DataFrame(
        {"VIX": np.linspace(15, 25, 50),
         "DXY": np.linspace(90, 100, 50),
         "TNX": np.linspace(20, 30, 50)},
        index=idx,
    )
    feats = build_macro_features(levels)
    assert list(feats.columns) == ["log_vix", "dlog_dxy", "d10y_pctpt"]
    assert len(feats) == 49
    assert (feats["dlog_dxy"] > 0).all()


def test_bandwidth_rule_targets_a_fraction_of_the_pool():
    resid = _resid()
    macro = _macro(resid.index)
    base = exp_half_life_weights(resid.index, 52)
    w, diag = select_macro_bandwidth(macro, macro.iloc[-1], base, min_ess_fraction=0.35)
    assert np.isclose(w.sum(), 1.0)
    if not diag.bandwidth_binding:
        assert diag.ess_fraction >= 0.35 - 1e-9
    tight = macro_kernel_weights(macro, macro.iloc[-1], base, 0.5)
    wide = macro_kernel_weights(macro, macro.iloc[-1], base, 4.0)
    assert effective_sample_size(wide) >= effective_sample_size(tight)


def test_prepare_pool_recenters_under_final_probabilities():
    resid = _resid()
    macro = _macro(resid.index)
    pool, prob, diag = prepare_residual_pool(
        resid, mode="macro", macro_features=macro, origin=resid.index[-1], recenter=True
    )
    assert np.allclose(np.sum(pool.values * prob[:, None], axis=0), 0.0, atol=1e-12)
    assert diag.recentered is True


def test_prepare_pool_without_recentering_keeps_conditional_drift():
    resid = _resid()
    macro = _macro(resid.index)
    pool, prob, diag = prepare_residual_pool(
        resid, mode="macro", macro_features=macro, origin=resid.index[-1], recenter=False
    )
    assert not np.allclose(np.sum(pool.values * prob[:, None], axis=0), 0.0, atol=1e-12)
    assert diag.recentered is False


def test_equal_mode_has_full_ess():
    _, prob, diag = prepare_residual_pool(_resid(), mode="equal")
    assert np.isclose(diag.ess_fraction, 1.0)
    assert np.isclose(prob.sum(), 1.0)


def test_macro_mode_requires_origin():
    with pytest.raises(ValueError):
        prepare_residual_pool(_resid(), mode="macro")


def test_kernel_standardization_uses_only_supplied_history():
    """A shifted future block must not change weights computed on the past."""
    resid = _resid()
    macro = _macro(resid.index)
    base = exp_half_life_weights(resid.index, 52)
    w1 = macro_kernel_weights(macro, macro.iloc[-1], base, 1.0)
    extended = pd.concat([macro, macro.iloc[-20:] + 10.0])
    w2 = macro_kernel_weights(
        extended.iloc[: len(macro)], macro.iloc[-1], base, 1.0
    )
    assert np.allclose(w1, w2)
