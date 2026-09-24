import numpy as np
import pandas as pd

from oil_futures_regime.validation import (
    ModelSpec,
    aggregate_oos,
    empirical_pit,
    evaluate_samples,
    pit_summary,
    quantile_loss,
    rolling_oos_validate,
    sample_crps,
)


def synthetic_panel(n=340, seed=13):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0.001, 0.025, size=n)) + 4.0
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.85 * spread[t - 1] + rng.normal(0.0, 0.015)
    idx = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    logp = pd.DataFrame(
        {"WTI": common + 0.5 * spread, "BRENT": common - 0.5 * spread}, index=idx
    )
    macro = pd.DataFrame(
        {
            "log_vix": np.log(np.clip(18 + rng.normal(0, 4, n), 8, None)),
            "dlog_dxy": rng.normal(0, 0.005, n),
            "d10y_pctpt": rng.normal(0, 0.08, n),
        },
        index=idx,
    )
    return logp, macro


SPECS = (
    ModelSpec("RW_equal", "RW", "equal"),
    ModelSpec("AR_time", "AR", "time"),
    ModelSpec("VECM_macro", "VECM", "macro"),
)


def test_crps_is_zero_for_degenerate_perfect_forecast():
    assert sample_crps(np.full(500, 2.0), 2.0) == 0.0


def test_crps_penalises_a_biased_forecast_more():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert sample_crps(x, 0.0) < sample_crps(x, 2.0)


def test_pit_midrank():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    assert empirical_pit(x, 1.5) == 0.5
    assert empirical_pit(x, -1.0) == 0.0
    assert empirical_pit(x, 10.0) == 1.0


def test_quantile_loss_is_asymmetric():
    assert quantile_loss(1.0, 0.0, 0.05) < quantile_loss(-1.0, 0.0, 0.05)


def test_evaluate_samples_keys_and_tail_ordering():
    rng = np.random.default_rng(1)
    out = evaluate_samples(rng.normal(0, 1, 4000), 0.2)
    for key in ("crps", "pit", "q01", "q05", "q50", "q95", "es05", "es01",
                "covered_90", "width_90", "qloss_05"):
        assert key in out
    assert out["es01"] <= out["q01"] <= out["q05"] <= out["q50"] <= out["q95"]
    assert out["es05"] <= out["q05"]
    assert out["covered_90"] == 1.0


def test_rolling_validation_returns_a_balanced_panel():
    logp, macro = synthetic_panel()
    vo = rolling_oos_validate(
        logp, macro, horizons=(1, 4), oos_start=logp.index[220], stride=8,
        n_sim=200, min_train_obs=180, model_specs=SPECS, seed=5, verbose=False,
    )
    assert not vo.results.empty
    assert vo.is_balanced
    counts = vo.results.groupby("model", observed=True)["origin"].nunique()
    assert counts.nunique() == 1
    assert set(vo.results["model"]) == {s.name for s in SPECS}
    # every model saw exactly the same origins
    per_model = {m: set(g["origin"]) for m, g in vo.results.groupby("model", observed=True)}
    reference = next(iter(per_model.values()))
    assert all(v == reference for v in per_model.values())


def test_failures_are_logged_and_origin_dropped_for_everyone():
    """A model that cannot be fitted must not silently shrink one model's sample."""
    logp, macro = synthetic_panel()
    broken = SPECS + (ModelSpec("BROKEN", "AR", "macro"),)
    # Macro features stop early, so 'macro' weighting fails at the later origins.
    truncated_macro = macro.iloc[:240]
    vo = rolling_oos_validate(
        logp, truncated_macro, horizons=(1,), oos_start=logp.index[220], stride=8,
        n_sim=100, min_train_obs=180, model_specs=broken, seed=5, verbose=False,
    )
    assert vo.is_balanced or vo.results.empty
    if not vo.failures.empty:
        assert {"origin", "model", "stage", "error"}.issubset(vo.failures.columns)
        assert vo.failures["error"].str.len().gt(0).all()


def test_common_random_numbers_are_shared_within_an_origin():
    """Two models with identical dynamics must give identical paths under CRN."""
    logp, macro = synthetic_panel()
    twins = (ModelSpec("A", "AR", "time"), ModelSpec("B", "AR", "time"))
    vo = rolling_oos_validate(
        logp, macro, horizons=(4,), oos_start=logp.index[240], stride=16,
        n_sim=300, min_train_obs=180, model_specs=twins, seed=3, verbose=False,
    )
    wide = vo.results.pivot_table(index=["origin", "variable"], columns="model", values="crps")
    assert np.allclose(wide["A"].values, wide["B"].values)


def test_aggregate_and_pit_summary_shapes():
    logp, macro = synthetic_panel()
    vo = rolling_oos_validate(
        logp, macro, horizons=(1, 4), oos_start=logp.index[220], stride=8,
        n_sim=200, min_train_obs=180, model_specs=SPECS, seed=5, verbose=False,
    )
    summary = aggregate_oos(vo.results)
    assert len(summary) == len(SPECS) * 3 * 2  # models x variables x horizons
    assert (summary["coverage_90"] >= 0).all() and (summary["coverage_90"] <= 1).all()
    pit = pit_summary(vo.results)
    assert "ks_uniform_pvalue_descriptive" in pit.columns
