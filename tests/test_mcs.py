import numpy as np
import pandas as pd
import pytest

from oil_futures_regime.mcs import (
    mcs_by_cell,
    mcs_summary,
    model_confidence_set,
    romano_wolf_stepdown,
)


def _losses(n=150, gaps=(0.0, 0.0, 0.0), noise=0.3, seed=0):
    """Common component plus per-model shifts, so differences are the signal."""
    rng = np.random.default_rng(seed)
    common = rng.gamma(2.0, 1.0, n)
    return pd.DataFrame(
        {f"m{i}": common + g + rng.normal(0, noise, n) for i, g in enumerate(gaps)}
    )


def test_mcs_keeps_several_models_when_they_are_indistinguishable():
    L = _losses(gaps=(0.0, 0.001, -0.002, 0.0015), noise=0.4, seed=1)
    out = model_confidence_set(L, alpha=0.10, horizon=4, stride=4, n_boot=800)
    assert out["in_mcs_90"].sum() >= 3
    assert out["mcs_pvalue"].max() == 1.0


def test_mcs_eliminates_a_clearly_worse_model():
    L = _losses(gaps=(0.0, 0.02, 3.0), noise=0.3, seed=2)
    out = model_confidence_set(L, alpha=0.10, horizon=1, stride=1, n_boot=800)
    worst = out.set_index("model").loc["m2"]
    assert not worst["in_mcs_90"]
    assert worst["elimination_step"] == 1


def test_mcs_pvalues_are_monotone_in_elimination_order():
    L = _losses(gaps=(0.0, 0.3, 0.8, 1.5, 2.5), noise=0.4, seed=3)
    out = model_confidence_set(L, horizon=4, stride=4, n_boot=600)
    ordered = out.sort_values("elimination_step")
    p = ordered["mcs_pvalue"].values
    assert np.all(np.diff(p) >= -1e-12)   # non-decreasing as elimination proceeds


def test_mcs_is_more_conservative_with_overlapping_windows():
    """Longer blocks acknowledge dependence and should not shrink the set."""
    rng = np.random.default_rng(4)
    n = 200
    common = np.convolve(rng.normal(0, 1, n + 10), np.ones(6) / 6, mode="valid")[:n] + 5
    L = pd.DataFrame({f"m{i}": common + g + rng.normal(0, 0.15, n)
                      for i, g in enumerate((0.0, 0.05, 0.1, 0.2))})
    short = model_confidence_set(L, horizon=1, stride=4, n_boot=800)["in_mcs_90"].sum()
    long = model_confidence_set(L, horizon=20, stride=4, n_boot=800)["in_mcs_90"].sum()
    assert long >= short


def test_mcs_requires_at_least_two_models():
    with pytest.raises(ValueError):
        model_confidence_set(_losses(gaps=(0.0,)), horizon=1, stride=1)


def test_romano_wolf_rejects_true_winners_and_not_losers():
    L = _losses(gaps=(0.0, -0.8, 0.9), noise=0.3, seed=5)
    out = romano_wolf_stepdown(L, baseline="m0", horizon=1, stride=1, n_boot=1000)
    res = out.set_index("model")
    assert res.loc["m1", "reject_at_5pct"]        # genuinely better
    assert not res.loc["m2", "reject_at_5pct"]    # genuinely worse


def test_romano_wolf_adjusted_pvalues_dominate_raw_ones():
    L = _losses(gaps=(0.0, -0.2, -0.15, -0.05, 0.1), noise=0.5, seed=6)
    out = romano_wolf_stepdown(L, baseline="m0", horizon=4, stride=4, n_boot=1000)
    assert (out["p_familywise"] >= out["p_raw"] - 1e-12).all()


def test_romano_wolf_controls_familywise_error_under_the_null():
    """With no true difference, rejections must be rare across many families."""
    rejections = 0
    trials = 30
    for seed in range(trials):
        L = _losses(n=120, gaps=(0.0,) * 6, noise=0.4, seed=100 + seed)
        out = romano_wolf_stepdown(L, baseline="m0", horizon=1, stride=1, n_boot=400)
        rejections += int(out["reject_at_5pct"].any())
    assert rejections <= 0.25 * trials


def test_mcs_by_cell_and_summary_shapes():
    rng = np.random.default_rng(7)
    origins = pd.date_range("2018-01-05", periods=60, freq="4W-FRI")
    rows = []
    for variable in ["WTI", "SPREAD"]:
        for horizon in [1, 20]:
            common = rng.gamma(2.0, 0.02, len(origins))
            for model, shift in [("A", 0.0), ("B", 0.001), ("C", 0.02)]:
                for o, c in zip(origins, common):
                    rows.append({"origin": o, "variable": variable, "model": model,
                                 "horizon_weeks": horizon,
                                 "crps": c + shift + rng.normal(0, 0.002)})
    results = pd.DataFrame(rows)
    table = mcs_by_cell(results, stride=4, n_boot=400)
    assert len(table) == 2 * 2 * 3
    summary = mcs_summary(table)
    assert len(summary) == 4
    assert (summary["n_in_mcs_90"] >= 1).all()
    assert (summary["n_in_mcs_90"] <= summary["n_models"]).all()


def test_rolling_johansen_rank_reports_a_distribution():
    from oil_futures_regime import johansen_rank_summary, rolling_johansen_rank

    rng = np.random.default_rng(41)
    n = 560
    tau = np.cumsum(rng.normal(0.001, 0.03, n)) + 4.2
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.88 * s[t - 1] + rng.normal(0, 0.02)
    idx = pd.date_range("2013-01-04", periods=n, freq="W-FRI")
    Y = pd.DataFrame({"WTI": tau + 0.5 * s, "BRENT": tau - 0.5 * s}, index=idx)

    table = rolling_johansen_rank(Y, oos_start=idx[300], stride=16)
    assert len(table) > 5
    assert {"origin", "rank_95", "rank_99"}.issubset(table.columns)
    assert table["rank_95"].between(0, 2).all()
    summary = johansen_rank_summary(table)
    assert set(summary["level"]) == {"95%", "99%"}
    assert np.isclose(summary[summary["level"] == "95%"]["share_pct"].sum(), 100.0)
