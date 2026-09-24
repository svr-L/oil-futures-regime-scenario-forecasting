import numpy as np
import pandas as pd

from oil_futures_regime.risk_tests import (
    acerbi_szekely_z2,
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
    non_overlapping_subset,
    var_es_report,
)
from oil_futures_regime.significance import (
    best_model_summary,
    block_bootstrap_mean_ci,
    diebold_mariano,
    newey_west_lrv,
    overlap_lag,
    paired_score_table,
)


# --------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------


def test_overlap_lag_matches_origin_spacing():
    assert overlap_lag(1, 4) == 0      # 1-week targets never overlap at stride 4
    assert overlap_lag(20, 4) == 4     # 20-week targets overlap 4 origins deep
    assert overlap_lag(12, 4) == 2
    assert overlap_lag(4, 4) == 0


def test_dm_finds_no_difference_between_identical_losses():
    rng = np.random.default_rng(0)
    loss = rng.gamma(2.0, 1.0, 200)
    out = diebold_mariano(loss, loss.copy(), horizon=1, stride=1)
    assert out["mean_diff"] == 0.0
    assert not np.isfinite(out["dm_stat"]) or abs(out["dm_stat"]) < 1e-8


def test_dm_detects_a_real_and_persistent_advantage():
    rng = np.random.default_rng(1)
    base = rng.gamma(2.0, 1.0, 400)
    better = base - 0.35 + rng.normal(0, 0.05, 400)
    out = diebold_mariano(better, base, horizon=1, stride=1)
    assert out["dm_stat"] < 0            # negative => first argument is better
    assert out["p_value"] < 0.01


def test_dm_is_more_conservative_when_errors_overlap():
    """Autocorrelated differentials must not be treated as independent draws."""
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 500)
    d = np.convolve(e, np.ones(5) / 5, mode="same") + 0.05   # strongly serially correlated
    naive = diebold_mariano(d, np.zeros_like(d), horizon=1, stride=1, lag=0)
    hac = diebold_mariano(d, np.zeros_like(d), horizon=20, stride=4)
    assert abs(hac["dm_stat"]) < abs(naive["dm_stat"])
    assert hac["hac_lag"] == 4


def test_newey_west_matches_sample_variance_at_lag_zero():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 2, 1000)
    assert np.isclose(newey_west_lrv(x, 0), np.var(x), rtol=1e-10)


def test_block_bootstrap_interval_brackets_the_mean():
    rng = np.random.default_rng(4)
    d = rng.normal(0.5, 1.0, 300)
    out = block_bootstrap_mean_ci(d, horizon=4, stride=4, n_boot=1000)
    assert out["lo"] < out["mean"] < out["hi"]
    assert out["p_two_sided"] < 0.05


def _fake_results(seed=0):
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2018-01-05", periods=60, freq="4W-FRI")
    rows = []
    for variable in ["WTI", "SPREAD"]:
        for horizon in [1, 4]:
            for model, shift in [("RW_equal", 0.0), ("GOOD", -0.02), ("BAD", 0.03)]:
                for o in origins:
                    crps = 0.05 + shift + rng.normal(0, 0.004)
                    rows.append({"origin": o, "variable": variable, "model": model,
                                 "horizon_weeks": horizon, "crps": crps})
    return pd.DataFrame(rows)


def test_paired_score_table_ranks_and_flags_significance():
    tab = paired_score_table(_fake_results(), baseline="RW_equal", stride=4, n_boot=400)
    good = tab[tab["model"] == "GOOD"]
    bad = tab[tab["model"] == "BAD"]
    assert (good["improvement_pct"] > 0).all()
    assert (bad["improvement_pct"] < 0).all()
    assert good["significant_5pct"].all()
    assert (tab["eff_independent_n"] <= tab["n_origins"]).all()

    best = best_model_summary(tab)
    assert set(best["best_model"]) == {"GOOD"}
    assert best["beats_baseline_at_5pct"].all()


# --------------------------------------------------------------------------
# VaR / ES backtests
# --------------------------------------------------------------------------


def test_kupiec_accepts_a_correctly_calibrated_tail():
    rng = np.random.default_rng(5)
    hits = (rng.random(500) < 0.05).astype(int)
    out = kupiec_pof(hits, 0.05)
    assert out["p_value"] > 0.05


def test_kupiec_rejects_a_badly_undersized_tail():
    rng = np.random.default_rng(6)
    hits = (rng.random(500) < 0.15).astype(int)
    out = kupiec_pof(hits, 0.05)
    assert out["p_value"] < 0.01
    assert out["rate"] > 0.10


def test_christoffersen_detects_clustered_exceptions():
    clustered = np.zeros(200, dtype=int)
    clustered[50:60] = 1
    clustered[120:130] = 1
    rng = np.random.default_rng(31)
    independent = (rng.random(200) < 0.1).astype(int)
    assert christoffersen_independence(clustered)["p_value"] < 0.01
    # Perfectly regular spacing is dependence too, so the comparison case must
    # be genuinely i.i.d. rather than periodic.
    assert christoffersen_independence(independent)["p_value"] > 0.05


def test_conditional_coverage_combines_both_tests():
    rng = np.random.default_rng(7)
    hits = (rng.random(400) < 0.05).astype(int)
    cc = christoffersen_cc(hits, 0.05)
    assert 0.0 <= cc["p_value"] <= 1.0


def test_non_overlapping_subset_thins_by_horizon():
    origins = pd.Series(pd.date_range("2018-01-05", periods=40, freq="4W-FRI"))
    assert non_overlapping_subset(origins, horizon=1, stride=4).sum() == 40
    assert non_overlapping_subset(origins, horizon=20, stride=4).sum() == 8
    assert non_overlapping_subset(origins, horizon=12, stride=4).sum() == 14


def test_es_z2_is_near_zero_when_the_tail_is_right():
    rng = np.random.default_rng(8)
    n = 3000
    y = rng.normal(0, 1, n)
    var = np.full(n, -1.6449)          # 5% normal quantile
    es = np.full(n, -2.0627)           # normal ES at 5%
    out = acerbi_szekely_z2(y, var, es, 0.05, n_boot=500)
    assert abs(out["z2"]) < 0.5
    assert out["p_value"] > 0.05


def test_es_z2_is_negative_when_realized_tails_are_worse():
    rng = np.random.default_rng(9)
    n = 3000
    y = rng.standard_t(3, n) * 1.2     # much fatter tails than assumed
    var = np.full(n, -1.6449)
    es = np.full(n, -2.0627)
    out = acerbi_szekely_z2(y, var, es, 0.05, n_boot=500)
    assert out["z2"] < 0
    assert out["n_exceptions"] > 0.05 * n


def test_var_es_report_produces_one_row_per_cell_and_level():
    rng = np.random.default_rng(10)
    origins = pd.date_range("2018-01-05", periods=48, freq="4W-FRI")
    rows = []
    for model in ["A", "B"]:
        for h in [1, 20]:
            for o in origins:
                actual = rng.normal(0, 1)
                rows.append({
                    "origin": o, "model": model, "variable": "WTI", "horizon_weeks": h,
                    "actual": actual, "q05": -1.64, "q01": -2.33,
                    "es05": -2.06, "es01": -2.67,
                })
    rep = var_es_report(pd.DataFrame(rows), stride=4)
    assert len(rep) == 2 * 2 * 2                      # models x horizons x alphas
    assert (rep.loc[rep["horizon_weeks"] == 20, "n_used"] == 10).all()   # thinned 5:1
    assert (rep.loc[rep["horizon_weeks"] == 1, "n_used"] == 48).all()
    assert {"kupiec_p", "christoffersen_cc_p", "es_z2"}.issubset(rep.columns)
