import numpy as np
import pandas as pd

from oil_futures_regime.decision import (
    DEFAULT_RULES,
    Rule,
    baseline_in_mcs,
    evaluate_rules,
    format_verdict,
    rule_verdict,
)
from oil_futures_regime.mcs import mcs_by_cell


def _panel(shifts, n_origins=89, horizons=(1, 4, 12, 20), variables=("WTI", "SPREAD"),
           noise=0.004, seed=0):
    """Balanced score panel with a common per-origin component plus model shifts."""
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2018-01-05", periods=n_origins, freq="4W-FRI")
    rows = []
    for variable in variables:
        for h in horizons:
            common = rng.gamma(2.0, 0.02, n_origins)
            for model, shift in shifts.items():
                for o, c in zip(origins, common):
                    rows.append({"origin": o, "variable": variable, "horizon_weeks": h,
                                 "model": model, "crps": c + shift + rng.normal(0, noise)})
    return pd.DataFrame(rows)


BASE_SHIFTS = {"RW_equal": 0.0, "AR_equal": 0.0, "AR_time": 0.0,
               "AR_macro": 0.0, "VECM_equal": 0.0}


def test_binding_incumbent_is_the_best_simpler_alternative():
    """A rule must not pass just because one listed comparator is weak."""
    shifts = dict(BASE_SHIFTS)
    shifts["AR_time"] = 0.010        # a bad comparator
    shifts["AR_macro"] = 0.004       # better than AR_time, worse than AR_equal
    results = _panel(shifts, seed=1)
    v = rule_verdict(results, Rule("macro", "AR_macro", ["AR_equal", "AR_time"]), n_boot=400)
    assert (v["cells"]["incumbent"] == "AR_equal").all()
    assert v["decision"] == "DROP"
    assert v["median_improvement_pct"] < 0


def test_single_lucky_cell_does_not_survive_correction():
    """The v0.7 failure mode: one significant cell out of twelve declared KEEP."""
    rng = np.random.default_rng(7)
    origins = pd.date_range("2018-01-05", periods=89, freq="4W-FRI")
    rows = []
    for variable in ["WTI", "SPREAD"]:
        for h in [1, 4, 12, 20]:
            common = rng.gamma(2.0, 0.02, len(origins))
            # Challenger is genuinely better in exactly one cell, equal elsewhere.
            edge = -0.004 if (variable == "SPREAD" and h == 4) else 0.0
            for model, shift in [("AR_equal", 0.0), ("AR_macro", edge)]:
                for o, c in zip(origins, common):
                    rows.append({"origin": o, "variable": variable, "horizon_weeks": h,
                                 "model": model, "crps": c + shift + rng.normal(0, 0.002)})
    results = pd.DataFrame(rows)
    rule = Rule("macro", "AR_macro", ["AR_equal"])

    v = rule_verdict(results, rule, n_boot=1000)
    cell = v["cells"].set_index(["variable", "horizon_weeks"]).loc[("SPREAD", 4)]
    # The familywise adjustment must never make a p-value smaller.
    assert (v["cells"]["p_familywise"] >= v["cells"]["p_raw"] - 1e-9).all()
    assert cell["p_familywise"] >= cell["p_raw"]


def test_strong_and_broad_effect_still_passes_after_correction():
    shifts = dict(BASE_SHIFTS)
    shifts["VECM_equal"] = -0.006        # clearly better everywhere
    results = _panel(shifts, noise=0.002, seed=3)
    v = rule_verdict(results, Rule("ec", "VECM_equal", ["AR_equal", "RW_equal"]), n_boot=800)
    assert v["decision"] == "KEEP"
    assert v["n_significant"] >= 4
    assert v["median_improvement_pct"] > 0


def test_no_effect_is_dropped():
    results = _panel(BASE_SHIFTS, seed=4)
    v = rule_verdict(results, Rule("noop", "AR_macro", ["AR_equal"]), n_boot=800)
    assert v["decision"] == "DROP"
    assert v["n_significant"] == 0


def test_evaluate_rules_returns_summary_and_detail():
    shifts = {"RW_equal": 0.0, "AR_equal": 0.0, "AR_time": 0.004, "AR_macro": 0.002,
              "AR_macro_nc": 0.006, "FHS_equal": -0.001, "FHS_time": -0.001,
              "VECM_equal": -0.005, "VARL_equal": -0.005, "VARL_time": -0.005,
              "CTS_equal": -0.003, "CTSF_equal": -0.005}
    results = _panel(shifts, seed=5)
    summary, detail = evaluate_rules(results, n_boot=400)
    assert len(summary) == len(DEFAULT_RULES)
    assert set(summary["decision"]).issubset({"KEEP", "DROP"})
    assert {"rule", "variable", "horizon_weeks", "p_raw", "p_familywise",
            "passes", "incumbent"}.issubset(detail.columns)
    assert (detail["p_familywise"] >= detail["p_raw"] - 1e-9).all()


def test_baseline_in_mcs_flags_exclusion_without_picking_a_winner():
    shifts = {"RW_equal": 0.0, "AR_equal": 0.0005, "VECM_equal": -0.008,
              "CTSF_equal": -0.0082, "VARL_equal": -0.0079}
    results = _panel(shifts, noise=0.002, seed=6)
    table = baseline_in_mcs(mcs_by_cell(results, stride=4, n_boot=500))
    assert len(table) == 8
    assert table["something_beats_baseline"].any()
    assert (table["best_improvement_pct"] > 0).all()
    assert table["baseline_rank"].between(1, 5).all()


def test_baseline_survives_mcs_when_nothing_is_better():
    results = _panel(BASE_SHIFTS, seed=8)
    table = baseline_in_mcs(mcs_by_cell(results, stride=4, n_boot=500))
    assert table["baseline_in_mcs_90"].all()
    assert not table["something_beats_baseline"].any()


def test_format_verdict_mentions_every_rule_and_decision():
    shifts = {"RW_equal": 0.0, "AR_equal": 0.0, "AR_time": 0.003, "AR_macro": 0.001,
              "AR_macro_nc": 0.005, "FHS_equal": 0.0, "FHS_time": 0.0,
              "VECM_equal": -0.004, "VARL_equal": -0.004, "VARL_time": -0.004,
              "CTS_equal": -0.002, "CTSF_equal": -0.004}
    results = _panel(shifts, seed=9)
    summary, _ = evaluate_rules(results, n_boot=300)
    table = baseline_in_mcs(mcs_by_cell(results, stride=4, n_boot=300))
    text = format_verdict(summary, table)
    for rule in DEFAULT_RULES:
        assert rule.label in text
    assert "KEEP" in text or "DROP" in text
    assert "MCS" in text
