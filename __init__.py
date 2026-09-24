"""Oil futures regime-aware scenario forecasting toolkit."""

__version__ = "1.0.0"

from .cointegration import (
    estimate_spread_half_life,
    fit_vecm,
    johansen_rank_summary,
    rolling_johansen_rank,
    var_johansen_summary,
)
from .data import (
    EIA_CUSHING_XLS,
    EIA_WTI_CURVE_XLS,
    download_weekly_close,
    get_weekly_state,
    load_eia_cushing,
    load_eia_wti_curve,
    load_long_oil_history,
    load_or_download,
    roll_gap_diagnostics,
    spread_jump_summary,
)
from .common_trend import (
    CommonTrendSpread,
    CTSScenarioModel,
    cts_state_summary,
    filtered_states,
    fit_cts_model,
    simulate_cts_paths,
)
from .decision import (
    DEFAULT_RULES,
    STABILITY_RULES,
    Rule,
    baseline_in_mcs,
    evaluate_rules,
    format_verdict,
    rule_verdict,
)
from .diagnostics import adf_kpss_table, fit_ar1_summary, independence_diagnostics
from .mcs import (
    mcs_by_cell,
    mcs_summary,
    model_confidence_set,
    romano_wolf_stepdown,
)
from .models import (
    arima_recursion_params,
    choose_p_bic_with_lb,
    extract_ar_params,
    fit_ar_orders,
    fit_final_ar,
    simulate_ar_bootstrap,
)
from .regimes import (
    WeightDiagnostics,
    build_macro_features,
    effective_sample_size,
    exp_half_life_weights,
    macro_kernel_weights,
    prepare_residual_pool,
    select_macro_bandwidth,
    weighted_mean_cov,
)
from .robustness import (
    exclude_period,
    influence_of_worst_origins,
    mean_vs_median_table,
    ranking_stability,
    relative_gain_by_period,
    split_dev_holdout,
)
from .risk_tests import (
    acerbi_szekely_z2,
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
    non_overlapping_subset,
    var_es_report,
)
from .scenario_models import (
    ARSeriesSpec,
    FHSScenarioModel,
    JointARScenarioModel,
    RWScenarioModel,
    VARLevelsScenarioModel,
    VECMScenarioModel,
    common_uniforms,
    fit_fhs_model,
    fit_joint_ar_model,
    fit_rw_model,
    fit_var_levels_model,
    fit_vecm_scenario_model,
    path_frame,
    simulate,
    simulate_fhs_paths,
    simulate_joint_ar_paths,
    simulate_rw_paths,
    simulate_var_levels_paths,
    simulate_vecm_paths,
)
from .significance import (
    best_model_summary,
    block_bootstrap_mean_ci,
    diebold_mariano,
    overlap_lag,
    paired_score_table,
)
from .stability import (
    EXPORT_BAN_REPEAL,
    TVPRegression,
    dols,
    dols_break_test,
    fit_tvp_beta,
    format_stability,
    holm_adjust,
    rolling_dols_beta,
    spread_dynamics_break,
    stability_report,
    sup_wald_break,
)
from .commodity_state import (
    DEFAULT_STATE_FEATURES,
    StateDependentSpreadModel,
    build_commodity_state,
    build_inventory_features,
    build_wti_curve_features,
    conditional_reversion_table,
    fit_state_dependent_spread,
    interaction_wald,
    rolling_state_dependent_oos,
    simulate_spread_paths,
)
from .state_space import fit_dynamic_factor_model
from .validation import (
    DEFAULT_MODEL_SPECS,
    FAST_MODEL_SPECS,
    ModelSpec,
    ValidationOutput,
    aggregate_oos,
    stability_model_specs,
    empirical_pit,
    evaluate_samples,
    pit_summary,
    quantile_loss,
    rolling_oos_validate,
    sample_crps,
)

__all__ = [
    "__version__",
    # data
    "load_or_download", "download_weekly_close", "get_weekly_state",
    "load_long_oil_history", "load_eia_wti_curve", "load_eia_cushing",
    "EIA_WTI_CURVE_XLS", "EIA_CUSHING_XLS",
    "roll_gap_diagnostics", "spread_jump_summary",
    # diagnostics
    "adf_kpss_table", "independence_diagnostics", "fit_ar1_summary",
    # AR utilities
    "fit_ar_orders", "choose_p_bic_with_lb", "fit_final_ar",
    "arima_recursion_params", "extract_ar_params", "simulate_ar_bootstrap",
    # probabilities
    "WeightDiagnostics", "exp_half_life_weights", "effective_sample_size",
    "weighted_mean_cov", "build_macro_features", "macro_kernel_weights",
    "select_macro_bandwidth", "prepare_residual_pool",
    # cointegration / state space
    "var_johansen_summary", "estimate_spread_half_life", "fit_vecm",
    "rolling_johansen_rank", "johansen_rank_summary",
    "fit_dynamic_factor_model",
    # cointegration-consistent state space
    "CommonTrendSpread", "CTSScenarioModel", "fit_cts_model",
    "simulate_cts_paths", "cts_state_summary", "filtered_states",
    # scenario models
    "ARSeriesSpec", "RWScenarioModel", "JointARScenarioModel",
    "VECMScenarioModel", "FHSScenarioModel", "VARLevelsScenarioModel",
    "fit_rw_model", "fit_joint_ar_model", "fit_vecm_scenario_model", "fit_fhs_model",
    "fit_var_levels_model", "simulate_var_levels_paths",
    "simulate_rw_paths", "simulate_joint_ar_paths", "simulate_vecm_paths",
    "simulate_fhs_paths", "simulate", "common_uniforms", "path_frame",
    # validation
    "ModelSpec", "ValidationOutput", "DEFAULT_MODEL_SPECS", "FAST_MODEL_SPECS",
    "rolling_oos_validate", "aggregate_oos", "pit_summary", "evaluate_samples",
    "sample_crps", "quantile_loss", "empirical_pit",
    # significance
    "diebold_mariano", "block_bootstrap_mean_ci", "paired_score_table",
    "best_model_summary", "overlap_lag",
    # multiple-comparison control
    "model_confidence_set", "romano_wolf_stepdown", "mcs_by_cell", "mcs_summary",
    # stability of the cointegrating relation
    "EXPORT_BAN_REPEAL", "dols", "dols_break_test", "sup_wald_break", "fit_tvp_beta",
    "TVPRegression", "spread_dynamics_break", "rolling_dols_beta", "stability_report",
    "format_stability", "holm_adjust", "stability_model_specs",
    # commodity-specific physical / curve state
    "DEFAULT_STATE_FEATURES", "StateDependentSpreadModel", "build_wti_curve_features",
    "build_inventory_features", "build_commodity_state", "fit_state_dependent_spread",
    "simulate_spread_paths", "rolling_state_dependent_oos", "interaction_wald",
    "conditional_reversion_table",
    # pre-specified decision rule
    "Rule", "DEFAULT_RULES", "STABILITY_RULES", "rule_verdict", "evaluate_rules", "baseline_in_mcs",
    "format_verdict",
    # robustness
    "exclude_period", "split_dev_holdout", "mean_vs_median_table",
    "influence_of_worst_origins", "ranking_stability", "relative_gain_by_period",
    # risk tests
    "kupiec_pof", "christoffersen_independence", "christoffersen_cc",
    "acerbi_szekely_z2", "var_es_report", "non_overlapping_subset",
]
