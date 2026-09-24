# Changelog

## v1.0.0 — frozen core, corrected inference, stable public release

### Changed

- **Core empirical claim frozen.** The 19-model 2013-2025 rolling experiment remains the validated headline: no robust edge on outright WTI/Brent levels; structural multivariate models improve WTI-Brent spread CRPS by about 20% at 12 weeks and 25-28% at 20 weeks.
- **Multiple-testing language corrected.** Romano-Wolf stepdown controls FWER within each variable × horizon family of models versus the random-walk baseline. The repository no longer describes this as one global correction across all 216 model-cell entries.
- **Free-loading common-trend state space is now the descriptive default.** The real run estimates `a = 1.0162` and rejects `a = 1` at `p = 0.0002`; the free-loading latent half-life is 6.71 weeks versus 24.92 weeks under the rejected restriction.
- **Validation terminology fixed.** The 2023-2024 block is a late development/validation split. The genuine external block is post-7-March-2025 and remains isolated in `run_forward_holdout.py`.
- **Commodity-state branch is experimental and disabled by default.** The real EIA fit shows strong in-sample state interaction, but the recursively iterated unconstrained persistence becomes unstable in multi-week OOS simulation. No real-data commodity-state OOS claim is part of v1.0.

### Structural-stability run

- Stability diagnostics now use a dedicated 2008-2025 history while preserving the 2013 core forecasting sample.
- The five known-date 2015 export-ban-repeal tests are Holm-corrected as one family. The smallest raw p-value is spread volatility (`0.014`), which becomes `0.070`; none rejects at 5%.
- The unknown-date sup-Wald bootstrap p-value is `0.166`.
- Post-repeal VECM, rolling-window VECM and post-repeal CTSF each produce 0/4 significant horizon improvements after correction; rolling VECM is materially worse on average.

### Decision framework

- Recency weighting: DROP.
- Macro conditioning: formally KEEP but localized to one spread h=4 cell; median effect is essentially zero and it loses to GARCH/FHS.
- Probability re-centring: KEEP.
- Error correction: KEEP, driven by spread h=12/h=20.
- Cointegration-rank restriction and incremental state-space superiority: DROP.

### Testing / reproducibility

- 92 network-free tests remain in the suite.
- Live market-data snapshots are not committed; `scripts/fetch_data.py` populates the local cache.
- Synthetic core and commodity-state runs remain committed as correctness/falsification checks, not as substitutes for real-data evidence.

## v0.9.0 — stability of the cointegrating relation

### Added

- **`stability.py`.** Separate tests for the three things the 18 December 2015
  export-ban repeal should have moved: the level of the WTI/Brent relation, its
  slope, and the spread's mean, persistence and volatility.
  - `dols` / `dols_break_test`: dynamic OLS with a known-date break in level and
    slope, the level regressor centred at the break-date price so the level test
    measures the jump in the relation rather than an extrapolated intercept.
  - `spread_dynamics_break`: pre/post mean, persistence, half-life and volatility.
  - `sup_wald_break`: existence of a break at an unknown date (sup-Wald, 15%
    trimming) and its location (Bai least-squares estimator), kept apart because
    they behave very differently.
  - `fit_tvp_beta`: intercept and slope as random walks by Kalman filter, with a
    boundary-corrected likelihood-ratio test of a constant slope.
  - `rolling_dols_beta`, `stability_report`, `format_stability`.
- **AR-sieve bootstrap calibration for every test.** Simulated under no break
  with WTI/Brent-like persistence, the asymptotic known-date test rejected in 41%
  of samples at a nominal 5% (54% at persistence 0.95); the bootstrap brings every
  test to 3-7%. A moving-block bootstrap was tried first and discarded because it
  understated location uncertainty enough to reject a correctly specified break
  date. The calibration result is frozen as a regression test.
- **Location reported as a hypothesis test, not a point.** Bootstrap under the
  hypothesised date gives the range where the estimator would land if the repeal
  were the true break, and a p-value for that hypothesis. In simulation this
  rejects a correctly specified date at the nominal 5% rate.
- **Estimation windows in the harness.** `ModelSpec` gains `train_start` and
  `window`; models sharing a family and a sample are fitted once. An origin whose
  window is too short is dropped for every model, so the panel stays balanced.
- **`stability_model_specs` and `STABILITY_RULES`**: a separate, targeted grid
  (expanding, post-repeal and rolling-window VECM; expanding and post-repeal
  free-beta state space) as a distinct hypothesis family about parameter instability rather than additional
  candidates in the primary forecast-selection exercise.
- **`relative_gain_by_period`**: challenger gain by origin year, because a pooled
  average hides transient effects. In simulation a genuine break made
  post-break estimation 25-32% better near the break and irrelevant three years
  later, with a slightly negative pooled average.
- **`synthetic_panel(break_date=...)`** plants a repeal-shaped break (discount
  narrowing from -0.08 to -0.03, persistence falling from 0.95 to 0.80); the
  default path is unchanged so earlier seeds reproduce.
- Notebook sections **3b** (stability evidence with plots and a reading guide)
  and **8b** (forecasting consequence), writing `beta_stability.md`,
  `stability_rules.csv` and `stability_forecast_rows.csv.gz`.
- Twelve tests: planted level break found and slope break not, bootstrap
  correcting the oversized asymptotic test, faster reversion detected, quiet
  under the null, location logic, AR-sieve persistence, estimation windows
  respected and balance preserved, short windows dropping origins for everyone.

### Documented from simulation

- The Kalman constant-slope test has ~3% size, ~15% power against genuine slope
  drift, and rejects under volatility clustering alone (p = 0.026 on a GARCH panel
  without a break). It is labelled descriptive.
- With about three years of pre-repeal data, the pre-repeal spread mean has a
  standard deviation of ~0.03 across simulated panels, so a real shift in the
  level can fail to reach significance on sample length alone.
- A slope rejection alongside a level jump can reflect a gradual transition to
  the new mean leaking into the step-dummy regression.

## v0.8.0 — the decision rule, corrected; and the write-up

### Added

- **`decision.py`.** The decision rule (then described as pre-registered) is now a module rather than a
  notebook cell, with two corrections.
  - *Multiplicity inside each clause.* Each clause is tested across twelve cells,
    so a single significant cell is weak evidence (~0.6 spurious wins per clause
    are expected at 5%). Because all cells share the same forecast origins, their
    loss differentials form a panel and a Romano-Wolf stepdown corrects across
    them while preserving their dependence. On the real-data run this reverses
    the verdict on macro conditioning, which the uncorrected version kept on one
    cell out of twelve while its median effect was negative.
  - *No post-hoc winner.* `baseline_in_mcs` replaces "the best model beats the
    baseline" with "is the baseline excluded from the 90% Model Confidence Set",
    which requires no choice made after seeing the data and is already corrected
    for the size of the grid.
- `Rule`, `DEFAULT_RULES`, `rule_verdict`, `evaluate_rules` and `format_verdict`,
  with the binding incumbent reported per cell and the clause's rationale printed
  alongside its decision.
- **`reports/RESULTS_wti_brent.md`**: the written study for the 2013-2025 run —
  structure of the system, level results, spread results, what does not work,
  calibration and tails, robustness, verdict, limitations and what follows.
- Eight tests covering the new logic: the binding incumbent is the best simpler
  alternative, a lucky single cell does not survive correction, a broad effect
  still passes, familywise p-values never fall below raw ones, and the baseline
  is flagged as excluded only when something genuinely beats it.

### Changed

- The notebook's verdict section prints the corrected rule and the
  baseline-versus-MCS table, and writes `decision_rule_summary.csv`,
  `decision_rule_cells.csv` and `baseline_vs_mcs.csv`.
- Within each clause, `p_raw` and `p_familywise` are one-sided bootstrap
  p-values on the same statistic; the two-sided Diebold-Mariano p-value is kept
  as a separate reference column rather than mixed with them.
- README leads with the result instead of the apparatus.

## v0.7.0 — fixes forced by the first real-data run

The full grid on WTI/Brent 2013-2025 gave a clean result (error correction beats
a random walk on the spread at 12 and 20 weeks, surviving Romano-Wolf, crisis
exclusion and the holdout) and exposed three specification problems.

### Added

- **`CTSF_*`: state space with a free trend loading.** The symmetric restriction
  `beta = [1, -1]` is rejected on real data (a = 1.016, LR p = 2e-4), and
  imposing it leaves part of a near-unit-root trend inside the relative
  component: the restricted fit put the latent spread half-life at 25 weeks
  against 5.3 observed, with spread PITs centred on 0.34 instead of 0.5. The free
  variant is now a first-class member of the grid. A new test reproduces the bias
  mechanism on synthetic data with a known loading, asserting that the restricted
  fit over-estimates persistence and the free one does not.
- **`VARL_*`: unrestricted VAR in log levels.** Johansen can marginally reject
  `r <= 1` as well as `r = 0` on this sample, implying a stationary system. Rather
  than arguing the point, the model that such a decision implies now competes in
  the ablation.
- **`rolling_johansen_rank` / `johansen_rank_summary`.** The rank decision is
  re-run at every forecast origin and reported as a distribution, with the
  marginal trace statistic plotted through time, so the rank imposed in the
  scenario models rests on more than one draw.

### Changed

- **The decision rule now compares against the best of the simpler
  alternatives**, not one fixed incumbent, and prints which incumbent was binding
  in each cell. The old chain reported "macro beats recency" as a win while the
  real finding was that half-life weighting *hurts*: `AR_time` lost to
  `AR_equal`, and macro conditioning merely recovered to the equal-weight
  baseline while losing to the GARCH filter.
- Rules added for the rank sensitivity (`VECM` vs `VARL`) and for the free-beta
  state space; the "vs random walk" rule now tests the actual best model rather
  than a fixed one.
- The grid grows from 14 to 19 models; the notebook writes
  `johansen_rank_stability.csv` alongside the other artefacts.

## v0.6.0 — one notebook, and results that survive correction

### Added

- **`notebooks/RUN_EVERYTHING.ipynb`.** The whole study in a single notebook:
  data, structural analysis, rolling ablation, multiplicity control, robustness,
  holdout, calibration, tail backtests and an automated verdict against the
  then-current decision rule. One configuration cell holds every knob, and
  `USE_SYNTHETIC = True` runs the entire pipeline offline against a known DGP.
  The library is meant to stay closed.
- **`mcs.py`: multiple-comparison control.** Hansen-Lunde-Nason Model Confidence
  Set (T_max statistic, circular block bootstrap, monotone MCS p-values) and
  Romano-Wolf stepdown for familywise error control against a baseline. Both
  respect the serial dependence induced by overlapping forecast windows.
  `mcs_by_cell` and `mcs_summary` run them across the whole grid.
- **`robustness.py`.** Mean-versus-median rankings, influence of the worst
  origins, crisis-window exclusion, ranking stability between panels, and a
  development/holdout split.
- The runner writes `mcs.csv` and `mcs_summary.csv`, and `RESULTS.md` gains a
  Model Confidence Set section placed *before* the winner table, so a reader sees
  how many models are indistinguishable before seeing which one ranked first.
- Nine new tests: MCS keeps indistinguishable models and eliminates clearly worse
  ones, MCS p-values are monotone in elimination order, longer blocks do not
  shrink the set, Romano-Wolf adjusted p-values dominate raw ones and control
  familywise error under the null across repeated families.

- **Self-bootstrapping notebook.** `RUN_EVERYTHING.ipynb` now locates the library
  on its own — from a clone, from the project zip, or via `git clone` — installs
  any missing dependencies (with a PEP 668 fallback), and only then imports.
  Verified end-to-end from an empty directory containing nothing but the notebook
  and the zip.

### Changed

- The decision rule is now applied mechanically in code rather than by reading a
  table, so the conclusion cannot drift toward whatever the data produced.
- README leads with "everything lives in one notebook" and documents that a
  crowded confidence set is itself the finding.

## v0.5.0 — the state space becomes the model

### Added

- **`common_trend.py`: cointegration-consistent state space.** An integrated
  common oil trend plus a stationary relative component, estimated by Kalman
  filter / MLE, whose implied cointegrating vector `[a, -1]'` can be compared
  directly with the Johansen estimate. `a` is restricted to 1 by default and can
  be freed and tested by likelihood ratio, so the symmetric `[1, -1]` spread is a
  checked restriction rather than an assumption.
- **Innovations-form simulation.** Standardized one-step prediction errors are
  bootstrapped with the project's flexible probabilities and re-inflated by the
  steady-state `F^{1/2}`, making this the state-space analogue of filtered
  historical simulation. `CTS_equal`, `CTS_time` and `CTS_macro` therefore enter
  the ablation under exactly the same rules — common random numbers, balanced
  panel, identical scoring — as every other model.
- **`cts_state_summary` / `filtered_states`**: implied beta, latent spread
  half-life, unconditional spread dispersion, signal-to-noise ratio, filtered
  and smoothed states.
- **CTS vs VECM head-to-head** written to `significance_cts_vs_vecm.csv` and to
  a dedicated section of `RESULTS.md`, because "both beat the random walk" is not
  the same claim as "the filtered model beats the raw one".
- **Notebook `03_common_trend_state_space.ipynb`**: fit, restriction test,
  agreement with Johansen/VECM, filtered-state plots against the roll-artefact
  diagnostic, horizon variance decomposition, and spread fan charts.
- Nine new tests: parameter recovery on data from the model, correlation of the
  filtered state with the latent one, implied-beta identity, unit-scale
  standardized innovations, registry dispatch, and the two behavioural checks
  that matter — the simulated level diffuses while the simulated spread
  converges, and a forced dislocation reverts.

### Changed

- The one-factor `DynamicFactor` model in `state_space.py` is retained for
  provenance but is no longer the project's state-space story.
- The then-current decision rule gains a clause: the state space replaces the
  VECM only if it wins the head-to-head, not merely if both beat the baseline.
- README documents that multiplicity across 168 comparisons is not yet
  controlled, and that a Model Confidence Set is the next required step.

### Result on the simulated panel

`CTS_*` beats the random walk on the spread at all four horizons — including 1
and 4 weeks, where the VECM was not significant — and improves on the VECM by
1.5–4% CRPS at every horizon, significant only at h=4. Consistent sign,
insufficient sample to establish it at long horizons; reported as such.

## v0.4.0 — validity of the comparison

Methodological fixes, in order of how much they change the conclusions.

### Fixed

- **Unbalanced comparison.** v0.3's per-model `try/except: continue` meant a
  model that failed at some origins was scored on a different subset of dates
  than its competitors, and failures concentrate in turbulent periods. An origin
  is now committed only if every model produced a forecast; drops are recorded
  in `failures.csv` with the exception type and message. `ValidationOutput`
  exposes `is_balanced` and the harness raises if the panel is ever unbalanced.
- **Independent random draws across models.** Simulators now consume a shared
  `(horizon, n_sim)` uniform array via inverse-CDF sampling instead of a
  per-model seed, so paired score differences are not inflated by Monte Carlo
  noise.
- **Absolute ESS floor for the macro kernel.** `min_macro_ess=75` made the
  conditioning progressively tighter as the pool grew from ~250 to ~600 weeks.
  Replaced with `min_macro_ess_fraction=0.35`, and the diagnostics now flag when
  the bandwidth grid is exhausted.
- **Duplicated probability code.** `exp_half_life_weights`,
  `effective_sample_size`, `weighted_mean_cov` and `macro_kernel_weights`
  existed in both `models.py` and `regimes.py`. `regimes.py` is now the single
  source of truth.

### Added

- **Significance testing** (`significance.py`): Diebold–Mariano with Newey–West
  HAC variance at the overlap lag implied by horizon and stride, plus the
  Harvey–Leybourne–Newbold small-sample correction; circular moving-block
  bootstrap intervals; `paired_score_table` and `best_model_summary` for
  leaderboards that report whether a gap is real.
- **External benchmarks**: driftless random walk (`RW`) and filtered historical
  simulation with an AR mean and a GARCH(1,1)-t variance filter (`FHS`), whose
  conditional variance evolves along each simulated path.
- **`VECM_equal`**, completing the 2×3 dynamics × probabilities design, and
  **`AR_macro_nc`**, which ablates the residual re-centering convention instead
  of assuming it.
- **Tail backtests** (`risk_tests.py`): Kupiec POF, Christoffersen independence
  and conditional coverage, Acerbi–Székely Z2 for expected shortfall, panel
  thinning to non-overlapping targets, and an `underpowered` flag for cells with
  fewer than two expected exceptions.
- **Data caching** (`data.py`): parquet snapshots via `load_or_download`, so
  notebooks are deterministic and offline-reproducible; `roll_gap_diagnostics`
  and `spread_jump_summary` quantify front-month splicing artefacts.
- **Scripts**: `fetch_data.py` (download and cache once) and `run_validation.py`
  (full ablation to disk, with a `--synthetic` mode that needs no network).
- **Executed evidence**: `reports/synthetic_demo/` holds a complete run on a
  simulated panel with a known DGP, in which the harness correctly finds the
  cointegration effect on the spread and correctly finds no significant edge over
  a random walk on the levels.
- **Tests**: 44 tests including a regression test for the ARIMA intercept bug, a
  CRN determinism test, a VECM spread-compression test, an FHS
  volatility-response test, DM power and size checks, and Kupiec/Christoffersen
  behaviour on constructed hit sequences. CI runs the tests plus an end-to-end
  synthetic validation on Python 3.11 and 3.12.

### Changed

- `pit_summary` now documents that its KS p-values are descriptive only, since
  PIT values from overlapping windows are serially dependent.
- The default baseline for all comparisons is `RW_equal` rather than `AR_equal`.
- README leads with the then-current decision rule and states plainly that a
  negative result is a result.

## v0.3.0

- Added rolling pseudo-OOS validation for WTI, Brent and WTI/Brent spread
  distributions.
- Added CRPS, pinball loss, 90% interval coverage/width, lower-tail hit rate and
  PIT diagnostics.
- Added explicit ablation across `AR_equal`, `AR_time`, `AR_macro`, `VECM_time`,
  `VECM_macro`.
- Macro conditioning uses log VIX, weekly DXY log changes and weekly 10Y yield
  changes.
- Residuals re-centered under final scenario probabilities by default.
- Added synthetic-data tests and GitHub Actions CI.

## v0.2.0

- Added joint WTI/Brent AR residual-vector bootstrap benchmark.
- Added VECM residual-bootstrap scenario generator with path-by-path error
  correction.
- Added reusable flexible-probability / macro-kernel module.
- Corrected ARIMA constant handling for direct AR recursion in simulation.

## v0.1.0

- Original cleaned ARPM-derived project: stationarity diagnostics, low-order AR
  innovation extraction, half-life flexible probabilities, macro-conditioned WTI
  residual bootstrap, Johansen/VECM analysis, spread half-life and one-factor
  Kalman interpretation.
