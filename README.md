# Oil Futures Regime-Aware Scenario Forecasting

A research framework for **WTI, Brent, and WTI-Brent relative-price dynamics** under the physical measure. The project compares random-walk, AR/GARCH, VECM, unrestricted level-VAR, and common-trend state-space specifications in a balanced rolling out-of-sample design with probabilistic scoring, Model Confidence Sets, familywise multiple-testing control, robustness checks, and VaR/ES calibration diagnostics.

The project started from an ARPM time-series assignment and evolved into a standalone empirical study. The main result is deliberately narrow:

> **Outright WTI and Brent prices are not robustly predictable in this design, but medium-horizon relative-price dynamics are.** Structural multivariate models reduce WTI-Brent spread CRPS by about **20% at 12 weeks** and **25-28% at 20 weeks** versus a driftless random walk, while the random-walk baseline remains in the 90% Model Confidence Set for every outright-price cell.

Full write-up: [`reports/RESULTS_wti_brent.md`](reports/RESULTS_wti_brent.md).

---

## Headline evidence

Core experiment:

- **636 weekly observations:** 2013-01-04 to 2025-03-07.
- **89 balanced forecast origins:** 2018-01-05 onward, stride 4 weeks, zero dropped origins.
- **19 models**, 3 targets, 4 horizons (1, 4, 12, 20 weeks), 2,000 scenarios per forecast.
- Scores are computed on **WTI, Brent, and the WTI-Brent log spread**.
- Every model is re-estimated at each origin using information available at or before that date.

### Outright prices: no robust edge

At every WTI and Brent horizon, the driftless random walk remains inside the **90% Model Confidence Set**. The sets retain 18-19 of 19 models, so the correct conclusion is not that one specification wins, but that the data do not separate them.

### Relative prices: robust medium-horizon structure

| Target | Horizon | Best model | CRPS improvement vs RW |
|---|---:|---|---:|
| WTI-Brent spread | 12 weeks | `CTSF_equal` | **19.7%** |
| WTI-Brent spread | 20 weeks | `CTSF_equal` | **27.6%** |

The random walk is excluded from the 90% Model Confidence Set **only** in those two spread cells. After Romano-Wolf stepdown control **within each variable × horizon model family**, multiple VECM / free-loading common-trend / level-VAR specifications still beat the baseline at 12 and 20 weeks. At 20 weeks the strongest adjusted p-values are below 0.01.

This matters because the result is not a single selected winner. Several structurally related specifications agree that the forecastable object is the **relative component**, while the data do not sharply distinguish rank-one cointegration from an unrestricted near-unit-root level VAR.

### The result is not a 2020 artefact

Removing the February-June 2020 crisis window strengthens the spread gains: the best free-loading common-trend specification improves CRPS by about **21.8% at 12 weeks** and **30.5% at 20 weeks**.

---

## Structural interpretation

The common-trend state-space model is

```text
tau_t = tau_{t-1} + mu + eta_t              # integrated common oil trend
s_t   = phi s_{t-1} + nu_t                  # stationary relative component

log WTI_t   = tau_t + 0.5 s_t + eps_1t
log Brent_t = a tau_t - 0.5 s_t + eps_2t
```

The symmetric loading restriction `a = 1` is strongly rejected:

- free trend loading on Brent: **a = 1.0162**;
- likelihood-ratio test of `a = 1`: **p = 0.0002**;
- restricted latent half-life: **24.9 weeks**;
- free-loading latent half-life: **6.7 weeks**.

Accordingly, v1.0 uses the **free-loading CTSF** model for descriptive state-space quantities. Under that fit, the common trend explains roughly **96% of latent outright variance at 1 week** and about **99% at 20 weeks**, while the stationary relative component saturates. This is consistent with the empirical forecasting result: outright uncertainty accumulates, while relative deviations can mean-revert.

The rank decision itself is unstable. Rolling Johansen inference selects rank 1 in 44% of origins at the 95% level and 84% at the 99% level. The forecast comparison therefore treats rank as a model hypothesis rather than a fact: VECM and unrestricted level-VAR predictive distributions are statistically indistinguishable.

---

## 2015 U.S. crude-export-ban repeal: a tested macro hypothesis

The 18 December 2015 repeal is used as a known institutional event that could plausibly have changed the WTI-Brent relationship. Stability diagnostics use a **dedicated 2008-2025 history** so the pre-event sample is not limited to the 2013 start of the forecasting experiment.

Five primary known-date tests are treated as one family and corrected with Holm FWER control:

| Hypothesis | Raw p | Holm p |
|---|---:|---:|
| level shift | 0.598 | 1.000 |
| cointegrating-slope change | 0.430 | 1.000 |
| joint level + slope | 0.710 | 1.000 |
| spread reversion-speed change | 0.526 | 1.000 |
| spread-volatility change | 0.014 | **0.070** |

The unknown-date sup-Wald test also does not reject (**bootstrap p = 0.166**). The correct reading is therefore **no familywise-robust evidence of a structural break at the repeal date**, not proof that the relationship is immutable.

The forecasting consequence is tested separately. Post-repeal-only VECM, rolling-window VECM, and post-repeal free-loading state-space estimation each produce **0/4 significant horizon improvements after correction**. The rolling-window VECM is materially worse on average, supporting the expanding-window core specification.

---

## What survives the ablation

The v1.0 decision framework is pre-specified for this release and corrects each clause across its 12 target × horizon cells.

| Component | Decision | Reading |
|---|---|---|
| half-life recency weighting | DROP | median effect -1.16% |
| macro-state conditioning | KEEP, localized | 1/12 cells survives correction; median effect -0.09% |
| macro conditioning vs GARCH filter | DROP | median -3.07% |
| probability re-centring | KEEP | spread h=12 and h=20 survive correction |
| error correction | KEEP | spread h=12 and h=20 survive correction |
| rank-one VECM vs level VAR | DROP | 0/12 cells; rank not forecast-relevant |
| restricted state space vs VECM | DROP | rejected loading restriction hurts |
| free-loading state space vs VECM | DROP as incremental edge | reaches VECM performance, does not significantly beat it |

The macro result is intentionally not advertised as a broad forecasting edge: the formal rule keeps it because one spread h=4 cell survives, but the median effect is essentially zero and the GARCH filter is the stronger volatility treatment.

---

## Calibration and tail risk

The project evaluates more than average CRPS:

- PIT and interval-coverage diagnostics;
- Kupiec unconditional coverage;
- Christoffersen independence and conditional coverage;
- Acerbi-Szekely ES severity tests;
- overlap-thinning before tail tests;
- explicit underpowered-cell flags.

At the 5% VaR level, **18 of 114 adequately powered cells** reject Christoffersen conditional coverage, concentrated around the 4-week horizon; **342 cells are excluded as underpowered** rather than reported as passes. The issue is mainly clustering of exceptions rather than extreme exception rates.

---

## Validation discipline

The 2023-2024 block is a **late development/validation split**, not an untouched holdout. It has been inspected across successive versions and is used only as a ranking-stability check. Spread model rankings remain materially more stable across the split than outright-price rankings.

The genuine external block is **post-7-March-2025** and is intentionally kept outside `RUN_EVERYTHING.ipynb`. Once run, it should be treated as spent evidence rather than a new tuning sample:

```bash
python scripts/fetch_data.py
python scripts/run_forward_holdout.py --out reports/forward_holdout
```

---

## Experimental commodity-state branch

The repository also contains an **experimental** oil-specific extension based on public EIA WTI C1-C4 curve information and publication-lagged Cushing inventories. It asks whether spread mean-reversion speed changes with curve / inventory state.

The real-data run finds strong in-sample interaction evidence, but the current recursively iterated linear specification becomes unstable at extreme states in multi-week OOS simulation. **No real-data OOS claim from this branch is part of v1.0.** It is disabled by default in the main notebook and retained as an explicit research branch for the next specification step (direct-horizon/local-projection or bounded-persistence dynamics).

This separation is deliberate: the validated core result is not mixed with an unfinished extension.

---

## Model grid

| Model | Dynamics | Scenario weighting |
|---|---|---|
| `RW_equal` | driftless random walk | equal |
| `AR_equal`, `AR_time`, `AR_macro` | marginal AR + joint empirical residual rows | equal / recency / macro |
| `AR_macro_nc` | AR macro variant without final re-centring | macro |
| `VECM_equal`, `VECM_time`, `VECM_macro` | rank-1 error correction | equal / recency / macro |
| `FHS_equal`, `FHS_time`, `FHS_macro` | AR + GARCH(1,1)-t filtered historical simulation | equal / recency / macro |
| `CTS_equal`, `CTS_time`, `CTS_macro` | common-trend state space with `a=1` imposed | equal / recency / macro |
| `CTSF_equal`, `CTSF_time`, `CTSF_macro` | common-trend state space with free trend loading | equal / recency / macro |
| `VARL_equal`, `VARL_time` | unrestricted VAR in log levels | equal / recency |

Common random numbers are used across models at each origin so paired loss differences reflect model choice rather than independent Monte Carlo noise.

---

## Repository structure

```text
.
├── notebooks/
│   ├── 00_original_arpm_assignment_executed.ipynb
│   ├── 01_oil_futures_regime_scenario_forecasting_clean.ipynb
│   ├── 02_joint_vecm_oos_validation.ipynb
│   ├── 03_common_trend_state_space.ipynb
│   └── RUN_EVERYTHING.ipynb            # v1.0 main entry point
├── scripts/
│   ├── fetch_data.py
│   ├── run_validation.py               # frozen core ablation
│   ├── run_commodity_state.py          # experimental EIA mechanism branch
│   └── run_forward_holdout.py          # frozen post-2025 external validation
├── src/oil_futures_regime/
│   ├── data.py
│   ├── diagnostics.py
│   ├── models.py
│   ├── regimes.py
│   ├── cointegration.py
│   ├── common_trend.py
│   ├── scenario_models.py
│   ├── validation.py
│   ├── significance.py
│   ├── mcs.py
│   ├── risk_tests.py
│   ├── robustness.py
│   ├── decision.py
│   ├── stability.py
│   └── commodity_state.py
├── reports/
│   ├── RESULTS_wti_brent.md
│   ├── V1_0_UPGRADE.md
│   ├── synthetic_demo/
│   └── commodity_state_synthetic/
├── tests/                              # 92 tests, no network required
└── data/cache/README.md                # cache schema / provenance; data fetched locally
```

---

## Reproduce the core study

```bash
python -m pip install -r requirements.txt
python scripts/fetch_data.py
python scripts/run_validation.py --out reports/oos
```

Or open `notebooks/RUN_EVERYTHING.ipynb` and run all cells. The main notebook is configured for the validated core; the experimental commodity-state branch is **off by default**.

The repository does not commit live market-data snapshots. `scripts/fetch_data.py` populates the local cache and records the inputs used by the run.

### Offline correctness checks

`reports/synthetic_demo/` contains an executed simulated-data run of the full validation harness. `reports/commodity_state_synthetic/` contains a separate planted-interaction check for the experimental oil-state branch. These are regression / falsification checks, not substitutes for the real-data results above.

---

## Limitations

1. `CL=F` and `BZ=F` are concatenated front-month proxies, not a contract-level roll-aware curve.
2. The longest forecast horizon has only about **18 effectively independent evaluations** after accounting for overlap.
3. The core study evaluates probabilistic accuracy and risk calibration, not trading P&L, transaction costs, margin, or capacity.
4. The 2023-2024 split is validation data, not an untouched external holdout.
5. The public EIA commodity-state extension ends in April 2024 and its current recursive specification is experimental.

The next material data upgrade is **full contract-level WTI + Brent curves**, followed by either scenario-to-P&L/risk attribution for a Strat application or a cost-aware relative-value decision layer for systematic research.

---

## Version history

The detailed progression from the original assignment through v1.0 is in [`CHANGELOG.md`](CHANGELOG.md). The core design evolved in response to failures found in real runs: recursion conventions, balanced panels, external benchmarks, multiplicity, rank sensitivity, state-space restrictions, validation labeling, and structural-stability inference were all changed when the evidence required it.

## License

MIT.
