# Where is forecastable structure in WTI and Brent?

A rolling out-of-sample study of WTI, Brent and their relative price under the physical measure, with multiple-comparison control, tail backtests, structural-stability tests and a pre-specified v1.0 decision framework.

## Executive result

The core forecasting experiment uses **636 weekly observations (2013-01-04 to 2025-03-07)**, **89 balanced forecast origins** from 2018-01-05, **19 models**, horizons of 1/4/12/20 weeks and 2,000 scenarios per forecast.

The result is asymmetric:

- **Outright WTI and Brent:** no robust evidence that any model improves on a driftless random walk. The 90% Model Confidence Set retains 18-19 of 19 models in every level cell.
- **WTI-Brent spread:** structural multivariate models improve CRPS by about **20% at 12 weeks** and **25-28% at 20 weeks**. The random walk is excluded from the 90% MCS only in those two cells.
- **Interpretation:** the data support medium-horizon structure in the relative component, but do not sharply identify rank-one VECM versus an unrestricted near-unit-root level VAR as the unique representation.

The late 2023-2024 split is development/validation evidence, not an untouched holdout. The genuinely external post-7-March-2025 block remains separated in `scripts/run_forward_holdout.py`.

---

## 1. Core design

**Targets:** WTI log price, Brent log price, WTI-Brent log spread.  
**Horizons:** 1, 4, 12 and 20 weeks.  
**Forecast origins:** 89, monthly stride, zero dropped origins, balanced panel.  
**Effective independent evaluations at h=20:** approximately 18.  
**Models:** random walk; AR variants; GARCH-filtered historical simulation; VECM; restricted/free-loading common-trend state space; unrestricted level VAR.  
**Conditioning:** equal weights, half-life recency weights and macro-state similarity where applicable.  
**Inference:** DM/HAC + HLN correction, circular block bootstrap, Model Confidence Sets, Romano-Wolf stepdown within each variable × horizon model family, and familywise correction inside the pre-specified decision clauses.

---

## 2. Outright prices: the random walk survives

Selected WTI mean CRPS:

| model | h=1 | h=4 | h=12 | h=20 |
|---|---:|---:|---:|---:|
| `RW_equal` | 0.03448 | 0.06489 | 0.11600 | **0.14052** |
| `FHS_equal` | **0.03436** | **0.06325** | 0.11525 | 0.14175 |
| `FHS_macro` | 0.03455 | 0.06346 | **0.11454** | 0.14082 |
| `VECM_equal` | 0.03528 | 0.06505 | 0.12043 | 0.14619 |
| `CTSF_equal` | 0.03502 | 0.06547 | 0.11873 | 0.14571 |

The small numerical edges at 1-12 weeks do not survive inference. The MCS retains all 19 models in every WTI cell and 18-19 in every Brent cell. At Brent h=20, for example, the level VAR is numerically best by less than 1% with no statistical separation from the random walk.

The conclusion is therefore intentionally negative: **the study does not establish outright-price density predictability.**

---

## 3. Relative prices: the signal is at 12-20 weeks

Selected spread mean CRPS:

| model | h=1 | h=4 | h=12 | h=20 |
|---|---:|---:|---:|---:|
| `RW_equal` | 0.01045 | 0.01567 | 0.02457 | 0.02921 |
| `CTSF_equal` | 0.01059 | 0.01492 | **0.01973** | **0.02114** |
| `VECM_equal` | 0.01059 | **0.01486** | 0.02012 | 0.02146 |
| `VARL_equal` | 0.01085 | 0.01500 | 0.02038 | 0.02184 |
| `CTS_equal` | **0.01022** | 0.01589 | 0.02374 | 0.02731 |
| `FHS_equal` | 0.01110 | 0.01719 | 0.02834 | 0.03444 |

The best-model improvements versus the random walk are:

- **12 weeks:** `CTSF_equal` **+19.68%** CRPS improvement;
- **20 weeks:** `CTSF_equal` **+27.60%**.

The 90% Model Confidence Set narrows from essentially the whole grid at short horizons to **9/19 models at 12 weeks** and **7/19 at 20 weeks**. The random walk is excluded only in these two spread cells.

### Multiple-comparison control

Romano-Wolf correction is applied **within each variable × horizon family** of models tested against the same baseline. It is not a single global 216-hypothesis correction across unrelated cells.

Examples that survive the within-cell familywise correction:

| cell | model | improvement | adjusted p |
|---|---|---:|---:|
| SPREAD h=12 | `CTSF_equal` | 19.68% | 0.0315 |
| SPREAD h=12 | `VECM_equal` | 18.11% | 0.0270 |
| SPREAD h=12 | `VECM_macro` | 17.82% | 0.0420 |
| SPREAD h=12 | `VECM_time` | 17.49% | 0.0470 |
| SPREAD h=20 | `CTSF_equal` | 27.60% | 0.0030 |
| SPREAD h=20 | `VECM_equal` | 26.53% | 0.0005 |
| SPREAD h=20 | `VARL_equal` | 25.22% | 0.0030 |

Across the full reporting table there are 216 model-cell comparisons; 12 entries survive their **cell-level** familywise control. That count should not be read as global FWER control over all 216 simultaneously.

### Crisis exclusion

Excluding 2020-02-01 through 2020-06-30 strengthens the result. `CTSF_equal` improves on the random walk by approximately **21.8% at 12 weeks** and **30.5% at 20 weeks**. The finding is therefore not driven by the April 2020 dislocation alone.

---

## 4. State-space specification: the restriction matters, the implementation does not

The common-trend representation is

```text
tau_t = tau_{t-1} + mu + eta_t
s_t   = phi s_{t-1} + nu_t

log WTI_t   = tau_t + 0.5 s_t + eps_1t
log Brent_t = a tau_t - 0.5 s_t + eps_2t
```

The data reject the symmetric restriction `a = 1`:

- VECM/Johansen vector normalized on WTI: approximately `[1, -1.013]`;
- free state-space trend loading: **a = 1.0162**;
- LR test of `a = 1`: **14.239, p = 0.0002**.

This changes the latent persistence materially:

| estimate | persistence | half-life |
|---|---:|---:|
| raw WTI-Brent AR(1) | 0.8785 | 5.35 weeks |
| restricted CTS | 0.9726 | 24.92 weeks |
| **free-loading CTSF** | **0.9019** | **6.71 weeks** |

v1.0 therefore uses the free-loading model for descriptive filtering and variance decomposition. Under that fit, the common trend contributes about **96.3-96.4%** of latent outright variance at one week and roughly **99%** at 20 weeks.

The observed free-loading cointegrating residual has standard deviation 0.0467, versus 0.0458 for the Kalman-filtered component and 0.0021 for the residual measurement/noise component.

### No incremental CTSF victory over VECM

Freeing the loading fixes a bad restriction, but the free-loading state space does **not** significantly beat the VECM head-to-head. The decision rule records 0/12 significant cells and a median incremental effect of -0.34% against the best VECM/restricted-CTS incumbent. The defensible claim is that CTSF joins the structural winner set, not that Kalman filtering dominates error correction.

---

## 5. Cointegration rank: economically relevant structure, statistically ambiguous representation

Full-sample Johansen testing rejects both `r=0` and marginally `r<=1`. Rolling inference shows why the single full-sample decision is fragile:

| significance level | rank 0 | rank 1 | rank 2 |
|---|---:|---:|---:|
| 95% | 0% | 44% | 56% |
| 99% | 16% | 84% | 0% |

The forecast test settles the practical question: `VECM_equal` versus unrestricted level-VAR is significant in **0/12 cells** after correction, with a median difference of -0.22%. The data support multivariate long-run structure in the relative price, but do not identify rank-one cointegration as uniquely necessary for forecasting.

---

## 6. Structural stability around the 2015 export-ban repeal

The core 2013 forecasting sample is left untouched. Stability diagnostics instead use **896 weekly observations from 2008-01-04 to 2025-03-07** to increase pre-event power.

Known event: **18 December 2015**, repeal of the U.S. crude-oil export ban.

The five primary known-date tests are treated as one family:

| test | pre | post | raw p | Holm p |
|---|---:|---:|---:|---:|
| relation level | +3.6110 | +3.6418 | 0.598 | 1.000 |
| cointegrating slope | 1.0164 | 0.9567 | 0.430 | 1.000 |
| joint level+slope | — | — | 0.710 | 1.000 |
| spread persistence | 0.9238 | 0.8750 | 0.526 | 1.000 |
| weekly spread volatility | 0.0302 | 0.0219 | 0.014 | **0.070** |

The corresponding spread half-life falls from 8.75 to 5.19 weeks, but the speed-change test is not significant. The raw volatility result is suggestive but does not survive familywise correction.

An unknown-date sup-Wald search also fails to reject (**bootstrap p = 0.166**). Its least-squares date is 2010-12-31, but location uncertainty is broad; this is not evidence for a specific alternative break date.

The correct conclusion is **failure to find a familywise-robust break at the repeal**, not proof of structural invariance.

### Does shorter-memory estimation help forecasting?

No targeted window variant improves the spread forecasts significantly after correction:

- `VECM_post` vs expanding VECM: **0/4**, median **+1.39%**;
- `VECM_roll` vs expanding VECM: **0/4**, median **-13.98%**;
- `CTSF_post` vs expanding CTSF: **0/4**, median **-2.67%**.

This supports retaining the expanding-window core rather than discarding pre-2015 history.

---

## 7. Decision framework

Each v1.0 clause is evaluated across its 12 target × horizon cells with familywise correction inside the clause.

| clause | decision | evidence |
|---|---|---|
| recency weighting | DROP | 0/12, median -1.16% |
| macro conditioning | KEEP, localized | 1/12, SPREAD h=4; median -0.09% |
| macro vs GARCH filter | DROP | 0/12, median -3.07% |
| re-centring | KEEP | 2/12, SPREAD h=12/h=20; median +2.72% |
| error correction | KEEP | 2/12, SPREAD h=12/h=20 |
| cointegration rank | DROP | 0/12 |
| restricted state space | DROP | 0/12 |
| free-beta state space incremental edge | DROP | 0/12 |

The macro-condition `KEEP` should not be read as a broad macro forecasting claim: it is driven by one spread h=4 cell, its median effect is essentially zero, and it does not beat the GARCH/FHS volatility treatment.

---

## 8. Calibration and tails

Ninety-percent interval coverage is broadly reasonable on levels but often too wide for non-error-correcting spread models at long horizons. Error-correcting families are closer to nominal coverage.

At the 5% VaR level, after thinning overlapping targets:

- **18 of 114 adequately powered cells** reject Christoffersen conditional coverage;
- failures cluster around the **4-week horizon**;
- **342 cells** are explicitly excluded as underpowered (fewer than two expected exceptions).

The main issue is clustered tail exceptions rather than extreme unconditional exception rates. This matters for risk applications even when mean CRPS looks competitive.

---

## 9. Robustness and validation labeling

Development/validation split:

- early development: 66 origins, 2018-01-05 to 2022-12-30;
- late validation: 23 origins, 2023-01-27 to 2024-10-04.

Spearman early-vs-late ranking correlations:

| target | h=1 | h=4 | h=12 | h=20 |
|---|---:|---:|---:|---:|
| SPREAD | 0.775 | 0.554 | 0.772 | 0.772 |
| BRENT | 0.261 | 0.360 | 0.658 | 0.344 |
| WTI | 0.060 | -0.174 | 0.158 | 0.458 |

Spread rankings are materially more stable than outright-price rankings. Because the late block informed successive model-development versions, it is **not** described as an untouched holdout.

The external block starts after **7 March 2025** and is kept outside the automatic notebook run:

```bash
python scripts/fetch_data.py
python scripts/run_forward_holdout.py --out reports/forward_holdout
```

Once inspected, that block should be reported rather than used to tune the v1.0 grid.

---

## 10. Experimental commodity-state extension

The EIA extension replaces generic regime proxies with oil-specific state variables:

- WTI NYMEX C1-C4 curve slope and curvature;
- contango/backwardation;
- publication-lagged Cushing inventories.

The current model allows the one-step spread autoregressive coefficient to vary linearly with those states. On the real sample ending 2024-04-05, the joint HAC Wald test of spread × state interactions is strongly significant (**p ≈ 4.6e-5**), and the fitted conditional persistence varies substantially across curve states.

However, recursively iterating that unconstrained linear persistence over 4-20 weeks can produce `|phi(z)| >= 1` at extreme states. The resulting real-data multi-horizon OOS forecasts are numerically unstable and are **not treated as economic evidence against or for state dependence**.

For that reason:

- the branch is **disabled by default** in `RUN_EVERYTHING.ipynb`;
- no real-data OOS commodity-state claim appears in the README/CV result;
- the next specification should use either direct-horizon/local-projection dynamics or a bounded persistence map.

The committed synthetic mechanism check remains useful as a regression test, not as evidence about real oil markets.

---

## 11. Limitations

1. Yahoo `CL=F` / `BZ=F` are concatenated front-month proxies, not roll-aware contract histories.
2. Long-horizon inference is sample-limited: h=20 has roughly 18 effectively independent evaluations.
3. Core losses are probabilistic/risk metrics rather than trading P&L or transaction-cost-adjusted returns.
4. The 2023-2024 block is validation, not external holdout evidence.
5. The EIA commodity-state branch ends in April 2024 and its current recursive multi-horizon specification is experimental.

The next material data upgrade is full **contract-level WTI + Brent curves**. That would make curve factors, calendar spreads and roll yield explicit and open the two downstream paths: scenario-to-P&L/risk attribution for Strat, or a cost-aware relative-value decision layer for QR/QP.
