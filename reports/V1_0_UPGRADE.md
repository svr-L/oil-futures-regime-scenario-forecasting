# v1.0 upgrade note

v1.0 freezes the validated core WTI/Brent forecasting experiment and separates three things that should not be conflated: the core forecast result, structural-stability diagnostics, and an experimental commodity-state extension.

## Core result retained

The 19-model rolling experiment remains unchanged: 636 weekly observations, 89 balanced origins, horizons 1/4/12/20 weeks, and targets WTI / Brent / WTI-Brent spread.

The headline is unchanged but now stated more precisely:

- no robust density-prediction edge over the random walk on outright WTI or Brent;
- roughly **20%** spread CRPS improvement at 12 weeks and **25-28%** at 20 weeks from structural multivariate models;
- the random walk is excluded from the 90% MCS only in those two spread cells;
- Romano-Wolf FWER control is **within each variable × horizon family**, not a single global correction across all 216 model-cell entries.

## State-space correction

The free-loading common-trend model is the descriptive default whenever `a=1` is rejected.

Executed v1.0 diagnostics:

- `a = 1.0162`;
- LR p-value for `a=1`: `0.0002`;
- restricted latent half-life: `24.92` weeks;
- free-loading latent half-life: `6.71` weeks;
- common-trend share of latent outright variance: about 96% at 1 week and 99% at 20 weeks.

The restricted result is retained only as evidence of the cost of a bad loading restriction.

## Stability correction and executed longer-history run

Stability diagnostics use 896 weekly observations from 2008-01-04 to 2025-03-07, while the core forecast sample stays fixed at 2013 onward.

Five known-date tests around the 18 December 2015 export-ban repeal are Holm-corrected as one family. The smallest raw p-value is the spread-volatility test (`0.014`), which becomes `0.070`; none rejects at 5%. The unknown-date sup-Wald bootstrap p-value is `0.166`.

Forecast-window challengers also fail to improve the core significantly after correction:

- post-repeal VECM: 0/4 horizons, median +1.39%;
- rolling-window VECM: 0/4, median -13.98%;
- post-repeal CTSF: 0/4, median -2.67%.

## Validation language

The 2023-2024 block is a late development/validation split, not an untouched holdout. The genuine external block starts after 7 March 2025 and remains isolated in `scripts/run_forward_holdout.py`.

## Experimental commodity-state branch

The public EIA branch is intentionally **not part of the v1.0 empirical claim**.

The real-data fit finds a strong in-sample interaction between WTI curve / Cushing state and spread reversion, but the current recursively iterated linear-persistence specification becomes unstable at extreme states in multi-week OOS simulation. That is a model-specification failure, not a validated negative economic result.

Accordingly:

- `RUN_COMMODITY_STATE = False` by default in the main notebook;
- the branch remains available through `scripts/run_commodity_state.py`;
- the next version should replace recursive unconstrained persistence with either direct-horizon/local-projection forecasts or a bounded persistence map.

## Testing

The repository contains 92 network-free tests. The code-level test suite and synthetic checks remain separate from the real-data conclusions above.
