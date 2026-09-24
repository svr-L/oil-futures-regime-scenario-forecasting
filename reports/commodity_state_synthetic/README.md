# Commodity-state extension — synthetic DGP check

This directory is an **offline correctness check**, not evidence about real WTI/Brent.
The synthetic spread is generated with persistence that changes with planted curve /
inventory state; the code then estimates the same state-dependent error-correction
model used by the public-data branch.

Executed configuration: 43 balanced OOS origins, horizons 1/4/12/20 weeks, 800
scenarios per origin, stride 8. The joint HAC Wald test strongly detects the planted
spread×state interaction in the full sample. Conditional half-lives move in the
expected direction: about 2.34 weeks under a one-sigma backwardation state versus
6.31 weeks under a one-sigma contango state.

OOS CRPS improvement of `SD_EC` over the unconditional `SPREAD_AR1` benchmark:

| horizon | improvement | DM p | Holm p across horizons |
|---:|---:|---:|---:|
| 1w | -1.61% | 0.654 | 0.982 |
| 4w | +5.13% | 0.327 | 0.982 |
| 12w | +12.54% | 0.217 | 0.867 |
| 20w | +9.28% | 0.391 | 0.982 |

So the implementation detects the **mechanism** but does not manufacture a
forecast-significance claim from a sparse OOS sample. Real-data results are not
committed here; they require the EIA/Yahoo caches and should be reported only after
running `scripts/run_commodity_state.py` on those public inputs.
