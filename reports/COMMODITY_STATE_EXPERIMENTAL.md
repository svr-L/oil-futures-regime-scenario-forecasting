# Experimental commodity-state branch

This branch is **not part of the validated v1.0 headline result**.

It uses public EIA WTI C1-C4 curve variables and publication-lagged Cushing inventories to test whether WTI-Brent spread mean reversion changes with oil-market state.

## What the real run says

On the common sample ending 2024-04-05, the joint HAC Wald test on spread × state interactions is strongly significant (`p ≈ 4.6e-5`). The fitted one-step persistence varies materially across curve states.

However, the current model lets persistence vary linearly and without a stability bound. When that one-step law is recursively iterated for 4-20 week density forecasts, extreme states can imply `|phi(z)| >= 1`, producing explosive multi-step paths. The resulting OOS CRPS degradation is therefore a **specification failure**, not a reliable economic rejection of state dependence.

## Release treatment

- `RUN_COMMODITY_STATE = False` in the main notebook.
- No real-data commodity-state OOS result is used in the README or CV claim.
- `scripts/run_commodity_state.py` and the synthetic planted-interaction test remain available for research development.

## Next specification

Two defensible routes:

1. direct-horizon/local-projection models for `h = 1, 4, 12, 20`, avoiding recursive explosion; or
2. a bounded persistence map such as `phi(z) = tanh(eta0 + eta'z)` for scenario-path generation.

The public EIA C1-C4 history ends in April 2024, so this remains a historical mechanism exercise until a full contract-level WTI + Brent dataset is introduced.
