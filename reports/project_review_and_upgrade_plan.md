> **Historical document (v0.1).** This is the original review that set the
> roadmap. Items 'no rolling OOS validation', 'macro conditioning not
> benchmarked' and 'VECM not fully exploited' were addressed in v0.2-v0.4;
> see `CHANGELOG.md` and the README for the current state. The futures-curve
> and decision-layer items remain open.

# Project Review and Upgrade Plan

## Verdict

This project is worth keeping and publishing, especially for a Market Risk Quant / Risk Strat profile. It is not yet a flagship Quant Research project, but it is a credible bridge project because it combines:

- market-risk style scenario generation;
- empirical residual bootstrapping rather than Gaussian-only assumptions;
- AR filtering of marginal dynamics;
- half-life flexible probabilities;
- macro/risk-state conditioning;
- WTI/Brent cointegration;
- Kalman/state-space interpretability.

## What is strong

### 1. The theme is coherent

Oil futures are a natural market-risk object. WTI/Brent common-vs-spread behavior is a natural cointegration/state-space topic. Flexible probabilities and weighted bootstrap are coherent with ARPM-style scenario generation.

### 2. It links market risk and QR language

For Market Risk Quant, the project says: "I can build a scenario engine and validate joint risk-driver dynamics."

For QR/Systematic, the project says: "I can condition distributions on macro/risk regimes and think probabilistically."

### 3. The actual outputs support the core claims

The executed notebook supports these claims:

- WTI and Brent return dynamics need AR(1) filtering.
- Residuals remain heteroskedastic, motivating empirical/bootstrap treatment.
- Half-life flexible probabilities are implemented and calibrated by ESS.
- WTI forecast distributions are generated via weighted residual bootstrap.
- Johansen rank is 1 for the WTI/Brent/NVDA system.
- WTI-Brent spread half-life is around 5.2 weeks.
- A one-factor Kalman/dynamic-factor model extracts a common oil component.

## What is weaker

### 1. The original assignment still leaks through

NVIDIA is in the original assignment and makes the narrative less clean. For GitHub, keep it as a risk proxy or move it into an appendix. The main story should be WTI/Brent.

### 2. VECM is not fully exploited

Johansen cointegration is tested, but the project should add an explicit VECM fit and compare its forecasts with the AR/bootstrap approach.

### 3. No rolling OOS validation yet

This is the biggest gap. For QR credibility, the model must be compared against baselines using rolling forecasts and proper scoring rules.

Recommended metrics:

- mean log predictive density / NLL;
- CRPS;
- PIT histogram;
- VaR exceedance rate;
- ES backtesting;
- quantile loss.

### 4. Macro-regime conditioning is not benchmarked

The VIX/USD/rates kernel is intuitive, but the notebook should show whether it improves predictive performance relative to:

- unconditional empirical bootstrap;
- time-decay-only bootstrap;
- Gaussian AR;
- AR-GARCH Student-t.

### 5. Futures curve is missing

Front-month Yahoo proxies are fine for a first pass, but not enough for a serious oil futures research project. The next version should include:

- multiple maturities;
- rolling contract logic;
- calendar spreads;
- roll yield;
- contango/backwardation regimes.

## Suggested version roadmap

### v0.1 — current GitHub-ready version

Goal: clean portfolio project.

- Keep original executed notebook as evidence.
- Provide clean reproducible notebook.
- Modularize functions.
- Publish README and limitations honestly.

### v0.2 — validation version

Goal: make it defensible in interviews.

- Add rolling OOS forecast evaluation.
- Compare against baselines.
- Add PIT / quantile calibration diagnostics.
- Report a compact results table.

### v0.3 — oil futures curve version

Goal: make it less like an assignment and more like a real commodities quant project.

- Add curve maturities.
- Model WTI/Brent spreads and calendar spreads.
- Add roll yield / term-structure state variables.
- Condition scenarios on inventory/proxy variables if data is available.

### v0.4 — QR/systematic version

Goal: move from forecast engine to decision engine.

- Translate forecasts into risk-aware positions.
- Add transaction costs and slippage.
- Use position sizing based on forecast distribution, drawdown risk or expected shortfall.
- Evaluate strategy OOS.

## Recommended positioning

### Market Risk Quant

Use confidently. The project is highly relevant.

### Risk Strat

Use confidently if framed as scenario/risk-driver modeling and validation.

### FO Strat / QIS

Use as a supporting project, especially if paired with pricing/rates work.

### Quant Research / Systematic

Use carefully. It is promising, but do not oversell it as alpha research until OOS validation and baselines are added.
