# Oil Futures Regime-Aware Scenario Forecasting

This repository turns an ARPM-style time-series assignment into a compact quantitative finance project on **WTI/Brent futures scenario forecasting**, **flexible probabilities**, **macro-regime conditioning**, **cointegration**, and **Kalman/state-space interpretation**.

The project is designed to sit on a GitHub profile as a market-risk / quant-research bridge: it starts from weekly WTI, Brent and NVIDIA prices, models marginal log-return dynamics, extracts empirical innovations, applies half-life flexible probabilities, conditions scenarios on macro/risk states, and validates the WTI/Brent joint structure through Johansen cointegration and a one-factor dynamic factor model.

## Repository structure

```text
.
├── notebooks/
│   ├── 00_original_arpm_assignment_executed.ipynb
│   └── 01_oil_futures_regime_scenario_forecasting_clean.ipynb
├── reports/
│   └── project_review_and_upgrade_plan.md
├── src/
│   └── oil_futures_regime/
│       ├── data.py
│       ├── diagnostics.py
│       ├── models.py
│       ├── cointegration.py
│       └── state_space.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── LICENSE
```

## Methods

### 1. Data and stationarity

The executed notebook uses weekly closes from **2013-01-01 to 2023-01-01** for:

- WTI crude oil futures (`CL=F`)
- Brent crude oil futures (`BZ=F`)
- NVIDIA (`NVDA`) as a high-beta equity/risk proxy in the original assignment

Log-prices are non-stationary, while log-price differences are approximately stationary. This is supported by AR(1) persistence close to one in log-prices and by ADF/KPSS diagnostics.

### 2. Marginal dynamics and innovation extraction

Weekly log-price differences are modeled with low-order AR models selected via BIC and residual Ljung-Box diagnostics.

Executed results:

| Series | Selected model | Residual Ljung-Box p-value | Residual \|shock\| Ljung-Box p-value | ARCH LM p-value |
|---|---:|---:|---:|---:|
| WTI | AR(1) | 0.1187 | 4.24e-59 | 7.42e-33 |
| Brent | AR(1) | 0.0581 | 1.03e-70 | 4.61e-32 |
| NVDA | AR(0) | 0.9545 | 1.05e-09 | 6.83e-03 |

Interpretation: the AR filters remove most linear autocorrelation in returns, but volatility clustering remains strong. That makes empirical residual bootstrap / flexible probabilities more appropriate than an i.i.d. Gaussian assumption.

### 3. Flexible probabilities

The project uses exponentially decaying flexible probabilities. A half-life of **52 weeks** is selected to target an effective sample size close to 150.

Executed result:

| Half-life | ESS |
|---:|---:|
| 52 weeks | 149.75 |

### 4. Macro-conditioned residual bootstrap

For the WTI forecast, the notebook combines:

- AR(1) conditional mean dynamics
- centered empirical WTI residuals
- time-decay flexible probabilities
- macro/risk-state kernel conditioning using VIX, USD index and US 10Y yield proxies
- weighted bootstrap over residual scenarios

Executed result:

- Forecast origin: **2025-03-05**
- Horizon: **20 weeks**
- Number of scenarios: **10,000**
- Macro kernel bandwidth: **2.0**
- Macro-conditioned ESS: **≈ 76.3**

The 20-week WTI log-price forecast quantiles are:

| Quantile | Log-price | Price equivalent |
|---:|---:|---:|
| 5% | 3.5814 | 35.92 |
| 50% | 4.1072 | 60.78 |
| 95% | 4.6485 | 104.42 |

### 5. Joint WTI/Brent structure

A VAR(1) is fitted on `[log(WTI), log(Brent), log(NVDA)]`.

Executed results:

- VAR(1) eigenvalue moduli: **0.8717, 0.9929, 0.9929**
- Heuristic number of stochastic trends: **2**
- Implied cointegration rank: **1**
- Johansen trace statistics: **[48.53, 15.08, 1.40]**
- Johansen 95% critical values: **[29.80, 15.49, 3.84]**
- Johansen implied rank at 95%: **1**

A direct WTI-Brent spread AR(1) gives:

- `rho = 0.8755`
- long-run mean `mu = -0.0770`
- spread half-life: **≈ 5.21 weeks**

### 6. Kalman / dynamic factor interpretation

A one-factor dynamic factor model is fitted to `[log(WTI), log(Brent)]`.

Executed results:

- WTI loading: **-0.1207**
- Brent loading: **-0.1230**
- latent factor AR coefficient: **0.9874**

The near-identical loadings show that the Kalman-filtered factor is essentially a common oil component. The residual spread proxy captures relative WTI/Brent deviations around that common trend.

Important caveat: residual diagnostics still show autocorrelation, heavy tails and heteroskedasticity. The state-space model is useful as an interpretable factor decomposition, but it should not be sold as a fully validated production forecasting model without further residual modeling.

## Resume bullets

### Market Risk Quant / Risk Strat

```latex
\item Regime-aware WTI/Brent scenario forecasting: AR-based innovation extraction with half-life flexible probabilities and macro-conditioned weighted bootstrap to produce multi-horizon WTI distributions; validated joint WTI/Brent dynamics via Johansen cointegration and interpreted common-vs-spread components using a one-factor Kalman state-space model (spread half-life $\approx$ 5.2w).
```

### Quant Research / Systematic Investing

```latex
\item Oil futures under macro regimes (USD/rates/risk): regime-conditioned residual bootstrap for probabilistic WTI forecasts, Johansen cointegration analysis of WTI/Brent common trends, and Kalman-filtered one-factor decomposition of common oil vs relative spread dynamics.
```

## How to run

```bash
git clone <repo-url>
cd oil-futures-regime-scenario-forecasting
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_oil_futures_regime_scenario_forecasting_clean.ipynb
```

## Current limitations

This is a strong GitHub project for market-risk quant / risk-strat positioning, but it is not yet a full systematic trading research project.

Main limitations:

1. The forecast is scenario-based but not yet evaluated with a proper rolling out-of-sample backtest.
2. The macro-regime conditioning is kernel-based and intuitive, but not compared against baselines.
3. The futures curve is not modeled: no term structure, calendar spreads, roll yield, contango/backwardation or contract rolling logic.
4. The state-space model is useful for interpretation, but residual diagnostics are not clean.
5. The project does not yet produce a tradeable signal or portfolio strategy.

## Best next upgrades

1. Add rolling OOS validation: log score, CRPS, PIT calibration, VaR/ES exceedance tests.
2. Add baselines: unconditional bootstrap, pure time-decay bootstrap, Gaussian AR model, GARCH/Student-t.
3. Extend from front-month proxies to a proper futures curve dataset.
4. Add VECM forecasts for WTI/Brent and compare against independent AR forecasts.
5. Add transaction-cost-aware spread or calendar-spread strategy only after the forecasting layer is validated.

## Bottom line

As a **Market Risk Quant / Risk Strat** project, this is worth publishing after cleanup: it shows scenario generation, empirical innovations, flexible probabilities, macro conditioning, cointegration and Kalman filtering.

As a **Quant Research** project, it is promising but should be framed carefully. The current version demonstrates a research pipeline and probabilistic modeling skill; it becomes genuinely QR-grade only after rolling OOS validation, baselines, futures-curve treatment and clearer decision/use-case design.
