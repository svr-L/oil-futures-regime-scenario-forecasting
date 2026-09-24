from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def fit_ar1_summary(series: pd.Series) -> dict:
    """Fit AR(1) and return compact diagnostics."""
    x = series.dropna()
    model = AutoReg(x, lags=1, old_names=False).fit()
    params = model.params
    return {
        "n": int(model.nobs),
        "const": float(params.get("const", np.nan)),
        "b_lag1": float(params.iloc[1]) if len(params) > 1 else np.nan,
        "b_pvalue": float(model.pvalues.iloc[1]) if len(model.pvalues) > 1 else np.nan,
        "AIC": float(model.aic),
        "BIC": float(model.bic),
    }


def independence_diagnostics(x: pd.Series, lags: int = 10) -> dict:
    """Ljung-Box on returns and absolute returns, plus Engle ARCH LM test."""
    x = x.dropna()
    lb_ret = acorr_ljungbox(x, lags=[lags], return_df=True)
    lb_abs = acorr_ljungbox(x.abs(), lags=[lags], return_df=True)
    arch = het_arch(x)
    return {
        "LB_pvalue_returns": float(lb_ret["lb_pvalue"].iloc[0]),
        "LB_pvalue_abs": float(lb_abs["lb_pvalue"].iloc[0]),
        "ARCH_LM_pvalue": float(arch[1]),
    }


def adf_kpss_table(logp: pd.DataFrame, dlogp: pd.DataFrame) -> pd.DataFrame:
    """ADF/KPSS stationarity table for log-prices and log-differences."""
    rows = []
    for col in logp.columns:
        adf_lp = adfuller(logp[col].dropna(), autolag="AIC")[1]
        adf_dl = adfuller(dlogp[col].dropna(), autolag="AIC")[1]
        kpss_lp = kpss(logp[col].dropna(), regression="c", nlags="auto")[1]
        kpss_dl = kpss(dlogp[col].dropna(), regression="c", nlags="auto")[1]
        rows.append({
            "series": col,
            "ADF_p_logp": adf_lp,
            "ADF_p_dlogp": adf_dl,
            "KPSS_p_logp": kpss_lp,
            "KPSS_p_dlogp": kpss_dl,
        })
    return pd.DataFrame(rows)
