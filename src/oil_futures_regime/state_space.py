from __future__ import annotations

import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor


def fit_dynamic_factor_model(Y_oil: pd.DataFrame, k_factors: int = 1, factor_order: int = 1, error_order: int = 0):
    """Fit a one-factor dynamic factor model, estimated by Kalman filter / MLE."""
    mod = DynamicFactor(Y_oil.dropna(), k_factors=k_factors, factor_order=factor_order, error_order=error_order)
    return mod.fit(disp=False)
