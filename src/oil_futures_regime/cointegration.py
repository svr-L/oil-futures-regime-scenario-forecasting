from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM


def var_johansen_summary(Y: pd.DataFrame, var_lags: int = 1, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Fit VAR and Johansen trace test; return compact summary."""
    Y = Y.dropna()
    var_res = VAR(Y).fit(var_lags)
    A1 = var_res.coefs[0]
    eigvals = np.linalg.eigvals(A1)
    joh = coint_johansen(Y, det_order=det_order, k_ar_diff=k_ar_diff)

    rank95 = int(np.sum(joh.lr1 > joh.cvt[:, 1]))
    return {
        "var_res": var_res,
        "A1": pd.DataFrame(A1, index=Y.columns, columns=Y.columns),
        "eigenvalues": eigvals,
        "moduli": np.abs(eigvals),
        "johansen": joh,
        "johansen_trace": joh.lr1,
        "johansen_crit_95": joh.cvt[:, 1],
        "rank95": rank95,
    }


def fit_vecm(Y: pd.DataFrame, coint_rank: int = 1, k_ar_diff: int = 1, deterministic: str = "ci"):
    """Fit a VECM after a Johansen rank decision."""
    return VECM(Y.dropna(), k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic).fit()


def estimate_spread_half_life(spread: pd.Series) -> dict:
    """AR(1) spread mean-reversion estimates and half-life."""
    spread = spread.dropna().rename("spread")
    res = AutoReg(spread, lags=1, trend="c").fit()
    alpha = float(res.params["const"])
    rho = float(res.params["spread.L1"])
    mu = alpha / (1 - rho) if abs(1 - rho) > 1e-8 else np.nan
    half_life = np.log(0.5) / np.log(rho) if 0 < rho < 1 else np.nan
    return {"model": res, "alpha": alpha, "rho": rho, "mu": mu, "half_life_weeks": half_life}


def rolling_johansen_rank(
    Y: pd.DataFrame,
    oos_start,
    stride: int = 4,
    det_order: int = 0,
    k_ar_diff: int = 1,
    min_train_obs: int = 180,
) -> pd.DataFrame:
    """Re-run the Johansen rank decision at every forecast origin.

    A rank read once on the full sample is a single draw.  With near-unit-root
    data the trace statistic for ``r <= n-1`` sits close to its critical value,
    so the implied rank can flip between origins.  Reporting the *distribution*
    of the decision, rather than one number, is the honest way to justify the
    rank actually imposed in the scenario models.
    """
    Y = Y.dropna(how="any")
    idx = pd.DatetimeIndex(Y.index)
    positions = np.where(idx >= pd.Timestamp(oos_start))[0][::stride]

    rows = []
    for pos in positions:
        train = Y.iloc[: pos + 1]
        if len(train) < min_train_obs:
            continue
        try:
            joh = coint_johansen(train, det_order=det_order, k_ar_diff=k_ar_diff)
        except Exception:  # noqa: BLE001
            continue
        rank95 = int(np.sum(joh.lr1 > joh.cvt[:, 1]))
        rank99 = int(np.sum(joh.lr1 > joh.cvt[:, 2]))
        row = {"origin": idx[pos], "n_train": len(train),
               "rank_95": rank95, "rank_99": rank99}
        for i, (stat, crit) in enumerate(zip(joh.lr1, joh.cvt[:, 1])):
            row[f"trace_r<={i}"] = float(stat)
            row[f"crit95_r<={i}"] = float(crit)
        rows.append(row)
    return pd.DataFrame(rows)


def johansen_rank_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Frequency of each rank decision across origins, at 95% and 99%."""
    if table.empty:
        return table
    out = []
    for level in ("rank_95", "rank_99"):
        counts = table[level].value_counts().sort_index()
        for rank, n in counts.items():
            out.append({"level": level.replace("rank_", "") + "%", "rank": int(rank),
                        "n_origins": int(n), "share_pct": 100.0 * n / len(table)})
    return pd.DataFrame(out)
