"""Formal VaR / ES backtests.

Coverage *rates* alone do not tell you whether a tail is mis-calibrated: with 80
overlapping forecasts a 7% exceedance rate against a 5% target is well inside
sampling noise.  This module adds the tests a market-risk reviewer expects:

* Kupiec unconditional-coverage LR test (are there too many exceptions?);
* Christoffersen independence LR test (do exceptions cluster?);
* Christoffersen conditional-coverage LR test (both jointly);
* Acerbi-Szekely Z2 severity statistic for expected shortfall, with a
  block-bootstrap interval.

Overlapping forecasts violate the independence assumption these tests rest on.
:func:`non_overlapping_subset` therefore thins the panel so that consecutive
retained origins have non-overlapping target windows; the report functions use
it by default and record how many observations survived.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .significance import block_bootstrap_mean_ci


def non_overlapping_subset(origins: pd.Series | np.ndarray, horizon: int, stride: int):
    """Boolean mask keeping every ``ceil(horizon/stride)``-th origin."""
    step = max(int(math.ceil(horizon / max(stride, 1))), 1)
    n = len(origins)
    mask = np.zeros(n, dtype=bool)
    mask[::step] = True
    return mask


def kupiec_pof(hits: np.ndarray, alpha: float) -> dict:
    """Kupiec proportion-of-failures (unconditional coverage) LR test."""
    x = np.asarray(hits, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    k = int(x.sum())
    if n == 0:
        return {"n": 0, "exceptions": 0, "rate": np.nan, "lr_uc": np.nan, "p_value": np.nan}
    pi_hat = k / n
    if k == 0:
        lr = -2.0 * (n * math.log(1 - alpha))
    elif k == n:
        lr = -2.0 * (n * math.log(alpha))
    else:
        ll0 = k * math.log(alpha) + (n - k) * math.log(1 - alpha)
        ll1 = k * math.log(pi_hat) + (n - k) * math.log(1 - pi_hat)
        lr = -2.0 * (ll0 - ll1)
    return {"n": n, "exceptions": k, "rate": pi_hat, "lr_uc": float(lr),
            "p_value": float(1 - stats.chi2.cdf(lr, df=1))}


def christoffersen_independence(hits: np.ndarray) -> dict:
    """LR test for first-order Markov dependence in the exception sequence."""
    x = np.asarray(hits, dtype=int)
    x = x[np.isfinite(x.astype(float))]
    if len(x) < 3:
        return {"lr_ind": np.nan, "p_value": np.nan, "n00": 0, "n01": 0, "n10": 0, "n11": 0}

    prev, cur = x[:-1], x[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))

    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return {"lr_ind": 0.0, "p_value": 1.0, "n00": n00, "n01": n01, "n10": n10, "n11": n11}

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _ll(p, a, b):
        if p <= 0 or p >= 1:
            return 0.0
        return a * math.log(1 - p) + b * math.log(p)

    ll0 = _ll(pi, n00 + n10, n01 + n11)
    ll1 = _ll(pi01, n00, n01) + _ll(pi11, n10, n11)
    lr = -2.0 * (ll0 - ll1)
    return {"lr_ind": float(lr), "p_value": float(1 - stats.chi2.cdf(lr, df=1)),
            "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def christoffersen_cc(hits: np.ndarray, alpha: float) -> dict:
    """Conditional coverage: Kupiec + independence, chi2 with 2 df."""
    uc = kupiec_pof(hits, alpha)
    ind = christoffersen_independence(hits)
    if not np.isfinite(uc["lr_uc"]) or not np.isfinite(ind["lr_ind"]):
        return {"lr_cc": np.nan, "p_value": np.nan}
    lr = uc["lr_uc"] + ind["lr_ind"]
    return {"lr_cc": float(lr), "p_value": float(1 - stats.chi2.cdf(lr, df=2))}


def acerbi_szekely_z2(
    actual: np.ndarray,
    var_q: np.ndarray,
    es_q: np.ndarray,
    alpha: float,
    n_boot: int = 2_000,
    seed: int = 7,
) -> dict:
    """Acerbi-Szekely Z2 severity statistic for a lower-tail ES forecast.

    Working with a *level* variable (log price), a loss is a fall below the
    ``alpha`` quantile.  Define per-observation contributions

        c_t = 1 + (actual_t - VaR_t) * 1{actual_t < VaR_t} / (alpha * (VaR_t - ES_t))

    so that ``Z2 = mean(c_t)`` is zero in expectation under a correctly
    specified tail (the unshifted term averages to -1 because
    E[(Y - VaR) 1{Y < VaR}] = -alpha * (VaR - ES)), negative when realized shortfalls are *worse* than predicted
    and positive when the predicted tail is too wide.  The p-value comes from a
    block bootstrap of the contributions rather than an asymptotic law, because
    the underlying forecasts overlap.
    """
    y = np.asarray(actual, dtype=float)
    v = np.asarray(var_q, dtype=float)
    e = np.asarray(es_q, dtype=float)
    ok = np.isfinite(y) & np.isfinite(v) & np.isfinite(e)
    y, v, e = y[ok], v[ok], e[ok]
    if len(y) == 0:
        return {"n": 0, "z2": np.nan, "boot_lo": np.nan, "boot_hi": np.nan,
                "p_value": np.nan, "mean_shortfall": np.nan, "mean_predicted_es": np.nan}

    tail_depth = np.maximum(v - e, 1e-12)
    contrib = 1.0 + np.where(y < v, (y - v) / (alpha * tail_depth), 0.0)
    bs = block_bootstrap_mean_ci(contrib, block_length=2, n_boot=n_boot, seed=seed)

    exceed = y < v
    return {
        "n": int(len(y)),
        "n_exceptions": int(exceed.sum()),
        "z2": float(np.mean(contrib)),
        "boot_lo": bs["lo"],
        "boot_hi": bs["hi"],
        "p_value": bs["p_two_sided"],
        "mean_shortfall": float(np.mean(y[exceed])) if exceed.any() else np.nan,
        "mean_predicted_es": float(np.mean(e[exceed])) if exceed.any() else np.nan,
    }


def var_es_report(
    results: pd.DataFrame,
    stride: int = 4,
    alphas=(0.05, 0.01),
    thin_overlap: bool = True,
    min_expected_exceptions: float = 2.0,
) -> pd.DataFrame:
    """Kupiec / Christoffersen / ES report per model, variable, horizon and level.

    ``results`` must contain, for each ``alpha`` in ``alphas``, the columns
    ``q{aa}`` and ``es{aa}`` (e.g. ``q05``/``es05``), plus ``actual``.

    Cells where ``alpha * n_used`` falls below ``min_expected_exceptions`` are
    marked ``underpowered``: at the 1% level with a thinned 20-week panel you
    may expect fewer than one exception, so a passing Kupiec test there carries
    no information and should not be reported as evidence of a well-specified
    tail.
    """
    rows = []
    for (model, variable, horizon), g in results.groupby(
        ["model", "variable", "horizon_weeks"], observed=True
    ):
        g = g.sort_values("origin")
        mask = (
            non_overlapping_subset(g["origin"], int(horizon), stride)
            if thin_overlap
            else np.ones(len(g), dtype=bool)
        )
        gg = g.loc[mask]
        for alpha in alphas:
            tag = f"{int(round(alpha * 100)):02d}"
            qcol, escol = f"q{tag}", f"es{tag}"
            if qcol not in gg.columns or escol not in gg.columns:
                continue
            hits = (gg["actual"].values < gg[qcol].values).astype(int)
            uc = kupiec_pof(hits, alpha)
            ind = christoffersen_independence(hits)
            cc = christoffersen_cc(hits, alpha)
            es = acerbi_szekely_z2(
                gg["actual"].values, gg[qcol].values, gg[escol].values, alpha
            )
            rows.append({
                "model": model,
                "variable": variable,
                "horizon_weeks": int(horizon),
                "alpha": alpha,
                "n_used": uc["n"],
                "n_total": int(len(g)),
                "expected_exceptions": alpha * uc["n"],
                "underpowered": bool(alpha * uc["n"] < min_expected_exceptions),
                "exceptions": uc["exceptions"],
                "exception_rate": uc["rate"],
                "kupiec_p": uc["p_value"],
                "christoffersen_ind_p": ind["p_value"],
                "christoffersen_cc_p": cc["p_value"],
                "es_z2": es["z2"],
                "es_z2_p": es["p_value"],
                "es_z2_lo": es["boot_lo"],
                "es_z2_hi": es["boot_hi"],
            })
    return pd.DataFrame(rows).sort_values(
        ["variable", "horizon_weeks", "alpha", "model"]
    ).reset_index(drop=True)
