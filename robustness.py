"""Robustness checks on a scored forecast panel.

Two failure modes that a single mean-CRPS table hides:

**A handful of origins can decide the ranking.** With roughly 85 origins and a
20-week horizon, March-April 2020 can contribute a small number of enormous
losses. A model can win the average while losing the median, which means it won a
few exceptional weeks rather than the full sample.

**A late split is not automatically an untouched holdout.** The rolling validation
is pseudo-out-of-sample because later versions of the code were developed with the
full historical sample visible. ``split_dev_holdout`` is retained as a backwards-
compatible function name, but v1.0 uses it only for an early-vs-late ranking-
stability check. Genuine external validation is handled by the frozen post-2025
forward script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def exclude_period(results: pd.DataFrame, start, end, on: str = "target_date") -> pd.DataFrame:
    """Drop rows whose target (or origin) falls inside a date window."""
    col = on if on in results.columns else "origin"
    dates = pd.to_datetime(results[col])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return results.loc[~mask].copy()


def split_dev_holdout(results: pd.DataFrame, holdout_start, on: str = "origin"):
    """Split a scored panel into an early development part and a late validation block."""
    dates = pd.to_datetime(results[on])
    cut = pd.Timestamp(holdout_start)
    return results.loc[dates < cut].copy(), results.loc[dates >= cut].copy()


def mean_vs_median_table(results: pd.DataFrame, score: str = "crps") -> pd.DataFrame:
    """Mean and median score side by side, with the ranks each one implies.

    A model whose mean rank is much better than its median rank is winning on a
    few extreme origins rather than on typical ones.
    """
    grp = results.groupby(["variable", "horizon_weeks", "model"], observed=True)[score]
    out = grp.agg(mean_score="mean", median_score="median", n="size").reset_index()
    out["rank_mean"] = out.groupby(["variable", "horizon_weeks"], observed=True)["mean_score"].rank()
    out["rank_median"] = out.groupby(["variable", "horizon_weeks"], observed=True)["median_score"].rank()
    out["rank_gap"] = out["rank_mean"] - out["rank_median"]
    return out.sort_values(["variable", "horizon_weeks", "mean_score"]).reset_index(drop=True)


def influence_of_worst_origins(
    results: pd.DataFrame,
    score: str = "crps",
    top_k: int = 3,
) -> pd.DataFrame:
    """How much of the average loss comes from the ``top_k`` worst origins.

    Computed per variable and horizon on the origins that are worst *on average
    across models*, so the same dates are removed for everybody and the
    comparison stays balanced.
    """
    rows = []
    for (variable, horizon), g in results.groupby(["variable", "horizon_weeks"], observed=True):
        by_origin = g.groupby("origin", observed=True)[score].mean().sort_values(ascending=False)
        worst = list(by_origin.index[:top_k])
        kept = g[~g["origin"].isin(worst)]
        for model, gm in g.groupby("model", observed=True):
            full = float(gm[score].mean())
            trimmed = float(kept[kept["model"] == model][score].mean())
            rows.append({
                "variable": variable,
                "horizon_weeks": int(horizon),
                "model": model,
                "mean_full": full,
                f"mean_excl_worst_{top_k}": trimmed,
                "share_from_worst_pct": 100.0 * (full - trimmed) / full if full else np.nan,
                "worst_origins": ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in worst),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_full"] = out.groupby(["variable", "horizon_weeks"], observed=True)["mean_full"].rank()
    out["rank_trimmed"] = out.groupby(
        ["variable", "horizon_weeks"], observed=True
    )[f"mean_excl_worst_{top_k}"].rank()
    return out.sort_values(["variable", "horizon_weeks", "mean_full"]).reset_index(drop=True)


def ranking_stability(
    full: pd.DataFrame,
    variant: pd.DataFrame,
    score: str = "crps",
    label: str = "variant",
) -> pd.DataFrame:
    """Compare model rankings between the full panel and a robustness variant."""
    def _ranks(df):
        g = df.groupby(["variable", "horizon_weeks", "model"], observed=True)[score].mean()
        return g.groupby(level=[0, 1]).rank()

    a, b = _ranks(full).rename("rank_full"), _ranks(variant).rename(f"rank_{label}")
    out = pd.concat([a, b], axis=1).reset_index()
    out["rank_change"] = out[f"rank_{label}"] - out["rank_full"]
    return out.sort_values(["variable", "horizon_weeks", "rank_full"]).reset_index(drop=True)


def relative_gain_by_period(
    results: pd.DataFrame,
    challenger: str,
    incumbent: str,
    variable: str = "SPREAD",
    freq: str = "Y",
    score: str = "crps",
) -> pd.DataFrame:
    """Challenger's score improvement over the incumbent, grouped by origin period.

    A pooled average can hide a gain that is real but transient.  The motivating
    case is estimation after a structural break: in simulation with a genuine
    break planted at the export-ban repeal, estimating on post-break data only
    improved spread CRPS by 25-32% at origins within two years of the break, then
    faded to noise as the expanding window filled with post-break observations.
    Pooled over 2018-2024 origins the effect was slightly *negative*.  This table
    is how that pattern becomes visible.

    Positive values mean the challenger is better.
    """
    g = results[results["variable"] == variable]
    wide = g.pivot_table(index=["origin", "horizon_weeks"], columns="model",
                         values=score).reset_index()
    if challenger not in wide or incumbent not in wide:
        raise ValueError("challenger or incumbent missing from results")
    wide["gain_pct"] = 100.0 * (wide[incumbent] - wide[challenger]) / wide[incumbent]
    period = pd.to_datetime(wide["origin"]).dt.to_period(freq).astype(str)
    wide["period"] = period
    table = wide.groupby(["period", "horizon_weeks"])["gain_pct"].mean().unstack()
    counts = wide.groupby("period")["origin"].nunique().rename("n_origins")
    return table.join(counts)
