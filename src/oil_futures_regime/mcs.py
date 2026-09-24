"""Multiple-comparison control for forecast-model selection.

Every Diebold-Mariano test in :mod:`oil_futures_regime.significance` is
individually correct, but the *selection* of a winner from a model family is not.
The v1.0 grid contains 19 models across three variables and four horizons.  For
baseline comparisons, Romano-Wolf correction is applied separately inside each
variable x horizon family (18 challengers against one baseline), where the loss
differentials share the same forecast origins and bootstrap dependence
structure.  The resulting p-values therefore control familywise error *within a
cell*; they are not a single global FWER correction across all model-cell
entries.  Reporting a selected winner without this distinction would overstate
the evidence.

Two standard corrections are provided:

* :func:`model_confidence_set` -- Hansen, Lunde and Nason (2011).  Instead of
  naming one winner, it returns the *set* of models that cannot be
  distinguished from the best at a given confidence level.  With overlapping
  forecast windows and fewer than 20 effectively independent observations, an
  honest answer is usually a set, not a point.

* :func:`romano_wolf_stepdown` -- Romano and Wolf (2005).  Familywise-error
  control for many models tested against one baseline, which is strictly more
  powerful than Bonferroni because it exploits the dependence between the test
  statistics.

Both use a circular moving-block bootstrap whose block length defaults to the
overlap implied by horizon and stride, so serial dependence in the loss
differentials is respected.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .significance import overlap_lag


def _block_bootstrap_indices(n: int, block_length: int, n_boot: int, rng) -> np.ndarray:
    """Circular moving-block bootstrap index matrix of shape (n_boot, n)."""
    block_length = int(max(1, min(block_length, n)))
    n_blocks = int(math.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    horizon: int = 1,
    stride: int = 1,
    block_length: int | None = None,
    n_boot: int = 2_000,
    seed: int = 20260201,
) -> pd.DataFrame:
    """Hansen-Lunde-Nason Model Confidence Set using the T_max statistic.

    Parameters
    ----------
    losses:
        Rows are forecast origins, columns are models, entries are per-origin
        losses (CRPS, pinball, ...).  Must be a balanced panel: no NaNs.
    alpha:
        Models with an MCS p-value below ``alpha`` are excluded from the set.
        ``alpha = 0.10`` gives the 90% MCS.

    Returns
    -------
    DataFrame with one row per model: mean loss, elimination order, MCS
    p-value, and membership flags at the 90% and 75% levels.  The MCS p-value is
    the largest level at which a model still belongs to the set, and is
    monotone in the elimination order by construction.

    Notes
    -----
    The procedure iteratively tests the null that all surviving models are
    equally good.  When it rejects, the model with the worst standardized
    average loss differential is eliminated.  A model's MCS p-value is the
    running maximum of the p-values observed up to its elimination, which is
    what makes the set interpretation valid.
    """
    L = losses.dropna(how="any")
    if L.shape[1] < 2:
        raise ValueError("need at least two models to build a confidence set")
    n, m = L.shape
    if n < 8:
        raise ValueError("too few origins for a meaningful confidence set")

    if block_length is None:
        block_length = overlap_lag(horizon, stride) + 1
    rng = np.random.default_rng(seed)
    boot_idx = _block_bootstrap_indices(n, block_length, n_boot, rng)

    values = L.values
    names = list(L.columns)
    # Bootstrap replicates of each model's mean loss: (n_boot, m)
    boot_means = np.stack([values[idx].mean(axis=0) for idx in boot_idx])
    means = values.mean(axis=0)

    alive = list(range(m))
    elimination_order: dict[str, int] = {}
    mcs_p: dict[str, float] = {}
    running_p = 0.0
    step = 0

    while len(alive) > 1:
        step += 1
        idx = np.array(alive)
        mu = means[idx]
        boot_mu = boot_means[:, idx]
        k = len(idx)

        # Average loss differential of model i against the surviving set.
        d = mu - (mu.sum() - mu) / (k - 1)
        boot_d = boot_mu - (boot_mu.sum(axis=1, keepdims=True) - boot_mu) / (k - 1)

        centered = boot_d - d
        zeta = np.sqrt(np.mean(centered ** 2, axis=0))
        zeta = np.where(zeta < 1e-14, 1e-14, zeta)

        t_stat = d / zeta
        T_obs = float(np.max(t_stat))
        T_boot = np.max(centered / zeta, axis=1)
        p = float(np.mean(T_boot >= T_obs))

        running_p = max(running_p, p)
        worst_local = int(np.argmax(t_stat))
        worst = int(idx[worst_local])

        mcs_p[names[worst]] = running_p
        elimination_order[names[worst]] = step
        alive.remove(worst)

    last = names[alive[0]]
    mcs_p[last] = 1.0
    elimination_order[last] = step + 1

    out = pd.DataFrame({
        "model": names,
        "mean_loss": means,
        "elimination_step": [elimination_order[nm] for nm in names],
        "mcs_pvalue": [mcs_p[nm] for nm in names],
    })
    out["in_mcs_90"] = out["mcs_pvalue"] >= 0.10
    out["in_mcs_75"] = out["mcs_pvalue"] >= 0.25
    out["in_mcs"] = out["mcs_pvalue"] >= alpha
    return out.sort_values("mean_loss").reset_index(drop=True)


def romano_wolf_stepdown(
    losses: pd.DataFrame,
    baseline: str,
    horizon: int = 1,
    stride: int = 1,
    block_length: int | None = None,
    n_boot: int = 2_000,
    seed: int = 20260202,
    one_sided: bool = True,
) -> pd.DataFrame:
    """Romano-Wolf stepdown: familywise-error control against a single baseline.

    Tests, for every model, whether its expected loss is lower than the
    baseline's, controlling the probability of *any* false rejection across the
    whole family.  Strictly more powerful than Bonferroni because the bootstrap
    preserves the dependence between the statistics.

    Returns a DataFrame with the raw and familywise-adjusted p-values and a
    rejection flag at 5%.
    """
    L = losses.dropna(how="any")
    if baseline not in L.columns:
        raise ValueError(f"baseline {baseline!r} is not a column of losses")
    others = [c for c in L.columns if c != baseline]
    if not others:
        raise ValueError("no competing models to test")

    n = len(L)
    if block_length is None:
        block_length = overlap_lag(horizon, stride) + 1
    rng = np.random.default_rng(seed)
    boot_idx = _block_bootstrap_indices(n, block_length, n_boot, rng)

    D = (L[others].values - L[[baseline]].values)      # negative => model better
    d_bar = D.mean(axis=0)
    boot_d = np.stack([D[idx].mean(axis=0) for idx in boot_idx])
    centered = boot_d - d_bar
    sd = np.sqrt(np.mean(centered ** 2, axis=0))
    sd = np.where(sd < 1e-14, 1e-14, sd)

    # One-sided: evidence that the model beats the baseline => large -d/sd.
    stat = (-d_bar / sd) if one_sided else np.abs(d_bar / sd)
    boot_stat = (-centered / sd) if one_sided else np.abs(centered / sd)

    order = np.argsort(-stat)
    adj_p = np.empty(len(others))
    remaining = list(order)
    running = 0.0
    while remaining:
        cols = np.array(remaining)
        max_boot = boot_stat[:, cols].max(axis=1)
        j = remaining[0]
        p = float(np.mean(max_boot >= stat[j]))
        running = max(running, p)
        adj_p[j] = running
        remaining.pop(0)

    raw_p = np.array([float(np.mean(boot_stat[:, i] >= stat[i])) for i in range(len(others))])

    out = pd.DataFrame({
        "model": others,
        "mean_loss": L[others].mean().values,
        "delta_vs_baseline": d_bar,
        "improvement_pct": 100.0 * (-d_bar) / L[baseline].mean(),
        "statistic": stat,
        "p_raw": raw_p,
        "p_familywise": adj_p,
    })
    out["reject_at_5pct"] = out["p_familywise"] < 0.05
    return out.sort_values("mean_loss").reset_index(drop=True)


def mcs_by_cell(
    results: pd.DataFrame,
    score: str = "crps",
    alpha: float = 0.10,
    stride: int = 4,
    n_boot: int = 1_000,
    seed: int = 20260201,
) -> pd.DataFrame:
    """Run a Model Confidence Set separately for every variable x horizon cell."""
    frames = []
    for (variable, horizon), g in results.groupby(["variable", "horizon_weeks"], observed=True):
        wide = g.pivot(index="origin", columns="model", values=score).dropna(how="any")
        mcs = model_confidence_set(
            wide, alpha=alpha, horizon=int(horizon), stride=stride,
            n_boot=n_boot, seed=seed,
        )
        mcs.insert(0, "horizon_weeks", int(horizon))
        mcs.insert(0, "variable", variable)
        frames.append(mcs)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mcs_summary(mcs_table: pd.DataFrame) -> pd.DataFrame:
    """One line per cell: how many models survive, and which."""
    rows = []
    for (variable, horizon), g in mcs_table.groupby(["variable", "horizon_weeks"], observed=True):
        keep = g[g["in_mcs_90"]].sort_values("mean_loss")
        rows.append({
            "variable": variable,
            "horizon_weeks": int(horizon),
            "n_models": len(g),
            "n_in_mcs_90": len(keep),
            "best_model": g.sort_values("mean_loss").iloc[0]["model"],
            "mcs_90_members": ", ".join(keep["model"].tolist()),
        })
    return pd.DataFrame(rows).sort_values(["variable", "horizon_weeks"]).reset_index(drop=True)
