"""The pre-specified decision rule, with multiplicity control.

v0.7 applied each rule cell by cell and declared ``KEEP`` whenever a single cell
reached p < 0.05.  With twelve cells per rule that threshold is too lenient: at
the 5% level roughly 0.6 spurious wins per rule are expected by chance, so a
one-cell victory carries almost no evidence.  On the real-data run this fired
exactly as feared — macro conditioning was kept on 1 cell out of 12 while its
median effect across cells was *negative*.

Two corrections:

1. :func:`rule_verdict` applies a Romano-Wolf stepdown **across the cells of a
   single rule**.  Because all cells share the same forecast origins, the loss
   differentials form a genuine panel and the bootstrap can preserve their
   cross-cell dependence rather than assuming independence (Holm) or ignoring
   the problem entirely.

2. :func:`baseline_in_mcs` replaces "the best model beats the baseline".
   Picking the winner and then testing it on the same data is the selection
   problem in miniature; the multiplicity-correct version of the question
   "does anything beat the random walk?" is "is the random walk excluded from
   the Model Confidence Set?", which needs no post-hoc choice at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .mcs import _block_bootstrap_indices
from .significance import diebold_mariano, overlap_lag


@dataclass
class Rule:
    """One clause of the pre-specified rule.

    ``challenger`` must beat the *best* of ``incumbents`` -- the best simpler
    alternative, not a fixed one, so a rule cannot be passed merely because the
    comparator chosen for it happens to be weak.
    """

    label: str
    challenger: str
    incumbents: list
    note: str = ""


DEFAULT_RULES: tuple = (
    Rule("recency weighting", "AR_time", ["AR_equal"],
         "does half-life weighting beat equal weighting?"),
    Rule("macro conditioning", "AR_macro", ["AR_equal", "AR_time"],
         "does the macro kernel add anything over the best simple weighting?"),
    Rule("macro vs GARCH filter", "AR_macro", ["FHS_equal", "FHS_time"],
         "or is it a worse proxy for conditional volatility?"),
    Rule("re-centering", "AR_macro", ["AR_macro_nc"],
         "is centring residuals under the final probabilities the right default?"),
    Rule("error correction", "VECM_equal", ["AR_equal", "RW_equal"],
         "does the cointegrating relation carry forecastable information?"),
    Rule("cointegration rank", "VECM_equal", ["VARL_equal", "VARL_time"],
         "does imposing rank 1 beat an unrestricted level VAR?"),
    Rule("state space, restricted", "CTS_equal", ["VECM_equal"],
         "does filtering the relative component beat the raw error-correction term?"),
    Rule("state space, free beta", "CTSF_equal", ["VECM_equal", "CTS_equal"],
         "and does freeing the trend loading matter?"),
)


STABILITY_RULES: tuple = (
    Rule("post-break estimation (VECM)", "VECM_post", ["VECM_equal"],
         "does discarding pre-repeal data improve spread forecasts?"),
    Rule("rolling-window estimation (VECM)", "VECM_roll", ["VECM_equal"],
         "or does a moving window do better than a fixed break?"),
    Rule("post-break estimation (state space)", "CTSF_post", ["CTSF_equal"],
         "same question for the free-beta state space"),
)


def _cell_panel(results: pd.DataFrame, score: str = "crps"):
    """Per-origin losses as {(variable, horizon): wide DataFrame of models}."""
    panel = {}
    for (variable, horizon), g in results.groupby(["variable", "horizon_weeks"], observed=True):
        wide = g.pivot(index="origin", columns="model", values=score).dropna(how="any")
        panel[(variable, int(horizon))] = wide
    return panel


def rule_verdict(
    results: pd.DataFrame,
    rule: Rule,
    stride: int = 4,
    score: str = "crps",
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int = 20260301,
) -> dict:
    """Evaluate one rule across all cells with familywise control within the rule.

    Returns a dict with a per-cell table and an overall ``KEEP`` / ``DROP``
    decision.  ``p_raw`` and ``p_familywise`` are one-sided bootstrap p-values on
    the same statistic, before and after the stepdown; ``dm_pvalue_two_sided`` is
    the ordinary Diebold-Mariano p-value, reported for reference only.

    A rule is kept only if at least one cell shows a *positive* improvement whose
    familywise-adjusted p-value is below ``alpha``.  Ranking first is not enough,
    and neither is a single uncorrected win.
    """
    panel = _cell_panel(results, score=score)

    cells, diffs, rows = [], [], []
    max_h = 1
    for (variable, horizon), wide in sorted(panel.items()):
        incs = [m for m in rule.incumbents if m in wide.columns]
        if rule.challenger not in wide.columns or not incs:
            continue
        binding = min(incs, key=lambda m: wide[m].mean())
        d = wide[rule.challenger].values - wide[binding].values
        base = float(wide[binding].mean())
        dm = diebold_mariano(wide[rule.challenger].values, wide[binding].values,
                             horizon=horizon, stride=stride)
        cells.append((variable, horizon))
        diffs.append(d)
        max_h = max(max_h, horizon)
        rows.append({
            "variable": variable,
            "horizon_weeks": horizon,
            "incumbent": binding,
            "challenger_score": float(wide[rule.challenger].mean()),
            "incumbent_score": base,
            "improvement_pct": 100.0 * (base - float(wide[rule.challenger].mean())) / base,
            "dm_stat": dm["dm_stat"],
            "dm_pvalue_two_sided": dm["p_value"],
        })

    if not rows:
        return {"rule": rule.label, "decision": "N/A", "cells": pd.DataFrame(),
                "n_cells": 0, "n_significant": 0, "median_improvement_pct": np.nan}

    table = pd.DataFrame(rows)
    D = np.column_stack(diffs)                      # origins x cells
    n = len(D)

    # One block length for the whole rule: the most conservative of its cells.
    block_length = overlap_lag(max_h, stride) + 1
    rng = np.random.default_rng(seed)
    boot_idx = _block_bootstrap_indices(n, block_length, n_boot, rng)

    d_bar = D.mean(axis=0)
    boot_d = np.stack([D[idx].mean(axis=0) for idx in boot_idx])
    centered = boot_d - d_bar
    sd = np.sqrt(np.mean(centered ** 2, axis=0))
    sd = np.where(sd < 1e-14, 1e-14, sd)

    stat = -d_bar / sd                              # large => challenger better
    boot_stat = -centered / sd

    order = list(np.argsort(-stat))
    adj = np.empty(len(stat))
    running = 0.0
    while order:
        cols = np.array(order)
        j = order[0]
        p = float(np.mean(boot_stat[:, cols].max(axis=1) >= stat[j]))
        running = max(running, p)
        adj[j] = running
        order.pop(0)

    # One-sided bootstrap p-values, uncorrected and corrected, on the same
    # statistic.  The Diebold-Mariano column is two-sided and reported for
    # reference only: mixing the two in one comparison would be incoherent.
    table["p_raw"] = [float(np.mean(boot_stat[:, i] >= stat[i])) for i in range(len(stat))]
    table["p_familywise"] = adj
    table["passes"] = (table["improvement_pct"] > 0) & (table["p_familywise"] < alpha)

    passed = table[table["passes"]]
    return {
        "rule": rule.label,
        "note": rule.note,
        "challenger": rule.challenger,
        "incumbents": rule.incumbents,
        "decision": "KEEP" if len(passed) else "DROP",
        "cells": table.sort_values(["variable", "horizon_weeks"]).reset_index(drop=True),
        "n_cells": len(table),
        "n_significant": int(len(passed)),
        "median_improvement_pct": float(table["improvement_pct"].median()),
        "block_length": block_length,
    }


def evaluate_rules(
    results: pd.DataFrame,
    rules=DEFAULT_RULES,
    stride: int = 4,
    score: str = "crps",
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int = 20260301,
):
    """Run every rule; return a one-line-per-rule summary and the per-cell detail."""
    summaries, details = [], []
    for rule in rules:
        v = rule_verdict(results, rule, stride=stride, score=score,
                         n_boot=n_boot, alpha=alpha, seed=seed)
        if v["decision"] == "N/A":
            continue
        cells = v["cells"].copy()
        cells.insert(0, "rule", rule.label)
        details.append(cells)
        won = cells[cells["passes"]]
        summaries.append({
            "rule": rule.label,
            "challenger": rule.challenger,
            "incumbents": "/".join(rule.incumbents),
            "decision": v["decision"],
            "n_significant": v["n_significant"],
            "n_cells": v["n_cells"],
            "median_improvement_pct": v["median_improvement_pct"],
            "winning_cells": ", ".join(f"{r.variable}/h{r.horizon_weeks}"
                                       for r in won.itertuples()),
            "binding_incumbent": cells["incumbent"].mode().iat[0],
        })
    summary = pd.DataFrame(summaries)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary, detail


def baseline_in_mcs(mcs_table: pd.DataFrame, baseline: str = "RW_equal") -> pd.DataFrame:
    """Per cell: is the baseline inside the 90% Model Confidence Set?

    This replaces "the best model beats the baseline", which requires choosing a
    winner after seeing the data and then testing it on the same data.  Asking
    whether the baseline survives the confidence set needs no post-hoc choice and
    is already corrected for the size of the grid: if the random walk is excluded,
    *something* beats it at that level, whichever model it is.
    """
    rows = []
    for (variable, horizon), g in mcs_table.groupby(["variable", "horizon_weeks"], observed=True):
        if baseline not in set(g["model"]):
            continue
        indexed = g.set_index("model")
        row = indexed.loc[baseline]
        best = g.sort_values("mean_loss").iloc[0]
        rows.append({
            "variable": variable,
            "horizon_weeks": int(horizon),
            "baseline_in_mcs_90": bool(row["in_mcs_90"]),
            "baseline_mcs_pvalue": float(row["mcs_pvalue"]),
            "baseline_rank": int(indexed["mean_loss"].rank().loc[baseline]),
            "n_in_mcs_90": int(g["in_mcs_90"].sum()),
            "n_models": int(len(g)),
            "best_model": best["model"],
            "best_improvement_pct": 100.0 * (row["mean_loss"] - best["mean_loss"]) / row["mean_loss"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["something_beats_baseline"] = ~out["baseline_in_mcs_90"]
    return out.sort_values(["variable", "horizon_weeks"]).reset_index(drop=True)


def format_verdict(summary: pd.DataFrame, baseline_table: pd.DataFrame,
                   rules=DEFAULT_RULES, baseline: str = "RW_equal") -> str:
    """Human-readable verdict block."""
    notes = {r.label: r.note for r in rules}
    lines = ["=" * 84,
             "PRE-SPECIFIED DECISION RULE",
             "challenger vs the best simpler alternative; p-values familywise-corrected",
             "across the cells of each rule (Romano-Wolf stepdown, block bootstrap)",
             "=" * 84]
    for r in summary.itertuples():
        lines.append("")
        lines.append(f"[{r.decision}] {r.rule}: {r.challenger} vs {r.incumbents}")
        if notes.get(r.rule):
            lines.append(f"        {notes[r.rule]}")
        lines.append(f"        significant in {r.n_significant}/{r.n_cells} cells after "
                     f"correction   median {r.median_improvement_pct:+.2f}%")
        if r.winning_cells:
            lines.append(f"        wins: {r.winning_cells}")
        else:
            lines.append(f"        binding incumbent in most cells: {r.binding_incumbent}")

    lines += ["", "=" * 84,
              f"IS THE BASELINE ({baseline}) EXCLUDED FROM THE 90% MCS?",
              "the selection-free version of 'does anything beat the random walk?'",
              "=" * 84]
    for r in baseline_table.itertuples():
        mark = "EXCLUDED -> something beats it" if r.something_beats_baseline else "survives"
        lines.append(f"{r.variable:7s} h={r.horizon_weeks:<3d} {mark:32s} "
                     f"| MCS keeps {r.n_in_mcs_90}/{r.n_models} | best: {r.best_model} "
                     f"({r.best_improvement_pct:+.1f}%)")
    return "\n".join(lines)
