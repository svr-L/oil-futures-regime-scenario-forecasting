"""Data access with an on-disk cache.

v0.3 called yfinance directly from the notebook, which made every result
dependent on a live network call and on whatever Yahoo returns that day: the
repository was not reproducible by a reader six months later, and CI could
never touch the real pipeline.

Here every download goes through :func:`load_or_download`, which writes a
parquet snapshot under ``data/cache`` and reads from it on subsequent runs.
Commit the cache and the notebooks become deterministic.

A second addition is :func:`roll_gap_diagnostics`.  ``CL=F`` and ``BZ=F`` are
*concatenated front-month* series, not roll-adjusted ones, so returns spanning a
contract roll contain a price jump that is not a tradeable return.  Those jumps
contaminate the WTI/Brent spread and therefore the VECM beta and the estimated
spread half-life.  The function flags the largest weekly moves so the caveat can
be quantified rather than merely mentioned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")


def _cache_path(name: str, start: str, end: str, anchor: str, cache_dir: Path) -> Path:
    safe = f"{name}_{start}_{end}_{anchor}".replace(":", "").replace("/", "-")
    return Path(cache_dir) / f"{safe}.parquet"


def _download_weekly(tickers, start, end, anchor, auto_adjust) -> pd.DataFrame:
    from yfinance import download  # imported lazily so the package works offline

    if isinstance(tickers, Mapping):
        yf_tickers = list(tickers.values())
        rename_map = {v: k for k, v in tickers.items()}
    else:
        yf_tickers = list(tickers)
        rename_map = {v: v for v in yf_tickers}

    df = download(yf_tickers, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in df.columns.get_level_values(0) else "Close"
        px = df[field]
    else:
        field = "Adj Close" if "Adj Close" in df.columns else "Close"
        px = df[[field]]
        px.columns = [yf_tickers[0]]

    if isinstance(px, pd.Series):
        px = px.to_frame(name=yf_tickers[0])
    px = px.rename(columns=rename_map)
    return px.resample(anchor).last()


def load_or_download(
    name: str,
    tickers: Mapping[str, str] | Sequence[str],
    start: str,
    end: str,
    anchor: str = "W-FRI",
    auto_adjust: bool = False,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    dropna: str = "any",
) -> pd.DataFrame:
    """Return weekly closes, downloading only when the cache is missing or stale.

    Parameters
    ----------
    name:
        Cache key, e.g. ``"oil"`` or ``"macro_state"``.
    refresh:
        Force a fresh download and overwrite the snapshot.
    dropna:
        ``"any"``, ``"all"`` or ``"none"``.
    """
    path = _cache_path(name, start, end, anchor, Path(cache_dir))
    if path.exists() and not refresh:
        px = pd.read_parquet(path)
    else:
        px = _download_weekly(tickers, start, end, anchor, auto_adjust)
        path.parent.mkdir(parents=True, exist_ok=True)
        px.to_parquet(path)

    px.index = pd.DatetimeIndex(px.index)
    if dropna in {"any", "all"}:
        px = px.dropna(how=dropna)
    return px.sort_index()


# Backwards-compatible thin wrappers used by notebooks 01 and 02.
def download_weekly_close(tickers, start, end, anchor="W-FRI", auto_adjust=False,
                          cache_dir=DEFAULT_CACHE_DIR, refresh=False) -> pd.DataFrame:
    return load_or_download("prices", tickers, start, end, anchor, auto_adjust,
                            cache_dir, refresh, dropna="any")


def get_weekly_state(tickers, start, end, anchor="W-WED", auto_adjust=False,
                     cache_dir=DEFAULT_CACHE_DIR, refresh=False) -> pd.DataFrame:
    return load_or_download("state", tickers, start, end, anchor, auto_adjust,
                            cache_dir, refresh, dropna="all")


def roll_gap_diagnostics(prices: pd.DataFrame, z_threshold: float = 4.0) -> pd.DataFrame:
    """Flag weekly log returns whose size is consistent with a contract-roll jump.

    Returns one row per (series, date) whose absolute log return exceeds
    ``z_threshold`` robust standard deviations, using a median/MAD scale so a
    handful of jumps do not inflate the threshold that is supposed to catch them.
    """
    logp = np.log(prices.dropna(how="any").astype(float))
    rets = logp.diff().dropna()
    rows = []
    for col in rets.columns:
        r = rets[col]
        med = float(r.median())
        mad = float((r - med).abs().median())
        scale = 1.4826 * mad if mad > 0 else float(r.std())
        if scale <= 0:
            continue
        z = (r - med) / scale
        flagged = z[z.abs() > z_threshold]
        for date, zz in flagged.items():
            rows.append({
                "series": col,
                "date": pd.Timestamp(date),
                "log_return": float(r.loc[date]),
                "robust_z": float(zz),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("robust_z", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def spread_jump_summary(prices: pd.DataFrame, a: str = "WTI", b: str = "BRENT",
                        z_threshold: float = 4.0) -> pd.DataFrame:
    """Same diagnostic applied to the log spread, which is what the VECM models."""
    logp = np.log(prices[[a, b]].dropna(how="any").astype(float))
    spread = (logp[a] - logp[b]).rename("SPREAD").to_frame()
    return roll_gap_diagnostics(np.exp(spread), z_threshold=z_threshold)


# ---------------------------------------------------------------------------
# EIA physical / futures-curve state
# ---------------------------------------------------------------------------

EIA_WTI_CURVE_XLS = {
    "C1": "https://www.eia.gov/dnav/pet/hist_xls/RCLC1w.xls",
    "C2": "https://www.eia.gov/dnav/pet/hist_xls/RCLC2w.xls",
    "C3": "https://www.eia.gov/dnav/pet/hist_xls/RCLC3w.xls",
    "C4": "https://www.eia.gov/dnav/pet/hist_xls/RCLC4w.xls",
}
EIA_CUSHING_XLS = (
    "https://www.eia.gov/dnav/pet/hist_xls/"
    "W_EPC0_SAX_YCUOK_MBBLw.xls"
)


def _read_eia_weekly_xls(source, value_name: str) -> pd.Series:
    """Read one EIA ``hist_xls`` weekly series.

    ``source`` may be a local XLS path, file-like object, or an HTTP(S) URL.
    EIA's history workbooks use sheet ``Data 1`` with two metadata rows, then a
    date column and a single value column.  Keeping this parser here makes the
    public-data branch reproducible without an EIA API key.
    """
    frame = pd.read_excel(source, sheet_name="Data 1", skiprows=2)
    if frame.shape[1] < 2:
        raise ValueError("unexpected EIA workbook layout: expected date + value columns")
    date = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    value = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    out = pd.Series(value.to_numpy(), index=pd.DatetimeIndex(date), name=value_name)
    out = out[~out.index.isna()].dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.astype(float)


def load_eia_wti_curve(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    sources: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Weekly EIA WTI futures curve, contracts 1--4.

    The EIA discontinued these futures-price series after 5 April 2024.  The
    function therefore exposes them as a *historical state variable*, not as a
    current live-data feed.  A parquet snapshot is cached for deterministic OOS
    research.  ``sources`` exists mainly for tests / air-gapped workflows and may
    map ``C1``...``C4`` to local XLS files.
    """
    cache = Path(cache_dir) / "eia_wti_curve_c1_c4_weekly.parquet"
    if cache.exists() and not refresh:
        out = pd.read_parquet(cache)
        out.index = pd.DatetimeIndex(out.index)
        return out.sort_index()

    src = dict(EIA_WTI_CURVE_XLS if sources is None else sources)
    missing = {"C1", "C2", "C3", "C4"}.difference(src)
    if missing:
        raise ValueError(f"missing EIA curve sources: {sorted(missing)}")
    cols = [_read_eia_weekly_xls(src[k], k) for k in ("C1", "C2", "C3", "C4")]
    out = pd.concat(cols, axis=1).dropna(how="any").sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def load_eia_cushing(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    source: object | None = None,
) -> pd.Series:
    """Weekly Cushing crude stocks excluding SPR, thousand barrels.

    EIA history begins in 2004 and is still maintained.  The value for a week
    ending Friday is published after that Friday; forecasting code should lag it
    before using it at a Friday origin.
    """
    cache = Path(cache_dir) / "eia_cushing_stocks_weekly.parquet"
    if cache.exists() and not refresh:
        frame = pd.read_parquet(cache)
        frame.index = pd.DatetimeIndex(frame.index)
        if isinstance(frame, pd.Series):
            return frame.sort_index().rename("CUSHING")
        return frame.iloc[:, 0].sort_index().rename("CUSHING")

    ser = _read_eia_weekly_xls(EIA_CUSHING_XLS if source is None else source, "CUSHING")
    cache.parent.mkdir(parents=True, exist_ok=True)
    ser.to_frame().to_parquet(cache)
    return ser


def load_long_oil_history(
    start: str = "2008-01-01",
    end: str = "2025-03-08",
    anchor: str = "W-FRI",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Longer WTI/Brent history reserved for structural-stability diagnostics.

    The main experiment deliberately keeps its original 2013 start.  Extending
    only the stability sample gives the pre-2015 break tests more power without
    silently changing the forecasting experiment that generated the headline
    results.
    """
    return load_or_download(
        "oil_stability", {"WTI": "CL=F", "BRENT": "BZ=F"}, start, end, anchor,
        auto_adjust=False, cache_dir=cache_dir, refresh=refresh, dropna="any",
    )
