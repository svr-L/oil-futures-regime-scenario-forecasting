# Data cache

`scripts/fetch_data.py` writes parquet snapshots here for the weekly Yahoo series,
the dedicated longer stability history, the latest post-freeze snapshot, the EIA
WTI C1-C4 futures curve and Cushing inventories.

The repository does not need to commit proprietary data. Public snapshots may be
committed when licensing permits; otherwise this directory can contain only this
README and each loader will recreate its cache from the documented public source.
Delete a snapshot or pass `--refresh` to refresh it.

Key v1.0 cache names include:

- `oil.parquet`, `state.parquet` — frozen 2013-2025 core experiment;
- `oil_stability.parquet` — longer history used only by structural diagnostics;
- `oil_latest.parquet`, `state_latest.parquet` — forward external validation;
- `eia_wti_curve_c1_c4_weekly.parquet` — public historical WTI curve state;
- `eia_cushing_stocks_weekly.parquet` — public physical inventory state.
