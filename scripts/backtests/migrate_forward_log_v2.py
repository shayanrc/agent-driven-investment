"""One-time migration: forward_predictions_log.csv v1 → v2 unified schema.

v1 (12 cols, sp500-only): gate columns named ``spx_close`` / ``spx_sma200``.
v2 (16 cols): adds ``cell`` / ``universe`` / ``gate_index`` / ``deployed`` and
renames the two gate columns ``spx_*`` → ``gate_*`` (universe-aware).

Value-preserving for the existing champion rows — every shared column is
byte-identical; the migration only renames the two gate columns and adds the
new descriptor columns. Idempotent (a no-op once the log is already v2). Run once
before the candidate backfill (``daily_forward_predictions.py``).
"""
from __future__ import annotations

import pandas as pd

from scripts.backtests.daily_forward_predictions import LOG, LOG_COLUMNS

# The only models present in v1 — both sp500 champions, deployed.
_V1_CHAMPIONS = {
    "sp500_50": "sp500_up_50pct_50d_dd25pct_agentloop",
    "sp500_20": "sp500_up_20pct_25d_dd10pct_agentloop",
}


def main() -> None:
    df = pd.read_csv(LOG)
    if set(LOG_COLUMNS).issubset(df.columns):
        print(f"already v2 ({len(df)} rows) — no-op.")
        return
    unexpected = set(df["model"]) - set(_V1_CHAMPIONS)
    if unexpected:
        raise SystemExit(f"unexpected models in v1 log (not migratable): {unexpected}")
    df["cell"] = df["model"].map(_V1_CHAMPIONS)
    df["universe"] = "sp500"
    df["gate_index"] = "INDEX:^SPX"
    df["deployed"] = True
    df = df.rename(columns={"spx_close": "gate_close", "spx_sma200": "gate_sma200"})
    out = df.reindex(columns=LOG_COLUMNS)
    out.to_csv(LOG, index=False)
    print(f"migrated {len(out)} champion rows → v2 ({len(LOG_COLUMNS)} cols): "
          f"{LOG.relative_to(LOG.parents[3])}")


if __name__ == "__main__":
    main()
