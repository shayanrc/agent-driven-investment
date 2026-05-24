"""V5.A.2 — fat-tail panel for the ensemble run.

Thin wrapper that delegates to ``scripts/compute_fat_tail_eval.py`` against the
synthetic ensemble run dir produced by ``scripts/v5/ensemble_paths.py``. Kept
as a separate entry point so the V5_EXPERIMENTS_PLAN.md deliverable list is
satisfied with an explicit script per stage, and so the canonical defaults
(ensemble dir + v2.4 baseline JSON + ``v5_a2`` label) are baked in.

Usage:
    uv run python scripts/v5/compute_v5_a2_fat_tail.py
    # equivalent to:
    uv run python scripts/compute_fat_tail_eval.py \\
        --run-dir runs/analog_mc/v5_a2_ensemble \\
        --label v5_a2 \\
        --baseline-json results/analog_mc/data/fat_tail_baseline_v24.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir", default="runs/analog_mc/v5_a2_ensemble",
        help="ensemble run dir produced by scripts/v5/ensemble_paths.py",
    )
    p.add_argument("--label", default="v5_a2")
    p.add_argument(
        "--baseline-json",
        default="results/analog_mc/data/fat_tail_baseline_v24.json",
    )
    args = p.parse_args()

    # Delegate by importing the underlying script's main() and rewriting argv.
    sys.path.insert(0, str(REPO / "scripts"))
    import compute_fat_tail_eval  # noqa: WPS433 (intentional dynamic import)

    sys.argv = [
        "compute_fat_tail_eval.py",
        "--run-dir", args.run_dir,
        "--label", args.label,
        "--baseline-json", args.baseline_json,
    ]
    compute_fat_tail_eval.main()


if __name__ == "__main__":
    main()
