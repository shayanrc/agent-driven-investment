"""CLI orchestrator: ``python -m gbdt experiment <spec.yaml>``.

Loads a spec, builds the universe panel + 279-col feature matrix + binary
target, runs the walk-forward driver with the default algorithmic FS+HP
fallback (the ``/gbdt-experiment`` skill overrides this with agent loops),
applies the calibration policy, and emits the full per-experiment artifact
directory at ``results/gbdt/experiments/<experiment_name>/``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Re-exports: the implementation moved to gbdt.spec / gbdt.agent_cycles /
# gbdt.experiment_runner in the runner split. Tests, gbdt.experiment, and
# downstream scripts keep importing these names from gbdt.__main__.
from gbdt.agent_cycles import (  # noqa: F401
    _build_combine_fit_one,
    _build_fs_prefit_runner_fit_one,
    _build_scout_metrics_blocks,
    _build_scout_runner_fit_one,
    _handle_scout_cycles_agent_mode,
    _load_and_apply_resume,
    _make_agent_file_protocol_callback,
    _resolve_callback,
)
from gbdt.experiment_runner import (  # noqa: F401
    _clear_stale_loop_decisions,
    _collect_preflight,
    _compute_headline,
    _data_hash,
    _format_preflight_line,
    _format_test_split_warning,
    _has_runner_artifacts,
    _project_test_rows,
    _sanitize_path_for_emission,
    _sanitize_preflight_for_emission,
    run_experiment,
)
from gbdt.spec import (  # noqa: F401
    _DEFAULT_CALLBACK_MODE,
    _DEFAULT_DEGENERATE_SINK_THRESHOLD,
    _DEFAULT_SWEEP_CSV_RELPATH,
    _HP_SEARCH_ITER_THRESHOLD,
    _TEST_ROWS_WARNING_THRESHOLD,
    _VALID_BACKENDS,
    _VALID_CALLBACK_MODES,
    _VALID_CAL_METHODS,
    _VALID_DIRECTIONS,
    _deep_merge,
    _spec_hash,
    _strip_internal_keys,
    _validate_spec,
    load_spec,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gbdt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("experiment", help="Run one gbdt experiment end-to-end")
    p_exp.add_argument("spec", type=Path, help="Path to spec YAML")
    p_exp.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing non-empty artifact dir")
    p_exp.add_argument(
        "--callback-mode",
        choices=sorted(_VALID_CALLBACK_MODES),
        default=None,
        help="Override backend.fs_hp_loop.callback_mode from the spec "
             "(default: use the spec's value, or 'default' if absent).",
    )
    p_exp.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="Resume a paused agent-driven FS+HP run (callback_mode="
             "agent_file_protocol). Loads the run's checkpoint + the agent's "
             "loop/iter_<N>_decision.json, validates + applies it, and "
             "continues at iteration N+1. RUN_ID is the value printed in the "
             "pause hint.",
    )
    p_exp.add_argument(
        "--snapshot-end",
        metavar="YYYY-MM-DD",
        default=None,
        help="Pin the date_range.end for this run (overrides the spec). "
             "Use the SAME value across every cell of a sweep so the universe-"
             "level feature cache key stays stable cell-to-cell (an "
             "auto-fetch between cells will otherwise drift the panel "
             "signature and force a cold rebuild on every sibling). See "
             "bug #226.",
    )

    args = parser.parse_args(argv)
    if args.cmd == "experiment":
        snapshot_end_val: date | None = None
        if args.snapshot_end is not None:
            try:
                snapshot_end_val = date.fromisoformat(args.snapshot_end)
            except ValueError as exc:
                print(
                    f"[experiment] --snapshot-end: invalid ISO date "
                    f"{args.snapshot_end!r} ({exc})",
                    file=sys.stderr,
                )
                return 2
        run_experiment(
            args.spec,
            overwrite=args.overwrite,
            callback_mode_override=args.callback_mode,
            resume=args.resume,
            snapshot_end=snapshot_end_val,
        )
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
