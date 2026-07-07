"""``python -m gbdt.experiment <spec.yaml>`` entry point.

Thin wrapper over :func:`gbdt.experiment_runner.run_experiment` so the
``/gbdt-experiment`` skill and the docs' ``python -m gbdt.experiment <spec>``
invocation work without going through the top-level subcommand.

The v1 Stage 8 orchestrator originally lived in ``__main__``; the runner
split moved it to ``gbdt.experiment_runner`` (with spec handling in
``gbdt.spec`` and the agent/scout cycles in ``gbdt.agent_cycles``) and left
``__main__`` as the CLI + back-compat re-export surface. Importing
``gbdt.experiment`` still re-exports ``run_experiment`` and ``load_spec``
for programmatic use.
"""

from __future__ import annotations

import sys
from pathlib import Path

from gbdt.__main__ import load_spec, run_experiment  # noqa: F401  (re-export)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python -m gbdt.experiment <spec.yaml> [--overwrite]")
        return 0 if argv else 2
    overwrite = False
    args = list(argv)
    if "--overwrite" in args:
        args.remove("--overwrite")
        overwrite = True
    if len(args) != 1:
        print("error: expected one positional spec path", file=sys.stderr)
        return 2
    run_experiment(Path(args[0]), overwrite=overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
