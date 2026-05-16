"""CLI entry point: ``python -m analog_mc <subcommand> [opts]``.

Currently the only subcommand is ``walk-forward``. Putting the CLI here
(rather than in ``walk_forward.py`` directly) avoids the import-order
RuntimeWarning that fires when ``python -m analog_mc.walk_forward`` triggers
the package ``__init__`` before executing the module as ``__main__``.
"""

from __future__ import annotations

import sys

from analog_mc.walk_forward import _cli as walk_forward_cli


SUBCOMMANDS = {
    "walk-forward": walk_forward_cli,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(f"usage: python -m analog_mc <subcommand> [opts]")
        print(f"subcommands: {', '.join(SUBCOMMANDS)}")
        return 0 if argv else 2
    sub = argv[0]
    if sub not in SUBCOMMANDS:
        print(f"unknown subcommand: {sub}; available: {', '.join(SUBCOMMANDS)}", file=sys.stderr)
        return 2
    return SUBCOMMANDS[sub](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
