"""``python -m backtesting`` entry point.

Delegates to :func:`backtesting.cli.main`; see that module's docstring
for the YAML config schema and usage.
"""

from __future__ import annotations

import sys

from backtesting.cli import main

if __name__ == "__main__":
    sys.exit(main())
