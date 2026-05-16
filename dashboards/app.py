"""Global dashboard launcher.

Lists every module that has a ``dashboards/<module>/app.py`` and lets the user
pick which to open. Currently the only module is ``analog_mc``; future modules
plug in by adding their own ``dashboards/<module>/`` directory.

Run with::

    streamlit run dashboards/app.py
"""

from __future__ import annotations

import importlib
from pathlib import Path

import streamlit as st


def _discover_modules() -> dict[str, str]:
    """Map module name -> import path for every dashboards/<module>/app.py found."""
    root = Path(__file__).parent
    modules: dict[str, str] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if (sub / "app.py").exists():
            modules[sub.name] = f"dashboards.{sub.name}.app"
    return modules


def main() -> None:
    st.set_page_config(page_title="agent-driven-investment", layout="wide")
    modules = _discover_modules()
    if not modules:
        st.error("No dashboards/<module>/app.py found.")
        return

    if len(modules) == 1:
        # Single module — go straight to it, no module-picker noise.
        only = next(iter(modules.values()))
        importlib.import_module(only).main()
        return

    st.sidebar.title("modules")
    choice = st.sidebar.selectbox("module", options=list(modules.keys()))
    importlib.import_module(modules[choice]).main()


if __name__ == "__main__":
    main()
