"""Streamlit entry point for the analog_mc module.

Run with::

    streamlit run dashboards/analog_mc/app.py

A thin dispatcher over three views: config_editor, run_experiment, diagnostics.
All real work lives in ``src/analog_mc/``; this file only routes UI.
"""

from __future__ import annotations

import streamlit as st

from dashboards.analog_mc.views import config_editor, diagnostics, run_experiment


VIEWS = {
    "Diagnostics": diagnostics.render,
    "Run experiment": run_experiment.render,
    "Config editor": config_editor.render,
}


def main() -> None:
    st.set_page_config(page_title="analog_mc", layout="wide")
    st.sidebar.title("analog_mc")
    choice = st.sidebar.radio("view", options=list(VIEWS.keys()))
    VIEWS[choice]()


# Streamlit runs the file as __main__. The global launcher imports this
# module and calls main() explicitly, so the else-branch is intentionally
# absent — it must NOT auto-run on plain import.
if __name__ == "__main__":
    main()
