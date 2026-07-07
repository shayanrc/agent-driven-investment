"""Streamlit view: load, edit, validate, and save analog_mc YAML configs."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import streamlit as st
import yaml

from analog_mc import Config

from dashboards.analog_mc.views._shared import list_configs as _list_configs


def _parse_tuple_input(s: str, item_type: type) -> tuple:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(item_type(p) for p in parts)


def render(configs_root: Path = Path("configs/analog_mc")) -> None:
    st.title("analog_mc — config editor")

    configs_root.mkdir(parents=True, exist_ok=True)
    configs = _list_configs(configs_root)

    options = ["<new>"] + [c.name for c in configs]
    choice = st.sidebar.selectbox("config", options=options)

    if choice == "<new>":
        cfg = Config()
        load_path: Path | None = None
    else:
        load_path = configs_root / choice
        try:
            cfg = Config.from_yaml(load_path)
        except Exception as exc:
            st.error(f"Failed to parse {load_path.name}: {exc}")
            return

    st.caption(
        "Edit any field, then click **Validate & save**. Invariant violations "
        "(forecast_horizon ≠ n_blocks × block_length, etc.) block the save."
    )

    # Build form: one widget per Config field.
    values: dict = {}
    with st.form("config_form"):
        cols = st.columns(2)
        col_idx = 0
        for f in fields(Config):
            current = getattr(cfg, f.name)
            label = f.name
            with cols[col_idx % 2]:
                if isinstance(current, bool):
                    values[f.name] = st.checkbox(label, value=current)
                elif isinstance(current, int) and not isinstance(current, bool):
                    values[f.name] = st.number_input(label, value=int(current), step=1)
                elif isinstance(current, float):
                    values[f.name] = st.number_input(label, value=float(current), format="%.6f")
                elif isinstance(current, str):
                    values[f.name] = st.text_input(label, value=current)
                elif isinstance(current, tuple):
                    item_type = type(current[0]) if current else float
                    values[f.name] = (",".join(str(v) for v in current), item_type)
                    text = st.text_input(
                        f"{label} (comma-separated)",
                        value=",".join(str(v) for v in current),
                    )
                    values[f.name] = _parse_tuple_input(text, item_type)
                else:
                    st.text(f"{label}: <unsupported type {type(current).__name__}>")
            col_idx += 1

        new_name = st.text_input(
            "save as (filename, .yaml will be appended)",
            value=load_path.stem if load_path else "new_config",
        )
        submitted = st.form_submit_button("Validate & save")

    if not submitted:
        return

    try:
        new_cfg = Config(**values)
    except (TypeError, ValueError) as exc:
        st.error(f"Invalid config: {exc}")
        return

    if not new_name.endswith(".yaml"):
        new_name += ".yaml"
    out_path = configs_root / new_name
    new_cfg.to_yaml(out_path)
    st.success(f"Saved to `{out_path}`.")
    with st.expander("YAML preview"):
        st.code(yaml.safe_dump(new_cfg.to_dict(), sort_keys=False), language="yaml")
