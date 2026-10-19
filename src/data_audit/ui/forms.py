"""Widgets derived from the same callable signatures used by binding."""

import streamlit as st
from ..plugins import parameter_schema


def parameter_form(function, *, key, skip=0, supplied=None):
    values = dict(supplied or {})
    for p in parameter_schema(function, skip, supplied):
        widget_key = f"{key}.{p.name}"
        disabled = False
        help_text = None
        if getattr(function, "__name__", "") == "NeighborAudit":
            mode = values.get("radius_mode", "auto")
            if p.name in ("max_radius", "radius_quantile"):
                active_mode = "fixed" if p.name == "max_radius" else "auto"
                disabled = mode != active_mode
                help_text = f"Used only with radius_mode={active_mode}"
        if p.nullable:
            enabled = st.checkbox(
                f"Set {p.name}",
                value=p.default is not None,
                key=widget_key + ".set",
                disabled=disabled,
                help=help_text,
            )
            if not enabled:
                values[p.name] = None
                continue
        value = p.default
        if p.choices:
            values[p.name] = st.selectbox(
                p.name, p.choices, index=p.choices.index(value) if value in p.choices else 0, key=widget_key
            )
        elif p.kind is bool:
            values[p.name] = st.checkbox(p.name, value=bool(value), key=widget_key)
        elif p.kind in (int, float):
            initial = value if value is not None else (0 if p.kind is int else 0.0)
            values[p.name] = st.number_input(
                p.name,
                value=p.kind(initial),
                step=1 if p.kind is int else 0.05,
                key=widget_key,
                disabled=disabled,
                help=help_text,
            )
        else:
            values[p.name] = st.text_input(p.name, value="" if value is None else str(value), key=widget_key)
        if p.required and p.kind is str and not values[p.name]:
            st.caption(f"{p.name} is required")
    return values
