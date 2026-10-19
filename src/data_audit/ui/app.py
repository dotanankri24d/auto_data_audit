"""Local investigation UI. All data validation, projections, and math live in the library."""

from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
from queue import Empty
import pandas as pd
import plotly.express as px
import streamlit as st
from data_audit import (
    load_csv,
    load_excel,
    list_excel_sheets,
    map_columns,
    load_run,
    Identity,
    Standardize,
    PCAEmbedding,
    AutoencoderEmbedding,
    NeighborAudit,
    OOFAudit,
    ReviewPolicy,
    ValidationError,
)
from data_audit.plugins import discover_plugins
from data_audit.runner import describe
from data_audit.views import (
    demo_data,
    reference_projection,
    category_summary,
    compare_methods,
    metadata_summary,
    compare_decisions,
)
from data_audit.ui.forms import parameter_form
from data_audit.ui.jobs import AuditJob


def chart_selected(key):
    event = st.session_state[key]
    points = event.get("selection", {}).get("points", [])
    ids = list(dict.fromkeys(p["customdata"][0] for p in points if p.get("customdata")))
    # Plotly clears its transient selection when highlight traces change. Keep
    # stable sample IDs until a new selection or the explicit Clear action.
    if ids:
        st.session_state.selection = ids


def table_selected(key, visible_ids):
    rows = st.session_state[key].get("selection", {}).get("rows", [])
    ids = [visible_ids[i] for i in rows if 0 <= i < len(visible_ids)]
    if ids:
        st.session_state.selection = ids


def choose_component(kind, builtins, plugins, key):
    choices = {**builtins, **{f"Extension: {p.name}": p for p in plugins if p.kind == kind}}
    name = st.selectbox(kind.capitalize(), list(choices), key=key)
    component = choices[name]
    with st.expander(f"{name} parameters"):
        if hasattr(component, "bind"):
            st.caption(component.function.__doc__ or component.name)
            params = parameter_form(
                component.function,
                skip={"embedding": 1, "distance": 2, "method": 1, "policy": 1}[kind],
                key=key + name,
            )
            if kind == "method":
                params["name"] = st.text_input("Instance name", value=component.name, key=key + ".instance")
            return component.bind(**params)
        if component is PCAEmbedding:
            mode = st.radio("PCA component selection", ["Count", "Explained variance"], key=key + ".mode")
            n = (
                st.number_input("n_components", min_value=1, value=2, step=1, key=key + ".n")
                if mode == "Count"
                else st.number_input(
                    "variance_fraction", min_value=0.01, max_value=0.999, value=0.95, key=key + ".fraction"
                )
            )
            params = parameter_form(component, key=key + name, supplied={"n_components": n})
        else:
            params = parameter_form(
                component, key=key + name, supplied={"estimator": None} if component is OOFAudit else None
            )
        return component(**params)


def upload_data():
    st.subheader("1 · Load data")
    if st.button("Use synthetic demonstration dataset"):
        st.session_state.source = demo_data()
        st.session_state.source_kind = "Synthetic demonstration"
        st.session_state.source_bytes = None
    uploaded = st.file_uploader("Source data · CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        raw = uploaded.getvalue()
        header = st.number_input("Header row (zero based)", min_value=0, value=0, step=1)
        id_column = st.text_input("ID column (optional; preserves textual IDs)", value="") or None
        types_text = st.text_input("Column type overrides (JSON object)", value="{}")
        try:
            dtype = json.loads(types_text)
            if not isinstance(dtype, dict):
                raise ValidationError("Column type overrides must be a JSON object")
            options = dict(header=header, id_column=id_column, dtype=dtype)
            if uploaded.name.lower().endswith(".csv"):
                options["delimiter"] = st.text_input("CSV delimiter", value=",")
                options["encoding"] = st.selectbox(
                    "CSV encoding", ["utf-8-sig", "utf-8", "cp1255", "latin-1"]
                )
                reader = load_csv
            else:
                sheets = list_excel_sheets(io.BytesIO(raw))
                sheet = st.selectbox(
                    "Excel sheet (choose explicitly)", [None] + sheets if len(sheets) > 1 else sheets
                )
                if sheet is None:
                    st.info("Choose one Excel sheet to preview and load.")
                    return
                options["sheet_name"] = sheet
                reader = load_excel
            preview = reader(io.BytesIO(raw), **options)
            st.dataframe(preview.head(25), width="stretch")
            st.caption("Inferred types: " + ", ".join(f"{c}: {t}" for c, t in preview.dtypes.items()))
            st.caption(
                "Excel IDs with meaningful leading zeros must be stored as text cells. Formulas use cached values only."
            )
            if st.button("Load source data", type="primary"):
                st.session_state.source = preview
                st.session_state.source_kind = uploaded.name
                st.session_state.source_bytes = raw
        except Exception as exc:
            st.error(str(exc))
    with st.expander("Open an existing .audit.zip run"):
        bundle = st.file_uploader("Saved run", type=["zip"], key="bundle")
        if bundle is not None and st.button("Open saved run"):
            try:
                st.session_state.result = load_run(io.BytesIO(bundle.getvalue()))
                st.session_state.pop("selection", None)
                st.success("Restored results without fitting or loading extension code.")
            except Exception as exc:
                st.error(str(exc))


def extension_controls():
    plugins = st.session_state.get("plugins", [])
    with st.sidebar:
        st.header("Local extensions")
        path = st.text_input("Trusted extension directory", value=os.environ.get("DATA_AUDIT_EXTENSIONS", ""))
        st.caption(
            "Loading executes local Python code. Choose a directory you trust; uploaded data is never code."
        )
        if st.button("Load / refresh extensions"):
            plugins, errors = discover_plugins(path)
            st.session_state.plugins = plugins
            for file, error in errors.items():
                st.error(f"{file}: {error}")
            st.success(f"{len(plugins)} components discovered")
    return plugins


def configure(plugins):
    source = st.session_state.get("source")
    if source is None:
        return
    st.subheader("2 · Map columns and configure")
    st.caption(
        f"{st.session_state.get('source_kind', 'Selection')} · {len(source):,} samples · {len(source.columns)} columns"
    )
    columns = list(source.columns)
    label = st.selectbox("Label column", columns, index=columns.index("label") if "label" in columns else 0)
    groups = st.selectbox("Group column (optional)", [None] + [c for c in columns if c != label])
    meta = st.multiselect("Metadata columns", [c for c in columns if c not in [label, groups]])
    allowed = [c for c in columns if c not in [label, groups] + meta]
    features = st.multiselect(
        "Numeric feature columns",
        allowed,
        default=[c for c in allowed if pd.api.types.is_numeric_dtype(source[c])],
    )
    try:
        data = map_columns(source, label=label, features=features, groups=groups, metadata=meta)
        st.dataframe(data.y.value_counts(sort=False).rename("samples"), width="stretch")
        embedding = choose_component(
            "embedding",
            {
                "Identity": Identity,
                "Standardize": Standardize,
                "PCA": PCAEmbedding,
                "Autoencoder (optional)": AutoencoderEmbedding,
            },
            plugins,
            "embedding",
        )
        from data_audit.distances import Euclidean

        distance = choose_component("distance", {"Euclidean": Euclidean}, plugins, "distance")
        method_options = ["Neighbors", "OOF"] + [
            f"Extension: {p.name}" for p in plugins if p.kind == "method"
        ]
        selected = st.multiselect("Analysis methods", method_options, default=["Neighbors"])
        methods = []
        for i, name in enumerate(selected):
            if name.startswith("Extension:"):
                candidates = {
                    name: next(p for p in plugins if p.kind == "method" and name == f"Extension: {p.name}")
                }
            else:
                candidates = {name: NeighborAudit if name == "Neighbors" else OOFAudit}
            methods.append(choose_component("method", candidates, [], f"method-{i}"))
        recipe = {
            "embedding": describe(embedding),
            "distance": describe(distance),
            "methods": [describe(m) for m in methods],
            "features": features,
            "label": label,
            "groups": groups,
            "metadata": meta,
        }
        recipe_key = hashlib.sha256(repr(recipe).encode()).hexdigest()
        if "result_recipe" in st.session_state and recipe_key != st.session_state.result_recipe:
            st.warning(
                "Configuration changed. Run again to analyze it. Existing results retain their saved configuration."
            )
        st.caption("Run scope: all mapped samples. Features: " + ", ".join(map(str, features)))
        job = st.session_state.get("job")
        running = job is not None and not job.future.done()
        if st.button("Run audit", type="primary", disabled=running or not methods):
            st.session_state.job = AuditJob.start(
                data, embedding=embedding, distance=distance, methods=methods, continue_on_error=True
            )
            st.session_state.pending_recipe = recipe_key
        if running and st.button("Cancel audit"):
            job.token.cancel()
            st.info("Cancellation requested; waiting for the current unit of work to finish.")
    except Exception as exc:
        st.error(str(exc))


@st.fragment(run_every=0.5)
def poll_job():
    job = st.session_state.get("job")
    if job is None:
        return
    events = []
    while True:
        try:
            events.append(job.events.get_nowait())
        except Empty:
            break
    if events:
        st.session_state.last_progress = events[-1]
    st.caption("Progress: " + str(st.session_state.get("last_progress", {})))
    if job.future.done():
        try:
            result = job.future.result()
            st.session_state.result = result
            st.session_state.result_recipe = st.session_state.pending_recipe
            st.session_state.pop("selection", None)
        except Exception as exc:
            st.session_state.run_error = str(exc)
        st.session_state.job = None
        st.rerun()
    else:
        st.info("Analysis running. Forms and existing results remain available.")


def investigation(plugins):
    result = st.session_state.get("result")
    if result is None:
        return
    st.subheader("3 · Investigate")
    st.caption(f"Run {result.run_id} · {result.status} · dataset {result.dataset_id[:12]}")
    st.info(
        "Scores rank evidence relative to selected features, representation, method and scale. They are not probabilities of incorrect labels or proof of inseparability."
    )
    if result.status != "completed":
        st.warning(f"Run status: {result.status}. Failed/cancelled rows have no fabricated scores.")
    if result.run_info.get("errors"):
        with st.expander("Component errors and technical tracebacks"):
            st.json(result.run_info["errors"])
    if result.run_info.get("restored_view_only"):
        st.caption(result.run_info["rerun_note"])
    method = st.selectbox("Result method", result.samples.method_id.unique())
    with st.expander("Decision policy · initial presets require calibration", expanded=True):
        policy = choose_component("policy", {"ReviewPolicy": ReviewPolicy}, plugins, "review-policy")
        # Replace previous UI policy for this method to avoid growing history on every rerun.
        signature = repr((method, describe(policy)))
        cache = st.session_state.get("decision_cache")
        if cache is None or cache[0] != result.run_id or cache[1] != signature:
            decision = result.apply_policy(method, policy)
            st.session_state.decision_cache = (result.run_id, signature, decision)
        else:
            decision = cache[2]
    base = result.samples[result.samples.method_id == method].copy()
    diag = result.diagnostics
    if not diag.empty:
        pivot = diag[diag.method_id == method].pivot(
            index="sample_id", columns="metric_name", values="numeric_value"
        )
        base = base.join(pivot, on="sample_id")
    base = base.join(decision.flags, on="sample_id")
    if result.data is not None and result.data.metadata is not None:
        base = base.join(result.data.metadata.add_prefix("metadata."), on="sample_id")
    search = st.text_input("Search sample ID")
    labels = st.multiselect("Original labels", list(pd.unique(base.original_label)))
    alternatives = st.multiselect("Alternative labels", list(pd.unique(base.alternative_label.dropna())))
    support = st.selectbox("Support filter", ["All", "Sufficient", "Insufficient / unknown"])
    filtered = base
    if search:
        filtered = filtered[filtered.sample_id.astype(str).str.contains(search, regex=False)]
    if labels:
        filtered = filtered[filtered.original_label.isin(labels)]
    if alternatives:
        filtered = filtered[filtered.alternative_label.isin(alternatives)]
    if support != "All":
        valid = filtered.sufficient_support.fillna(False)
        filtered = filtered[valid if support == "Sufficient" else ~valid]
    meta_cols = [c for c in base if c.startswith("metadata.")]
    for c in meta_cols:
        values = st.multiselect(f"Filter {c}", list(pd.unique(base[c].dropna())))
        if values:
            filtered = filtered[filtered[c].isin(values)]
    st.caption(
        "Display filters leave analysis reference population and scores unchanged. Missing flags mean abstention."
    )
    selection = st.session_state.get("selection", [])
    mode = st.radio("Graph/table selection action", ["Highlight", "Filter to selection"], horizontal=True)
    if st.button("Clear selection"):
        st.session_state.selection = []
        selection = []
    shown = (
        filtered[filtered.sample_id.isin(selection)]
        if mode == "Filter to selection" and selection
        else filtered
    )
    scatter(result, shown, method, selection)
    shown = shown.copy()
    shown.insert(0, "selected", shown.sample_id.isin(selection))
    display_cols = st.multiselect("Table columns", list(shown.columns), default=list(shown.columns[:13]))
    table_key = "findings-" + result.run_id + method
    st.dataframe(
        shown[display_cols],
        width="stretch",
        on_select=lambda: table_selected(table_key, tuple(shown.sample_id)),
        selection_mode="multi-row",
        key=table_key,
    )
    st.caption(f"Selection: {len(selection):,} samples")
    choices = shown.sample_id.tolist()
    if choices:
        sid = st.selectbox(
            "Inspect sample",
            choices,
            index=choices.index(selection[0]) if selection and selection[0] in choices else 0,
        )
        sample_details(result, base, sid, method)
    if selection and result.data is not None and st.button("Analyze selection as a new dataset"):
        d = result.data
        ids = [i for i in d.X.index if i in selection]
        source = d.X.loc[ids].copy()
        # Reserved UI role name chosen to avoid overwriting a feature column.
        name = "label"
        while name in source:
            name = "_" + name
        source[name] = d.y.loc[ids]
        st.session_state.source = source
        st.session_state.source_kind = "Explicit selection from " + result.run_id
        st.rerun()
    with st.expander("Summaries and method comparison"):
        st.dataframe(category_summary(result, method, decision), width="stretch")
        st.dataframe(
            result.overlaps[result.overlaps.method_id == method]
            .groupby(["class_a", "class_b"], sort=False)
            .pair_overlap.mean()
            .reset_index(),
            width="stretch",
        )
        st.caption("Method comparison: percentile ranks and latest policy flags, without score averaging.")
        st.dataframe(compare_methods(result), width="stretch")
        st.dataframe(compare_decisions(result), width="stretch")
        if result.data is not None and result.data.metadata is not None:
            summary_column = st.selectbox("Metadata summary column", list(result.data.metadata.columns))
            st.dataframe(metadata_summary(result, method, summary_column, decision), width="stretch")
    with st.expander("Saved configuration and provenance"):
        st.json(json.loads(json.dumps(result.run_info, default=str)))
    downloads(result, shown, selection, method)


def scatter(result, shown, method, selection):
    st.subheader("Data space")
    options = ["Saved embedding"] if result.representations.get(method) else []
    if result.data is not None:
        options += ["Original features", "Display PCA"]
    if not options:
        st.info("Source data and representations omitted; graph unavailable.")
        return
    space = st.selectbox("Display space", options)
    try:
        if space == "Original features":
            coords = result.data.X
        elif space == "Saved embedding":
            reps = result.representations[method]
            rep_name = st.selectbox("Representation", list(reps))
            rep = reps[rep_name]
            st.caption("Fit scope: " + rep["fitted_scope"])
            coords = rep["coordinates"]
        else:
            key = "display_projection_" + result.run_id
            if key not in st.session_state:
                existing = result.representations.get("__display__", {}).get("reference")
                st.session_state[key] = existing if existing is not None else reference_projection(result)
                result.representations.setdefault("__display__", {})["reference"] = st.session_state[key]
            rep = st.session_state[key]
            coords = rep["coordinates"]
            st.caption(rep["note"])
        names = list(coords.columns)
        x = st.selectbox("X axis", names)
        y = st.selectbox("Y axis", names, index=min(1, len(names) - 1))
        colors = [
            c
            for c in shown
            if c in ["original_label", "label_issue_score", "max_overlap_score", "sufficient_support"]
            or c.startswith(method + ".")
        ]
        color = st.selectbox("Color", colors)
        graph = shown.merge(
            coords[[x, y] if x != y else [x]].rename(columns={x: "display_x", y: "display_y"}),
            left_on="sample_id",
            right_index=True,
            how="left",
        )
        if x == y:
            graph["display_x"] = coords[x].reindex(graph.sample_id).to_numpy()
            graph["display_y"] = graph["display_x"]
        if len(graph) > 20000:
            graph = graph.sample(20000, random_state=42)
            st.caption(
                f"Displaying a deterministic random sample of 20,000 / {len(shown):,} points (seed 42). Export retains full scope."
            )
        if color in ["sufficient_support", "original_label"]:
            graph[color] = graph[color].astype(str)
        fig = px.scatter(
            graph,
            x="display_x",
            y="display_y",
            color=color,
            custom_data=["sample_id"],
            hover_data=[c for c in ["sample_id", "label_issue_score", "max_overlap_score"] if c in graph],
            labels={"display_x": str(x), "display_y": str(y)},
            template="plotly_white",
        )
        if selection:
            # Highlight selected samples and known neighbors by stable IDs.
            neighbors = set(selection)
            for sid in selection:
                table = result.sample_evidence(sid, method).get("neighbors")
                if table is not None:
                    neighbors.update(table.neighbor_sample_id)
            fig.update_traces(marker={"opacity": 0.4})
            selected = graph[graph.sample_id.isin(neighbors)]
            if not selected.empty:
                import plotly.graph_objects as go

                fig.add_trace(
                    go.Scatter(
                        x=selected.display_x,
                        y=selected.display_y,
                        mode="markers",
                        marker={"size": 14, "symbol": "circle-open", "color": "black"},
                        customdata=selected[["sample_id"]].to_numpy(),
                        name="Selected / neighbors",
                    )
                )
        chart_key = "scatter-" + result.run_id + method + space
        st.plotly_chart(
            fig,
            width="stretch",
            on_select=lambda: chart_selected(chart_key),
            selection_mode=["points", "box", "lasso"],
            key=chart_key,
        )
        st.caption(
            "Distances and scores use the saved analysis space, including when this view is projected to 2D."
        )
    except Exception as exc:
        st.warning(f"Projection unavailable: {exc}")


def sample_details(result, base, sid, method):
    with st.expander("Sample evidence and review", expanded=True):
        st.dataframe(base[base.sample_id == sid], width="stretch")
        if result.data is not None:
            st.dataframe(result.data.X.loc[[sid]], width="stretch")
        st.dataframe(
            result.class_evidence[
                (result.class_evidence.method_id == method) & (result.class_evidence.sample_id == sid)
            ],
            width="stretch",
        )
        for name, table in result.sample_evidence(sid, method).items():
            st.caption(name)
            st.dataframe(table, width="stretch")
        st.caption("These tables describe the computed evidence; they do not establish a causal explanation.")
        status = st.selectbox("Human review", ["unreviewed", "confirmed", "rejected"])
        label = st.text_input("Proposed label")
        note = st.text_area("Review note")
        if st.button("Save review"):
            row = dict(sample_id=sid, method_id=method, review_status=status, proposed_label=label, note=note)
            keep = ~((result.reviews.sample_id == sid) & (result.reviews.method_id == method))
            result.reviews = pd.concat([result.reviews[keep], pd.DataFrame([row])], ignore_index=True)
            st.success("Saved separately from the input snapshot.")


def downloads(result, shown, selection, method):
    st.subheader("4 · Export and save")
    scope = st.selectbox("CSV export scope", ["All results", "Filtered rows", "Selected samples"])
    ids = (
        None
        if scope == "All results"
        else (shown.sample_id.tolist() if scope == "Filtered rows" else selection)
    )
    method_id = method if scope != "All results" else None
    count = (
        len(result.samples)
        if ids is None
        else len(result.samples[(result.samples.method_id == method) & result.samples.sample_id.isin(ids)])
    )
    st.caption(f"{scope}: {count:,} rows")
    features = st.checkbox("Include original features", disabled=result.data is None)
    metadata = st.checkbox("Include metadata", disabled=result.data is None)
    st.download_button(
        "Download findings CSV",
        lambda: result.export_csv(
            sample_ids=ids,
            method_id=method_id,
            include_decisions=True,
            include_features=features,
            include_metadata=metadata,
        ),
        file_name="findings.csv",
        mime="text/csv",
        on_click="ignore",
    )
    st.download_button(
        "Download all tables (CSV ZIP)", result.export_tables, file_name="audit-tables.zip", on_click="ignore"
    )
    include = st.checkbox(
        "Include source snapshot in saved run", value=result.data is not None, disabled=result.data is None
    )
    if not include:
        st.info(
            "Without source data, original feature details and new display projections will be unavailable on reopening."
        )
    st.download_button(
        "Save complete run",
        lambda: result.save(include_data=include),
        file_name="run.audit.zip",
        mime="application/zip",
        on_click="ignore",
    )


def main():
    st.set_page_config(page_title="Data Audit", page_icon="◈", layout="wide")
    st.title("Data Audit")
    st.caption("Inspect label quality · understand class overlap · follow the evidence")
    if os.environ.get("DATA_AUDIT_RESULT") and "result" not in st.session_state:
        st.session_state.result = load_run(Path(os.environ["DATA_AUDIT_RESULT"]))
    plugins = extension_controls()
    upload_data()
    configure(plugins)
    poll_job()
    if st.session_state.get("run_error"):
        st.error(st.session_state.run_error)
    try:
        investigation(plugins)
    except Exception as exc:
        st.error(f"Investigation: {exc}")


if __name__ == "__main__":
    main()
