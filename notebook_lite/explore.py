"""Notebook helpers. All results are ordinary pandas tables or Plotly figures."""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from representations import PCAEmbedding, fit_transformer


def load_csv(path, *, id_column=None, **kwargs):
    """Read a CSV, preserving textual IDs (including leading zeroes)."""
    if id_column is not None:
        kwargs["dtype"] = {**kwargs.get("dtype", {}), id_column: "string"}
    frame = pd.read_csv(path, **kwargs)
    if id_column is not None:
        frame = frame.set_index(id_column, verify_integrity=True)
    if frame.index.hasnans:
        raise ValueError("Sample IDs cannot be missing")
    return frame


def review_flags(
    samples, issue_threshold=0.5, overlap_threshold=0.6, require_support=True, min_alternative_agreement=None
):
    """Thresholds only; missing flags mean abstention. Does not refit."""
    for threshold in (issue_threshold, overlap_threshold, min_alternative_agreement):
        if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("Thresholds must lie in [0, 1]")
    valid = pd.Series(True, index=samples.index)
    if require_support:
        valid &= samples.sufficient_support.fillna(False)
    if min_alternative_agreement is not None:
        valid &= samples.get("alternative_agreement", pd.Series(np.nan, index=samples.index)).ge(
            min_alternative_agreement
        )
    flags = pd.DataFrame(index=samples.index)
    for name, score, threshold in [
        ("label_issue", "label_issue_score", issue_threshold),
        ("overlap", "max_overlap_score", overlap_threshold),
    ]:
        flags[name] = samples[score].ge(threshold).astype("boolean").where(valid & samples[score].notna())
    return flags


def project(X, embedding=None):
    """Display projection fitted on all rows; never used as OOF evidence."""
    embedding = embedding or PCAEmbedding(n_components=min(2, *X.shape))
    transform = fit_transformer(embedding, X.to_numpy(dtype=float), X.columns)
    return pd.DataFrame(transform(X), index=X.index, columns=transform.feature_names)


def scatter(coordinates, samples, *, color="original_label", axes=None, selected=()):
    """Interactive hover/zoom/lasso. Use selected IDs for explicit highlights."""
    axes = axes or list(coordinates.columns[:2])
    x, y = axes[0], axes[-1]
    graph = coordinates.loc[samples.index].copy()
    graph["_x"], graph["_y"] = graph[x], graph[y]
    graph["_id"] = graph.index.astype(str)
    graph["_color"] = samples[color]
    if color in ("original_label", "alternative_label", "sufficient_support"):
        graph["_color"] = graph["_color"].astype(str)
    fig = px.scatter(
        graph,
        x="_x",
        y="_y",
        color="_color",
        hover_name="_id",
        labels={"_x": x, "_y": y, "_color": color},
        template="plotly_white",
    )
    highlighted = graph.loc[graph.index.isin(selected)]
    if not highlighted.empty:
        import plotly.graph_objects as go

        fig.add_trace(
            go.Scatter(
                x=highlighted["_x"],
                y=highlighted["_y"],
                mode="markers",
                marker=dict(size=15, symbol="circle-open", color="black"),
                name="Selected",
            )
        )
    return fig


def inspect_sample(result, sample_id):
    """Return score, class evidence, neighbors and repeated OOF predictions."""
    detail = {"sample": result["samples"].loc[[sample_id]]}
    for name in ("class_evidence", "neighbors", "oof", "overlaps"):
        table = result.get(name)
        if table is not None:
            key = "query_sample_id" if name == "neighbors" else "sample_id"
            detail[name] = table.loc[table[key].eq(sample_id)].copy()
    return detail


def summary(samples, by="original_label"):
    grouped = samples.groupby(by, dropna=False, sort=False)
    table = grouped.agg(
        count=("label_issue_score", "size"),
        supported=("sufficient_support", "sum"),
        mean_issue=("label_issue_score", "mean"),
        mean_overlap=("max_overlap_score", "mean"),
    )
    for flag in ("label_issue", "overlap"):
        if flag in samples:
            table[flag + "_count"] = grouped[flag].sum(min_count=1)
            table[flag + "_abstentions"] = grouped[flag].agg(lambda values: values.isna().sum())
    return table


def compare_methods(results):
    """Compare percentile ranks without averaging scores between methods."""
    return pd.concat(
        {
            name: result["samples"][["label_issue_score", "max_overlap_score"]].rank(pct=True)
            for name, result in results.items()
        },
        axis=1,
    )


def export_tables(result, directory):
    """Explicit CSV export, including split provenance; no custom bundle format."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(
                directory / f"{name}.csv", index=name in ("samples", "coordinates"), encoding="utf-8-sig"
            )
    if "run_info" in result:
        import json

        (directory / "run_info.json").write_text(json.dumps(result["run_info"], indent=2, default=str))
