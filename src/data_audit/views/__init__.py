"""Reusable projection and summary operations; UI keeps no analysis mathematics."""
import numpy as np
import pandas as pd
from ..contracts import ValidationError
from ..embeddings import PCAEmbedding, fit_transformer


def reference_projection(result, embedding=None):
    """Fit a separate all-data view. Never feeds OOF predictions or merges fold axes."""
    if result.data is None:
        raise ValidationError("Source snapshot unavailable for a new display projection")
    X = result.data.X
    if embedding is None:
        embedding = PCAEmbedding(n_components=min(2, *X.shape), standardize=True)
    transform = fit_transformer(embedding, X.to_numpy(dtype=float), X.columns)
    z = transform(X)
    coordinates = pd.DataFrame(z, index=X.index, columns=transform.feature_names)
    if coordinates.shape[1] == 1:
        coordinates["z1"] = 0.0
    from ..runner import describe
    return {"coordinates": coordinates, "fitted_scope": "all_data_display_only",
            "feature_names": list(coordinates.columns), "sample_ids": X.index.tolist(),
            "component": describe(embedding), "seed": getattr(embedding, "random_state", None),
            "note": "Display-only reference fit; distances/scores remain in the analysis space"}


def category_summary(result, method_id, decision=None):
    s = result.samples[result.samples.method_id == method_id].copy()
    if decision is not None:
        s = s.join(decision.flags, on="sample_id")
    agg = {"samples": ("sample_id", "size"), "supported": ("sufficient_support", "sum")}
    for flag in ["label_issue", "overlap"]:
        if flag in s:
            agg[flag] = (flag, "sum")
            agg[f"{flag}_decided"] = (flag, "count")
    summary = s.groupby("original_label", sort=False, dropna=False).agg(**agg).reset_index()
    for flag in ["label_issue", "overlap"]:
        if flag in summary:
            summary[f"{flag}_rate"] = summary[flag] / summary[f"{flag}_decided"].replace(0, np.nan)
    return summary


def compare_methods(result):
    """Compare ranks without averaging scores from different methods."""
    return result.samples.pivot(index="sample_id", columns="method_id", values="label_issue_score").rank(pct=True)


def demo_data(seed=42, n_samples=240):
    """Marked synthetic data: separated/overlapping regions and swapped labels."""
    rng = np.random.default_rng(seed)
    y = np.array(["A", "B", "C"])[np.arange(n_samples) % 3]
    X = rng.normal(size=(n_samples, 5))
    X[:, :2] += np.array([[-2, 0], [2, 0], [0, 2]])[np.arange(n_samples) % 3]
    y[::31] = np.roll(y, 1)[::31]
    out = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)],
                       index=[f"sample_{i:04d}" for i in range(n_samples)])
    out["label"] = y
    out["recording"] = np.arange(n_samples) // 6
    out.index.name = "sample_id"
    return out
