"""Snapshot orchestration, output validation, progress, and provenance."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import copy
import importlib.metadata
import platform
import traceback
import uuid
import numpy as np
import pandas as pd
from .contracts import AnalysisInput, CancellationToken, CancelledError, STANDARD_COLUMNS, ValidationError
from .distances import Euclidean
from .embeddings import Identity
from .result import AuditResult
from .validation import snapshot, fingerprint


def describe(component):
    if hasattr(component, "describe"):
        return component.describe()
    if is_dataclass(component):
        params = {
            f.name: describe(getattr(component, f.name))
            if hasattr(getattr(component, f.name), "fit")
            else getattr(component, f.name)
            for f in fields(component)
        }
    elif hasattr(component, "get_params"):
        params = component.get_params(deep=False)
    else:
        params = {}
    return {"component": type(component).__name__, "module": type(component).__module__, "parameters": params}


def _normalize(output, data, method_id):
    if not isinstance(output, Mapping) or not isinstance(output.get("samples"), pd.DataFrame):
        raise ValidationError("method must return a mapping with a samples DataFrame")
    output = copy.deepcopy(dict(output))
    samples = output["samples"]
    if "sample_id" not in samples and samples.index.name == "sample_id":
        samples = samples.reset_index()
    if not samples.columns.is_unique or "sample_id" not in samples:
        raise ValidationError("method.samples requires unique columns and sample_id")
    ids = pd.Index(samples.sample_id)
    if ids.hasnans or not ids.is_unique or len(ids) != len(data.X) or set(ids) != set(data.sample_ids):
        raise ValidationError("method.samples must contain exactly one row per requested sample_id")
    samples = samples.set_index("sample_id").loc[list(data.sample_ids)].reset_index()
    unsupported = []
    for col in STANDARD_COLUMNS:
        if col not in samples:
            samples[col] = pd.NA
            if col not in ("sample_id", "method_id", "original_label", "status"):
                unsupported.append(col)
    samples["method_id"] = method_id
    samples["original_label"] = data.y
    samples["status"] = samples.status.fillna("completed")
    for col in ["label_issue_score", "max_overlap_score"]:
        try:
            samples[col] = pd.to_numeric(samples[col]).astype(float)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{col}: must be numeric or missing") from exc
        valid = samples[col].dropna()
        if not np.isfinite(valid).all() or not valid.between(0, 1).all():
            raise ValidationError(f"{col}: scores must be in [0,1]")
    if not all(pd.isna(v) or isinstance(v, (bool, np.bool_)) for v in samples.sufficient_support):
        raise ValidationError("sufficient_support must be boolean or missing")
    samples["sufficient_support"] = samples.sufficient_support.astype("boolean")
    extra = {
        c: f"{method_id}.{c}"
        for c in samples
        if c not in STANDARD_COLUMNS + ["reason", "score_definition_id"]
    }
    samples = samples.rename(columns=extra)
    output["samples"] = samples
    for name, defaults in {
        "overlaps": ["sample_id", "class_a", "class_b", "pair_overlap"],
        "class_evidence": ["sample_id", "class_label", "class_score", "source"],
        "diagnostics": ["sample_id", "metric_name", "numeric_value", "text_value"],
    }.items():
        table = output.get(name, pd.DataFrame(columns=defaults))
        if not isinstance(table, pd.DataFrame):
            raise ValidationError(f"{name} must be a DataFrame")
        if not set(defaults[:2]).issubset(table.columns) or not table.columns.is_unique:
            raise ValidationError(f"{name}: required columns {defaults[:2]}")
        if "sample_id" in table and not set(table.sample_id).issubset(set(data.sample_ids)):
            raise ValidationError(f"{name}: unknown sample_id")
        for col in defaults:
            if col not in table:
                table[col] = pd.NA
        for score in ["pair_overlap", "class_score"]:
            if score in table:
                values = pd.to_numeric(table[score], errors="raise").dropna()
                if not np.isfinite(values).all() or not values.between(0, 1).all():
                    raise ValidationError(f"{name}.{score}: expected values in [0,1]")
        if name == "class_evidence" and not table.empty:
            if table.duplicated(["sample_id", "class_label"]).any():
                raise ValidationError("class_evidence: duplicate sample/category")
            sums = table.groupby("sample_id", sort=False).class_score.sum(min_count=1).dropna()
            if not np.allclose(sums, 1, atol=1e-6):
                raise ValidationError("class_evidence must sum to one per sample")
            for _, part in table.groupby("sample_id", sort=False):
                if set(part.class_label) != set(data.classes):
                    raise ValidationError("class_evidence must identify all original categories explicitly")
        if name == "diagnostics" and table.duplicated(["sample_id", "metric_name"]).any():
            raise ValidationError("diagnostics: duplicate sample/metric")
        if name == "overlaps" and not table.empty:
            if not set(table.class_a).union(table.class_b).issubset(set(data.classes)):
                raise ValidationError("overlaps: unknown category")
            if (table.class_a == table.class_b).any():
                raise ValidationError("overlaps: pair must contain distinct categories")
        table["method_id"] = method_id
        output[name] = table
    output.setdefault("run_info", {})["unsupported_fields"] = unsupported
    evidence = output.setdefault("evidence", {})
    if not isinstance(evidence, dict):
        raise ValidationError("evidence must map names to DataFrames")
    for name, table in evidence.items():
        if not isinstance(table, pd.DataFrame) or not table.columns.is_unique:
            raise ValidationError(f"evidence {name}: expected a DataFrame with unique columns")
        for col in ["sample_id", "query_sample_id", "neighbor_sample_id"]:
            if col in table and not set(table[col]).issubset(set(data.sample_ids)):
                raise ValidationError(f"evidence {name}: unknown {col}")
    for name, representation in output.setdefault("representations", {}).items():
        coords = representation.get("coordinates")
        if not isinstance(coords, pd.DataFrame) or not coords.index.equals(pd.Index(data.sample_ids)):
            raise ValidationError(f"representation {name}: coordinates must preserve sample IDs/order")
        if not coords.columns.is_unique or not np.isfinite(coords.to_numpy(dtype=float)).all():
            raise ValidationError(f"representation {name}: expected unique columns and finite coordinates")
        if "fitted_scope" not in representation:
            raise ValidationError(f"representation {name}: fitted_scope is required")
        representation.update(
            feature_names=list(coords.columns),
            sample_ids=data.sample_ids.tolist(),
            component=describe(data.embedding),
            seed=getattr(data.embedding, "random_state", None),
        )
    return output


def audit(
    X,
    y,
    *,
    groups=None,
    metadata=None,
    sample_ids=None,
    feature_names=None,
    embedding=None,
    distance=None,
    methods=None,
    continue_on_error=False,
    progress=None,
    cancellation_token=None,
):
    """Audit a private snapshot. Components and thresholds are ordinary Python objects."""
    from .methods import NeighborAudit

    data = snapshot(X, y, groups, metadata, sample_ids, feature_names)
    # Snapshot recipes too: external mutation must not alter an in-flight run.
    embedding = copy.deepcopy(embedding if embedding is not None else Identity())
    distance = copy.deepcopy(distance if distance is not None else Euclidean())
    methods = copy.deepcopy(list(methods if methods is not None else [NeighborAudit()]))
    names = [m.name for m in methods]
    if not methods or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValidationError("methods must have unique nonempty instance names")
    token = cancellation_token or CancellationToken()
    callback = progress or (lambda event: None)
    callback({"stage": "queued", "status": "queued"})
    arrays = [
        data.X.to_numpy(dtype=float, copy=True),
        data.y.to_numpy(copy=True),
        data.X.index.to_numpy(copy=True),
    ]
    group_array = None if data.groups is None else data.groups.to_numpy(copy=True)
    for a in arrays + ([group_array] if group_array is not None else []):
        a.setflags(write=False)
    aid = AnalysisInput(
        *arrays, tuple(data.X.columns), group_array, data.metadata, embedding, distance, token, callback
    )
    results, errors, cancelled = {}, {}, False
    for method in methods:
        try:
            token.check()
            callback({"stage": "running", "status": "running", "method_id": method.name})
            # Attribute every emitted stage to its method instance.
            scoped = AnalysisInput(
                aid.X,
                aid.y,
                aid.sample_ids,
                aid.feature_names,
                aid.groups,
                aid._metadata,
                aid.embedding,
                aid.distance,
                token,
                lambda event, name=method.name: callback({"method_id": name, **event}),
            )
            results[method.name] = _normalize(method.analyze(scoped), aid, method.name)
            callback({"stage": "method_completed", "method_id": method.name})
        except CancelledError:
            cancelled = True
            break
        except Exception as exc:
            if not continue_on_error:
                raise
            errors[method.name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            missing = pd.DataFrame(
                {"sample_id": aid.sample_ids, "status": "method_failed", "reason": str(exc)}
            )
            results[method.name] = _normalize({"samples": missing}, aid, method.name)
    if cancelled:
        for name in names:
            if name not in results:
                results[name] = _normalize(
                    {
                        "samples": pd.DataFrame(
                            {
                                "sample_id": aid.sample_ids,
                                "status": "cancelled",
                                "reason": "Cooperative cancellation",
                            }
                        )
                    },
                    aid,
                    name,
                )
    status = (
        "cancelled"
        if cancelled
        else ("partial" if errors and len(errors) < len(methods) else "failed" if errors else "completed")
    )
    tables = {}
    for key in ["samples", "overlaps", "class_evidence", "diagnostics"]:
        parts = [r[key] for r in results.values() if not r[key].empty]
        tables[key] = (
            pd.concat(parts, ignore_index=True) if parts else next(iter(results.values()))[key].copy()
        )
    info = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        feature_names=list(data.X.columns),
        sample_ids=aid.sample_ids.tolist(),
        classes=aid.classes,
        embedding=describe(embedding),
        distance=describe(distance),
        methods={
            m.name: {"recipe": describe(m), **results.get(m.name, {}).get("run_info", {})} for m in methods
        },
        errors=errors,
        versions={
            p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "data-audit"]
        },
        python=platform.python_version(),
        import_info=data.import_info,
        groups_policy="OOF group isolation; neighbor exclusion is independently configured",
        reproducibility="Same environment, seeds and data; numerical tolerance 1e-8 for CPU built-ins",
    )
    callback({"stage": "finished", "status": status})
    return AuditResult(
        str(uuid.uuid4()),
        fingerprint(data),
        status,
        **tables,
        data=data,
        run_info=info,
        representations={m: r.get("representations", {}) for m, r in results.items()},
        evidence={m: r["evidence"] for m, r in results.items()},
    )
