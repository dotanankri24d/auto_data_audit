"""Strict role mapping and snapshot creation. Never silently align rows."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from .contracts import DataSnapshot, ValidationError


def validate_ids(index, field="sample_id"):
    if index.hasnans or not index.is_unique:
        raise ValidationError(f"{field}: IDs must be unique and non-missing; offending IDs: "
                              f"{list(index[index.duplicated(keep=False)])[:5]}")
    if any(not isinstance(v, (str, int, float, np.integer, np.floating)) or isinstance(v, bool) for v in index):
        raise ValidationError(f"{field}: IDs must be strings or numbers")


def snapshot(X, y, groups=None, metadata=None, sample_ids=None, feature_names=None):
    if isinstance(X, pd.DataFrame):
        if sample_ids is not None or feature_names is not None:
            raise ValidationError("sample_ids/feature_names apply only to NumPy X")
        frame = X.copy(deep=True)
    else:
        a = np.asarray(X)
        if a.ndim != 2:
            raise ValidationError("X must be a two-dimensional matrix")
        frame = pd.DataFrame(a, index=sample_ids, columns=feature_names)
        if feature_names is None:
            frame.columns = [f"feature_{i}" for i in range(a.shape[1])]
    if frame.empty:
        raise ValidationError("X must have samples and features")
    validate_ids(frame.index)
    if not frame.columns.is_unique or frame.columns.hasnans:
        raise ValidationError("feature_names must be unique and non-missing")
    for col in frame:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValidationError(f"X column {col!r} must be numeric")
    a = frame.to_numpy(dtype=float)
    if not np.isfinite(a).all():
        rows, cols = np.where(~np.isfinite(a))
        raise ValidationError(f"X must be finite; sample {frame.index[rows[0]]!r}, column {frame.columns[cols[0]]!r}")

    def vector(v, name):
        if v is None:
            return None
        if isinstance(v, pd.Series):
            if not v.index.equals(frame.index):
                raise ValidationError(f"{name}: index must match X in the same order")
            out = v.copy(deep=True)
        else:
            v = np.asarray(v)
            if v.ndim != 1 or len(v) != len(frame):
                raise ValidationError(f"{name}: expected a vector with {len(frame)} rows")
            out = pd.Series(v, index=frame.index, name=name)
        if out.isna().any():
            raise ValidationError(f"{name}: missing values at samples {out.index[out.isna()].tolist()[:5]}")
        return out

    labels = vector(y, "y")
    if labels is None:
        raise ValidationError("y is required")
    try:
        list(pd.unique(labels))
    except TypeError as exc:
        raise ValidationError("y: labels must be scalar hashable values") from exc
    group = vector(groups, "groups")
    if metadata is not None:
        if not isinstance(metadata, pd.DataFrame) or not metadata.index.equals(frame.index):
            raise ValidationError("metadata: DataFrame index must match X in the same order")
        if not metadata.columns.is_unique:
            raise ValidationError("metadata: duplicate column names")
        metadata = metadata.copy(deep=True)
    return DataSnapshot(frame, labels, group, metadata, dict(frame.attrs.get("import_info", {})))


def fingerprint(data):
    h = hashlib.sha256()
    for obj in [data.X, data.y, data.groups, data.metadata]:
        if obj is not None:
            h.update(pd.util.hash_pandas_object(obj, index=True).values.tobytes())
            h.update(repr(list(obj.dtypes) if isinstance(obj, pd.DataFrame) else obj.dtype).encode())
            if isinstance(obj, pd.DataFrame):
                h.update(repr(list(obj.columns)).encode())
    return h.hexdigest()
