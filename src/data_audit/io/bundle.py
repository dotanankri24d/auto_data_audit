"""Versioned JSON + ZIP persistence. No pickle, code loading, or archive extraction."""

from __future__ import annotations
from datetime import date, datetime
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from ..contracts import SCHEMA_VERSION, DataSnapshot, ValidationError
from ..result import AuditResult, DecisionResult

MAX_BUNDLE_BYTES = 512 * 1024 * 1024


def _scalar(value):
    if value is None:
        return {"t": "none"}
    if value is pd.NA:
        return {"t": "na"}
    if value is pd.NaT:
        return {"t": "nat"}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return {"t": "nan"}
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return {"t": "timestamp", "v": value.isoformat()}
    if isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValidationError("Cannot save infinite scalar values")
        return {"t": type(value).__name__, "v": value}
    raise ValidationError(f"Unsupported saved scalar type {type(value).__name__}")


def _unscalar(value):
    t = value["t"]
    if t == "none":
        return None
    if t == "na":
        return pd.NA
    if t == "nat":
        return pd.NaT
    if t == "nan":
        return np.nan
    if t == "timestamp":
        return pd.Timestamp(value["v"])
    if t not in ("bool", "int", "float", "str"):
        raise ValidationError(f"Unknown scalar tag {t!r}")
    v = value["v"]
    if type(v).__name__ != t:
        raise ValidationError(f"Invalid scalar value for {t}")
    return v


def _frame(frame):
    # Column-wise encoding retains nullable and categorical dtypes and avoids row coercion.
    columns = []
    for i in range(len(frame.columns)):
        s = frame.iloc[:, i]
        descriptor = {
            "name": _scalar(frame.columns[i]),
            "dtype": str(s.dtype),
            "values": [_scalar(v) for v in s],
        }
        if isinstance(s.dtype, pd.CategoricalDtype):
            descriptor.update(categories=[_scalar(v) for v in s.cat.categories], ordered=s.cat.ordered)
        if isinstance(s.dtype, pd.StringDtype):
            descriptor["storage"] = s.dtype.storage
        columns.append(descriptor)
    return {
        "columns": columns,
        "columns_name": _scalar(frame.columns.name),
        "index": [_scalar(v) for v in frame.index],
        "index_name": _scalar(frame.index.name),
        "index_dtype": str(frame.index.dtype),
    }


def _unframe(obj):
    index = pd.Index([_unscalar(v) for v in obj["index"]], name=_unscalar(obj["index_name"]))
    if index.name is pd.NA:
        index.name = None
    try:
        index = index.astype(obj["index_dtype"])
        series = []
        for c in obj["columns"]:
            values = [_unscalar(v) for v in c["values"]]
            dtype = (
                pd.CategoricalDtype([_unscalar(v) for v in c["categories"]], ordered=c["ordered"])
                if "categories" in c
                else c["dtype"]
            )
            if "storage" in c:
                dtype = pd.StringDtype(storage=c["storage"])
            s = pd.Series(values, index=index, dtype=dtype, name=_unscalar(c["name"]))
            series.append(s)
        result = pd.concat(series, axis=1) if series else pd.DataFrame(index=index)
        result.columns.name = _unscalar(obj.get("columns_name", {"t": "none"}))
        return result
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"Invalid bundle table dtype/shape: {exc}") from exc


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    try:
        return {"__scalar__": _scalar(value)}
    except ValidationError:
        # Configuration-only objects: describe them, never serialize executable code.
        from ..runner import describe

        return {"__unavailable_component__": describe(value)}


def _unjsonable(value):
    if isinstance(value, list):
        return [_unjsonable(v) for v in value]
    if isinstance(value, dict):
        if "__scalar__" in value:
            return _unscalar(value["__scalar__"])
        return {k: _unjsonable(v) for k, v in value.items()}
    return value


def save_run(result, destination=None, *, include_data=True):
    frames = {}

    def put(frame):
        key = f"tables/{len(frames)}.json"
        frames[key] = _frame(frame)
        return key

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": result.run_id,
        "dataset_id": result.dataset_id,
        "status": result.status,
        "run_info": _jsonable(result.run_info),
        "data": None,
        "tables": {
            k: put(getattr(result, k))
            for k in ["samples", "overlaps", "class_evidence", "diagnostics", "reviews"]
        },
        "evidence": {m: {n: put(t) for n, t in tables.items()} for m, tables in result.evidence.items()},
        "representations": {},
        "decisions": [],
    }
    if include_data and result.data is not None:
        d = result.data
        manifest["data"] = {
            "X": put(d.X),
            "y": put(d.y.to_frame()),
            "y_name": _scalar(d.y.name),
            "groups": put(d.groups.to_frame()) if d.groups is not None else None,
            "groups_name": _scalar(d.groups.name) if d.groups is not None else None,
            "metadata": put(d.metadata) if d.metadata is not None else None,
            "import_info": _jsonable(d.import_info),
        }
    for m, reps in result.representations.items():
        manifest["representations"][m] = {
            name: {
                "coordinates": put(rep["coordinates"]),
                "info": _jsonable({k: v for k, v in rep.items() if k != "coordinates"}),
            }
            for name, rep in reps.items()
        }
    for d in result.decisions:
        manifest["decisions"].append(
            dict(
                policy_id=d.policy_id,
                method_id=d.method_id,
                flags=put(d.flags),
                policy_info=_jsonable(d.policy_info),
            )
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, allow_nan=False))
        for name, frame in frames.items():
            archive.writestr(name, json.dumps(frame, ensure_ascii=False, allow_nan=False))
    raw = buffer.getvalue()
    if destination is not None:
        Path(destination).write_bytes(raw)
    return raw


def load_run(source):
    """Restore viewable data only, even when extension source is absent."""
    from .readers import _bytes

    try:
        with zipfile.ZipFile(io.BytesIO(_bytes(source))) as archive:
            if (
                len(archive.infolist()) > 10000
                or sum(x.file_size for x in archive.infolist()) > MAX_BUNDLE_BYTES
            ):
                raise ValidationError("Bundle exceeds size or member-count limit")
            if len(set(archive.namelist())) != len(archive.namelist()):
                raise ValidationError("Bundle has duplicate members")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest["schema_version"] != SCHEMA_VERSION:
                raise ValidationError(
                    f"Unsupported bundle schema {manifest['schema_version']}; expected {SCHEMA_VERSION}"
                )

            def get(key):
                if not key.startswith("tables/") or ".." in key or not key.endswith(".json"):
                    raise ValidationError("Invalid table reference")
                return _unframe(json.loads(archive.read(key)))

            data = manifest["data"]
            if data is not None:
                labels = get(data["y"]).iloc[:, 0].rename(_unscalar(data["y_name"]))
                groups = (
                    get(data["groups"]).iloc[:, 0].rename(_unscalar(data["groups_name"]))
                    if data["groups"]
                    else None
                )
                data = DataSnapshot(
                    get(data["X"]),
                    labels,
                    groups,
                    get(data["metadata"]) if data["metadata"] else None,
                    _unjsonable(data["import_info"]),
                )
            result = AuditResult(
                run_id=manifest["run_id"],
                dataset_id=manifest["dataset_id"],
                status=manifest["status"],
                **{k: get(v) for k, v in manifest["tables"].items()},
                data=data,
                evidence={
                    m: {n: get(t) for n, t in tables.items()} for m, tables in manifest["evidence"].items()
                },
                representations={
                    m: {
                        n: {"coordinates": get(r["coordinates"]), **_unjsonable(r["info"])}
                        for n, r in reps.items()
                    }
                    for m, reps in manifest["representations"].items()
                },
                run_info=_unjsonable(manifest["run_info"]),
            )
            result.decisions = [
                DecisionResult(
                    d["policy_id"],
                    result.run_id,
                    d["method_id"],
                    get(d["flags"]),
                    _unjsonable(d["policy_info"]),
                )
                for d in manifest["decisions"]
            ]
            result.run_info["restored_view_only"] = True
            result.run_info["rerun_note"] = (
                "Rerun requires matching component code and dependencies; bundle contains no executable code"
            )
            return result
    except ValidationError:
        raise
    except (KeyError, TypeError, ValueError, OSError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"Invalid audit bundle: {exc}") from exc
