"""CSV/Excel readers used identically by the code API and UI."""

from __future__ import annotations
import csv
import io
from pathlib import Path
import numpy as np
import pandas as pd
from ..contracts import ValidationError
from ..validation import validate_ids


def _bytes(source):
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValidationError(f"Local file not found: {path}")
        return path.read_bytes()
    if not hasattr(source, "seek") or not hasattr(source, "read"):
        raise ValidationError("source must be a local path or seekable binary file")
    source.seek(0)
    raw = source.read()
    source.seek(0)
    if not isinstance(raw, bytes):
        raise ValidationError("source must be binary")
    return raw


def _columns(columns):
    names = pd.Index(columns)
    if names.hasnans or not names.is_unique or any(str(c).strip() == "" for c in names):
        raise ValidationError("Header: duplicate or missing column names")


def _finish(frame, id_column, info, dtype):
    if id_column is not None:
        if id_column not in frame:
            raise ValidationError(f"id_column {id_column!r} does not exist")
        ids = frame[id_column]
        if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
            raise ValidationError(f"id_column {id_column!r}: missing ID")
        frame = frame.set_index(id_column)
    frame.index.name = "sample_id"
    validate_ids(frame.index)
    for col, typ in (dtype or {}).items():
        if col == id_column:
            continue  # identifiers always retain textual spelling
        if col not in frame:
            raise ValidationError(f"dtype: unknown column {col!r}")
        try:
            frame[col] = frame[col].astype(typ)
        except (ValueError, TypeError) as exc:
            for i, value in enumerate(frame[col]):
                try:
                    pd.Series([value]).astype(typ)
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"{info.get('sheet_name', 'CSV')}: source row "
                        f"{i + info['header'] + 2}, column {col!r}, value {value!r}: {exc}"
                    ) from exc
            raise ValidationError(f"column {col!r}: {exc}") from exc
    frame.attrs["import_info"] = info
    return frame


def load_csv(source, *, id_column=None, delimiter=",", encoding="utf-8-sig", header=0, dtype=None):
    """Read CSV without normalizing duplicate headers or losing textual IDs."""
    if type(header) is not int or header < 0 or len(delimiter) != 1:
        raise ValidationError("header must be nonnegative and delimiter must have one character")
    raw = _bytes(source)
    try:
        rows = list(csv.reader(io.StringIO(raw.decode(encoding)), delimiter=delimiter, strict=True))
        if len(rows) <= header:
            raise ValidationError("CSV: header row does not exist")
        _columns(rows[header])
        for i, row in enumerate(rows[header + 1 :], header + 2):
            if len(row) != len(rows[header]):
                raise ValidationError(
                    f"CSV: source row {i} has {len(row)} fields; expected {len(rows[header])}"
                )
        types = dict(dtype or {})
        if id_column is not None:
            types[id_column] = "string"
        frame = pd.read_csv(
            io.BytesIO(raw),
            sep=delimiter,
            encoding=encoding,
            header=header,
            dtype=types or None,
            skip_blank_lines=False,
        )
    except (UnicodeError, csv.Error, pd.errors.ParserError, TypeError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        # Read untyped data for detailed per-cell conversion diagnostics.
        if dtype:
            frame = pd.read_csv(
                io.BytesIO(raw),
                sep=delimiter,
                encoding=encoding,
                header=header,
                dtype={id_column: "string"} if id_column else None,
                skip_blank_lines=False,
            )
        else:
            raise ValidationError(f"CSV: {exc}") from exc
    if id_column is not None and id_column in frame:
        position = rows[header].index(id_column)
        frame[id_column] = pd.Series(
            [row[position] for row in rows[header + 1 :]], dtype="string", index=frame.index
        )
    return _finish(
        frame,
        id_column,
        {
            "format": "csv",
            "delimiter": delimiter,
            "encoding": encoding,
            "header": header,
            "id_column": id_column,
        },
        dtype,
    )


def _excel(raw):
    if raw.startswith(b"PK"):
        return pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
    if raw.startswith(bytes.fromhex("d0cf11e0a1b11ae1")):
        return pd.ExcelFile(io.BytesIO(raw), engine="xlrd")
    raise ValidationError("Unsupported Excel format; expected .xlsx or .xls")


def list_excel_sheets(source):
    """List workbook sheet names in source order."""
    try:
        with _excel(_bytes(source)) as book:
            return book.sheet_names
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Excel workbook: {exc}") from exc


def load_excel(source, *, sheet_name=None, id_column=None, header=0, dtype=None):
    """Read exactly one sheet; multi-sheet workbooks require an explicit name."""
    if type(header) is not int or header < 0:
        raise ValidationError("header must be a nonnegative integer")
    try:
        with _excel(_bytes(source)) as book:
            if sheet_name is None:
                if len(book.sheet_names) != 1:
                    raise ValidationError(f"Select sheet_name explicitly from {book.sheet_names}")
                sheet_name = book.sheet_names[0]
            if sheet_name not in book.sheet_names:
                raise ValidationError(f"Sheet {sheet_name!r} not found; available: {book.sheet_names}")
            # Read header as cells, before pandas can mangle duplicate names.
            raw_header = book.parse(sheet_name, header=None, nrows=header + 1).iloc[header].tolist()
            _columns(raw_header)
            frame = book.parse(
                sheet_name, header=header, dtype={id_column: "string"} if id_column is not None else None
            )
            if id_column is not None and id_column in frame:
                # ID text is literal: tokens such as "NA" are not missing IDs.
                ids = book.parse(
                    sheet_name,
                    header=header,
                    usecols=[id_column],
                    dtype={id_column: "string"},
                    keep_default_na=False,
                )
                frame[id_column] = ids[id_column]
            return _finish(
                frame,
                id_column,
                {"format": book.engine, "sheet_name": sheet_name, "header": header, "id_column": id_column},
                dtype,
            )
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Excel sheet {sheet_name!r}: {exc}") from exc


def map_columns(frame, *, label, features, groups=None, metadata=None):
    """Explicit column roles, including source-aware numeric conversion errors."""
    from ..validation import snapshot

    features, metadata = list(features), list(metadata or [])
    if not features or len(features) != len(set(features)):
        raise ValidationError("Select unique feature columns")
    forbidden = {label, groups} | set(metadata)
    if forbidden.intersection(features):
        raise ValidationError("Label, group, and metadata columns cannot be selected as features")
    X = frame[features].copy()
    info = frame.attrs.get("import_info", {})
    for col in features:
        try:
            X[col] = pd.to_numeric(X[col], errors="raise")
        except (ValueError, TypeError) as exc:
            bad = pd.to_numeric(X[col], errors="coerce").isna() & X[col].notna()
            pos = next(i for i, v in enumerate(bad) if v)
            raise ValidationError(
                f"{info.get('sheet_name', 'CSV')}: source row {pos + info.get('header', 0) + 2}, "
                f"column {col!r}, value {X[col].iloc[pos]!r} is not numeric"
            ) from exc
        invalid = ~np.isfinite(X[col].to_numpy(dtype=float, na_value=np.nan))
        if invalid.any():
            pos = int(np.flatnonzero(invalid)[0])
            raise ValidationError(
                f"{info.get('sheet_name', 'CSV')}: source row {pos + info.get('header', 0) + 2}, "
                f"column {col!r}, value {X[col].iloc[pos]!r} must be finite"
            )
    X.attrs = frame.attrs.copy()
    return snapshot(
        X, frame[label], None if groups is None else frame[groups], frame[metadata] if metadata else None
    )
