"""Ordinary DataFrame results, independent decision layers and explicit export scopes."""
from __future__ import annotations
from dataclasses import dataclass, field
import io
import uuid
import zipfile
import pandas as pd
from .contracts import SCHEMA_VERSION, ValidationError


@dataclass
class DecisionResult:
    policy_id: str
    score_run_id: str
    method_id: str
    flags: pd.DataFrame
    policy_info: dict = field(default_factory=dict)


@dataclass
class AuditResult:
    run_id: str
    dataset_id: str
    status: str
    samples: pd.DataFrame
    overlaps: pd.DataFrame
    class_evidence: pd.DataFrame
    diagnostics: pd.DataFrame
    representations: dict
    data: object
    evidence: dict
    run_info: dict
    decisions: list = field(default_factory=list)
    reviews: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["sample_id", "method_id", "review_status", "proposed_label", "note"]))
    schema_version: str = SCHEMA_VERSION

    def apply_policy(self, method_id, policy):
        from .runner import describe
        samples = self.samples[self.samples.method_id == method_id].copy().set_index("sample_id")
        if samples.empty:
            raise ValidationError(f"Unknown method_id {method_id!r}")
        diag = self.diagnostics
        if not diag.empty:
            d = diag[diag.method_id == method_id].pivot(index="sample_id", columns="metric_name", values="numeric_value")
            samples = samples.join(d[[c for c in d if c not in samples]], how="left")
        flags = policy.decide(samples.copy(deep=True))
        if not isinstance(flags, pd.DataFrame) or not flags.index.equals(samples.index):
            raise ValidationError("policy must return a DataFrame with exactly the same sample_id index/order")
        if not flags.columns.is_unique:
            raise ValidationError("policy returned duplicate flag columns")
        for col in flags:
            if not pd.api.types.is_bool_dtype(flags[col].dtype):
                raise ValidationError(f"policy flag {col!r} must be nullable boolean")
            flags[col] = flags[col].astype("boolean")
        decision = DecisionResult(str(uuid.uuid4()), self.run_id, method_id, flags.copy(), describe(policy))
        self.decisions.append(decision)
        return decision

    def sample_evidence(self, sample_id, method_id):
        tables = self.evidence.get(method_id, {})
        return {name: table[table["query_sample_id" if "query_sample_id" in table else "sample_id"] == sample_id].copy()
                for name, table in tables.items() if "query_sample_id" in table or "sample_id" in table}

    def export_csv(self, destination=None, *, table="samples", sample_ids=None, method_id=None,
                   include_decisions=False, include_features=False, include_metadata=False, encoding="utf-8-sig"):
        if table not in ("samples", "overlaps", "class_evidence", "diagnostics", "reviews"):
            raise ValidationError(f"Unknown result table {table}")
        frame = getattr(self, table).copy()
        if method_id is not None:
            frame = frame[frame.method_id == method_id]
        if sample_ids is not None:
            frame = frame[frame.sample_id.isin(sample_ids)]
        if table == "samples":
            if include_decisions and self.decisions:
                for m in frame.method_id.unique():
                    matches = [d for d in self.decisions if d.method_id == m]
                    if matches:
                        flags = matches[-1].flags
                        for col in flags:
                            if col not in frame:
                                frame[col] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
                            mask = frame.method_id == m
                            frame.loc[mask, col] = frame.loc[mask, "sample_id"].map(flags[col])
            if include_features or include_metadata:
                if self.data is None:
                    raise ValidationError("Source snapshot was omitted; features/metadata are unavailable")
                if include_features:
                    frame = frame.join(self.data.X.add_prefix("feature.").rename_axis("sample_id"), on="sample_id")
                if include_metadata and self.data.metadata is not None:
                    frame = frame.join(self.data.metadata.add_prefix("metadata.").rename_axis("sample_id"), on="sample_id")
        text = frame.to_csv(index=False)
        if destination is not None:
            from pathlib import Path
            Path(destination).write_bytes(text.encode(encoding))
        return text.encode(encoding)

    def export_tables(self, destination=None):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ["samples", "overlaps", "class_evidence", "diagnostics", "reviews"]:
                archive.writestr(f"{name}.csv", self.export_csv(table=name, include_decisions=name == "samples"))
            for i, decision in enumerate(self.decisions):
                archive.writestr(f"decisions/{i}.csv", decision.flags.to_csv().encode("utf-8-sig"))
            for method, tables in self.evidence.items():
                for i, (name, frame) in enumerate(tables.items()):
                    # Names enter only a manifest-like filename; never filesystem extraction.
                    archive.writestr(f"evidence/{method.replace('/', '_')}-{i}.csv", frame.to_csv(index=False).encode("utf-8-sig"))
        raw = buffer.getvalue()
        if destination:
            from pathlib import Path
            Path(destination).write_bytes(raw)
        return raw

    def save(self, destination=None, *, include_data=True):
        from .io.bundle import save_run
        return save_run(self, destination, include_data=include_data)
