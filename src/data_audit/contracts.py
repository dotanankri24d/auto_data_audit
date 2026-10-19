"""Small data contracts shared by the API, built-ins, and extensions."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable
import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"
STANDARD_COLUMNS = ["sample_id", "method_id", "original_label", "label_issue_score",
                    "alternative_label", "max_overlap_score", "overlap_class_a", "overlap_class_b",
                    "sufficient_support", "support_kind", "status"]


class ValidationError(ValueError):
    """Invalid data, component parameters, or component output."""


class CancelledError(RuntimeError):
    """A cooperative cancellation boundary was reached."""


@dataclass
class CancellationToken:
    _event: Event = field(default_factory=Event, repr=False)

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()

    def check(self):
        if self.cancelled:
            raise CancelledError("Run cancelled")


@dataclass(frozen=True)
class DataSnapshot:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series | None = None
    metadata: pd.DataFrame | None = None
    import_info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisInput:
    """Read-only raw data; fitting is explicitly scoped to training indices."""
    X: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray
    feature_names: tuple
    groups: np.ndarray | None
    _metadata: pd.DataFrame | None
    embedding: Any
    distance: Any
    token: CancellationToken
    progress: Callable

    @property
    def metadata(self):
        return None if self._metadata is None else self._metadata.copy(deep=True)

    @property
    def classes(self):
        return list(pd.unique(self.y))

    def fit_embedding(self, train_indices):
        self.token.check()
        indices = np.asarray(train_indices, dtype=int)
        if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= len(self.X)):
            raise ValidationError("train_indices must be nonempty valid row indices")
        from .embeddings import fit_transformer
        return fit_transformer(self.embedding, self.X[indices], self.feature_names)

    def emit(self, stage, **details):
        self.token.check()
        self.progress({"stage": stage, **details})
