"""Decision thresholds never refit an analysis."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReviewPolicy:
    issue_threshold: float = 0.5
    overlap_threshold: float = 0.6
    require_support: bool = True
    min_alternative_agreement: float | None = None

    def __post_init__(self):
        from ..plugins import validate_component_types

        validate_component_types(self)
        for name in ("issue_threshold", "overlap_threshold", "min_alternative_agreement"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be between 0 and 1")

    def decide(self, samples):
        valid = pd.Series(True, index=samples.index)
        if self.require_support:
            valid &= samples["sufficient_support"].fillna(False).astype(bool)
        if self.min_alternative_agreement is not None:
            if "alternative_agreement" not in samples:
                valid[:] = False
            else:
                valid &= samples["alternative_agreement"].ge(self.min_alternative_agreement).fillna(False)
        flags = pd.DataFrame(index=samples.index)
        for flag, score, threshold in [
            ("label_issue", "label_issue_score", self.issue_threshold),
            ("overlap", "max_overlap_score", self.overlap_threshold),
        ]:
            flags[flag] = samples[score].ge(threshold).astype("boolean").where(valid & samples[score].notna())
        return flags
