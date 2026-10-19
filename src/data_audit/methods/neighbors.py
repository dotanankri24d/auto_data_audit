"""Exact bounded-memory neighborhoods with explicit scale and support."""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from ..distances import Euclidean, checked_distance
from ..scoring import score_evidence


@dataclass(frozen=True)
class NeighborAudit:
    name: str = "neighbors"
    n_neighbors: int = 30
    weighting: Literal["uniform", "inverse_distance"] = "uniform"
    radius_mode: Literal["auto", "fixed", "none"] = "auto"
    radius_quantile: float = 0.95
    max_radius: float | None = None
    min_neighbors: int = 10
    min_effective_neighbors: float = 5.0
    exclude_same_group: bool = False
    distance_epsilon: float = 1e-8
    batch_size: int = 128
    save_evidence: bool = True

    def __post_init__(self):
        if min(self.n_neighbors, self.min_neighbors, self.batch_size) < 1:
            raise ValueError("n_neighbors, min_neighbors and batch_size must be positive")
        if self.min_effective_neighbors <= 0 or not np.isfinite(self.min_effective_neighbors):
            raise ValueError("min_effective_neighbors must be positive and finite")
        if not 0 <= self.radius_quantile <= 1 or not np.isfinite(self.distance_epsilon) or self.distance_epsilon <= 0:
            raise ValueError("radius_quantile must be in [0,1]; distance_epsilon must be positive")
        if self.radius_mode not in ("auto", "fixed", "none") or self.weighting not in ("uniform", "inverse_distance"):
            raise ValueError("Unknown radius_mode or weighting")
        if self.radius_mode == "fixed" and (self.max_radius is None or not np.isfinite(self.max_radius) or self.max_radius < 0):
            raise ValueError("fixed radius requires finite max_radius >= 0")

    def analyze(self, data):
        n = len(data.X)
        if self.exclude_same_group and data.groups is None:
            raise ValueError("exclude_same_group requires groups")
        transform = data.fit_embedding(np.arange(n))
        z = transform(data.X)
        metric = "euclidean" if isinstance(data.distance, Euclidean) else checked_distance(data.distance)
        nearest = []
        kth = []
        # Only a batch by n distance matrix is retained; never allocate n by n.
        for start in range(0, n, self.batch_size):
            data.emit("neighbors", completed=start, total=n)
            stop = min(n, start + self.batch_size)
            distances = cdist(z[start:stop], z, metric=metric)
            for local, i in enumerate(range(start, stop)):
                d = distances[local]
                d[i] = np.inf  # identity, not feature equality
                if self.exclude_same_group:
                    d[data.groups == data.groups[i]] = np.inf
                eligible = np.flatnonzero(np.isfinite(d))
                # Partial selection bounds sorting work, tie order is stable by row ID order.
                if len(eligible) > self.n_neighbors:
                    cutoff = np.partition(d[eligible], self.n_neighbors - 1)[self.n_neighbors - 1]
                    lower = eligible[d[eligible] < cutoff]
                    ties = eligible[d[eligible] == cutoff]
                    eligible = np.concatenate([lower, ties[:self.n_neighbors - len(lower)]])
                ix = eligible[np.lexsort((eligible, d[eligible]))]
                nearest.append((ix, d[ix].copy()))
                if len(ix) == self.n_neighbors:
                    kth.append(float(d[ix[-1]]))
        if self.radius_mode == "auto":
            if len(kth) != n:
                raise ValueError("Insufficient reference neighbors to calibrate auto radius; choose radius_mode='none' explicitly")
            radius = float(np.quantile(kth, self.radius_quantile))
        else:
            radius = self.max_radius if self.radius_mode == "fixed" else None
        classes = data.classes
        q = np.full((n, len(classes)), np.nan)
        supports, diagnostics, evidence = [], [], []
        for i, (ix, ds) in enumerate(nearest):
            data.token.check()
            if radius is not None:
                ix, ds = ix[ds <= radius], ds[ds <= radius]
            weights = np.ones(len(ix)) if self.weighting == "uniform" else 1 / np.maximum(ds, self.distance_epsilon)
            effective = float(weights.sum() ** 2 / (weights ** 2).sum()) if len(ix) else np.nan
            supports.append(bool(len(ix) >= self.min_neighbors and effective >= self.min_effective_neighbors))
            if len(ix):
                q[i] = [weights[data.y[ix] == c].sum() / weights.sum() for c in classes]
            metrics = dict(neighbor_count=len(ix), effective_neighbors=effective,
                           nearest_distance=float(ds[0]) if len(ds) else np.nan,
                           max_distance=float(ds[-1]) if len(ds) else np.nan,
                           median_distance=float(np.median(ds)) if len(ds) else np.nan,
                           radius_used=radius,
                           reference_group_count=len(pd.unique(data.groups[ix])) if data.groups is not None else None)
            for name, value in metrics.items():
                diagnostics.append(dict(sample_id=data.sample_ids[i], metric_name=name, numeric_value=value))
            if self.save_evidence:
                for rank, (j, distance, weight) in enumerate(zip(ix, ds, weights), 1):
                    evidence.append(dict(query_sample_id=data.sample_ids[i], neighbor_sample_id=data.sample_ids[j],
                                         rank=rank, distance=distance, weight=weight, neighbor_label=data.y[j]))
        result = score_evidence(data.sample_ids, data.y, classes, q, supports,
                                "local_neighbors" if radius is not None else "neighbor_count_and_weight_only")
        result.update(diagnostics=pd.DataFrame(diagnostics),
                      evidence={"neighbors": pd.DataFrame(evidence, columns=["query_sample_id", "neighbor_sample_id",
                                "rank", "distance", "weight", "neighbor_label"])},
                      representations={"reference": {"coordinates": pd.DataFrame(z, index=data.sample_ids,
                                                  columns=transform.feature_names), "fitted_scope": "all_reference"}},
                      run_info={"calibrated_radius": radius, "evidence_saved": self.save_evidence})
        return result
