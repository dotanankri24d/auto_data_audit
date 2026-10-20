"""Plain Python analyses: arrays/DataFrames in, dictionaries of DataFrames out.

Representations fit on the reference set for neighbors and inside each OOF fold.
No registry, UI contract, job runner, serialization schema or installed package.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from representations import fit_transformer
from scoring import score_evidence


def _inputs(X, y, groups):
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("Use a pandas DataFrame X and Series y with matching sample-ID indices")
    if not X.index.equals(y.index) or (groups is not None and not X.index.equals(groups.index)):
        raise ValueError("X, y and groups must have exactly matching indices and order")
    if X.empty or not X.columns.is_unique or not X.index.is_unique or X.index.hasnans:
        raise ValueError("Features and sample IDs must be nonempty, unique and present")
    if y.isna().any() or (groups is not None and groups.isna().any()):
        raise ValueError("Labels and groups must be present")
    values = X.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Features must be numeric and finite; clean missing values explicitly")
    return (
        values,
        y.to_numpy(copy=True),
        X.index.to_numpy(copy=True),
        list(pd.unique(y)),
        None if groups is None else groups.to_numpy(copy=True),
    )


def _finish(result):
    """Expose diagnostics as ordinary columns and remove presentation wrappers."""
    diagnostics = result.pop("diagnostics", pd.DataFrame())
    if not diagnostics.empty:
        result["samples"] = result["samples"].merge(
            diagnostics.pivot(index="sample_id", columns="metric_name", values="numeric_value"),
            left_on="sample_id",
            right_index=True,
            validate="one_to_one",
        )
    result["samples"] = result["samples"].set_index("sample_id")
    for name, table in result.pop("evidence", {}).items():
        result[name] = table
    reps = result.pop("representations", {})
    if reps:
        result["coordinates"] = reps["reference"]["coordinates"]
    return result


def neighbors(
    X,
    y,
    *,
    groups=None,
    embedding=None,
    metric="euclidean",
    n_neighbors=30,
    weighting="uniform",
    radius_mode="auto",
    radius_quantile=0.95,
    max_radius=None,
    min_neighbors=10,
    min_effective_neighbors=5.0,
    exclude_same_group=False,
    distance_epsilon=1e-08,
    batch_size=128,
    save_evidence=True,
):
    """Rank label issues and class overlap using local class evidence.

    X/y: aligned DataFrame/Series indexed by unique sample IDs.
    embedding: fit(X) -> transform, or None for original features.
    metric: scipy distance name or distance(a, b) callable.
    radius_mode: auto quantile of kth-neighbor distances, fixed, or none.
    Support requires min_neighbors and min_effective_neighbors; weak support
    keeps scores visible but causes review_flags to abstain by default.
    Returns samples, class_evidence, overlaps, neighbors, coordinates, run_info.
    """
    if min(n_neighbors, min_neighbors, batch_size) < 1:
        raise ValueError("n_neighbors, min_neighbors and batch_size must be positive")
    if min_effective_neighbors <= 0 or not np.isfinite(min_effective_neighbors):
        raise ValueError("min_effective_neighbors must be positive and finite")
    if not 0 <= radius_quantile <= 1 or not np.isfinite(distance_epsilon) or distance_epsilon <= 0:
        raise ValueError("radius_quantile must be in [0,1]; distance_epsilon must be positive")
    if radius_mode not in ("auto", "fixed", "none") or weighting not in ("uniform", "inverse_distance"):
        raise ValueError("Unknown radius_mode or weighting")
    if radius_mode == "fixed" and (max_radius is None or not np.isfinite(max_radius) or max_radius < 0):
        raise ValueError("fixed radius requires finite max_radius >= 0")
    values, labels, ids, classes, groups = _inputs(X, y, groups)
    feature_names = tuple(X.columns)

    def fit_embedding(train):
        return fit_transformer(embedding, values[train], feature_names)

    n = len(values)
    if exclude_same_group and groups is None:
        raise ValueError("exclude_same_group requires groups")
    transform = fit_embedding(np.arange(n))
    z = transform(values)
    if callable(metric):
        original_metric = metric

        def metric(a, b):
            value = original_metric(a, b)
            if not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError("Distance must be a finite nonnegative scalar")
            return float(value)

    # scipy accepts metric names or an ordinary distance(a, b) callable
    nearest = []
    kth = []
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        distances = cdist(z[start:stop], z, metric=metric)
        for local, i in enumerate(range(start, stop)):
            d = distances[local]
            d[i] = np.inf
            if exclude_same_group:
                d[groups == groups[i]] = np.inf
            eligible = np.flatnonzero(np.isfinite(d))
            if len(eligible) > n_neighbors:
                cutoff = np.partition(d[eligible], n_neighbors - 1)[n_neighbors - 1]
                lower = eligible[d[eligible] < cutoff]
                ties = eligible[d[eligible] == cutoff]
                eligible = np.concatenate([lower, ties[: n_neighbors - len(lower)]])
            ix = eligible[np.lexsort((eligible, d[eligible]))]
            nearest.append((ix, d[ix].copy()))
            if len(ix) == n_neighbors:
                kth.append(float(d[ix[-1]]))
    if radius_mode == "auto":
        if len(kth) != n:
            raise ValueError(
                "Insufficient reference neighbors to calibrate auto radius; choose radius_mode='none' explicitly"
            )
        radius = float(np.quantile(kth, radius_quantile))
    else:
        radius = max_radius if radius_mode == "fixed" else None
    q = np.full((n, len(classes)), np.nan)
    supports, diagnostics, evidence = ([], [], [])
    for i, (ix, ds) in enumerate(nearest):
        if radius is not None:
            ix, ds = (ix[ds <= radius], ds[ds <= radius])
        weights = np.ones(len(ix)) if weighting == "uniform" else 1 / np.maximum(ds, distance_epsilon)
        effective = float(weights.sum() ** 2 / (weights**2).sum()) if len(ix) else np.nan
        supports.append(bool(len(ix) >= min_neighbors and effective >= min_effective_neighbors))
        if len(ix):
            q[i] = [weights[labels[ix] == c].sum() / weights.sum() for c in classes]
        metrics = dict(
            neighbor_count=len(ix),
            effective_neighbors=effective,
            nearest_distance=float(ds[0]) if len(ds) else np.nan,
            max_distance=float(ds[-1]) if len(ds) else np.nan,
            median_distance=float(np.median(ds)) if len(ds) else np.nan,
            radius_used=radius,
            reference_group_count=len(pd.unique(groups[ix])) if groups is not None else None,
        )
        for name, value in metrics.items():
            diagnostics.append(dict(sample_id=ids[i], metric_name=name, numeric_value=value))
        if save_evidence:
            for rank, (j, distance, weight) in enumerate(zip(ix, ds, weights), 1):
                evidence.append(
                    dict(
                        query_sample_id=ids[i],
                        neighbor_sample_id=ids[j],
                        rank=rank,
                        distance=distance,
                        weight=weight,
                        neighbor_label=labels[j],
                    )
                )
    result = score_evidence(
        ids,
        labels,
        classes,
        q,
        supports,
        "local_neighbors" if radius is not None else "neighbor_count_and_weight_only",
    )
    result.update(
        diagnostics=pd.DataFrame(diagnostics),
        evidence={
            "neighbors": pd.DataFrame(
                evidence,
                columns=[
                    "query_sample_id",
                    "neighbor_sample_id",
                    "rank",
                    "distance",
                    "weight",
                    "neighbor_label",
                ],
            )
        },
        representations={
            "reference": {
                "coordinates": pd.DataFrame(z, index=ids, columns=transform.feature_names),
                "fitted_scope": "all_reference",
            }
        },
        run_info={"calibrated_radius": radius, "evidence_saved": save_evidence},
    )
    return _finish(result)


def oof(
    X,
    y,
    *,
    groups=None,
    embedding=None,
    estimator=None,
    n_splits=5,
    n_repeats=3,
    random_state=42,
    min_train_per_class=2,
    min_valid_repeats=None,
    save_evidence=True,
):
    """Repeated stratified OOF evidence with training-fold-local fitting.

    X/y: aligned DataFrame/Series indexed by unique sample IDs.
    groups: optional aligned Series, isolating groups between train and test.
    estimator: sklearn classifier with predict_proba; defaults to logistic regression.
    Returns samples (including stability), class_evidence, overlaps, oof
    (repeat/fold/class predictions), and run_info with exact split IDs.
    No final estimator is fitted on the full dataset.
    """
    if n_splits < 2 or n_repeats < 1 or min_train_per_class < 1:
        raise ValueError("n_splits >= 2, n_repeats >= 1 and min_train_per_class >= 1 are required")
    if min_valid_repeats is not None and (not 1 <= min_valid_repeats <= n_repeats):
        raise ValueError("min_valid_repeats must be in [1, n_repeats]")
    values, labels, ids, classes, groups = _inputs(X, y, groups)
    feature_names = tuple(X.columns)

    def fit_embedding(train):
        return fit_transformer(embedding, values[train], feature_names)

    if len(classes) < 2:
        return _finish(
            score_evidence(
                ids,
                labels,
                classes,
                np.full((len(values), len(classes)), np.nan),
                [False] * len(values),
                "prediction_coverage",
            )
        )
    codes = np.array([classes.index(v) for v in labels])
    if np.bincount(codes).min() < n_splits:
        raise ValueError("Every category needs at least n_splits samples for stratified OOF")
    if groups is not None and len(pd.unique(groups)) < n_splits:
        raise ValueError("groups: fewer distinct groups than n_splits")
    splits = []
    for repeat in range(n_repeats):
        seed = random_state + repeat
        splitter = (
            StratifiedKFold(n_splits, shuffle=True, random_state=seed)
            if groups is None
            else StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        )
        for fold, (train, test) in enumerate(splitter.split(values, codes, groups)):
            counts = np.bincount(codes[train], minlength=len(classes))
            if counts.min() < min_train_per_class:
                raise ValueError(
                    f"OOF repeat {repeat} fold {fold}: training category counts {counts.tolist()} below min_train_per_class={min_train_per_class}"
                )
            if groups is not None and set(groups[train]) & set(groups[test]):
                raise ValueError("OOF splitter leaked groups")
            splits.append((repeat, fold, seed, train, test, counts))
    probs = np.full((n_repeats, len(values), len(classes)), np.nan)
    evidence, split_info = ([], [])
    for repeat, fold, seed, train, test, counts in splits:
        transform = fit_embedding(train)
        estimator = clone(estimator if estimator is not None else LogisticRegression(max_iter=1000))
        if not hasattr(estimator, "predict_proba"):
            raise ValueError("OOF estimator must implement predict_proba or use an explicit evidence adapter")
        params = estimator.get_params()
        if "random_state" in params and params["random_state"] is None:
            estimator.set_params(random_state=seed)
        estimator.fit(transform(values[train]), codes[train])
        raw = np.asarray(estimator.predict_proba(transform(values[test])), dtype=float)
        model_classes = list(estimator.classes_)
        if raw.ndim != 2 or raw.shape != (len(test), len(model_classes)):
            raise ValueError("predict_proba returned an invalid matrix shape")
        if set(model_classes) != set(range(len(classes))):
            raise ValueError(
                "OOF model is missing required categories; no zero-filled probabilities are invented"
            )
        aligned = raw[:, [model_classes.index(c) for c in range(len(classes))]]
        if (
            aligned.shape != (len(test), len(classes))
            or not np.isfinite(aligned).all()
            or (aligned < 0).any()
            or (aligned > 1).any()
            or (not np.allclose(aligned.sum(axis=1), 1, atol=1e-06))
        ):
            raise ValueError("predict_proba must produce finite normalized class evidence")
        probs[repeat, test] = aligned
        split_info.append(
            dict(
                repeat_id=repeat,
                fold_id=fold,
                seed=seed,
                train_ids=ids[train].tolist(),
                test_ids=ids[test].tolist(),
                train_class_counts=counts.tolist(),
            )
        )
        if save_evidence:
            for local, idx in enumerate(test):
                for j, category in enumerate(classes):
                    evidence.append(
                        dict(
                            sample_id=ids[idx],
                            repeat_id=repeat,
                            fold_id=fold,
                            class_label=category,
                            class_score=aligned[local, j],
                        )
                    )
    coverage = np.isfinite(probs).all(axis=2).sum(axis=0)
    required = min_valid_repeats or n_repeats
    q = probs.mean(axis=0)
    result = score_evidence(ids, labels, classes, q, coverage >= required, "prediction_coverage")
    per_repeat_scores, per_repeat_overlaps, alternatives = ([], [], [])
    partitions = set()
    for repeat in range(n_repeats):
        partitions.add(
            tuple(sorted((tuple(sorted(test.tolist())) for r, _, _, _, test, _ in splits if r == repeat)))
        )
    varied_splits = len(partitions) > 1
    for repeat in range(n_repeats):
        scored = score_evidence(
            ids, labels, classes, probs[repeat], [True] * len(values), "prediction_coverage"
        )["samples"]
        per_repeat_scores.append(scored.label_issue_score.to_numpy())
        per_repeat_overlaps.append(scored.max_overlap_score.to_numpy())
        alternatives.append(scored.alternative_label.to_numpy())
    std = np.std(per_repeat_scores, axis=0, ddof=1) if varied_splits else np.full(len(values), np.nan)
    overlap_std = (
        np.std(per_repeat_overlaps, axis=0, ddof=1) if varied_splits else np.full(len(values), np.nan)
    )
    agreement = (
        np.mean(
            np.array(alternatives, dtype=object) == result["samples"].alternative_label.to_numpy(), axis=0
        )
        if varied_splits
        else np.full(len(values), np.nan)
    )
    diagnostics = []
    for i, sid in enumerate(ids):
        for metric, value in dict(
            valid_repeats=coverage[i],
            label_issue_std=std[i],
            overlap_std=overlap_std[i],
            alternative_agreement=agreement[i],
        ).items():
            diagnostics.append(dict(sample_id=sid, metric_name=metric, numeric_value=value))
    result.update(
        diagnostics=pd.DataFrame(diagnostics),
        evidence={
            "oof": pd.DataFrame(
                evidence, columns=["sample_id", "repeat_id", "fold_id", "class_label", "class_score"]
            )
        },
        run_info={
            "splits": split_info,
            "distinct_split_repeats": len(partitions),
            "stability_experiment": "repeated_stratified_group_folds"
            if groups is not None
            else "repeated_stratified_folds",
            "stability_note": "Between-repeat dispersion unavailable with one repeat"
            if n_repeats == 1
            else "Identical repeat partitions do not provide additional stability evidence"
            if not varied_splits
            else "Dispersion is not a confidence interval",
            "evidence_saved": save_evidence,
        },
    )
    return _finish(result)
