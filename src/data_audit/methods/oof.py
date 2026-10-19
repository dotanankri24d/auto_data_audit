"""Repeated out-of-fold evidence with fold-local fitting and group isolation."""
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from ..scoring import score_evidence


@dataclass(frozen=True)
class OOFAudit:
    name: str = "oof"
    estimator: Any = None
    n_splits: int = 5
    n_repeats: int = 3
    random_state: int = 42
    min_train_per_class: int = 2
    min_valid_repeats: int | None = None
    save_evidence: bool = True

    def __post_init__(self):
        if self.n_splits < 2 or self.n_repeats < 1 or self.min_train_per_class < 1:
            raise ValueError("n_splits >= 2, n_repeats >= 1 and min_train_per_class >= 1 are required")
        if self.min_valid_repeats is not None and not 1 <= self.min_valid_repeats <= self.n_repeats:
            raise ValueError("min_valid_repeats must be in [1, n_repeats]")

    def analyze(self, data):
        classes = data.classes
        if len(classes) < 2:
            return score_evidence(data.sample_ids, data.y, classes,
                                  np.full((len(data.X), len(classes)), np.nan),
                                  [False] * len(data.X), "prediction_coverage")
        # Integer codes prevent sklearn from sorting or coercing original labels.
        codes = np.array([classes.index(v) for v in data.y])
        if np.bincount(codes).min() < self.n_splits:
            raise ValueError("Every category needs at least n_splits samples for stratified OOF")
        if data.groups is not None and len(pd.unique(data.groups)) < self.n_splits:
            raise ValueError("groups: fewer distinct groups than n_splits")
        splits = []
        for repeat in range(self.n_repeats):
            seed = self.random_state + repeat
            splitter = (StratifiedKFold(self.n_splits, shuffle=True, random_state=seed) if data.groups is None
                        else StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=seed))
            for fold, (train, test) in enumerate(splitter.split(data.X, codes, data.groups)):
                counts = np.bincount(codes[train], minlength=len(classes))
                if counts.min() < self.min_train_per_class:
                    raise ValueError(f"OOF repeat {repeat} fold {fold}: training category counts {counts.tolist()} "
                                     f"below min_train_per_class={self.min_train_per_class}")
                if data.groups is not None and set(data.groups[train]) & set(data.groups[test]):
                    raise ValueError("OOF splitter leaked groups")
                splits.append((repeat, fold, seed, train, test, counts))
        probs = np.full((self.n_repeats, len(data.X), len(classes)), np.nan)
        evidence, split_info = [], []
        for repeat, fold, seed, train, test, counts in splits:
            data.emit("oof_fold", repeat=repeat, fold=fold, total=len(splits))
            transform = data.fit_embedding(train)
            estimator = clone(self.estimator if self.estimator is not None else LogisticRegression(max_iter=1000))
            if not hasattr(estimator, "predict_proba"):
                raise ValueError("OOF estimator must implement predict_proba or use an explicit evidence adapter")
            params = estimator.get_params()
            if "random_state" in params and params["random_state"] is None:
                estimator.set_params(random_state=seed)
            estimator.fit(transform(data.X[train]), codes[train])
            raw = np.asarray(estimator.predict_proba(transform(data.X[test])), dtype=float)
            model_classes = list(estimator.classes_)
            if set(model_classes) != set(range(len(classes))):
                raise ValueError("OOF model is missing required categories; no zero-filled probabilities are invented")
            aligned = raw[:, [model_classes.index(c) for c in range(len(classes))]]
            if (aligned.shape != (len(test), len(classes)) or not np.isfinite(aligned).all()
                    or (aligned < 0).any() or (aligned > 1).any()
                    or not np.allclose(aligned.sum(axis=1), 1, atol=1e-6)):
                raise ValueError("predict_proba must produce finite normalized class evidence")
            probs[repeat, test] = aligned
            split_info.append(dict(repeat_id=repeat, fold_id=fold, seed=seed,
                                   train_ids=data.sample_ids[train].tolist(), test_ids=data.sample_ids[test].tolist(),
                                   train_class_counts=counts.tolist()))
            if self.save_evidence:
                for local, idx in enumerate(test):
                    for j, category in enumerate(classes):
                        evidence.append(dict(sample_id=data.sample_ids[idx], repeat_id=repeat, fold_id=fold,
                                             class_label=category, class_score=aligned[local, j]))
        coverage = np.isfinite(probs).all(axis=2).sum(axis=0)
        required = self.min_valid_repeats or self.n_repeats
        q = probs.mean(axis=0)
        result = score_evidence(data.sample_ids, data.y, classes, q, coverage >= required, "prediction_coverage")
        per_repeat_scores, alternatives = [], []
        for repeat in range(self.n_repeats):
            scored = score_evidence(data.sample_ids, data.y, classes, probs[repeat],
                                    [True] * len(data.X), "prediction_coverage")["samples"]
            per_repeat_scores.append(scored.label_issue_score.to_numpy())
            alternatives.append(scored.alternative_label.to_numpy())
        std = np.std(per_repeat_scores, axis=0, ddof=1) if self.n_repeats > 1 else np.full(len(data.X), np.nan)
        agreement = (np.mean(np.array(alternatives, dtype=object) == result["samples"].alternative_label.to_numpy(), axis=0)
                     if self.n_repeats > 1 else np.full(len(data.X), np.nan))
        diagnostics = []
        for i, sid in enumerate(data.sample_ids):
            for metric, value in dict(valid_repeats=coverage[i], label_issue_std=std[i],
                                      alternative_agreement=agreement[i]).items():
                diagnostics.append(dict(sample_id=sid, metric_name=metric, numeric_value=value))
        result.update(diagnostics=pd.DataFrame(diagnostics),
                      evidence={"oof": pd.DataFrame(evidence, columns=["sample_id", "repeat_id", "fold_id",
                                                                      "class_label", "class_score"])},
                      run_info={"splits": split_info, "stability_experiment": "repeated_stratified_group_folds" if
                                data.groups is not None else "repeated_stratified_folds",
                                "stability_note": "Between-repeat dispersion unavailable with one repeat" if
                                self.n_repeats == 1 else "Dispersion is not a confidence interval",
                                "evidence_saved": self.save_evidence})
        return result
