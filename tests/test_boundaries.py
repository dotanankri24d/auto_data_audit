import numpy as np
import pandas as pd
import pytest
from data_audit import (
    audit,
    NeighborAudit,
    OOFAudit,
    ReviewPolicy,
    bind_method,
    bind_distance,
    PCAEmbedding,
    ValidationError,
    CancellationToken,
)
from data_audit.embeddings import fit_transformer
from data_audit import bind_embedding


def test_one_category_unsupported_and_small_k():
    r = audit(
        [[0], [1], [2]],
        ["A"] * 3,
        methods=[NeighborAudit(n_neighbors=10, radius_mode="none"), OOFAudit(n_splits=2)],
    )
    assert r.samples.label_issue_score.isna().all() and r.samples.max_overlap_score.isna().all()
    assert r.samples.status.eq("unsupported").all()
    assert r.apply_policy("neighbors", ReviewPolicy(require_support=False)).flags.isna().all().all()
    assert r.diagnostics.query("metric_name=='neighbor_count'").numeric_value.eq(2).all()


@pytest.mark.parametrize(
    "output,match",
    [
        ({"samples": pd.DataFrame({"sample_id": [0, 1], "label_issue_score": [2.0, 0.0]})}, "scores"),
        ({"samples": pd.DataFrame({"sample_id": [0, 0]})}, "exactly one"),
        ({"samples": pd.DataFrame({"sample_id": [0, 1], "sufficient_support": ["yes", "no"]})}, "boolean"),
        ({"samples": pd.DataFrame({"sample_id": [0, 1]}), "evidence": {"bad": np.zeros(2)}}, "DataFrame"),
    ],
)
def test_output_rejection(output, match):
    def analyze(data):
        return output

    with pytest.raises(ValidationError, match=match):
        audit([[0], [1]], ["A", "B"], methods=[bind_method(analyze)])


def test_custom_distance_exact_and_invalid():
    def distance(a, b, scale=1.0):
        return float(np.abs(a - b).sum()) * scale

    r = audit(
        [[0, 0], [1, 0], [0, 3]],
        ["A", "A", "B"],
        distance=bind_distance(distance, scale=2),
        methods=[NeighborAudit(n_neighbors=1, radius_mode="none")],
    )
    e = r.evidence["neighbors"]["neighbors"]
    assert e.iloc[0].distance == 2

    def bad(a, b):
        return -1

    with pytest.raises(ValueError, match="nonnegative"):
        audit(
            [[0], [1]], ["A", "B"], distance=bind_distance(bad), methods=[NeighborAudit(radius_mode="none")]
        )


def test_transform_shape_columns_and_fold_limits():
    def wrong(X):
        return lambda x: x[:-1]

    with pytest.raises(ValidationError, match="preserve rows"):
        fit_transformer(bind_embedding(wrong), np.ones((3, 2)), ["a", "b"])
    t = fit_transformer(PCAEmbedding(1), np.array([[1.0, 2.0], [2.0, 1.0], [4.0, 0.0]]), ["a", "b"])
    with pytest.raises(ValidationError, match="names/order"):
        t(pd.DataFrame([[1, 2]], columns=["b", "a"]))
    with pytest.raises(ValueError, match="n_splits"):
        audit(np.arange(5).reshape(-1, 1), ["A", "A", "A", "B", "B"], methods=[OOFAudit(n_splits=3)])
    with pytest.raises(ValueError, match="distinct groups"):
        audit(np.arange(6).reshape(-1, 1), ["A", "B"] * 3, groups=[1] * 6, methods=[OOFAudit(n_splits=3)])
    with pytest.raises(ValueError, match="training category counts"):
        audit(
            np.arange(4).reshape(-1, 1), ["A", "B"] * 2, methods=[OOFAudit(n_splits=2, min_train_per_class=2)]
        )
    from sklearn.svm import LinearSVC

    with pytest.raises(ValueError, match="predict_proba"):
        audit(
            np.arange(12).reshape(-1, 1),
            ["A", "B"] * 6,
            methods=[OOFAudit(n_splits=2, estimator=LinearSVC())],
        )


def test_cancel_during_progress_and_instance_names():
    token = CancellationToken()

    def progress(event):
        if event["stage"] == "oof_fold":
            token.cancel()

    r = audit(
        np.arange(60).reshape(30, 2),
        ["A", "B"] * 15,
        methods=[OOFAudit(n_splits=3)],
        progress=progress,
        cancellation_token=token,
    )
    assert r.status == "cancelled" and r.samples.status.eq("cancelled").all()
    with pytest.raises(ValidationError, match="unique nonempty"):
        audit([[0], [1]], ["A", "B"], methods=[NeighborAudit(), NeighborAudit()])
    r = audit(
        np.arange(12).reshape(-1, 1),
        ["A", "B"] * 6,
        methods=[
            NeighborAudit(name="near", radius_mode="none"),
            NeighborAudit(name="far", radius_mode="none"),
        ],
    )
    assert set(r.samples.method_id) == {"near", "far"}


@pytest.mark.parametrize(
    "constructor,params",
    [
        (NeighborAudit, {"n_neighbors": 1.5}),
        (OOFAudit, {"n_splits": True}),
        (ReviewPolicy, {"require_support": "yes"}),
        (PCAEmbedding, {"whiten": "yes"}),
    ],
)
def test_constructor_type_contract(constructor, params):
    with pytest.raises(ValueError, match="expected"):
        constructor(**params)


def test_identical_repeat_partitions_do_not_claim_stability(monkeypatch):
    from sklearn.model_selection import StratifiedKFold
    from data_audit.methods import oof

    class FixedSplit:
        def __init__(self, n_splits, **kwargs):
            self.splitter = StratifiedKFold(n_splits, shuffle=True, random_state=1)

        def split(self, X, y, groups):
            return self.splitter.split(X, y)

    monkeypatch.setattr(oof, "StratifiedKFold", FixedSplit)
    r = audit(np.arange(60).reshape(30, 2), ["A", "B"] * 15, methods=[OOFAudit(n_splits=3, n_repeats=2)])
    assert r.run_info["methods"]["oof"]["distinct_split_repeats"] == 1
    assert r.diagnostics.query("metric_name=='label_issue_std'").numeric_value.isna().all()
    assert r.diagnostics.query("metric_name=='alternative_agreement'").numeric_value.isna().all()
