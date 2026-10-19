import io
import sys
import subprocess
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from data_audit import (
    audit,
    bind_embedding,
    bind_method,
    NeighborAudit,
    OOFAudit,
    ReviewPolicy,
    PCAEmbedding,
    Standardize,
    EmbeddingPipeline,
    load_run,
    ValidationError,
    CancellationToken,
)
from data_audit.scoring import score_evidence
from data_audit.views import demo_data, reference_projection


@pytest.fixture
def frame():
    return demo_data(n_samples=90)


def test_core_import_no_ui():
    p = subprocess.run(
        [
            sys.executable,
            "-c",
            'import data_audit, sys; assert "streamlit" not in sys.modules; '
            'assert "plotly" not in sys.modules; assert "torch" not in sys.modules',
        ],
        capture_output=True,
    )
    assert p.returncode == 0, p.stderr


def test_input_validation_and_snapshot(frame):
    f = frame
    method = NeighborAudit(n_neighbors=4, radius_mode="none", min_neighbors=2, min_effective_neighbors=1)
    r = audit(f.iloc[:, :5], f.label, methods=[method])
    old = r.data.X.copy()
    f.iloc[0, 0] = 999
    assert_frame_equal(r.data.X, old)
    with pytest.raises(ValidationError, match="same order"):
        audit(f.iloc[:, :5], f.label.iloc[::-1])
    for bad in [np.nan, np.inf]:
        X = f.iloc[:, :5].copy()
        X.iloc[0, 0] = bad
        with pytest.raises(ValidationError, match="sample"):
            audit(X, f.label)
    X = f.iloc[:, :5].copy()
    X.index = ["duplicate"] * len(X)
    with pytest.raises(ValidationError, match="unique"):
        audit(X, f.label)
    with pytest.raises(ValidationError, match="y: missing"):
        audit(np.ones((3, 2)), ["a", None, "b"])
    with pytest.raises(ValidationError, match="unique"):
        audit(np.ones((3, 2)), ["a", "b", "b"], feature_names=["f", "f"])


def test_numpy_and_custom_contract():
    def analyze(data, power: float = 2.0):
        assert not data.X.flags.writeable
        z = data.fit_embedding(np.arange(len(data.X)))(data.X)
        return {
            "samples": pd.DataFrame(
                {"sample_id": data.sample_ids, "distance": np.linalg.norm(z, axis=1) ** power}
            )
        }

    def fit(X, epsilon=1e-8):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        mean, scale = X.mean(0), np.maximum(X.std(0), epsilon)
        return lambda x: (x - mean) / scale

    r = audit(
        np.array([[0, 1], [1, 0], [2, 1]]),
        ["a", "b", "a"],
        embedding=bind_embedding(fit),
        methods=[bind_method(analyze, name="center")],
    )
    assert "center.distance" in r.samples
    assert r.samples.label_issue_score.isna().all()
    assert r.samples.sufficient_support.isna().all()
    with pytest.raises(ValidationError, match="expected"):
        bind_method(analyze, power="bad")
    with pytest.raises(ValueError, match="epsilon must be positive"):
        audit(
            [[1], [2]], ["a", "b"], embedding=bind_embedding(fit, epsilon=-1), methods=[bind_method(analyze)]
        )


@pytest.mark.parametrize(
    "q,issue,overlap",
    [
        ([0.95, 0.05, 0], 0, 0.1),
        ([0.5, 0.5, 0], 0, 1),
        ([0.05, 0.95, 0], 0.9, 0.1),
        ([0.05, 0.475, 0.475], 0.425, 0.95),
    ],
)
def test_score_definitions(q, issue, overlap):
    r = score_evidence(["id"], ["A"], ["A", "B", "C"], np.array([q]), [True], "test")
    s = r["samples"].iloc[0]
    assert s.label_issue_score == pytest.approx(issue)
    assert s.max_overlap_score == pytest.approx(overlap)
    assert len(r["overlaps"]) == (3 if q[1] == q[2] else 2)


def test_neighbors_self_duplicates_radius_groups():
    X = pd.DataFrame({"x": [0.0, 0.0, 1.0, 100.0]}, index=["a", "b", "c", "d"])
    y = pd.Series(["A", "B", "B", "A"], index=X.index)
    groups = pd.Series(["g1", "g2", "g2", "g3"], index=X.index)
    m = NeighborAudit(
        n_neighbors=3,
        radius_mode="fixed",
        max_radius=2,
        min_neighbors=2,
        min_effective_neighbors=1,
        exclude_same_group=True,
        weighting="inverse_distance",
    )
    r = audit(X, y, groups=groups, methods=[m])
    e = r.evidence["neighbors"]["neighbors"]
    assert not (e.query_sample_id == e.neighbor_sample_id).any()
    assert ((e.query_sample_id == "a") & (e.neighbor_sample_id == "b") & (e.distance == 0)).any()
    assert not ((e.query_sample_id == "b") & (e.neighbor_sample_id == "c")).any()
    assert e.distance.max() <= 2
    assert not r.samples.set_index("sample_id").loc["d", "sufficient_support"]
    d = r.apply_policy("neighbors", ReviewPolicy())
    assert pd.isna(d.flags.loc["d", "label_issue"])
    before = r.samples.copy(deep=True)
    r.apply_policy("neighbors", ReviewPolicy(issue_threshold=0, require_support=False))
    assert_frame_equal(before, r.samples)
    with pytest.raises(ValueError, match="calibrate"):
        audit(X, y, methods=[NeighborAudit(n_neighbors=10)])


def test_neighbor_effective_support_after_radius():
    r = audit(
        [[0], [0], [1], [100]],
        ["A", "B", "B", "A"],
        methods=[
            NeighborAudit(
                n_neighbors=3, radius_mode="fixed", max_radius=0.5, min_neighbors=2, min_effective_neighbors=1
            )
        ],
    )
    assert not r.samples.sufficient_support.any()
    assert r.samples.iloc[0].label_issue_score == 1


def test_oof_fit_isolation_groups_class_alignment(frame):
    seen = []

    def fit(X):
        seen.append(set(X[:, 0]))
        return lambda x: x

    from sklearn.linear_model import LogisticRegression

    class ReversedEstimator(LogisticRegression):
        def fit(self, X, y):
            super().fit(X, y)
            self.classes_ = self.classes_[::-1]
            return self

        def predict_proba(self, X):
            # Temporarily restore training order, then explicitly reverse output columns.
            self.classes_ = self.classes_[::-1]
            p = super().predict_proba(X)
            self.classes_ = self.classes_[::-1]
            return p[:, ::-1]

    r = audit(
        frame.iloc[:, :5],
        frame.label,
        groups=frame.recording,
        embedding=bind_embedding(fit),
        methods=[OOFAudit(n_splits=3, n_repeats=2, estimator=ReversedEstimator(max_iter=1000))],
    )
    assert r.status == "completed"
    splits = r.run_info["methods"]["oof"]["splits"]
    assert len(seen) == 6
    for actual, split in zip(seen, splits):
        assert actual == set(frame.loc[split["train_ids"], "feature_0"])
        assert not actual & set(frame.loc[split["test_ids"], "feature_0"])
        assert not set(frame.loc[split["train_ids"], "recording"]) & set(
            frame.loc[split["test_ids"], "recording"]
        )
    for repeat, table in r.evidence["oof"]["oof"].groupby("repeat_id"):
        assert table.groupby("sample_id").class_score.sum().eq(1).all() or np.allclose(
            table.groupby("sample_id").class_score.sum(), 1
        )
        assert table.groupby("sample_id").fold_id.nunique().eq(1).all()
    normal = audit(
        frame.iloc[:, :5],
        frame.label,
        groups=frame.recording,
        embedding=bind_embedding(fit),
        methods=[OOFAudit(n_splits=3, n_repeats=2)],
    )
    assert np.allclose(r.class_evidence.class_score, normal.class_evidence.class_score)


def test_single_repeat_and_reference_display(frame):
    r = audit(frame.iloc[:, :5], frame.label, methods=[OOFAudit(n_splits=3, n_repeats=1)])
    assert r.diagnostics.query("metric_name == 'label_issue_std'").numeric_value.isna().all()
    p = reference_projection(r)
    assert p["fitted_scope"] == "all_data_display_only"
    assert p["coordinates"].shape == (90, 2)
    d = r.apply_policy("oof", ReviewPolicy(min_alternative_agreement=0.5))
    assert d.flags.isna().all().all()


def test_partial_cancellation_invalid_outputs(frame):
    def bad(data):
        raise ValueError("deliberate component error")

    def good(data):
        return {"samples": pd.DataFrame({"sample_id": data.sample_ids})}

    r = audit(
        frame.iloc[:, :5], frame.label, methods=[bind_method(good), bind_method(bad)], continue_on_error=True
    )
    assert r.status == "partial"
    assert r.samples.query("method_id=='bad'").label_issue_score.isna().all()
    assert r.samples.query("method_id=='bad'").status.eq("method_failed").all()
    token = CancellationToken()
    token.cancel()
    r = audit(frame.iloc[:, :5], frame.label, methods=[bind_method(good)], cancellation_token=token)
    assert r.status == "cancelled" and r.samples.status.eq("cancelled").all()

    def wrong(data):
        return {"samples": pd.DataFrame({"sample_id": data.sample_ids[:2]})}

    with pytest.raises(ValidationError, match="exactly one"):
        audit(frame.iloc[:, :5], frame.label, methods=[bind_method(wrong)])


def test_pca_transform_pipeline():
    with pytest.raises(ValueError, match="exceeds"):
        PCAEmbedding(n_components=10).fit(np.ones((3, 2)))
    with pytest.raises(ValueError, match="variance"):
        PCAEmbedding().fit(np.ones((4, 2)))
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 4))
    t = EmbeddingPipeline((Standardize(), PCAEmbedding(2, standardize=False, whiten=True))).fit(X)
    assert t(X).shape == (20, 2)
    assert np.allclose(np.cov(t(X).T), np.eye(2))


def test_bundle_roundtrip_and_export(frame, tmp_path):
    frame.index = pd.Index([f"{i:04}" for i in range(len(frame))], name="sample_id")
    frame.label = frame.label.astype("category")
    r = audit(frame.iloc[:, :5], frame.label, metadata=frame[["recording"]], methods=[NeighborAudit()])
    r.apply_policy("neighbors", ReviewPolicy())
    raw = r.save(tmp_path / "run.audit.zip")
    q = load_run(io.BytesIO(raw))
    for name in ["samples", "overlaps", "diagnostics", "class_evidence"]:
        assert_frame_equal(getattr(r, name), getattr(q, name))
    assert_frame_equal(r.data.X, q.data.X)
    pd.testing.assert_series_equal(r.data.y, q.data.y)
    assert_frame_equal(r.decisions[0].flags, q.decisions[0].flags)
    csv = r.export_csv(
        sample_ids=frame.index[:3], include_decisions=True, include_features=True, include_metadata=True
    )
    assert csv.startswith(b"\xef\xbb\xbf")
    df = pd.read_csv(io.BytesIO(csv), dtype={"sample_id": str})
    assert len(df) == 3 and df.sample_id.iloc[0] == "0000"
    assert "feature.feature_0" in df and "metadata.recording" in df
    assert load_run(io.BytesIO(r.save(include_data=False))).data is None
    with pytest.raises(ValidationError, match="snapshot"):
        load_run(io.BytesIO(r.save(include_data=False))).export_csv(include_features=True)


def test_metadata_strata_and_method_decisions(frame):
    from data_audit.views import metadata_summary, compare_decisions

    r = audit(
        frame.iloc[:, :5],
        frame.label,
        metadata=frame[["recording"]],
        methods=[NeighborAudit(name="first"), NeighborAudit(name="second", radius_mode="none")],
    )
    a = r.apply_policy("first", ReviewPolicy())
    r.apply_policy("second", ReviewPolicy(issue_threshold=0.7))
    summary = metadata_summary(r, "first", "recording", a)
    assert summary.samples.sum() == len(frame)
    comparison = compare_decisions(r)
    assert comparison.shape == (len(frame), 4)
    assert set(comparison.columns) == {
        "first.label_issue",
        "first.overlap",
        "second.label_issue",
        "second.overlap",
    }
