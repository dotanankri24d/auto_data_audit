from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / "src/data_audit/ui/app.py"


def by_label(widgets, label):
    return next(w for w in widgets if w.label == label)


def test_demo_run_policy_filter_export_and_review():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    by_label(at.button, "Use synthetic demonstration dataset").click().run()
    assert not at.exception
    assert by_label(at.multiselect, "Numeric feature columns").value
    by_label(at.button, "Run audit").click().run()
    job = at.session_state["job"]
    if job is not None:
        job.future.result(timeout=30)
        at.run()
    assert not at.exception
    assert "result" in at.session_state
    assert at.session_state["result"].status == "completed"
    assert not list(at.error)
    run_id = at.session_state["result"].run_id
    by_label(at.number_input, "issue_threshold").set_value(0.8).run()
    assert at.session_state["result"].run_id == run_id
    by_label(at.text_input, "Search sample ID").set_value("sample_000").run()
    assert not at.exception and not list(at.error)
    by_label(at.text_area, "Review note").set_value("Checked evidence").run()
    by_label(at.button, "Save review").click().run()
    assert at.session_state["result"].reviews.iloc[0]["note"] == "Checked evidence"
    by_label(at.selectbox, "CSV export scope").select("Filtered rows").run()
    assert not at.exception and not list(at.error)
    by_label(at.selectbox, "Display space").select("Display PCA").run()
    assert not at.exception and not list(at.error)


def test_pca_variance_and_extension_forms(tmp_path):
    (tmp_path / "embeddings").mkdir()
    (tmp_path / "embeddings" / "scale.py").write_text(
        "def fit(X, epsilon=1e-8):\n    return lambda x:x/(1+epsilon)\n"
    )
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    by_label(at.text_input, "Trusted extension directory").set_value(str(tmp_path)).run()
    by_label(at.button, "Load / refresh extensions").click().run()
    by_label(at.button, "Use synthetic demonstration dataset").click().run()
    by_label(at.selectbox, "Embedding").select("Extension: scale").run()
    assert by_label(at.number_input, "epsilon").value == 1e-8
    by_label(at.selectbox, "Embedding").select("PCA").run()
    by_label(at.radio, "PCA component selection").set_value("Explained variance").run()
    assert by_label(at.number_input, "variance_fraction").value == 0.95
    assert not at.exception and not list(at.error)
