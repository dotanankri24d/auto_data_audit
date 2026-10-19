import numpy as np
import pytest
from data_audit import AutoencoderEmbedding, audit, OOFAudit


def test_autoencoder_fit_reproducible_and_fold_scoped():
    pytest.importorskip("torch")
    X = np.random.default_rng(4).normal(size=(36, 4))
    model = AutoencoderEmbedding(bottleneck=2, epochs=3, batch_size=12, patience=2, random_state=8)
    a, b = model.fit(X), model.fit(X)
    assert a(X).shape == (36, 2)
    assert np.isfinite(a(X)).all()
    assert np.allclose(a(X), b(X), atol=1e-8)
    r = audit(X, ["A", "B"] * 18, embedding=model, methods=[OOFAudit(n_splits=3, n_repeats=1)])
    assert r.status == "completed"
