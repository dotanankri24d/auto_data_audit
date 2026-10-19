"""Training-scoped, composable representations; importing never fits a model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from ..contracts import ValidationError


class CheckedTransform:
    def __init__(self, transform, n_features, names, output_features):
        self._transform = transform
        self.n_features = n_features
        self.input_names = tuple(names)
        self.feature_names = tuple(f"z{i}" for i in range(output_features))

    def __call__(self, X):
        if isinstance(X, pd.DataFrame) and tuple(X.columns) != self.input_names:
            raise ValidationError("transform: feature names/order differ from training")
        a = np.asarray(X, dtype=float)
        if a.ndim != 2 or a.shape[1] != self.n_features or not np.isfinite(a).all():
            raise ValidationError("transform: wrong feature count or nonfinite input")
        out = np.asarray(self._transform(a), dtype=float)
        if out.ndim != 2 or out.shape != (len(a), len(self.feature_names)) or not np.isfinite(out).all():
            raise ValidationError("transform must preserve rows and return a finite matrix with consistent width")
        return out

    transform = __call__


def fit_transformer(component, X, feature_names):
    component = component or Identity()
    fitted = component.fit(np.array(X, dtype=float, copy=True))
    transform = fitted.transform if hasattr(fitted, "transform") else fitted
    if not callable(transform):
        raise ValidationError("embedding.fit must return a callable or an object with transform")
    probe = np.asarray(transform(np.array(X, copy=True)), dtype=float)
    if probe.ndim != 2 or probe.shape[0] != len(X) or not probe.shape[1] or not np.isfinite(probe).all():
        raise ValidationError("embedding: transform must preserve rows and return a nonempty finite matrix")
    return CheckedTransform(transform, X.shape[1], feature_names, probe.shape[1])


@dataclass(frozen=True)
class Identity:
    """Use the original numeric features."""
    def fit(self, X):
        return lambda x: np.array(x, dtype=float, copy=True)


@dataclass(frozen=True)
class Standardize:
    """Center and scale features using training statistics only."""
    epsilon: float = 1e-8

    def __post_init__(self):
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")

    def fit(self, X):
        mean = X.mean(axis=0)
        scale = np.maximum(X.std(axis=0), self.epsilon)
        return lambda x: (x - mean) / scale


@dataclass(frozen=True)
class PCAEmbedding:
    """Select a component count or explained-variance fraction, with optional whitening."""
    n_components: int | float = 2
    standardize: bool = True
    whiten: bool = False
    random_state: int = 42

    def __post_init__(self):
        n = self.n_components
        if isinstance(n, bool) or not ((type(n) is int and n >= 1) or (type(n) is float and 0 < n < 1)):
            raise ValueError("n_components must be a positive integer or variance fraction between 0 and 1")

    def fit(self, X):
        if isinstance(self.n_components, int) and self.n_components > min(X.shape):
            raise ValueError(f"PCA n_components={self.n_components} exceeds training limit {min(X.shape)}")
        if len(X) < 2 or not np.any(np.var(X, axis=0) > 0):
            raise ValueError("PCA requires at least two samples and nonzero variance")
        norm = Standardize().fit(X) if self.standardize else Identity().fit(X)
        model = PCA(n_components=self.n_components, whiten=self.whiten,
                    svd_solver="full", random_state=self.random_state).fit(norm(X))
        if self.whiten and np.any(model.explained_variance_ <= np.finfo(float).eps):
            raise ValueError("PCA whitening requires nonzero variance in every retained component")
        return lambda x: model.transform(norm(x))


@dataclass(frozen=True)
class EmbeddingPipeline:
    steps: tuple

    def fit(self, X):
        transforms = []
        z = X
        for step in self.steps:
            t = fit_transformer(step, z, tuple(range(z.shape[1])))
            z = t(z)
            transforms.append(t)

        def transform(x):
            for t in transforms:
                x = t(x)
            return x
        return transform


@dataclass(frozen=True)
class AutoencoderEmbedding:
    """Optional CPU autoencoder; validation and early stopping stay inside training data."""
    bottleneck: int = 2
    epochs: int = 50
    batch_size: int = 64
    regularization: float = 1e-4
    patience: int = 5
    validation_fraction: float = 0.2
    random_state: int = 42
    activation: Literal["relu", "tanh"] = "relu"

    def __post_init__(self):
        if min(self.bottleneck, self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("bottleneck, epochs, batch_size and patience must be positive")
        if self.regularization < 0 or not 0 < self.validation_fraction < 0.5:
            raise ValueError("regularization must be nonnegative; validation_fraction must be in (0, 0.5)")
        if self.activation not in ("relu", "tanh"):
            raise ValueError("activation must be relu or tanh")

    def fit(self, X):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install data-audit[ae] to use AutoencoderEmbedding") from exc
        if len(X) < 5:
            raise ValueError("Autoencoder needs at least five training samples")
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(X))
        n_val = max(1, int(len(X) * self.validation_fraction))
        train, val = X[order[n_val:]], X[order[:n_val]]
        norm = Standardize().fit(train)
        train = torch.tensor(norm(train), dtype=torch.float32)
        val = torch.tensor(norm(val), dtype=torch.float32)
        # fork_rng keeps the caller's random state intact.
        with torch.random.fork_rng():
            torch.manual_seed(self.random_state)
            hidden = max(self.bottleneck * 2, X.shape[1])
            act = torch.nn.ReLU if self.activation == "relu" else torch.nn.Tanh
            encoder = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), act(),
                                          torch.nn.Linear(hidden, self.bottleneck))
            model = torch.nn.Sequential(encoder, torch.nn.Linear(self.bottleneck, hidden), act(),
                                        torch.nn.Linear(hidden, X.shape[1]))
            optim = torch.optim.Adam(model.parameters(), weight_decay=self.regularization)
            best, stale, state = float("inf"), 0, None
            for _ in range(self.epochs):
                model.train()
                indices = torch.randperm(len(train))
                for start in range(0, len(train), self.batch_size):
                    batch = train[indices[start:start + self.batch_size]]
                    optim.zero_grad()
                    loss = torch.nn.functional.mse_loss(model(batch), batch)
                    loss.backward()
                    optim.step()
                model.eval()
                with torch.no_grad():
                    score = torch.nn.functional.mse_loss(model(val), val).item()
                if score < best:
                    best, stale = score, 0
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
            model.load_state_dict(state)
            encoder.eval()

        def transform(x):
            with torch.no_grad():
                return encoder(torch.tensor(norm(x), dtype=torch.float32)).numpy()
        return transform
