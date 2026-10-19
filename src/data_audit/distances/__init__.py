"""Distance is independent of neighborhood selection and decision policy."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Euclidean:
    def __call__(self, a, b):
        return float(np.linalg.norm(a - b))


def checked_distance(distance):
    def call(a, b):
        value = distance(a, b)
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError("distance must return a finite nonnegative scalar")
        return float(value)

    return call
