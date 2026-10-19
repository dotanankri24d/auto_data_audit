import numpy as np
import pandas as pd


def analyze(data, power=2.0):
    """Distance from the reference center; not a label-issue score."""
    if power <= 0:
        raise ValueError("power must be positive")
    transform = data.fit_embedding(np.arange(len(data.X)))
    Z = transform(data.X)
    return {
        "samples": pd.DataFrame(
            {
                "sample_id": data.sample_ids,
                "distance_from_center": np.linalg.norm(Z - Z.mean(axis=0), axis=1) ** power,
            }
        )
    }
