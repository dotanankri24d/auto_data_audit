"""Reproducible CPU checks; no claim about arbitrary user models or physical UI rendering."""

import argparse
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import plotly.express as px
from threadpoolctl import threadpool_limits
from data_audit import audit, NeighborAudit, OOFAudit, Standardize


def timed(function):
    start = time.perf_counter()
    output = function()
    return output, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/data-audit-benchmark.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    labels = np.arange(10000) % 5
    X = rng.normal(size=(10000, 50))
    X[:, 0] += labels
    stats = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "visible_cpus": os.cpu_count(),
        "thread_limit": 8,
        "seed": 42,
        "analysis_shape": [10000, 50],
        "classes": 5,
    }
    with threadpool_limits(limits=8):
        for method in [NeighborAudit(), OOFAudit()]:
            r, seconds = timed(lambda: audit(X, labels, embedding=Standardize(), methods=[method]))
            assert r.status == "completed" and len(r.samples) == 10000
            stats[method.name + "_seconds"] = seconds
    table = pd.DataFrame(
        {
            "sample_id": [f"sample_{i:06}" for i in range(100000)],
            "label": np.arange(100000) % 5,
            "score": rng.random(100000),
        }
    )
    filtered, seconds = timed(
        lambda: table[table.sample_id.str.contains("100", regex=False) & table.score.ge(0.5)]
    )
    stats["filter_100k_seconds"] = seconds
    stats["filtered_rows"] = len(filtered)
    points = pd.DataFrame(
        {"x": rng.normal(size=20000), "y": rng.normal(size=20000), "label": np.arange(20000) % 5}
    )
    fig, seconds = timed(lambda: px.scatter(points, x="x", y="y", color="label"))
    stats["plotly_20k_figure_seconds"] = seconds
    _, seconds = timed(fig.to_json)
    stats["plotly_20k_json_seconds"] = seconds
    stats["limits"] = (
        "Server-side timings only; browser rendering and selection latency require a real browser benchmark."
    )
    args.output.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
