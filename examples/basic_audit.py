"""Run from any directory after installing data-audit. Outputs stay in the chosen directory."""

import argparse
from pathlib import Path
from data_audit import audit, load_csv, PCAEmbedding, NeighborAudit, OOFAudit, ReviewPolicy
from data_audit.views import demo_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("audit-output"))
    args = parser.parse_args()
    frame = load_csv(args.csv, id_column="sample_id") if args.csv else demo_data()
    features = [c for c in frame if c.startswith("feature_")]
    result = audit(
        frame[features],
        frame["label"],
        embedding=PCAEmbedding(2, whiten=True),
        methods=[NeighborAudit(), OOFAudit(n_splits=3, n_repeats=2)],
    )
    for method in ["neighbors", "oof"]:
        result.apply_policy(method, ReviewPolicy())
    args.output.mkdir(parents=True, exist_ok=True)
    result.export_csv(args.output / "findings.csv", include_decisions=True)
    result.save(args.output / "run.audit.zip")
    print(result.samples.groupby("method_id").size())
    print("Run:", result.run_id, "Status:", result.status)


if __name__ == "__main__":
    main()
