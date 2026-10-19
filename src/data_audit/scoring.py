"""Versioned ranking definitions. These scores are not error probabilities."""
import numpy as np
import pandas as pd

SCORE_DEFINITION = "class-evidence-margin-and-pair-min-v1"


def score_evidence(ids, y, classes, q, support, support_kind):
    samples, overlaps, evidence = [], [], []
    for i, (sid, original) in enumerate(zip(ids, y)):
        row = dict(sample_id=sid, original_label=original,
                   sufficient_support=support[i], support_kind=support_kind, status="completed",
                   score_definition_id=SCORE_DEFINITION)
        if len(classes) < 2 or not np.isfinite(q[i]).all():
            row.update(label_issue_score=np.nan, alternative_label=None, max_overlap_score=np.nan,
                       overlap_class_a=None, overlap_class_b=None)
            if len(classes) < 2:
                row.update(status="unsupported", reason="At least two categories are required")
        else:
            # Stable sorting breaks ties using documented first-seen class order.
            order = np.argsort(-q[i], kind="stable")
            original_pos = classes.index(original)
            alt = next(j for j in order if j != original_pos)
            a, b = order[:2]
            row.update(label_issue_score=max(0.0, float(q[i, alt] - q[i, original_pos])),
                       alternative_label=classes[alt], max_overlap_score=2 * float(min(q[i, a], q[i, b])),
                       overlap_class_a=classes[a], overlap_class_b=classes[b])
            pairs = {tuple(sorted((original_pos, j))) for j in range(len(classes)) if j != original_pos}
            pairs.add(tuple(sorted((int(a), int(b)))))
            for ca, cb in sorted(pairs):
                overlaps.append(dict(sample_id=sid, class_a=classes[ca], class_b=classes[cb],
                                     pair_overlap=2 * float(min(q[i, ca], q[i, cb]))))
        samples.append(row)
        for j, c in enumerate(classes):
            evidence.append(dict(sample_id=sid, class_label=c, class_score=q[i, j], source=support_kind))
    out = pd.DataFrame(samples)
    out["sufficient_support"] = out.sufficient_support.astype("boolean")
    return {"samples": out,
            "overlaps": pd.DataFrame(overlaps, columns=["sample_id", "class_a", "class_b", "pair_overlap"]),
            "class_evidence": pd.DataFrame(evidence)}
