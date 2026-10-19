import pandas as pd


def decide(samples, threshold=0.7):
    """Threshold a label-issue score, abstaining when support is absent."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    flags = samples["label_issue_score"].ge(threshold).astype("boolean")
    flags = flags.where(samples["sufficient_support"].fillna(False), pd.NA)
    return pd.DataFrame({"label_issue": flags}, index=samples.index)
