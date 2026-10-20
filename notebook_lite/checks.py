"""Run with the local Python: python checks.py. No project/package imports."""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from analysis import neighbors, oof
from explore import review_flags, load_csv
from representations import Standardize


class AnalysisChecks(unittest.TestCase):
    def setUp(self):
        a, b = make_classification(n_samples=60, n_features=4, n_redundant=0, random_state=42)
        self.X = pd.DataFrame(a, index=[f"{i:03d}" for i in range(60)])
        self.y = pd.Series(b, index=self.X.index)

    def test_neighbors_exclude_self_and_preserve_duplicate_features(self):
        self.X.iloc[1] = self.X.iloc[0]
        r = neighbors(self.X, self.y, n_neighbors=5, radius_mode="none", min_neighbors=3)
        evidence = r["neighbors"]
        self.assertFalse((evidence.query_sample_id == evidence.neighbor_sample_id).any())
        duplicate = evidence.loc[(evidence.query_sample_id == "000") & (evidence.neighbor_sample_id == "001")]
        self.assertEqual(duplicate.distance.iloc[0], 0)

    def test_radius_and_abstention(self):
        r = neighbors(self.X, self.y, n_neighbors=5, radius_mode="fixed", max_radius=0)
        flags = review_flags(r["samples"])
        self.assertTrue(flags.isna().all().all())
        with self.assertRaises(ValueError):
            neighbors(self.X.iloc[:3], self.y.iloc[:3], n_neighbors=5)

    def test_grouped_oof_has_no_group_leakage_and_fold_local_fit(self):
        groups = pd.Series(np.arange(60) // 2, index=self.X.index)
        fitted_rows = []

        class RecordFit:
            def fit(inner, train):
                fitted_rows.append(len(train))
                return Standardize().fit(train)

        r = oof(self.X, self.y, groups=groups, embedding=RecordFit(), n_splits=3, n_repeats=2)
        self.assertEqual(len(fitted_rows), 6)
        self.assertTrue(all(0 < n < len(self.X) for n in fitted_rows))
        for split in r["run_info"]["splits"]:
            train, test = split["train_ids"], split["test_ids"]
            self.assertFalse(set(train) & set(test))
            self.assertFalse(set(groups.loc[train]) & set(groups.loc[test]))
        q = r["class_evidence"].groupby("sample_id").class_score.sum()
        np.testing.assert_allclose(q, 1)

    def test_alignment_and_missing_data_fail_explicitly(self):
        with self.assertRaises(ValueError):
            oof(self.X, self.y.iloc[::-1])
        self.X.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            neighbors(self.X, self.y)

    def test_one_repeat_dispersion_is_missing(self):
        r = oof(self.X, self.y, n_splits=3, n_repeats=1)
        self.assertTrue(r["samples"].label_issue_std.isna().all())

    def test_ids_preserve_leading_zeros(self):
        from io import StringIO

        f = load_csv(StringIO("id,x,label\n001,2,a\n002,3,b\n"), id_column="id")
        self.assertEqual(f.index.tolist(), ["001", "002"])


if __name__ == "__main__":
    unittest.main()
