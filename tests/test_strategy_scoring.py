from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from services.strategy_scoring import (
    fuse_prediction_percentiles, mean_percentile, validated_model_percentile,
)


class StrategyScoringTests(unittest.TestCase):
    def test_mean_ignores_nonfinite_and_out_of_range_values(self):
        self.assertEqual(mean_percentile([0.2, np.nan, np.inf, -0.1, 1.1, 0.8]), 0.5)
        self.assertIsNone(mean_percentile([np.nan, np.inf, -1, 2]))

    def test_only_validated_matching_target_models_are_included(self):
        models = [
            {"validated": True, "target_kind": "excess_return_vs_csi300", "prediction_percentile": 0.2},
            {"validated": True, "target_kind": "other", "prediction_percentile": 1.0},
            {"validated": False, "target_kind": "excess_return_vs_csi300", "prediction_percentile": 0.9},
            {"validated": True, "target_kind": "excess_return_vs_csi300", "prediction_percentile": 0.8},
        ]
        self.assertEqual(validated_model_percentile(models), 0.5)

    def test_fusion_uses_union_and_available_percentiles(self):
        date = pd.Timestamp("2026-01-05")
        frames = {
            "A": pd.DataFrame({"date": [date, date], "code": ["a", "b"],
                               "prediction_percentile": [0.0, 0.5]}),
            "B": pd.DataFrame({"date": [date, date], "code": ["b", "c"],
                               "prediction_percentile": [0.8, 1.0]}),
        }
        fused = fuse_prediction_percentiles(frames, ("A", "B")).set_index("code")
        self.assertEqual(set(fused.index), {"a", "b", "c"})
        self.assertEqual(fused.loc["a", "score"], 0.0)
        self.assertEqual(fused.loc["b", "score"], 0.65)
        self.assertEqual(fused.loc["c", "score"], 1.0)

    def test_duplicate_model_keys_are_rejected(self):
        date = pd.Timestamp("2026-01-05")
        frames = {
            "A": pd.DataFrame({"date": [date, date], "code": ["a", "a"],
                               "prediction_percentile": [0.1, 0.9]}),
            "B": pd.DataFrame({"date": [date], "code": ["a"], "prediction_percentile": [0.5]}),
        }
        with self.assertRaises(pd.errors.MergeError):
            fuse_prediction_percentiles(frames, ("A", "B"))


if __name__ == "__main__":
    unittest.main()
