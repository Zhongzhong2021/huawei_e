"""The recipe runner must not fit preprocessing on validation or access test."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from test_contract import fixture

spec = importlib.util.spec_from_file_location(
    "retrain_seed42", Path(__file__).resolve().parents[1] / "scripts/retrain_seed42.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RecipeDataTests(unittest.TestCase):
    def raw(self):
        train, valid = fixture(), fixture()
        train["id"] = ["train_a", "train_b"]
        valid["id"] = ["valid_a", "valid_b"]
        valid["audio"][:, 1:5] = 20
        valid["vision"][:, 1:5] = 40
        # A poison value ensures the runner cannot require/process a test split.
        return {"train": train, "valid": valid, "test": object()}

    def test_only_training_fits_scaler_and_test_is_unused(self):
        data, stats, audits = runner.fitting_data(self.raw())
        self.assertEqual(set(data), {"train", "valid"})
        np.testing.assert_array_equal(stats["audio_mean"], np.full(74, 2))
        np.testing.assert_array_equal(stats["vision_mean"], np.full(35, 4))
        self.assertEqual(audits["valid"]["n"], 2)
        self.assertEqual(data["valid"]["audio"][0, 1, 0], 10)

    def test_split_overlap_rejected_before_scaler_fit(self):
        raw = self.raw()
        raw["valid"]["id"][0] = "train_a"
        with patch.object(runner, "fit_scaler") as fit:
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                runner.fitting_data(raw)
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
