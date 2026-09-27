"""Paired diagnostics preserve replayable masks and predictions."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from q2.data import convert, digest, fit_scaler
from test_contract import fixture

spec = importlib.util.spec_from_file_location(
    "observation_export", Path(__file__).resolve().parents[1] / "scripts/evaluate_observation_policy.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class ObservationExportTests(unittest.TestCase):
    def test_export_replays_identical_policy_predictions_and_shared_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, valid = fixture(), fixture()
            train["id"] = ["train_a", "train_b"]
            valid["id"] = ["video_a$_$1", "video_b$_$1"]
            aligned, checkpoint, scaler = root / "input.pkl", root / "model.pt", root / "scaler.npz"
            with aligned.open("wb") as f:
                pickle.dump({"train": train, "valid": valid}, f)
            checkpoint.write_bytes(b"mock model: deployment loader is patched")
            converted, _ = convert(train, train["id"])
            np.savez(scaler, **fit_scaler(converted))
            argv = ["evaluate", "--aligned", str(aligned), "--checkpoint", str(checkpoint),
                    "--scaler", str(scaler), "--output", str(root / "report")]
            probabilities = np.array([[.7, .1, .2], [.1, .1, .8]])
            values = np.array([-.2, .7])
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    patch("q2v2.deploy.load_deployment", return_value=(object(), {})), \
                    patch("q2.engine.infer", return_value=(probabilities, values)) as infer:
                script.main()
            self.assertEqual(infer.call_count, 26)
            report = json.loads((root / "report/results.json").read_text())
            self.assertEqual(len(report["predictions"]), 26)
            for record in report["predictions"]:
                self.assertEqual(record["sha256"], digest(root / "report" / record["path"]))
            with np.load(root / "report/selected_positions.npz", allow_pickle=False) as masks:
                np.testing.assert_array_equal(masks["ids"], valid["id"])
                self.assertFalse(masks["clean"].any())
                self.assertEqual(int(masks["joint_50_scattered"].sum()), 4)
            self.assertEqual(report["bootstrap"]["clusters"], 2)
            for row in report["paired_intervals"]:
                self.assertEqual(row["difference"], 0)
                self.assertEqual(row["lower"], 0)
                self.assertEqual(row["upper"], 0)


if __name__ == "__main__":
    unittest.main()
