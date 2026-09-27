import hashlib
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from q2.missing import augmentation_mask, select_positions
from q2v2.model import PretrainedMultimodal
from q2.joint_study import decision


class JointMissingTests(unittest.TestCase):
    def test_original_augmentation_golden_masks_unchanged(self):
        valid = np.arange(50)[None, :] < np.arange(1, 51)[:, None]
        valid[:, 0] = False
        # Golden values from the preceding committed implementation.
        expected = {
            "span": "0febc00769d1c13917bfca1e302f7bc162d31d62ede70878b7aee19a082c5119",
            "scattered": "fd94f07bb53516eab3fb522fbcf121c2d90ad7d13c4afc206c813b9379a89423",
            "none": "d911be33fc2c112fdb782e50cd178f4287892e80839909ad3f013a776aab5401",
        }
        for mode, sha in expected.items():
            actual = augmentation_mask(valid, np.random.default_rng(42), mode, .7)
            self.assertEqual(hashlib.sha256(actual.tobytes()).hexdigest(), sha)

    def test_short_masks_exact_count_and_run_boundaries(self):
        for n in range(1, 50):
            valid = np.zeros(50, dtype=bool)
            valid[1:n + 1] = True
            for rate in (.1, .2, .3, .4, .5):
                pos = select_positions(valid, rate, "short", np.random.default_rng(n))
                self.assertEqual(len(set(pos)), max(1, int(np.floor(n * rate + .5))))
                self.assertTrue(valid[pos].all())
                runs = np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1)
                self.assertLessEqual(max(map(len, runs)), 4)

    def test_mixture_contains_joint_single_and_clean_rows(self):
        valid = np.ones((1000, 50), dtype=bool)
        valid[:, [0, 49]] = False
        drop = augmentation_mask(valid, np.random.default_rng(42), "mixed_joint")
        self.assertFalse(drop[~valid].any())
        modalities = drop.any(1).sum(1)
        self.assertEqual(set(modalities.tolist()), {0, 1, 3})
        for row in drop[modalities == 3]:
            np.testing.assert_array_equal(row[:, 0], row[:, 1])
            np.testing.assert_array_equal(row[:, 0], row[:, 2])
        empty = augmentation_mask(np.zeros((2, 50), bool), np.random.default_rng(0), "mixed_joint", 1.)
        self.assertFalse(empty.any())

    def test_training_drop_matches_masked_placeholder_inference(self):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        model = PretrainedMultimodal({"layers": [0], "multimodal": True},
                                     [0, 100, 101, 102, 103, 2001, 2002]).eval()
        tokens = torch.tensor([[101, 2001, 2002, 102] + [0] * 46])
        valid = torch.zeros((1, 50), dtype=torch.bool)
        valid[:, 1:3] = True
        obs = valid[:, :, None].expand(-1, -1, 3).clone()
        audio, vision = torch.randn(1, 50, 74), torch.randn(1, 50, 35)
        drop = torch.zeros_like(obs)
        drop[:, 1, :] = True
        with torch.no_grad():
            trained = model(tokens, audio, vision, valid, obs, drop=drop)
            tokens[:, 1] = 100
            audio[:, 1] = 0
            vision[:, 1] = 0
            observed = obs & ~drop
            predicted = model(tokens, audio, vision, valid, observed)
        for a, b in zip(trained, predicted):
            self.assertTrue(torch.equal(a, b))

    def test_heavy_only_improvement_cannot_pass_selection(self):
        protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/joint_missing_study.json").read_text())
        base = {g: {"accuracy": .6, "macro_f1": .6, "mae": .6, "pearson": .6}
                for g in ("clean", "single", "low_joint", "medium_joint", "heavy_joint", "low_medium")}
        base["score"] = .75
        candidate = copy.deepcopy(base)
        candidate["heavy_joint"]["macro_f1"] += .2
        result = decision([base] * 3, [candidate] * 3, protocol)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["mean_score"])
        candidate["score"] += .01
        self.assertTrue(decision([base] * 3, [candidate] * 3, protocol)["passed"])
        candidate["clean"]["mae"] += .021
        self.assertFalse(decision([base] * 3, [candidate] * 3, protocol)["passed"])
