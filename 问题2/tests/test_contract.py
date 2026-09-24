import unittest
import numpy as np
import torch
from q2.data import convert, fit_scaler, normalize
from q2.metrics import metrics
from q2.missing import select_positions, scenario_mask
from q2.model import SentimentModel


def fixture():
    bert = np.zeros((2, 3, 50), dtype=float)
    bert[:, 0, :6] = [101, 2023, 2003, 1037, 2204, 102]
    bert[:, 1, :6] = 1
    audio = np.zeros((2, 50, 74))
    vision = np.zeros((2, 50, 35))
    audio[:, 1:5] = 2
    vision[:, 1:5] = 4
    return dict(text_bert=bert, audio=audio, vision=vision,
                classification_labels=np.array([0., 2.]), regression_labels=np.array([-1., 1.]))


class ContractTests(unittest.TestCase):
    def test_masks_special_padding_and_partial_zero(self):
        raw = fixture()
        raw["audio"][0, 2, 0] = 0
        raw["audio"][0, 3] = 0
        ds, audit = convert(raw, ["a", "b"])
        self.assertEqual(ds["valid"].sum(), 8)
        self.assertTrue(ds["observed"][0, 2, 1])
        self.assertFalse(ds["observed"][0, 3, 1])
        self.assertFalse(ds["observed"][:, [0, 5, 49]].any())

    def test_label_mapping_and_integer_validation(self):
        raw = fixture()
        raw["classification_labels"][0] = 2
        with self.assertRaises(ValueError):
            convert(raw, ["a", "b"])

    def test_neutral_and_unique_ids(self):
        raw = fixture()
        raw["classification_labels"][0] = 1
        raw["regression_labels"][0] = 0
        ds, _ = convert(raw, ["a", "b"])
        self.assertEqual(ds["labels"][0], 1)
        with self.assertRaises(ValueError):
            convert(raw, ["a", "a"])

    def test_nonfinite_feature_does_not_poison_scaler(self):
        raw = fixture()
        raw["vision"][0, 2, 3] = np.inf
        ds, audit = convert(raw, ["a", "b"])
        self.assertFalse(ds["observed"][0, 2, 2])
        self.assertEqual(audit["modalities"]["vision"]["nonfinite_vectors"], 1)
        normalized = normalize(ds, fit_scaler(ds))
        self.assertTrue(np.isfinite(normalized["vision"]).all())
        raw = fixture()
        raw["text_bert"][0, 0, 1] += .1
        with self.assertRaises(ValueError):
            convert(raw, ["a", "b"])

    def test_train_only_stats(self):
        ds, _ = convert(fixture(), ["a", "b"])
        stats = fit_scaler(ds)
        self.assertTrue(np.all(stats["audio_mean"] == 2))
        self.assertTrue(np.all(normalize(ds, stats)["audio"] == 0))

    def test_missing_reproducible_and_exact_count(self):
        ds, _ = convert(fixture(), ["a", "b"])
        a = scenario_mask(ds, "text_50_random")
        self.assertTrue(np.array_equal(a, scenario_mask(ds, "text_50_random")))
        self.assertEqual(a.sum(), 4)
        self.assertFalse(a[:, [0, 5, 49]].any())
        for n in range(1, 51):
            valid = np.arange(50) < n
            for rate in [.1, .3, .5]:
                expected = max(1, int(np.floor(n * rate + .5)))
                for shape in ["front", "middle", "back", "multi", "scattered", "random"]:
                    selected = select_positions(valid, rate, shape, np.random.default_rng(1))
                    self.assertEqual(len(set(selected)), expected)

    def test_metrics_and_constant_pearson(self):
        labels = np.array([0, 1, 2])
        result = metrics(labels, np.array([-1., 0., 1.]), np.eye(3), np.array([-1., 0., 1.]))
        self.assertEqual(result["macro_f1"], 1.)
        self.assertEqual(result["mae"], 0.)
        self.assertAlmostEqual(result["pearson"], 1.)
        result = metrics(labels, np.array([-1., 0., 1.]), np.eye(3), np.zeros(3))
        self.assertIsNone(result["pearson"])
        result = metrics(labels, np.array([-1., 0., 1.]), np.eye(3), np.full(3, .345, dtype=np.float32))
        self.assertIsNone(result["pearson"])

    def test_no_leakage_and_all_unavailable(self):
        ds, _ = convert(fixture(), ["a", "b"])
        args = {k: torch.from_numpy(ds[k]) for k in ["tokens", "audio", "vision", "valid", "observed"]}
        for encoder in ["mean", "gru"]:
            for fusion in ["concat", "gate"]:
                model = SentimentModel(dict(encoder=encoder, fusion=fusion)).eval()
                args["drop"] = torch.ones_like(args["observed"])
                with torch.no_grad():
                    first = model(**args)
                    changed = dict(args)
                    changed["tokens"] = torch.full_like(args["tokens"], 12345)
                    changed["audio"] = torch.full_like(args["audio"], 99)
                    second = model(**changed)
                    for a, b in zip(first, second):
                        self.assertTrue(torch.isfinite(a).all())
                        self.assertTrue(torch.equal(a, b))
                    changed["valid"] = torch.zeros_like(args["valid"])
                    self.assertTrue(all(torch.isfinite(x).all() for x in model(**changed)))


if __name__ == "__main__":
    unittest.main()
