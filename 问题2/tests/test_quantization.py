import unittest
import torch
from q2v2.quantization import encode_tensor, decode_tensor


class QuantizationTests(unittest.TestCase):
    def test_roundtrip_odd_shapes_and_zeros(self):
        torch.manual_seed(13)
        for bits in [4, 8]:
            for group in [1, 7, 128]:
                source = torch.randn(3, 17)
                source[0] = 0
                record = encode_tensor(source, bits, group)
                restored = decode_tensor(record)
                self.assertEqual(restored.shape, source.shape)
                self.assertTrue(torch.isfinite(restored).all())
                self.assertTrue(torch.equal(restored[0], source[0]))
                self.assertLessEqual(float((source-restored).abs().max()), float(record["scale"].max())*.501)

    def test_reject_invalid_format(self):
        for bits in [0, 3, 16]:
            with self.assertRaises(ValueError):
                encode_tensor(torch.zeros(2, 2), bits)


if __name__ == "__main__":
    unittest.main()
