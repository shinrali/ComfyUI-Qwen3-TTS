import unittest

from progress import estimate_generation_seconds, stage_percent


class ProgressTests(unittest.TestCase):
    def test_stage_anchors(self):
        self.assertEqual(stage_percent("queued"), 5.0)
        self.assertEqual(stage_percent("loading_model"), 18.0)
        self.assertEqual(stage_percent("preparing_reference"), 28.0)
        self.assertEqual(stage_percent("validating"), 93.0)
        self.assertEqual(stage_percent("saving"), 95.0)
        self.assertEqual(stage_percent("complete"), 100.0)

    def test_generation_is_estimated_and_capped(self):
        self.assertEqual(stage_percent("generating", elapsed=0, estimate=20), 35.0)
        self.assertEqual(stage_percent("generating", elapsed=10, estimate=20), 63.5)
        self.assertEqual(stage_percent("generating", elapsed=200, estimate=20), 92.0)

    def test_estimate_scales_with_text_and_candidates(self):
        short = estimate_generation_seconds("design", text_length=90, candidate_count=1)
        candidates = estimate_generation_seconds("design", text_length=90, candidate_count=3)
        long_clone = estimate_generation_seconds("clone", text_length=360)
        self.assertEqual(short, 12.0)
        self.assertGreater(candidates, short)
        self.assertEqual(long_clone, 40.0)


if __name__ == "__main__":
    unittest.main()
