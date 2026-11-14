"""Synthetic checks for development-only data boundaries and failure accounting."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments.development_v4_metrics import measure, prepared_input


class DevelopmentV4Tests(unittest.TestCase):
    def test_rejected_probe_stays_in_operational_denominator(self):
        rows = [
            dict(subject_id="a", split="calibration", role="gallery", condition_index=11),
            dict(subject_id="b", split="calibration", role="gallery", condition_index=11),
            dict(subject_id="a", split="calibration", role="probe", condition_index=12),
            dict(subject_id="b", split="calibration", role="probe", condition_index=12),
        ]
        source = rows + [dict(subject_id="a", split="calibration", role="probe", condition_index=14)]
        features = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        result, _ = measure(features, rows, source)
        self.assertEqual(result["rank1_operational"], 2 / 3)
        self.assertEqual(result["rank1_conditional"], 1.)
        self.assertEqual(result["quality_rejected_probes"], 1)
        self.assertEqual(result["by_condition"]["14"]["rank1_operational"], 0.)

    def test_test_image_is_refused_before_any_cache_open(self):
        with patch("experiments.development_v4_metrics.np.load") as load:
            with self.assertRaisesRegex(ValueError, "Forbidden identity split"):
                prepared_input({"split": "test", "cache_path": "must_not_be_read.npz"})
        load.assert_not_called()

    def test_missing_strict_quality_flag_becomes_zero_mask(self):
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        archive = Archive(patch_names=np.array(["forehead"]),
                          skin_only_forehead=np.full((64, 64, 3), 128, np.uint8),
                          aligned_bgr=np.zeros((224, 224, 3), np.uint8),
                          skin_mask=np.ones((224, 224), np.uint8))
        record = {"split": "train", "cache_path": "synthetic.npz", "quality": {"patch_quality": {"forehead": {}}}}
        with patch("experiments.development_v4_metrics.np.load", return_value=archive):
            prepared = prepared_input(record)
        self.assertEqual(int(prepared[2].sum()), 0)


if __name__ == "__main__":
    unittest.main()
