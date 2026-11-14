"""Mathematical checks for the prespecified FEI-only search operating point."""
import unittest
import numpy as np

from experiments.calibrate_search_v3 import conservative_cutoff, open_set_metrics


class SearchOperatingPointTests(unittest.TestCase):
    def test_cutoff_is_strictly_above_every_impostor(self):
        y = np.array([1, 0, 0, 1])
        scores = np.array([-.2, -.7, -.4, -.3])
        threshold = conservative_cutoff(y, scores)
        self.assertEqual(threshold, np.nextafter(-.4, np.inf))
        self.assertFalse(np.any(scores[y == 0] >= threshold))
        self.assertTrue(np.all(scores[y == 1] >= threshold))

    def test_tied_best_identity_is_not_credited_as_correct(self):
        matrix = np.array([[-.2, -.2], [-.8, -.8]])
        report = open_set_metrics(matrix, ["a", "b"], ["a", "unknown"], -.5, ["a", "b"])
        self.assertEqual(report["fnir"], 1)
        self.assertEqual(report["fpir"], 0)

    def test_open_set_denominators_include_all_known_and_unknown_probes(self):
        matrix = np.array([[-.2, -.8], [-.9, -.3], [-.4, -.7], [-.8, -.9]])
        report = open_set_metrics(matrix, ["a", "b"], ["a", "b", "u1", "u2"], -.25, ["a", "b"])
        self.assertEqual(report["known_probe_count"], 2)
        self.assertEqual(report["unknown_probe_count"], 2)
        self.assertEqual(report["fnir"], .5)
        self.assertEqual(report["fpir"], 0)

    def test_cutoff_requires_genuine_and_impostor_pairs(self):
        with self.assertRaises(ValueError):
            conservative_cutoff([1, 1], [-.2, -.5])


if __name__ == "__main__":
    unittest.main()

