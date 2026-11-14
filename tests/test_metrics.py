"""Metric regression tests on mathematical fixtures, not biometric validation."""
import unittest
import numpy as np

from experiments.scientific_metrics import (
    _pairwise_scores_neg_l2, _verification_pairs, bootstrap_identity_intervals,
    cmc_from_gallery, compute_cmc, compute_eer, compute_far_frr,
    compute_verification_metrics, cross_session_pairs, select_threshold_at_far,
    verification_from_scores,
)


class MetricsTests(unittest.TestCase):
    def test_perfect_inverted_and_tied_scores(self):
        labels = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(compute_eer(labels, [1, .9, .1, 0])[0], 0)
        self.assertAlmostEqual(compute_eer(labels, [0, .1, .9, 1])[0], 1)
        self.assertAlmostEqual(compute_eer(labels, [1, 1, 1, 1])[0], .5)

    def test_self_comparisons_are_excluded(self):
        x = np.array([[0, 0], [.1, 0], [5, 0], [5.1, 0]])
        labels = np.array(["a", "a", "b", "b"])
        y, scores = _verification_pairs(_pairwise_scores_neg_l2(x), labels)
        self.assertEqual(len(scores), 6)
        self.assertEqual(sum(y), 2)
        metric = compute_verification_metrics(x, labels)
        self.assertEqual(metric.eer, 0)
        self.assertEqual(metric.auc, 1)
        self.assertEqual(metric.threshold_source, "same_set_descriptive_only")

    def test_missing_class_and_nonfinite_scores_raise(self):
        for labels, scores in (([1, 1], [0, 1]), ([0, 0], [0, 1]), ([0, 1], [0, np.nan]),
                               ([0, 2], [0, 1]), ([0, 1], [0])):
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaises(ValueError):
                    compute_eer(labels, scores)

    def test_frozen_calibration_threshold_is_used_on_test(self):
        threshold = select_threshold_at_far([1, 1, 0, 0], [-.1, -.2, -.6, -.7], 0)
        self.assertAlmostEqual(threshold, -.2)
        metric = verification_from_scores([1, 1, 0, 0], [-.15, -.5, -.1, -.8], threshold,
                                          "independent_calibration")
        self.assertEqual(metric.threshold, threshold)
        self.assertEqual(metric.far, .5)
        self.assertEqual(metric.frr, .5)

    def test_equal_threshold_accepts_and_reject_all_is_finite(self):
        self.assertEqual(compute_far_frr([1, 0], [1, 1], 1), (1, 0))
        threshold = select_threshold_at_far([1, 0], [1, 1], 0)
        self.assertTrue(np.isfinite(threshold))
        self.assertGreater(threshold, 1)
        self.assertEqual(compute_far_frr([1, 0], [1, 1], threshold), (0, 1))

    def test_cmc_ranks_identities_not_templates(self):
        cmc = cmc_from_gallery([[.95, .94, .8]], ["wrong", "wrong", "correct"], ["correct"], 3)
        np.testing.assert_array_equal(cmc, [0, 1, 1])

    def test_cmc_conservative_ties_and_missing_gallery(self):
        np.testing.assert_array_equal(cmc_from_gallery([[1, 1]], ["a", "b"], ["a"], 2), [0, 1])
        with self.assertRaises(ValueError):
            cmc_from_gallery([[1, 0]], ["a", "b"], ["missing"])
        with self.assertRaises(ValueError):
            compute_cmc([[0], [1]], ["a", "b"])

    def test_cross_scores_match_direct_distance(self):
        g, p = np.array([[0, 0], [3, 4]]), np.array([[0, 0], [3, 4]])
        y, scores, pi, gi, matrix = cross_session_pairs(g, ["a", "b"], p, ["a", "b"])
        np.testing.assert_array_equal(matrix, [[0, -5], [-5, 0]])
        np.testing.assert_array_equal(y, [1, 0, 0, 1])
        np.testing.assert_array_equal(pi, ["a", "a", "b", "b"])
        np.testing.assert_array_equal(gi, ["a", "b", "a", "b"])
        with self.assertRaises(ValueError):
            cross_session_pairs(g, ["a", "b"], p, ["a", "b"], max_pairs=3)

    def test_identity_bootstrap_reproducible(self):
        args = ([1, 0, 0, 1], [0, -5, -5, 0], ["a", "a", "b", "b"],
                ["a", "b", "a", "b"], -1)
        first = bootstrap_identity_intervals(*args, n_bootstrap=20, seed=7)
        second = bootstrap_identity_intervals(*args, n_bootstrap=20, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["intervals"]["eer"], [0, 0])
        self.assertEqual(first["intervals"]["auc"], [1, 1])

    def test_paired_identity_difference_and_pair_mismatch(self):
        from experiments.compare_methods import compare
        pairs = (np.array([1, 0, 0, 1]), np.array([0, -1, -1, 0]),
                 np.array(["a", "a", "b", "b"]), np.array(["a", "b", "a", "b"]))
        result = compare(pairs, pairs, replicates=10, seed=9)
        self.assertEqual(result["difference"], 0)
        self.assertEqual(result["percentile_95_interval"], [0, 0])
        mismatched = (pairs[0], pairs[1], pairs[3], pairs[2])
        with self.assertRaisesRegex(ValueError, "identical pair ordering"):
            compare(pairs, mismatched, replicates=10)


if __name__ == "__main__":
    unittest.main()

