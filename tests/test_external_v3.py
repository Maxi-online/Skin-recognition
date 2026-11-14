"""Synthetic-only external evaluator tests: no GT images, features or scores."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.external_evaluate_v3 import (
    conservative_ranks, load_freeze, load_protocol, open_set_decisions,
    load_prepared_cache, load_search_operating_point, rate_identity_bootstrap, run, save_artifacts, score_matrix, sha256,
    summarize_protocol, write_json,
)


def small_protocol():
    by_path = {f"{identity}_{role}": {"subject_id": identity} for identity in ("a", "b", "c", "d") for role in ("g", "p")}
    protocol = {"closed_set": {"gallery_paths": [f"{i}_g" for i in "abcd"], "probe_paths": [f"{i}_p" for i in "abcd"]},
                "open_set": {"enrolled_identities": ["a", "b"], "unknown_identities": ["c", "d"],
                             "gallery_paths": ["a_g", "b_g"], "known_probe_paths": ["a_p", "b_p"],
                             "unknown_probe_paths": ["c_p", "d_p"]}}
    features = {f"{identity}_{role}": np.eye(4, dtype=np.float32)[index] for index, identity in enumerate("abcd") for role in ("g", "p")}
    return protocol, by_path, features


class ExternalMetricsTests(unittest.TestCase):
    def test_perfect_fixed_model_with_disjoint_unknowns(self):
        protocol, rows, features = small_protocol()
        report, numeric, decisions = summarize_protocol(protocol, rows, features, -.5, n_bootstrap=30)
        self.assertEqual(report["closed_set"]["rank1_operational"], 1)
        self.assertEqual(report["verification_conditional"]["genuine_pairs"], 4)
        self.assertEqual(report["verification_conditional"]["impostor_pairs"], 12)
        self.assertEqual(report["verification_conditional"]["eer"], 0)
        self.assertEqual(report["open_set"]["fpir_operational"], 0)
        self.assertEqual(report["open_set"]["fnir_operational"], 0)
        np.testing.assert_array_equal(numeric["ranks"], [1, 1, 1, 1])

    def test_preprocessing_failures_preserve_operational_denominators(self):
        protocol, rows, features = small_protocol()
        del features["a_p"]
        del features["b_g"]
        del features["c_p"]
        report, numeric, decisions = summarize_protocol(protocol, rows, features, -.5, n_bootstrap=30)
        self.assertEqual(report["closed_set"]["operational_denominator"], 4)
        self.assertEqual(report["closed_set"]["conditional_denominator"], 1)
        self.assertEqual(report["closed_set"]["rank1_operational"], .25)
        self.assertEqual(report["closed_set"]["rank1_conditional"], 1)
        self.assertEqual(report["open_set"]["known_probe_count"], 2)
        self.assertEqual(report["open_set"]["unknown_probe_count"], 2)
        self.assertEqual(report["open_set"]["fnir_operational"], 1)
        self.assertEqual(report["open_set"]["fpir_operational"], 0)
        self.assertEqual(report["open_set"]["fnir_conditional_denominator"], 0)
        self.assertEqual(report["open_set"]["fpir_conditional_denominator"], 1)

    def test_all_failures_have_explicit_unavailable_pair_metrics(self):
        protocol, rows, _ = small_protocol()
        report, numeric, _ = summarize_protocol(protocol, rows, {}, -.5, n_bootstrap=10)
        self.assertEqual(report["verification_conditional"]["status"], "unavailable")
        self.assertEqual(report["closed_set"]["rank1_operational"], 0)
        self.assertIsNone(report["closed_set"]["rank1_conditional"])
        self.assertEqual(report["open_set"]["fnir_operational"], 1)
        self.assertEqual(report["open_set"]["fpir_operational"], 0)
        self.assertFalse(numeric["valid_comparisons"].any())
        json.dumps(report, allow_nan=False)

    def test_ties_are_conservative_and_not_arbitrary_identity_selection(self):
        scores = np.array([[-.2, -.2], [-.2, -.5], [-.6, -.5]])
        valid = np.ones_like(scores, dtype=bool)
        ranks = conservative_ranks(scores, valid, ["a", "b"], ["a", "a", "c"])
        np.testing.assert_array_equal(ranks, [2, 1, 0])
        result = open_set_decisions(scores, valid, ["a", "b"], ["a", "a", "c"], ["a", "b"], -.5)
        np.testing.assert_array_equal(result["accepted"], [False, True, True])
        np.testing.assert_array_equal(result["correct"], [False, True, False])
        self.assertEqual(result["reason"][0], "tied_maximum")
        self.assertEqual(result["reason"][2], "unknown_false_identification")

    def test_invalid_zero_storage_is_never_compared_as_high_similarity(self):
        matrix, valid, _, _ = score_matrix({"g": np.array([1., 0.]), "p": np.array([0., 1.])}, ["g", "missing"], ["p", "failed"])
        self.assertAlmostEqual(matrix[0, 0], -np.sqrt(2))
        self.assertEqual(matrix[0, 1], 0)
        result = open_set_decisions(matrix, valid, ["a", "b"], ["u", "a"], ["a", "b"], -.5)
        self.assertFalse(result["accepted"].any())

    def test_bootstrap_resamples_identities_and_is_reproducible(self):
        first = rate_identity_bootstrap([0, 0, 0, 1, 1, 1], ["a"] * 3 + ["b"] * 3, 200, 42)
        second = rate_identity_bootstrap([0, 0, 0, 1, 1, 1], ["a"] * 3 + ["b"] * 3, 200, 42)
        self.assertEqual(first, second)
        self.assertEqual(first["interval"], [0, 1])
        self.assertEqual(first["identity_count"], 2)

    def test_two_frozen_thresholds_change_acceptance_without_changing_scores(self):
        matrix = np.array([[-.3, -.8], [-.7, -.6], [-.8, -.7]])
        original = matrix.copy()
        valid = np.ones_like(matrix, bool)
        primary = open_set_decisions(matrix, valid, ["a", "b"], ["a", "b", "u"], ["a", "b"], -.75)
        strict = open_set_decisions(matrix, valid, ["a", "b"], ["a", "b", "u"], ["a", "b"], -.5)
        np.testing.assert_array_equal(primary["accepted"], [True, True, True])
        np.testing.assert_array_equal(strict["accepted"], [True, False, False])
        np.testing.assert_array_equal(original, matrix)

    def test_safe_numeric_artifacts_and_missing_score_csv(self):
        protocol, rows, features = small_protocol()
        del features["a_p"]
        report, numeric, decisions = summarize_protocol(protocol, rows, features, -.5, n_bootstrap=10)
        with TemporaryDirectory() as directory:
            save_artifacts(directory, protocol, rows, features, report, numeric, decisions)
            for filename in ("embeddings.npz", "scores.npz"):
                with np.load(Path(directory) / filename, allow_pickle=False) as bundle:
                    self.assertTrue(all(bundle[name].dtype.kind in "biuf" for name in bundle.files))
            with (Path(directory) / "scores.csv").open(newline="", encoding="utf-8") as stream:
                pairs = list(csv.DictReader(stream))
            self.assertEqual(len(pairs), 16)
            failed = [row for row in pairs if row["probe_path"] == "a_p"]
            self.assertTrue(all(row["score_negative_L2"] == "" and row["comparison_available"] == "0" for row in failed))
            self.assertTrue((Path(directory) / "external_evaluation.pdf").is_file())


class ExternalFreezeTests(unittest.TestCase):
    def test_prepared_cache_hashes_contract_and_preprocessing_failure_scope(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache.npz"
            np.savez_compressed(cache, aligned_bgr=np.zeros((32, 32, 3), np.uint8))
            rows = [{"path": "synthetic/image.jpg", "sha256": "a" * 64}]
            records = [{**rows[0], "status": "rejected", "preprocessing_status": "accepted", "rejection_codes": ["invalid_or_zero_embedding"],
                        "preprocessing_id": "synthetic", "cache_path": "cache.npz", "cache_sha256": sha256(cache)}]
            write_json(root / "manifest.json", records)
            loaded = load_prepared_cache(root / "manifest.json", rows, "synthetic")
            self.assertEqual(loaded[rows[0]["path"]]["preprocessing_status"], "accepted")
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                load_prepared_cache(root / "manifest.json", rows, "different")
            cache.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_prepared_cache(root / "manifest.json", rows, "synthetic")

    def test_secondary_point_requires_same_model_and_hashed_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "original.json", {"synthetic": True})
            point = {"source_freeze_path": "original.json", "source_freeze_sha256": sha256(root / "original.json"),
                     "model_sha256": "a" * 64, "preprocessing_id": "synthetic", "feature_space_id": "fixture",
                     "score_definition": "negative_euclidean_distance", "score_threshold": -.4}
            write_json(root / "point.json", point)
            freeze = {"search_operating_point_path": "point.json", "search_operating_point_sha256": sha256(root / "point.json"),
                      "model_sha256": "a" * 64, "preprocessing_id": "synthetic", "feature_space_id": "fixture"}
            self.assertEqual(load_search_operating_point(freeze, root)["score_threshold"], -.4)
            freeze["model_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "another model"):
                load_search_operating_point(freeze, root)

    def make_freeze(self, root):
        names = ["experiments/external_evaluate_v3.py", "experiments/development_v3.py", "experiments/scientific_metrics.py",
                 "src/utils/data_loader.py", "src/preprocessing_v3.py"]
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic implementation", encoding="utf-8")
        (root / "model.npz").write_bytes(b"synthetic model bytes; never deserialized")
        write_json(root / "protocol.json", {"synthetic_metadata": True})
        freeze = {"model_path": "model.npz", "model_sha256": sha256(root / "model.npz"),
                  "preprocessing_id": "synthetic-strict", "preprocessing_mode": "landmark_strict", "score_threshold": -.5,
                  "protocol_path": "protocol.json", "protocol_sha256": sha256(root / "protocol.json"),
                  "source_hashes": {name: sha256(root / name) for name in names},
                  "timestamp": datetime.now(timezone.utc).isoformat()}
        write_json(root / "freeze.json", freeze)
        return freeze

    def test_missing_required_freeze_is_rejected_before_external_image_loading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "freeze.json", {})
            with patch("experiments.external_evaluate_v3.read_image") as read:
                with self.assertRaisesRegex(ValueError, "Incomplete freeze"):
                    run(root / "freeze.json", root / "output", root)
                read.assert_not_called()

    def test_hash_tampering_rejected_before_external_image_loading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_freeze(root)
            load_freeze(root / "freeze.json", root)
            (root / "model.npz").write_bytes(b"modified")
            with patch("experiments.external_evaluate_v3.read_image") as read:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch: model"):
                    run(root / "freeze.json", root / "output", root)
                read.assert_not_called()

    def test_changed_implementation_or_nonfinite_threshold_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = self.make_freeze(root)
            (root / "experiments/development_v3.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_freeze(root / "freeze.json", root)
            freeze["score_threshold"] = "NaN"
            write_json(root / "freeze.json", freeze)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_freeze(root / "freeze.json", root)

    def test_full_protocol_metadata_validated_without_images(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subjects = [f"s{i:02d}" for i in range(1, 51)]
            rows = [{"path": f"synthetic/{sid}/{i:02d}.jpg", "subject_id": sid, "image_id": f"{i:02d}", "session_id": "unknown"}
                    for sid in subjects for i in range(1, 16)]
            with (root / "inventory.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            permutation = np.random.default_rng(42).permutation(subjects).tolist()
            enrolled, unknown = permutation[:25], permutation[25:]
            protocol = {"protocol_id": "georgia_tech_external_v1", "dataset_inventory": {"path": "inventory.csv", "sha256": sha256(root / "inventory.csv")},
                        "closed_set": {"identities": subjects, "gallery_paths": [r["path"] for r in rows if r["image_id"] == "01"],
                                       "probe_paths": [r["path"] for r in rows if r["image_id"] != "01"]},
                        "open_set": {"enrolled_identities": enrolled, "unknown_identities": unknown,
                                     "gallery_paths": [r["path"] for r in rows if r["subject_id"] in enrolled and r["image_id"] == "01"],
                                     "known_probe_paths": [r["path"] for r in rows if r["subject_id"] in enrolled and r["image_id"] != "01"],
                                     "unknown_probe_paths": [r["path"] for r in rows if r["subject_id"] in unknown and r["image_id"] != "01"]}}
            write_json(root / "protocol.json", protocol)
            with patch("experiments.external_evaluate_v3.read_image") as read:
                _, actual_rows, _ = load_protocol(root / "protocol.json", root)
                self.assertEqual(len(actual_rows), 750)
                read.assert_not_called()
            protocol["open_set"]["enrolled_identities"].reverse()
            write_json(root / "protocol.json", protocol)
            with self.assertRaisesRegex(ValueError, "assignment"):
                load_protocol(root / "protocol.json", root)


if __name__ == "__main__":
    unittest.main()
