"""Synthetic-only v4 evaluation tests; no external portraits or embeddings."""
import copy
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist

from experiments.external_evaluate_v4 import (
    canonical, evaluate_lfw, evaluate_orl, fit_accuracy_threshold,
    load_embeddings, load_protocol_data, operational_rates,
    pair_arrays, pair_identity_bootstrap, run, sha256, threshold_cross_validation,
    validate_freeze_bundle, write_json,
)


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tiny_closed():
    inventory = {f"{identity}_{role}": {"subject_id": identity}
                 for identity in "abcd" for role in ("g", "p")}
    protocol = {"closed_set": {"gallery_paths": [f"{i}_g" for i in "abcd"],
                               "probe_paths": [f"{i}_p" for i in "abcd"]},
                "open_set": {"enrolled_identities": ["a", "b"], "unknown_identities": ["c", "d"],
                             "gallery_paths": ["a_g", "b_g"], "known_probe_paths": ["a_p", "b_p"],
                             "unknown_probe_paths": ["c_p", "d_p"]}}
    features = {f"{identity}_{role}": np.eye(4)[i] for i, identity in enumerate("abcd") for role in ("g", "p")}
    return protocol, inventory, features


def synthetic_closed_inventory(root, dataset="orl"):
    indices = range(1, 11) if dataset == "orl" else (3, 4, 5, 6, 7, 8, 11, 12, 13, 14)
    gallery_index = 1 if dataset == "orl" else 11
    rows = [{"path": f"synthetic/s{subject}/{index}.pgm", "subject_id": f"s{subject}",
             "image_id": str(index), "sha256": "a" * 64}
            for subject in range(1, 41) for index in indices]
    write_csv(root / "inventory.csv", rows)
    subjects = sorted({r["subject_id"] for r in rows})
    order = np.random.default_rng(42).permutation(subjects).tolist()
    gallery = [r["path"] for r in rows if int(r["image_id"]) == gallery_index]
    probes = [r["path"] for r in rows if int(r["image_id"]) != gallery_index]
    by_path = {r["path"]: r for r in rows}
    protocol = {"protocol_id": dataset + "_external_v4_1", "inventory_path": "inventory.csv",
                "inventory_sha256": sha256(root / "inventory.csv"),
                "closed_set": {"identities": subjects, "gallery_paths": gallery, "probe_paths": probes},
                "open_set": {"enrolled_identities": order[:20], "unknown_identities": order[20:],
                             "gallery_paths": [p for p in gallery if by_path[p]["subject_id"] in order[:20]],
                             "known_probe_paths": [p for p in probes if by_path[p]["subject_id"] in order[:20]],
                             "unknown_probe_paths": [p for p in probes if by_path[p]["subject_id"] in order[20:]]}}
    if dataset == "fei_holdout":
        protocol["evaluation_scope"] = "within_domain_heldout_identities_prior_v2_frontal_exposure"
    return protocol, rows


def synthetic_bundle(root, include_features=True):
    protocol, rows = synthetic_closed_inventory(root)
    write_json(root / "protocol.json", protocol)
    source_hashes = {}
    for name in ("experiments/external_evaluate_v4.py", "experiments/external_evaluate_v3.py", "experiments/scientific_metrics.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture source\n", encoding="utf-8")
        source_hashes[name] = sha256(path)
    model = root / "model.npz"
    np.savez(model, projection=np.eye(2))
    method = {"method_id": "synthetic", "feature_space_id": "unit_synthetic", "preprocessing_id": "synthetic_gate",
              "scope": "whole_face", "score_threshold": -.5}
    freeze = {"schema_version": "study-v4.1", "timestamp": "2000-01-01T00:00:00+00:00",
              "source_hashes": source_hashes, "model_artifacts": {"model.npz": sha256(model)},
              "methods": [method], "protocols": {"orl": {"path": "protocol.json", "sha256": sha256(root / "protocol.json")}}}
    write_json(root / "freeze.json", freeze)
    quality = []
    vectors = []
    for row in rows:
        record = {"path": row["path"], "subject_id": row["subject_id"], "source_sha256": row["sha256"],
                  "status": "accepted" if include_features else "rejected", "rejection_codes": [] if include_features else ["synthetic_rejection"]}
        if include_features:
            record["embedding_index"] = len(vectors)
            vectors.append(np.eye(40)[int(row["subject_id"][1:]) - 1])
        quality.append(record)
    np.savez(root / "embeddings.npz", embeddings=np.asarray(vectors) if vectors else np.empty((0, 40)))
    write_json(root / "quality.json", quality)
    bundle = {"freeze_sha256": sha256(root / "freeze.json"), "protocol_id": protocol["protocol_id"],
              "protocol_sha256": sha256(root / "protocol.json"), "method_id": method["method_id"],
              "feature_space_id": method["feature_space_id"], "preprocessing_id": method["preprocessing_id"], "scope": method["scope"],
              "extraction_started_utc": "2000-01-02T00:00:00+00:00", "embeddings_path": "embeddings.npz",
              "embeddings_sha256": sha256(root / "embeddings.npz"), "quality_path": "quality.json", "quality_sha256": sha256(root / "quality.json")}
    write_json(root / "bundle.json", bundle)
    return freeze, bundle, protocol, rows


class PairMetricTests(unittest.TestCase):
    def test_pair_scores_match_live_float64_cdist_at_exact_boundary(self):
        rng = np.random.default_rng(17)
        raw = rng.normal(size=(2, 512))
        vectors = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32).astype(np.float64)
        expected = -cdist(vectors[:1], vectors[1:])[0, 0]
        labels, scores, valid = pair_arrays([{"path_a": "a", "path_b": "b", "is_same": 0}], {"a": vectors[0], "b": vectors[1]})
        self.assertEqual(scores[0], expected)
        self.assertEqual(operational_rates(labels, scores, valid, expected)["far"], 1)
        self.assertEqual(operational_rates(labels, scores, valid, np.nextafter(expected, np.inf))["far"], 0)

    def test_quality_failures_remain_in_all_denominators(self):
        result = operational_rates([1, 1, 0, 0], [-.1, 0, -.1, 0], [True, False, True, False], -.5)
        self.assertEqual(result["far"], .5)
        self.assertEqual(result["frr"], .5)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["intended_pairs"], 4)
        self.assertEqual(result["available_pairs"], 2)

    def test_failed_zero_scores_are_never_accepted(self):
        rates = operational_rates([1, 0], [0, 0], [False, False], -2)
        self.assertEqual((rates["far"], rates["frr"], rates["accuracy"]), (0, 1, .5))

    def test_threshold_sweep_uses_conservative_ties(self):
        threshold = fit_accuracy_threshold([1, 0], [-.5, -.5], [True, True])
        self.assertGreater(threshold, -.5)
        self.assertEqual(fit_accuracy_threshold([1, 0, 1], [-.1, -.9, 0], [True, True, False]), -.1)

    def test_cv_does_not_select_using_heldout_fold_scores(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        scores = np.array([-.1, -.8, -.2, -.7, -.3, -.9])
        folds = np.repeat([1, 2, 3], 2)
        result = threshold_cross_validation(y, scores, np.ones(6, bool), folds)
        changed = scores.copy()
        changed[:2] = [-1.5, -.01]
        changed_result = threshold_cross_validation(y, changed, np.ones(6, bool), folds)
        self.assertEqual(result["folds"][0]["threshold_from_other_folds"], changed_result["folds"][0]["threshold_from_other_folds"])
        self.assertNotEqual(result["folds"][0]["operational"]["accuracy"], changed_result["folds"][0]["operational"]["accuracy"])
        self.assertEqual(result["folds"][0]["train_pair_count"], 4)

    def test_cv_all_quality_failures_are_explicitly_unavailable(self):
        report = threshold_cross_validation([1, 0, 1, 0], [0] * 4, [False] * 4, [1, 1, 2, 2])
        self.assertEqual(report["status"], "incomplete")
        self.assertIsNone(report["mean_accuracy"])

    def test_identity_bootstrap_preserves_dependence_of_repeated_pairs(self):
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        scores = np.array([-.1, -.7, -.1, -.7, -.1, -.8, -.8, -.1])
        a = np.array(list("abcdabcd"))
        b = np.array(list("abcdbcda"))
        first = pair_identity_bootstrap(labels, scores, np.ones(8, bool), a, b, -.5, 100)
        repeated = pair_identity_bootstrap(np.repeat(labels, 10), np.repeat(scores, 10), np.ones(80, bool), np.repeat(a, 10), np.repeat(b, 10), -.5, 100)
        self.assertEqual(first["operational_intervals"], repeated["operational_intervals"])
        self.assertEqual(first["conditional_pair_intervals"], repeated["conditional_pair_intervals"])
        self.assertEqual(first["identity_count"], 4)

    def test_identity_bootstrap_rejects_false_label_mapping(self):
        with self.assertRaisesRegex(ValueError, "identity equality"):
            pair_identity_bootstrap([1, 0], [-.1, -.7], [True, True], ["a", "a"], ["b", "b"], -.5, 10)

    def test_lfw_secondary_cv_is_separate_from_frozen_operating_point(self):
        features = {f"{subject}{index}": np.eye(4)[i] for i, subject in enumerate("abcd") for index in (1, 2)}
        pairs = []
        for fold in (1, 2):
            for a, b, same in (("a1", "a2", 1), ("b1", "b2", 1), ("a1", "c2", 0), ("b1", "d2", 0)):
                pairs.append({"path_a": a, "path_b": b, "subject_id_a": a[0], "subject_id_b": b[0], "is_same": same, "fold_id": fold})
        report, numeric = evaluate_lfw(pairs, features, -2, bootstrap_replicates=30)
        self.assertEqual(report["fixed_threshold_operational"]["far"], 1)
        self.assertEqual(report["secondary_threshold_cross_validation"]["mean_accuracy"], 1)
        self.assertEqual(report["score_threshold"], -2)
        self.assertTrue(numeric["valid_pairs"].all())

    def test_closed_open_failures_and_ties_are_conservative(self):
        protocol, inventory, features = tiny_closed()
        features["b_g"] = features["a_g"].copy()
        del features["c_p"]
        report, numeric, decisions = evaluate_orl(protocol, inventory, features, -.5, 30)
        self.assertEqual(numeric["ranks"][0], 2)
        self.assertFalse(decisions["accepted"][0])
        self.assertEqual(decisions["reason"][0], "tied_maximum")
        self.assertEqual(report["verification_operational"]["genuine_pairs"], 4)
        self.assertEqual(report["verification_operational"]["impostor_pairs"], 12)
        self.assertEqual(report["closed_set"]["operational_denominator"], 4)
        self.assertEqual(report["open_set"]["unknown_probe_count"], 2)

    def test_no_successful_inputs_still_has_operational_outcomes(self):
        protocol, inventory, _ = tiny_closed()
        report, _, _ = evaluate_orl(protocol, inventory, {}, -.5, 10)
        self.assertEqual(report["verification_operational"]["frr"], 1)
        self.assertEqual(report["verification_conditional"]["status"], "unavailable")
        self.assertEqual(report["open_set"]["fnir_operational"], 1)
        json.dumps(report, allow_nan=False)


class ProtocolAndProvenanceTests(unittest.TestCase):
    def test_original_lfw_order_and_labels_are_checked_against_pairs_text(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = [{"path": f"lfw/{name}/{name}_{i:04d}.jpg", "subject_id": name, "image_id": i, "sha256": "a" * 64}
                         for name in ("Synthetic_A", "Synthetic_B") for i in (1, 2)]
            write_csv(root / "inventory.csv", inventory)
            lines, pairs = ["10 300"], []
            for i in range(6000):
                same = i % 600 < 300
                name_b, index_b = ("Synthetic_A", 2) if same else ("Synthetic_B", 1)
                lines.append("Synthetic_A\t1\t2" if same else "Synthetic_A\t1\tSynthetic_B\t1")
                pairs.append({"pair_id": i, "fold_id": i // 600 + 1, "is_same": int(same),
                              "path_a": "lfw/Synthetic_A/Synthetic_A_0001.jpg", "path_b": f"lfw/{name_b}/{name_b}_{index_b:04d}.jpg",
                              "subject_id_a": "Synthetic_A", "subject_id_b": name_b})
            (root / "pairs.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            write_csv(root / "pairs.csv", pairs)
            protocol = {"inventory_path": "inventory.csv", "inventory_sha256": sha256(root / "inventory.csv"),
                        "pair_manifest_path": "pairs.csv", "pair_manifest_sha256": sha256(root / "pairs.csv"),
                        "official_pairs_path": "pairs.txt", "official_pairs_sha256": sha256(root / "pairs.txt")}
            _, checked, required = load_protocol_data(protocol, "lfw", root)
            self.assertEqual(len(checked), 6000)
            self.assertEqual(len(required), 3)
            pairs[0]["is_same"] = 0
            write_csv(root / "pairs.csv", pairs)
            protocol["pair_manifest_sha256"] = sha256(root / "pairs.csv")
            with self.assertRaisesRegex(ValueError, "label differs"):
                load_protocol_data(protocol, "lfw", root)

    def test_closed_protocol_rejects_changed_open_roles(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, _ = synthetic_closed_inventory(root)
            load_protocol_data(copy.deepcopy(protocol), "orl", root)
            protocol["open_set"]["unknown_probe_paths"][0] = protocol["open_set"]["known_probe_paths"][0]
            with self.assertRaisesRegex(ValueError, "identity roles"):
                load_protocol_data(protocol, "orl", root)

    def test_fei_prior_exposure_scope_is_required(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, _ = synthetic_closed_inventory(root, "fei_holdout")
            inventory, _, required = load_protocol_data(copy.deepcopy(protocol), "fei_holdout", root)
            self.assertEqual((len(inventory), len(required)), (400, 400))
            protocol.pop("evaluation_scope")
            with self.assertRaisesRegex(ValueError, "prior v2"):
                load_protocol_data(protocol, "fei_holdout", root)

    def test_freeze_detects_source_tampering_before_loading_vectors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_bundle(root)
            validate_freeze_bundle(root / "freeze.json", root / "bundle.json", "orl", root)
            (root / "experiments/external_evaluate_v4.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                validate_freeze_bundle(root / "freeze.json", root / "bundle.json", "orl", root)

    def test_extraction_before_freeze_and_changed_scope_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, bundle, _, _ = synthetic_bundle(root)
            bundle["extraction_started_utc"] = "1999-01-01T00:00:00+00:00"
            write_json(root / "bundle.json", bundle)
            with self.assertRaisesRegex(ValueError, "predates"):
                validate_freeze_bundle(root / "freeze.json", root / "bundle.json", "orl", root)
            bundle["extraction_started_utc"] = "2000-01-02T00:00:00+00:00"
            bundle["scope"] = "skin_only"
            write_json(root / "bundle.json", bundle)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                validate_freeze_bundle(root / "freeze.json", root / "bundle.json", "orl", root)

    def test_missing_inputs_hash_mismatch_and_nonunit_vectors_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, bundle, protocol, _ = synthetic_bundle(root)
            inventory, _, required = load_protocol_data(protocol, "orl", root)
            quality = json.loads((root / "quality.json").read_text(encoding="utf-8"))
            write_json(root / "quality.json", quality[:-1])
            with self.assertRaisesRegex(ValueError, "omits"):
                load_embeddings(bundle, inventory, required, root)
            quality[0]["source_sha256"] = "b" * 64
            write_json(root / "quality.json", quality)
            with self.assertRaisesRegex(ValueError, "identity/hash"):
                load_embeddings(bundle, inventory, required, root)
            quality[0]["source_sha256"] = "a" * 64
            write_json(root / "quality.json", quality)
            np.savez(root / "embeddings.npz", embeddings=np.zeros((400, 40)))
            with self.assertRaisesRegex(ValueError, "L2-unit"):
                load_embeddings(bundle, inventory, required, root)

    def test_empty_accepted_embeddings_are_supported_with_explicit_rejections(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, bundle, protocol, _ = synthetic_bundle(root, include_features=False)
            inventory, _, required = load_protocol_data(protocol, "orl", root)
            features, records, coverage = load_embeddings(bundle, inventory, required, root)
            self.assertEqual(features, {})
            self.assertEqual(len(records), 400)
            self.assertEqual(coverage["rejection_codes"], {"synthetic_rejection": 400})
            self.assertEqual(coverage["image_success_rate"], 0)

    def test_complete_synthetic_run_writes_safe_artifacts_and_preserves_bundle(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_bundle(root)
            before = sha256(root / "bundle.json")
            original = evaluate_orl
            with patch("experiments.external_evaluate_v4.evaluate_orl", side_effect=lambda p, i, f, t: original(p, i, f, t, 10)):
                report = run(root / "freeze.json", root / "bundle.json", "orl", root / "results", root)
            self.assertEqual(report["closed_set"]["rank1_operational"], 1)
            self.assertEqual(report["verification_operational"]["intended_pairs"], 14400)
            self.assertEqual(sha256(root / "bundle.json"), before)
            self.assertEqual(json.loads((root / "results/status.json").read_text())["status"], "completed")
            with np.load(root / "results/scores.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["scores"].shape, (360, 40))
                self.assertTrue(all(array.dtype.kind != "O" for array in arrays.values()))
            with (root / "results/scores.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 14400)
            with self.assertRaisesRegex(ValueError, "new empty"):
                run(root / "freeze.json", root / "bundle.json", "orl", root / "results", root)


if __name__ == "__main__":
    unittest.main()
