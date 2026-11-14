"""Tests for development isolation and classical feature math, using fixtures."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.development_v3 import (
    ClassicalTextureDescriptorV3, FeatureConfig, config_id, development_rows, extract_features,
    fit_pca_family, normalize_rows, within_identity_whitener,
)


class DevelopmentV3Tests(unittest.TestCase):
    def test_test_rows_are_filtered_before_any_image_loading(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory)/"input.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["path", "subject_id", "session_id", "split", "role"])
                for split in ("train", "calibration", "test"):
                    for i in range(2):
                        for j in range(2):
                            role = "train" if split == "train" else "gallery" if j == 0 else "probe"
                            writer.writerow([f"NONEXISTENT_{split}_{i}_{j}.png", f"{split}_{i}", "unknown", split, role])
            rows = development_rows(manifest)
            self.assertEqual(len(rows), 8)
            self.assertFalse(any(row["split"] == "test" for row in rows))

    def test_pca_is_fitted_only_on_training(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(12, 20))
        first, _ = fit_pca_family(train, np.vstack([train, np.zeros((2, 20))]), 8, 42)
        second, _ = fit_pca_family(train, np.vstack([train, np.full((2, 20), 1e9)]), 8, 42)
        np.testing.assert_allclose(first.mean_, train.mean(axis=0))
        np.testing.assert_array_equal(first.components_, second.components_)

    def test_classical_features_are_finite_normalized_and_deterministic(self):
        rng = np.random.default_rng(11)
        prepared = (
            rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            rng.integers(0, 256, (4, 64, 64, 3), dtype=np.uint8),
            np.ones((4, 64, 64), np.uint8), np.ones((224, 224), np.uint8),
        )
        for descriptor in ("lbp_nri", "mslbp", "hog", "dct", "rootsift"):
            with self.subTest(descriptor=descriptor):
                config = FeatureConfig(descriptor, "skin_regions", "raw")
                a, b = extract_features(prepared, config), extract_features(prepared, config)
                self.assertTrue(np.isfinite(a).all())
                self.assertAlmostEqual(float(np.linalg.norm(a)), 1, places=5)
                np.testing.assert_array_equal(a, b)

    def test_empty_skin_mask_rejects_histogram(self):
        image = np.zeros((64, 64, 3), np.uint8)
        prepared = (image, np.stack([image]*4), np.zeros((4, 64, 64), np.uint8), np.zeros((64, 64), np.uint8))
        with self.assertRaisesRegex(ValueError, "empty"):
            extract_features(prepared, FeatureConfig("lbp_nri", "skin_regions", "raw"))

    def test_within_identity_whitening_uses_real_pairs(self):
        rng = np.random.default_rng(6)
        train = rng.normal(size=(8, 5))
        identities = np.repeat(["a", "b", "c", "d"], 2)
        transform = within_identity_whitener(train, identities)
        np.testing.assert_allclose(transform, transform.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(transform) > 0))
        with self.assertRaisesRegex(ValueError, ">=2"):
            within_identity_whitener(train, np.arange(8))

    def test_normalization_handles_zeros_and_config_id_is_order_independent(self):
        np.testing.assert_allclose(normalize_rows([[0, 0], [3, 4]]), [[0, 0], [.6, .8]])
        self.assertEqual(config_id({"a": 1, "b": 2}), config_id({"b": 2, "a": 1}))

    def test_fitted_model_safe_roundtrip_and_preprocessing_guard(self):
        rng = np.random.default_rng(5)
        faces = [(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
                  rng.integers(0, 256, (4, 64, 64, 3), dtype=np.uint8),
                  np.ones((4, 64, 64), np.uint8), np.ones((64, 64), np.uint8)) for _ in range(8)]
        configuration = {"features": {"descriptor": "dct", "scope": "skin_regions", "photometric": "raw", "grid": 4},
                         "transform": {"kind": "pca", "dimension": 4, "whitening_power": 1.0,
                                       "variance_floor_fraction": .001}}
        descriptor = ClassicalTextureDescriptorV3(configuration, "fixture_only", max_pca_components=4).fit(faces)
        original = descriptor.extract_features(faces[0], "fixture_only")
        with TemporaryDirectory() as directory:
            path = Path(directory)/"descriptor.npz"
            descriptor.save(path)
            loaded = ClassicalTextureDescriptorV3.load(path)
            self.assertEqual(descriptor.feature_space_id, loaded.feature_space_id)
            np.testing.assert_array_equal(original, loaded.extract_features(faces[0], "fixture_only"))
            with self.assertRaisesRegex(ValueError, "Preprocessing contract mismatch"):
                loaded.extract_features(faces[0], "different_pipeline")


if __name__ == "__main__":
    unittest.main()

