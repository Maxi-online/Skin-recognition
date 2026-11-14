"""Regression tests for feature correctness, isolation, and artifact integrity.

These synthetic tests verify software invariants, not biometric accuracy.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.signal import convolve2d

from src.descriptors import (
    Gabor2DFilterBank, GaborStatisticsPCA, GaborPCAICA,
    GLCMTextureAnalyzer, LocalSkinPatchExtractor, HybridSkinDescriptor,
)
from src.descriptors.lgjg_wpca import moment_statistics


class DescriptorTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)

    def small_descriptor(self, **kwargs):
        descriptor = HybridSkinDescriptor(**kwargs)
        descriptor.gabor_bank = Gabor2DFilterBank(
            frequencies=[0.1, 0.2], orientations=[0, np.pi / 2], scales=[1.0])
        descriptor.lgwg_wpca = GaborStatisticsPCA(n_components=4)
        descriptor.multifold_pca = GaborPCAICA(n_pca_components=3, n_ica_components=2)
        return descriptor

    def test_centered_gabor_kernel_is_odd_zero_dc_and_unit_energy(self):
        bank = Gabor2DFilterBank(frequencies=[0.1], orientations=[0], scales=[1.0])
        kernel = bank.filters[0]["filter"]
        self.assertEqual(kernel.shape[0] % 2, 1)
        self.assertEqual(kernel.shape[1] % 2, 1)
        np.testing.assert_allclose(kernel.sum(), 0, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(kernel)), 1, places=6)

    def test_gabor_response_matches_direct_complex_convolution_interior(self):
        bank = Gabor2DFilterBank(frequencies=[0.15], orientations=[np.pi / 6], scales=[1.0])
        image = self.rng.integers(0, 256, (32, 32), dtype=np.uint8)
        kernel = bank.filters[0]["filter"]
        reference = np.abs(convolve2d(image / 255.0, kernel, mode="same"))
        actual = bank.responses(image)[..., 0]
        radius = kernel.shape[0] // 2
        np.testing.assert_allclose(actual[radius:-radius, radius:-radius],
                                   reference[radius:-radius, radius:-radius], atol=2e-6)

    def test_glcm_accepts_arbitrary_levels_and_consistent_float_ranges(self):
        image = self.rng.integers(0, 256, (32, 32), dtype=np.uint8)
        glcm = GLCMTextureAnalyzer(levels=30)
        expected = glcm.extract_features(image)
        np.testing.assert_allclose(expected, glcm.extract_features(image.astype(float)))
        np.testing.assert_allclose(expected, glcm.extract_features(image.astype(float) / 255))
        self.assertEqual(expected.shape, (84,))
        self.assertTrue(np.isfinite(expected).all())

    def test_glcm_known_constant_image(self):
        features = GLCMTextureAnalyzer(distances=[1], angles=[0]).extract_features(np.full((8, 8), 100, np.uint8))
        np.testing.assert_allclose(features, [0, 0, 1, 1, 1, 1, 0], atol=1e-6)

    def test_reject_invalid_pixels_and_shapes(self):
        descriptor = self.small_descriptor()
        for image in [np.empty((0, 8)), np.zeros((1, 1)), np.full((32, 32), np.nan),
                      np.full((32, 32), -1.0), np.zeros((32, 32, 4)), np.full((32, 32), 256)]:
            with self.subTest(shape=image.shape):
                with self.assertRaises(ValueError):
                    descriptor.extract_features(image)

    def test_bbox_is_respected_and_invalid_bbox_is_rejected(self):
        image = self.rng.integers(0, 256, (100, 120, 3), dtype=np.uint8)
        extractor = LocalSkinPatchExtractor()
        by_bbox = extractor.extract_patches(image, (20, 10, 60, 70))
        by_crop = extractor.extract_patches(image[10:80, 20:80])
        for name in extractor.get_patch_names():
            np.testing.assert_array_equal(by_bbox[name], by_crop[name])
        with self.assertRaises(ValueError):
            extractor.extract_patches(image, (-10, 0, 60, 70))

    def test_weighted_pca_applies_feature_weights_consistently(self):
        responses = [self.rng.normal(i, 1 + i / 10, (8, 8, 3)) for i in range(6)]
        model = GaborStatisticsPCA(n_components=12).fit(responses)
        self.assertEqual(model.pca.components_.shape, (5, 12))
        sample = responses[0]
        expected = model.pca.transform((moment_statistics(sample) * np.sqrt(model.weights))[None, :])[0]
        actual = model.transform(sample)
        np.testing.assert_allclose(actual[:5], expected, atol=1e-6)
        np.testing.assert_array_equal(actual[5:], 0)
        self.assertEqual(actual.shape, (12,))

    def test_projections_reject_degenerate_training(self):
        for model in [GaborStatisticsPCA(), GaborPCAICA()]:
            with self.assertRaises(ValueError):
                model.fit([np.ones((8, 8, 2))] * 3)
            with self.assertRaises(ValueError):
                model.transform(np.ones((8, 8, 2)))

    def test_pca_ica_uses_fixed_train_space_and_pads_output(self):
        samples = [self.rng.normal(i / 3, 1 + i / 10, (8, 8, 3)) for i in range(8)]
        model = GaborPCAICA(n_pca_components=16, n_ica_components=2).fit(samples)
        mean = model.pca.mean_.copy()
        before = model.transform(samples[0])
        model.transform(samples[-1] * 4)
        np.testing.assert_array_equal(mean, model.pca.mean_)
        np.testing.assert_array_equal(before, model.transform(samples[0]))
        self.assertEqual(before.shape, (18,))
        self.assertTrue(np.isfinite(before).all())

    def test_descriptor_is_fixed_size_unit_length_and_order_independent(self):
        first, second = [self.rng.integers(0, 256, (96, 96, 3), dtype=np.uint8) for _ in range(2)]
        descriptor = self.small_descriptor(descriptor_dim=512)
        feature_id = descriptor.feature_space_id
        before = descriptor.extract_features(first)
        descriptor.extract_features(second)
        after = descriptor.extract_features(first)
        self.assertEqual(before.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(before)), 1, places=6)
        np.testing.assert_allclose(before, after, atol=1e-7)
        self.assertEqual(feature_id, descriptor.feature_space_id)

    def test_fit_uses_same_patches_as_inference_and_changes_space(self):
        images = [self.rng.integers(0, 256, (96, 96, 3), dtype=np.uint8) for _ in range(4)]
        descriptor = self.small_descriptor(use_multifold_pca=False)
        old_id = descriptor.feature_space_id
        expected = []
        for image in images:
            for patch in descriptor._patches(image):
                expected.append(moment_statistics(descriptor._get_gabor_responses(patch)))
        expected = np.stack(expected)
        descriptor.fit(images)
        self.assertNotEqual(old_id, descriptor.feature_space_id)
        fitted = descriptor.lgwg_wpca
        np.testing.assert_allclose(fitted.pca.mean_, (expected * np.sqrt(fitted.weights)).mean(axis=0), atol=1e-6)
        frozen_id = descriptor.feature_space_id
        descriptor.extract_features(images[0])
        self.assertEqual(frozen_id, descriptor.feature_space_id)

    def test_serialization_round_trip_fitted_and_unfitted(self):
        images = [self.rng.integers(0, 256, (96, 96, 3), dtype=np.uint8) for _ in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "descriptor.npz"
            descriptor = self.small_descriptor()
            for fitted in [False, True]:
                if fitted:
                    descriptor.fit(images)
                descriptor.save(path)
                restored = HybridSkinDescriptor.load(path)
                self.assertEqual(descriptor.feature_space_id, restored.feature_space_id)
                np.testing.assert_allclose(descriptor.extract_features(images[0]),
                                           restored.extract_features(images[0]), atol=1e-7)
                with np.load(path, allow_pickle=False) as archive:
                    self.assertTrue(all(not archive[key].dtype.hasobject for key in archive.files))

    def test_serialized_metadata_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "descriptor.npz"
            descriptor = self.small_descriptor()
            descriptor.save(path)
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key].copy() for key in archive.files}
            metadata = json.loads(data["metadata"].item())
            metadata["config"]["descriptor_dim"] = 127
            data["metadata"] = np.asarray(json.dumps(metadata))
            np.savez_compressed(path, **data)
            with self.assertRaisesRegex(ValueError, "digest"):
                HybridSkinDescriptor.load(path)

    def test_feature_space_id_distinguishes_configurations(self):
        first = self.small_descriptor()
        second = self.small_descriptor(use_glcm=False)
        self.assertNotEqual(first.feature_space_id, second.feature_space_id)

if __name__ == "__main__":
    unittest.main()

