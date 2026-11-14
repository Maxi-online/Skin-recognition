"""Engineering invariants, not biometric performance validation."""
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.descriptors.hybrid_v4 as module
from src.descriptors.hybrid_v4 import (
    ArcFaceDescriptorV4, HybridFaceSkinDescriptorV4, arcface_crop, fuse_embeddings,
    source_landmarks_from_cache, unit_vector,
    source_landmarks_for_single_face,
)


@pytest.mark.parametrize("weight", [0, 0.02, 0.2, 1])
def test_metric_fusion_has_intended_weight_and_float32_storage(weight):
    g1, g2 = np.array([3., 4., 0.]), np.array([0., 2., 1.])
    s1, s2 = np.array([1., 2.]), np.array([-2., 1.])
    first, second = fuse_embeddings(g1, s1, weight), fuse_embeddings(g2, s2, weight)
    expected = (1-weight)*np.sum((unit_vector(g1)-unit_vector(g2))**2)+weight*np.sum((unit_vector(s1)-unit_vector(s2))**2)
    assert np.sum((first.astype(float)-second.astype(float))**2) == pytest.approx(expected, abs=2e-7)
    assert np.linalg.norm(first) == pytest.approx(1, abs=1e-7)
    assert first.dtype == np.float32


@pytest.mark.parametrize("weight", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_fusion_weight_is_rejected(weight):
    with pytest.raises(ValueError):
        fuse_embeddings([1, 0], [0, 1], weight)


def test_original_landmarks_reconstructed_without_resizing_face_cache():
    source = np.array([[25., 30.], [75., 29.], [51., 55.], [32., 80.], [72., 81.]])
    matrix = cv2.getRotationMatrix2D((50, 50), 11, 1.7)
    aligned = source@matrix[:, :2].T+matrix[:, 2]
    np.testing.assert_allclose(source_landmarks_from_cache(aligned, matrix), source, atol=1e-12)
    with pytest.raises(ValueError, match="Singular"):
        source_landmarks_from_cache(aligned, np.zeros((2, 3)))


def test_arcface_crop_matches_upstream_original_image_contract():
    from insightface.utils.face_align import norm_crop
    image = np.random.default_rng(42).integers(0, 256, (160, 200, 3), dtype=np.uint8)
    kps = np.array([[60., 50.], [130., 52.], [96., 83.], [67., 120.], [124., 120.]])
    np.testing.assert_array_equal(arcface_crop(image, kps), norm_crop(image, kps, image_size=112))


def test_global_tensor_is_rgb_with_exact_range_and_normalized_output():
    observed = {}
    class Session:
        def run(self, outputs, inputs):
            observed.update(inputs)
            return [np.tile(np.arange(1, 513, dtype=np.float32), (1, 1))]
    model = ArcFaceDescriptorV4.__new__(ArcFaceDescriptorV4)
    model.session, model.input_name, model.output_name = Session(), "input.1", "683"
    bgr = np.full((112, 112, 3), (0, 127, 255), np.uint8)
    result = model.extract_aligned_batch([bgr])
    blob = observed["input.1"]
    assert blob.shape == (1, 3, 112, 112) and blob.dtype == np.float32
    np.testing.assert_allclose(blob[0, :, 0, 0], [1, -0.5/127.5, -1], atol=1e-7)
    assert result.dtype == np.float32 and result.shape == (1, 512)
    assert np.linalg.norm(result[0]) == pytest.approx(1, abs=1e-7)


@pytest.fixture
def stub_model(tmp_path, monkeypatch):
    from experiments.development_v3 import ClassicalTextureDescriptorV3
    global_path, skin_path = tmp_path/"global.onnx", tmp_path/"skin.npz"
    global_path.write_bytes(b"stub-onnx")
    skin_path.write_bytes(b"stub-frozen-skin")
    class Global:
        def __init__(self, path, *args):
            self.model_path = Path(path).resolve()
            self.model_sha256 = module.sha256(path)
        @property
        def feature_space_id(self):
            return "stub-global:"+self.model_sha256
        def extract_features(self, image, points):
            assert image.shape == (160, 200, 3)
            return unit_vector(np.arange(1, 513)).astype(np.float32)
    class Skin:
        descriptor_dim = 64
        preprocessing_id = "fixture:strict"
        feature_space_id = "fixture-skin64"
        def extract_features(self, packed, pid):
            assert pid == self.preprocessing_id
            assert packed[1].shape == (4, 64, 64, 3)
            return unit_vector(np.arange(1, 65))
    monkeypatch.setattr(module, "ArcFaceDescriptorV4", Global)
    monkeypatch.setattr(ClassicalTextureDescriptorV3, "load", lambda path: Skin())
    return HybridFaceSkinDescriptorV4(global_path, skin_path)


def test_hybrid_dynamic_dimension_safe_manifest_and_checksum(stub_model, tmp_path):
    assert stub_model.descriptor_dim == 576
    destination = tmp_path/"hybrid.json"
    stub_model.save(destination)
    restored = HybridFaceSkinDescriptorV4.load(destination)
    assert restored.feature_space_id == stub_model.feature_space_id
    config = json.loads(destination.read_text())
    config["skin_weight"] = .2
    destination.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="feature space"):
        HybridFaceSkinDescriptorV4.load(destination)
    stub_model.save(destination)
    stub_model.global_model.model_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        HybridFaceSkinDescriptorV4.load(destination)


def test_full_frame_is_preprocessed_once_and_preserved_for_global(stub_model, monkeypatch):
    image = np.zeros((160, 200, 3), np.uint8)
    kps = np.array([[60., 50.], [130., 52.], [96., 83.], [67., 120.], [124., 120.]])
    prepared = SimpleNamespace(kps5=kps, transform=np.array([[1., 0., 0.], [0., 1., 0.]]))
    observations = []
    class Processor:
        def process(self, full):
            assert full is image
            observations.append(full.shape)
            return prepared
    stub_model.preprocessor = Processor()
    monkeypatch.setattr(module, "prepared_tuple_from_face_v4", lambda face:
                        ((np.zeros((224, 224, 3), np.uint8), np.zeros((4, 64, 64, 3), np.uint8),
                          np.ones((4, 64, 64), np.uint8), np.ones((224, 224), np.uint8)), "fixture:strict"))
    features = stub_model.extract_features(image)
    assert features.shape == (576,) and features.dtype == np.float32
    expected = fuse_embeddings(unit_vector(np.arange(1, 513)).astype(np.float32),
                               unit_vector(np.arange(1, 65)).astype(np.float32), stub_model.skin_weight)
    np.testing.assert_array_equal(features, expected)
    assert observations == [(160, 200, 3)]
    assert stub_model.input_mode == "full_frame"


def test_empty_roi_without_usability_flag_is_a_zero_feature_block():
    names = ["forehead", "left_cheek", "right_cheek", "chin"]
    arrays = {"aligned_bgr": np.zeros((224, 224, 3), np.uint8), "skin_mask": np.ones((224, 224), np.uint8),
              "patch_names": np.asarray(names), **{"skin_only_"+name: np.zeros((64, 64, 3), np.uint8) for name in names}}
    quality = {name: {"strict_skin_usable": True} for name in names}
    quality["forehead"] = {"rejection_codes": ["empty_roi"]}
    prepared = SimpleNamespace(numeric_arrays=lambda: arrays,
                               metadata={"preprocessing_id": "fixture", "patch_quality": quality})
    packed, pid = module.prepared_tuple_from_face_v4(prepared)
    assert pid == "fixture:strict"
    assert not packed[2][0].any()
    assert packed[2][1:].all()


def test_detector_only_accepts_small_face_without_skin_or_106_gates():
    kps = np.array([[[10., 10.], [15., 10.], [12., 12.], [10., 15.], [15., 15.]]])
    detector = SimpleNamespace(detect=lambda image, **kwargs: (np.array([[5., 5., 20., 20., .9]]), kps))
    processor = SimpleNamespace(detector=detector, _lock=threading.RLock(), preprocessing_id="fixture")
    points, quality = source_landmarks_for_single_face(processor, np.zeros((64, 64, 3), np.uint8))
    np.testing.assert_array_equal(points, kps[0])
    assert quality["preprocessing_id"] == "fixture:detector_only"
    assert "skin" not in quality


@pytest.mark.parametrize("count,code", [(0, "no_face"), (2, "multiple_faces")])
def test_detector_only_ambiguity_rejected(count, code):
    from src.preprocessing_v3 import QualityRejection
    processor = SimpleNamespace(detector=SimpleNamespace(detect=lambda image, **kwargs:
                                (np.zeros((count, 5)), np.zeros((count, 5, 2)))),
                                _lock=threading.RLock(), preprocessing_id="fixture")
    with pytest.raises(QualityRejection, match=code):
        source_landmarks_for_single_face(processor, np.zeros((64, 64, 3), np.uint8))
