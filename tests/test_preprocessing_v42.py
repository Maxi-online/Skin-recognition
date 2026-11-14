"""Geometry/contract tests; synthetic inputs do not validate biometric accuracy."""
from pathlib import Path
import hashlib
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from src.preprocessing_v3 import QualityRejection
from src.preprocessing_v42 import (
    FacePreprocessorV42, NativeResolutionDetector, detector_canvas, prepared_tuple_from_face_v42,
    source_landmarks_for_target_face_v42,
)


@pytest.mark.parametrize("shape", [(480, 640, 3), (640, 480, 3), (720, 1280, 3)])
def test_full_size_detector_input_is_identity(shape):
    image = np.zeros(shape, np.uint8)
    canvas, translation = detector_canvas(image)
    assert canvas is image and translation == (0, 0)


def test_small_frame_is_padded_without_resampling_and_coordinates_are_restored():
    image = np.random.default_rng(0).integers(1, 256, (113, 91, 3), dtype=np.uint8)
    canvas, (x, y) = detector_canvas(image)
    assert canvas.shape == (640, 640, 3) and (x, y) == (274, 263)
    np.testing.assert_array_equal(canvas[y:y+113, x:x+91], image)
    assert not canvas[:y].any() and not canvas[:, :x].any()
    original_boxes = np.array([[-2, 3, 90, 112, .9]], np.float32)
    original_points = np.array([[[20, 25], [60, 25], [40, 45], [25, 70], [55, 70]]], np.float32)
    padded_boxes = original_boxes.copy()
    padded_boxes[:, :4] += [x, y, x, y]
    padded_points = original_points+[x, y]
    observed = []
    def detect(value, **kwargs):
        observed.append(value)
        return padded_boxes, padded_points
    adapter = NativeResolutionDetector(SimpleNamespace(detect=detect))
    boxes, points = adapter.detect(image)
    np.testing.assert_array_equal(boxes, original_boxes)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(observed[0], canvas)
    # The original detector-owned arrays are not modified in place.
    assert padded_boxes[0, 0] == original_boxes[0, 0]+x


def fake_base(count=1, rejection=None):
    calls = {}
    points = np.array([[[290, 290], [330, 290], [310, 310], [295, 330], [325, 330]]], np.float32)
    boxes = np.array([[280, 270, 360, 370, .9]], np.float32)
    def detect(image, **kwargs):
        calls["detector_input"] = image
        return np.repeat(boxes, count, axis=0), np.repeat(points, count, axis=0)
    def landmarks(image, face):
        calls["landmark_input"] = image
        calls["landmark_box"] = face.bbox
        return np.zeros((106, 2), np.float32)
    def process(image, box, keypoints, landmarks, score):
        calls["skin_input"] = image
        if rejection:
            raise QualityRejection([rejection], {"preprocessing_id": "legacy-fixture"})
        return SimpleNamespace(metadata={"preprocessing_id": "legacy-fixture"})
    base = SimpleNamespace(_lock=threading.RLock(), detector=SimpleNamespace(detect=detect),
                           landmark_model=SimpleNamespace(get=landmarks),
                           config=SimpleNamespace(detection_size=640), preprocessing_id="legacy-fixture",
                           runtime_providers={"detector": ["CPUExecutionProvider"]}, model_sha256={},
                           process_observation=process)
    return base, calls


def test_landmarks_and_skin_use_original_pixels_with_explicit_contract_mapping():
    base, calls = fake_base()
    image = np.full((112, 92, 3), 17, np.uint8)
    processor = FacePreprocessorV42(base_preprocessor=base)
    face = processor.process(image)
    assert calls["landmark_input"] is image and calls["skin_input"] is image
    np.testing.assert_array_equal(calls["landmark_box"], [6, 6, 86, 106])
    assert face.metadata["preprocessing_id"] == processor.preprocessing_id
    assert face.metadata["legacy_preprocessing_id"] == "legacy-fixture"
    assert face.metadata["input_geometry_adapter"]["detector_padding_xy"] == [274, 264]
    assert processor.preprocessing_id != base.preprocessing_id


@pytest.mark.parametrize("count,code", [(0, "no_face"), (2, "multiple_faces")])
def test_ambiguity_is_still_rejected(count, code):
    base, calls = fake_base(count=count)
    processor = FacePreprocessorV42(base_preprocessor=base)
    with pytest.raises(QualityRejection, match=code):
        processor.process(np.zeros((112, 92, 3), np.uint8))
    assert "landmark_input" not in calls


def test_existing_native_resolution_quality_gate_is_not_relaxed():
    base, _ = fake_base(rejection="insufficient_source_eye_distance")
    processor = FacePreprocessorV42(base_preprocessor=base)
    with pytest.raises(QualityRejection) as error:
        processor.process(np.zeros((112, 92, 3), np.uint8))
    assert error.value.codes == ["insufficient_source_eye_distance"]
    assert error.value.metadata["effective_preprocessing_id"] == processor.preprocessing_id
    assert error.value.metadata["target_selection"]["original_detection_count"] == 1


def test_skin_model_mapping_is_explicit_and_checked():
    base, _ = fake_base()
    processor = FacePreprocessorV42(base_preprocessor=base)
    face = processor.process(np.zeros((112, 92, 3), np.uint8))
    names = ["forehead", "left_cheek", "right_cheek", "chin"]
    arrays = {"patch_names": np.asarray(names), "aligned_bgr": np.zeros((224, 224, 3), np.uint8),
              "skin_mask": np.ones((224, 224), np.uint8),
              **{"skin_only_"+name: np.zeros((64, 64, 3), np.uint8) for name in names}}
    face.numeric_arrays = lambda: arrays
    face.metadata["patch_quality"] = {name: {"strict_skin_usable": True} for name in names}
    packed, legacy, effective = prepared_tuple_from_face_v42(face)
    assert legacy == "legacy-fixture:strict" and effective == processor.preprocessing_id+":strict"
    assert packed[1].shape == (4, 64, 64, 3)
    face.metadata.pop("legacy_preprocessing_id")
    with pytest.raises(ValueError, match="mapping"):
        prepared_tuple_from_face_v42(face)


def test_frozen_v41_source_hashes_are_unchanged():
    root = Path(__file__).resolve().parents[1]
    freeze = root/"research/frozen/study_v4_1.json"
    if not freeze.is_file():
        pytest.skip("Local V4.1 research freeze is not installed")
    for relative, expected in json.loads(freeze.read_text())["source_hashes"].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected, relative


@pytest.mark.parametrize("source_boxes,expected_count,selected_index", [
    ([[1, 1, 20, 20, .99], [30, 30, 70, 80, .7]], 1, 1),
    ([[1, 1, 20, 20, .99], [70, 70, 90, 90, .7]], 2, None),
    ([[20, 20, 60, 60, .99], [40, 40, 80, 80, .7]], 2, None),
    ([[1, 1, 50, 50, .99], [50, 50, 90, 90, .7]], 2, None),
])
def test_target_selection_uses_unique_original_center_with_inclusive_edges(source_boxes, expected_count, selected_index):
    image = np.zeros((100, 100, 3), np.uint8)
    boxes = np.asarray(source_boxes, np.float32)
    padded = boxes.copy(); padded[:, :4] += [270, 270, 270, 270]
    points = np.asarray([[[285, 285], [300, 285], [292, 295], [287, 305], [297, 305]]]*2, np.float32)
    raw = SimpleNamespace(detect=lambda *args, **kwargs: (padded, points))
    result, _, metadata = NativeResolutionDetector(raw).detect_with_metadata(image)
    assert len(result) == expected_count
    assert metadata["source_center_xy"] == [50, 50]
    assert metadata["original_detection_count"] == 2
    assert metadata["selected_index"] == selected_index
    if selected_index is not None:
        np.testing.assert_array_equal(result[0], boxes[selected_index])


def test_strict_and_detector_only_use_same_target_and_rejection_metadata():
    base, _ = fake_base()
    original_boxes = np.array([[0, 0, 15, 15, .99], [20, 20, 80, 90, .7]], np.float32)
    padded = original_boxes.copy(); padded[:, :4] += [274, 264, 274, 264]
    points = np.array([[[280, 270], [290, 270], [285, 280], [282, 285], [288, 285]],
                       [[300, 290], [335, 290], [317, 310], [305, 330], [330, 330]]], np.float32)
    base.detector.detect = lambda *args, **kwargs: (padded, points)
    processor = FacePreprocessorV42(base_preprocessor=base)
    image = np.zeros((112, 92, 3), np.uint8)
    prepared = processor.process(image)
    _, quality = source_landmarks_for_target_face_v42(processor, image)
    assert quality["target_selection"] == prepared.metadata["target_selection"]
    assert quality["target_selection"]["original_detection_count"] == 2
    assert quality["target_selection"]["selected_index"] == 1
    assert quality["preprocessing_id"].endswith(":detector_only")
