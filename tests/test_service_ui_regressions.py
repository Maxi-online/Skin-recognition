"""Persistence and webcam regressions; synthetic fixtures do not validate biometrics."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.services import SkinRecognitionSystem
from src.ui.dash_app import create_dash_app


def test_delete_rolls_back_memory_when_persistence_fails(tmp_path, monkeypatch):
    system = SkinRecognitionSystem(database_path=str(tmp_path / "gallery"))
    image = np.random.default_rng(3).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    system.register_person(image, "alice", "Alice")
    system.register_person(image, "bob", "Bob")
    path = Path(system.database.storage_path(system.database_path))
    before_file = path.read_bytes()
    before_stats = system.database.get_stats()
    before_ids = [item["id"] for item in system.database.metadata]

    def failed_save(path):
        raise OSError("simulated full disk")

    monkeypatch.setattr(system.database, "save", failed_save)
    with pytest.raises(OSError, match="full disk"):
        system.remove_person("alice")
    assert path.read_bytes() == before_file
    assert system.database.get_stats() == before_stats
    assert [item["id"] for item in system.database.metadata] == before_ids
    assert len(system.database.get_person_embeddings("alice")) == 1


def test_service_preprocessing_matches_experiment_resize_and_bgr():
    system = SkinRecognitionSystem()
    rng = np.random.default_rng(9)
    for shape in [(300, 180, 3), (80, 100, 3), (224, 224, 3)]:
        image = rng.integers(0, 256, shape, dtype=np.uint8)
        interpolation = cv2.INTER_AREA if max(shape[:2]) > 224 else cv2.INTER_LINEAR
        expected = cv2.resize(image, (224, 224), interpolation=interpolation)
        np.testing.assert_array_equal(system._preprocess_image(image), expected)


class RecordingSystem:
    def __init__(self):
        self.records = []

    def register_person(self, **kwargs):
        self.records.append(kwargs)
        return len(self.records) - 1


def make_callbacks(detector):
    system = RecordingSystem()
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    app = create_dash_app(
        init_system=lambda: system,
        decode_image_bytes=lambda data: image,
        detect_face_fn=lambda im: (0, 0, 100, 100),
        detect_face_info_fn=detector,
        database_path="unused-fixture-path",
    )
    callbacks = {}
    for entry in app.callback_map.values():
        callback = entry.get("callback")
        if callback and hasattr(callback, "__wrapped__"):
            callback = callback.__wrapped__
            callbacks[callback.__name__] = callback
    return callbacks, system


def test_haar_registration_finishes_three_frontal_frames_without_fake_landmarks():
    callbacks, system = make_callbacks(lambda image: {"bbox": (0, 0, 100, 100)})
    store, _, _ = callbacks["start_enroll"](1, "Alice")
    for frame_number in range(3):
        result = callbacks["handle_frame"](
            {"action": "enroll", "frame": "data:image/png;base64,AA==", "ts": frame_number}, store)
        store = result[3]
        assert result[2]["mesh"]["points"] == []
        assert result[2]["mesh_n"] == 0
    assert result[5] == 100
    assert store["active"] is False
    assert store["capture_mode"] == "frontal"
    assert store["need"] == {"center": 3}
    assert len(system.records) == 3
    assert "Фронтальная" in result[4]
    for record in system.records:
        metadata = record["additional_metadata"]
        assert metadata["capture_mode"] == "frontal"
        assert metadata["yaw_proxy"] is None and metadata["pitch_proxy"] is None


def test_multiple_faces_return_visible_error_without_writing():
    def multiple_faces(image):
        raise ValueError("Exactly one face is required")

    callbacks, system = make_callbacks(multiple_faces)
    store, _, _ = callbacks["start_enroll"](1, "Alice")
    result = callbacks["handle_frame"](
        {"action": "enroll", "frame": "data:image/png;base64,AA==", "ts": 1}, store)
    assert "однозначно выбрать лицо" in result[1]
    assert result[2]["ok"] is False
    assert system.records == []


def test_temporary_landmark_loss_does_not_fabricate_pose_or_reset_multiview():
    information = {"bbox": (0, 0, 100, 100),
                   "kps": [[25, 30], [75, 30], [50, 50], [35, 70], [65, 70]]}
    callbacks, system = make_callbacks(lambda image: dict(information))
    store, _, _ = callbacks["start_enroll"](1, "Alice")
    frame = {"action": "enroll", "frame": "data:image/png;base64,AA==", "ts": 1}
    result = callbacks["handle_frame"](frame, store)
    store = result[3]
    assert store["capture_mode"] == "multi_view"
    assert len(system.records) == 1
    del information["kps"]
    result = callbacks["handle_frame"](frame, store)
    assert "не определены" in result[1]
    assert len(system.records) == 1
    assert result[3]["got"]["center"] == 1

