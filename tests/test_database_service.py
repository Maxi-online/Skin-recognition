import json
from pathlib import Path
import numpy as np
import pytest
from src.database import SkinEmbeddingDatabase
from src.services import SkinRecognitionSystem


def test_stable_ids_after_multiple_removals_and_reload(tmp_path):
    db = SkinEmbeddingDatabase(3, feature_space_id="test-space")
    for vector, subject in zip(np.eye(3), ("a", "a", "b")):
        db.add_embedding(vector, subject, subject)
    for key, _ in db.get_person_embeddings("a"):
        assert db.remove_embedding(key)
    assert db.search(np.array([0, 0, 1]))[0][:2] == (2, 0.0)
    assert db.add_embedding(np.array([1, 0, 0]), "c", "c") == 3
    db.save(str(tmp_path / "gallery"))
    loaded = SkinEmbeddingDatabase(3, feature_space_id="test-space")
    loaded.load(str(tmp_path / "gallery"))
    assert loaded.search(np.array([1, 0, 0]))[0][0] == 3
    assert loaded.get_stats()["total_persons"] == 2


def test_rejects_metadata_override_and_invalid_vectors():
    db = SkinEmbeddingDatabase(2)
    for vector in (np.zeros(2), np.array([np.nan, 1]), np.ones((2, 1)), np.ones(3)):
        with pytest.raises(ValueError):
            db.add_embedding(vector, "a", "a")
    with pytest.raises(ValueError):
        db.add_embedding(np.ones(2), "a", "a", additional_metadata={"person_id": "b"})
    assert db.get_stats()["total_embeddings"] == 0


def test_load_rejects_wrong_space_without_changing_live_state(tmp_path):
    path = str(tmp_path / "gallery")
    a = SkinEmbeddingDatabase(2, feature_space_id="a")
    a.add_embedding(np.array([1, 0]), "a", "a")
    a.save(path)
    b = SkinEmbeddingDatabase(2, feature_space_id="b")
    b.add_embedding(np.array([0, 1]), "b", "b")
    with pytest.raises(ValueError, match="feature space"):
        b.load(path)
    assert b.search(np.array([0, 1]))[0][2]["person_id"] == "b"
    with pytest.raises(ValueError, match="pickle"):
        b.load(str(tmp_path / "legacy.pkl"))


def test_squared_l2_and_ip_semantics():
    l2 = SkinEmbeddingDatabase(2)
    l2.add_embedding(np.array([1, 0]), "a", "a")
    assert l2.search(np.array([0, 1]))[0][1] == 2
    assert l2.search(np.array([0, 1]), threshold=1.5) == []
    ip = SkinEmbeddingDatabase(2, index_type="IP")
    ip.add_embedding(np.array([10, 0]), "a", "a")
    assert ip.search(np.array([5, 0]), threshold=0.9)[0][1] == 1
    with pytest.raises(ValueError):
        ip.search(np.ones(2), k=0)


class StubDescriptor:
    descriptor_dim = 2
    feature_space_id = "stub"
    def extract_features(self, image):
        return np.array([np.cos(float(image[0, 0]) / 100), np.sin(float(image[0, 0]) / 100)], np.float32)


@pytest.fixture
def system(monkeypatch):
    monkeypatch.setattr("src.services.recognition.HybridSkinDescriptor", lambda **kwargs: StubDescriptor())
    return SkinRecognitionSystem(descriptor_dim=2, recognition_threshold=0.7)


def test_recognition_applies_threshold_once_and_returns_four_fields(system):
    image = np.zeros((32, 32), np.uint8)
    assert len(system.recognize(image, return_distances=True)) == 4
    system.register_person(image, "a", "Alice")
    # d^2 = 2 - 2*cos(.8) = .6066, below .7 but above the old hidden .35 threshold.
    name, score, candidates, distances = system.recognize(np.full_like(image, 80), return_distances=True)
    assert name == "Alice" and 0.60 < distances[0] < 0.61
    assert score == pytest.approx(1 - distances[0] / 4)
    assert system.recognize(np.full_like(image, 120))[0] is None


def test_service_removes_all_person_templates(system):
    image = np.zeros((32, 32), np.uint8)
    for subject in ("a", "a", "b", "a"):
        system.register_person(image, subject, subject)
    assert system.remove_person("a") == 3
    assert system.get_database_stats()["total_persons"] == 1
    assert system.recognize(image)[0] == "b"


def test_service_detects_descriptor_change(system):
    system.descriptor.feature_space_id = "changed"
    with pytest.raises(ValueError, match="changed"):
        system.recognize(np.zeros((32, 32), np.uint8))


def test_service_does_not_replace_corrupt_database(tmp_path):
    path = tmp_path / "broken.npz"
    content = b"not a valid archive"
    path.write_bytes(content)
    with pytest.raises(Exception):
        SkinRecognitionSystem(database_path=str(path))
    assert path.read_bytes() == content
