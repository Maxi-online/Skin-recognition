"""V4.2 freeze lineage and gallery isolation; no biometric accuracy claims."""
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from src.database import SkinEmbeddingDatabase
from src.preprocessing_v42 import FacePreprocessorV42
from src.services.recognition_v4 import RecognitionSystemV4
from src.services.recognition_v42 import RecognitionSystemV42

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_base_preprocessor():
    return SimpleNamespace(_lock=threading.RLock(), detector=SimpleNamespace(), landmark_model=SimpleNamespace(),
                           config=SimpleNamespace(detection_size=640), preprocessing_id="fixture-base",
                           runtime_providers={"detector": ["CPUExecutionProvider"]}, model_sha256={})


@pytest.fixture
def frozen_profile(tmp_path, monkeypatch):
    method = {"method_id": "whole_face", "scope": "whole_face", "skin_weight": 0.,
              "feature_space_id": "fixture-descriptor", "descriptor_dim": 2,
              "score_threshold": -1.2, "search_score_threshold": -.9,
              "score_definition": "negative_euclidean_distance", "preprocessing_id": "fixture-base:strict"}
    base = {"schema_version": "study-v4.1", "methods": [method],
            "runtime": {"global_provider": "cpu", "preprocessing_provider": "cpu"},
            "preprocessing_config": {"detection_size": 640}, "model_artifacts": {}}
    base_path = tmp_path/"base.json"
    base_path.write_text(json.dumps(base))
    effective = FacePreprocessorV42(base_preprocessor=make_base_preprocessor()).preprocessing_id+":strict"
    repaired_method = {**method, "preprocessing_id": effective, "legacy_preprocessing_id": "fixture-base:strict"}
    repaired = {"schema_version": "study-v4.2", "base_freeze_path": str(base_path),
                "base_freeze_sha256": digest(base_path), "runtime": base["runtime"],
                "preprocessing_config": base["preprocessing_config"], "model_artifacts": {},
                "methods": [repaired_method], "source_hashes": {path: digest(ROOT/path) for path in
                    ("src/preprocessing_v42.py", "src/services/recognition_v42.py")}}
    repaired_path = tmp_path/"repaired.json"
    repaired_path.write_text(json.dumps(repaired))
    def fake_init(self, freeze_path, method_id, database_path=None):
        assert database_path is None  # Never load/write an existing V4.1 gallery.
        self._lock = threading.RLock()
        self.calibration_path, self.calibration_id = base_path, digest(base_path)
        self.calibration, self.method = base, method
        self.preprocessor = make_base_preprocessor()
        self.preprocessing_suffix = ":strict"
        self.descriptor = SimpleNamespace(feature_space_id="fixture-descriptor")
        self._descriptor_feature_space_id = "fixture-descriptor"
        self.descriptor_dim = 2
        self.scope, self.profile = "whole_face", method_id
        self.primary_score_threshold, self.score_threshold = -1.2, -.9
        self.recognition_threshold = .81
    monkeypatch.setattr(RecognitionSystemV4, "__init__", fake_init)
    return repaired_path, repaired, base_path


def test_new_effective_gallery_space_keeps_model_and_thresholds(frozen_profile):
    path, document, _ = frozen_profile
    service = RecognitionSystemV42(path, "whole_face")
    assert service.feature_space_id.startswith("recognition-service-v4.2:")
    assert service._descriptor_feature_space_id == "fixture-descriptor"
    assert service.score_threshold == -.9 and service.primary_score_threshold == -1.2
    stats = service.get_database_stats()
    assert stats["version"] == "4.2"
    assert stats["effective_preprocessing_id"] == document["methods"][0]["preprocessing_id"]
    assert stats["legacy_descriptor_preprocessing_id"] == "fixture-base:strict"


@pytest.mark.parametrize("field,value", [("score_threshold", -1.1), ("skin_weight", .05),
                                        ("feature_space_id", "different")])
def test_geometry_repair_cannot_reselect_model_or_threshold(frozen_profile, field, value):
    path, document, _ = frozen_profile
    document["methods"][0][field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="preserve"):
        RecognitionSystemV42(path, "whole_face")


def test_runtime_or_quality_gate_change_is_rejected(frozen_profile):
    path, document, _ = frozen_profile
    document["preprocessing_config"] = {"detection_size": 320}
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="runtime, preprocessing gates"):
        RecognitionSystemV42(path, "whole_face")


def test_base_freeze_modification_is_rejected_before_initialization(frozen_profile):
    path, _, base_path = frozen_profile
    base_path.write_text(base_path.read_text()+" ")
    with pytest.raises(ValueError, match="base freeze checksum"):
        RecognitionSystemV42(path, "whole_face")


def test_existing_legacy_feature_gallery_is_not_overwritten(frozen_profile, tmp_path):
    path, _, _ = frozen_profile
    gallery_path = tmp_path/"legacy_gallery"
    old = SkinEmbeddingDatabase(2, index_type="L2", feature_space_id="v4.1-other-space")
    old.add_embedding(np.array([1., 0.], np.float32), "fixture", "Fixture")
    old.save(gallery_path)
    before = digest(gallery_path.with_suffix(".npz"))
    with pytest.raises(ValueError, match="[Ff]eature|space|[Ii]ncompatible"):
        RecognitionSystemV42(path, "whole_face", gallery_path)
    assert digest(gallery_path.with_suffix(".npz")) == before
