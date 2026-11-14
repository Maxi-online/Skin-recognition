"""Synthetic frozen-profile integration; no biometric holdout images are read."""
from dataclasses import asdict
import hashlib
import importlib
import io
import json
from types import SimpleNamespace
import threading

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.distance import cdist
from fastapi.testclient import TestClient

import src.services.recognition_v4 as module
from src.descriptors.hybrid_v4 import prepared_tuple_from_face_v4
from src.preprocessing_v3 import PreprocessingConfigV3, QualityRejection

BASE_ID = "synthetic-v4-preprocessor"
KPS = np.array([[30, 25], [60, 25], [45, 40], [35, 55], [55, 55]], np.float64)
SCOPES = ("skin_only", "whole_face", "whole_face_detector_only", "hybrid")


def frame(value=0):
    return np.full((80, 100, 3), value, np.uint8)


def vector(image):
    angle = float(image[0, 0, 0]) / 100
    return np.array([np.cos(angle), np.sin(angle)], np.float32)


class Prepared:
    def __init__(self, image, missing_flag=False):
        self.image = image
        self.kps5 = KPS.copy()
        self.transform = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        self.metadata = {"preprocessing_id": BASE_ID, "patch_quality": {
            "forehead": {} if missing_flag else {"strict_skin_usable": True}}}

    def numeric_arrays(self):
        return {"patch_names": np.array(["forehead"]), "aligned_bgr": self.image,
                "skin_only_forehead": np.full((64, 64, 3), 128, np.uint8),
                "skin_mask": np.ones(self.image.shape[:2], np.uint8)}


class StubPreprocessor:
    preprocessing_id = BASE_ID
    runtime_providers = {"detector": ["CPUExecutionProvider"], "landmarks": ["CPUExecutionProvider"]}

    def __init__(self, **kwargs):
        self.images = []
        self.missing_flag = False
        self._lock = threading.RLock()
        self.detector = SimpleNamespace(detect=self._detect)

    def _detect(self, image, **kwargs):
        assert image.shape == (80, 100, 3)
        return np.array([[20, 10, 75, 70, .9]], np.float32), KPS[None].copy()

    def process(self, image):
        if image.shape != (80, 100, 3):
            raise QualityRejection(["full_frame_expected"])
        self.images.append(image.copy())
        return Prepared(image, self.missing_flag)


class StubSkin:
    feature_space_id = "synthetic-frozen-skin-v4"
    descriptor_dim = 2

    def extract_features(self, prepared, preprocessing_id):
        assert preprocessing_id == BASE_ID + ":strict"
        self.last_masks = prepared[2]
        return vector(prepared[0])


class StubGlobal:
    feature_space_id = "synthetic-frozen-global-v4"
    descriptor_dim = 2
    runtime_providers = ["CPUExecutionProvider"]

    def __init__(self, *args, **kwargs):
        self.images = []

    def extract_features(self, image, kps):
        assert image.shape == (80, 100, 3)
        np.testing.assert_allclose(kps, KPS)
        self.images.append(image.copy())
        return vector(image)


class StubHybrid:
    feature_space_id = "synthetic-frozen-hybrid-v4"
    descriptor_dim = 4

    def __init__(self, *args, skin_weight, preprocessor, **kwargs):
        self.global_model = StubGlobal()
        self.skin_model = StubSkin()
        self.skin_weight = skin_weight

    def extract_components(self, image, prepared_face):
        packed, pid = prepared_tuple_from_face_v4(prepared_face)
        return self.global_model.extract_features(image, KPS), self.skin_model.extract_features(packed, pid)


@pytest.fixture
def frozen_v4(tmp_path, monkeypatch):
    skin, global_model = tmp_path / "skin.npz", tmp_path / "global.onnx"
    skin.write_bytes(b"synthetic-skin-loader-is-stubbed")
    global_model.write_bytes(b"synthetic-global-loader-is-stubbed")
    sources = ("src/preprocessing_v3.py", "src/descriptors/hybrid_v4.py",
               "experiments/development_v3.py", "src/services/recognition_v4.py",
               "src/services/texture_v3.py", "src/database/database.py")
    methods = []
    for scope in SCOPES:
        method = {"method_id": scope, "scope": scope, "score_threshold": -.8,
                  "feature_space_id": StubSkin.feature_space_id if scope == "skin_only" else
                  StubHybrid.feature_space_id if scope == "hybrid" else StubGlobal.feature_space_id,
                  "preprocessing_id": BASE_ID + (":detector_only" if scope == "whole_face_detector_only" else ":strict")}
        if scope in ("skin_only", "hybrid"):
            method["skin_model_path"] = str(skin)
        if scope != "skin_only":
            method["global_model_path"] = str(global_model)
        if scope == "hybrid":
            method["skin_weight"] = .2
        methods.append(method)
    record = {"schema_version": "study-v4.1", "methods": methods,
              "model_artifacts": {str(p): module.digest(p) for p in (skin, global_model)},
              "source_hashes": {p: module.digest(module.ROOT / p) for p in sources},
              "runtime": {"global_provider": "cpu", "preprocessing_provider": "cpu"},
              "preprocessing_config": asdict(PreprocessingConfigV3())}
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(module, "FacePreprocessorV3", StubPreprocessor)
    monkeypatch.setattr(module.ClassicalTextureDescriptorV3, "load", lambda path: StubSkin())
    monkeypatch.setattr(module, "ArcFaceDescriptorV4", StubGlobal)
    monkeypatch.setattr(module, "HybridFaceSkinDescriptorV4", StubHybrid)
    return path, record


def rewrite(frozen_v4):
    path, record = frozen_v4
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize("scope", SCOPES)
def test_scope_routes_full_frame_enroll_recognize_reload_delete(frozen_v4, tmp_path, scope):
    path, _ = frozen_v4
    gallery = tmp_path / ("gallery_" + scope)
    service = module.RecognitionSystemV4(path, scope, gallery)
    service.register_person(frame(), "a", "Alice")
    service.register_person(frame(10), "a", "Alice")
    service.register_person(frame(140), "b", "Bob")
    name, similarity, candidates, distances = service.recognize(frame(30), return_distances=True)
    assert name == "Alice"
    assert [candidate["person_id"] for candidate in candidates] == ["a", "b"]
    assert similarity == pytest.approx(1 - distances[0] / 4)
    assert service.input_mode == "full_frame"
    assert len(service.preprocessor.images) == (0 if scope == "whole_face_detector_only" else 4)
    loaded = module.RecognitionSystemV4(path, scope, gallery)
    assert loaded.recognize(frame())[0] == "Alice"
    assert loaded.remove_person("a") == 2
    assert module.RecognitionSystemV4(path, scope, gallery).get_database_stats()["total_embeddings"] == 1


@pytest.mark.parametrize("mutation,match", [
    ("missing_source", "omits required"), ("bad_source", "implementation changed"),
    ("missing_model", "absent from frozen model_artifacts"), ("bad_model", "model changed"),
    ("bad_feature", "feature space mismatch"), ("bad_preprocessing", "preprocessing identity mismatch"),
    ("duplicate_method", "absent or ambiguous"), ("bad_threshold", "Invalid unit-vector")])
def test_frozen_contract_tampering_fails_closed(frozen_v4, mutation, match):
    _, record = frozen_v4
    if mutation == "missing_source": record["source_hashes"].pop("src/preprocessing_v3.py")
    elif mutation == "bad_source": record["source_hashes"]["src/preprocessing_v3.py"] = "0" * 64
    elif mutation == "missing_model": record["model_artifacts"].pop(record["methods"][0]["skin_model_path"])
    elif mutation == "bad_model": record["model_artifacts"][record["methods"][0]["skin_model_path"]] = "0" * 64
    elif mutation == "bad_feature": record["methods"][0]["feature_space_id"] = "another-model"
    elif mutation == "bad_preprocessing": record["methods"][0]["preprocessing_id"] = "another-preprocessor"
    elif mutation == "duplicate_method": record["methods"].append(dict(record["methods"][0]))
    elif mutation == "bad_threshold": record["methods"][0]["score_threshold"] = .1
    rewrite(frozen_v4)
    with pytest.raises(ValueError, match=match):
        module.RecognitionSystemV4(frozen_v4[0], "skin_only")


def test_gallery_rejects_different_preprocessing_contract(frozen_v4, tmp_path):
    path, _ = frozen_v4
    gallery = tmp_path / "gallery"
    module.RecognitionSystemV4(path, "whole_face", gallery).register_person(frame(), "a", "Alice")
    with pytest.raises(ValueError, match="feature.space|Feature.space"):
        module.RecognitionSystemV4(path, "whole_face_detector_only", gallery)


def test_failed_writes_roll_back_enrollment_and_deletion(frozen_v4, tmp_path, monkeypatch):
    path, _ = frozen_v4
    gallery = tmp_path / "gallery"
    service = module.RecognitionSystemV4(path, "skin_only", gallery)
    service.register_person(frame(), "a", "Alice")
    before = service.get_database_stats()
    disk_before = gallery.with_suffix(".npz").read_bytes()
    next_id = service.database.next_id
    def failure(path): raise OSError("simulated storage failure")
    monkeypatch.setattr(service.database, "save", failure)
    with pytest.raises(OSError): service.register_person(frame(10), "b", "Bob")
    assert service.get_database_stats() == before
    assert service.database.next_id == next_id
    with pytest.raises(OSError): service.remove_person("a")
    assert service.get_database_stats() == before
    assert gallery.with_suffix(".npz").read_bytes() == disk_before


def test_float64_score_boundary_survives_float32_storage_reload(frozen_v4, tmp_path):
    path, record = frozen_v4
    score = float(-cdist(vector(frame(80)).astype(float)[None], vector(frame()).astype(float)[None])[0, 0])
    gallery = tmp_path / "gallery"
    for threshold, expected in ((score, "Alice"), (np.nextafter(score, np.inf), None)):
        record["methods"][0]["search_score_threshold"] = float(threshold)
        rewrite(frozen_v4)
        service = module.RecognitionSystemV4(path, "skin_only", gallery)
        if not service.get_database_stats()["total_embeddings"]:
            service.register_person(frame(), "a", "Alice")
        assert service.recognize(frame(80))[0] == expected
        assert module.RecognitionSystemV4(path, "skin_only", gallery).recognize(frame(80))[0] == expected


def test_missing_roi_quality_flag_zeroes_skin_block(frozen_v4):
    service = module.RecognitionSystemV4(frozen_v4[0], "skin_only")
    service.preprocessor.missing_flag = True
    service.register_person(frame(), "a", "Alice")
    assert not service.descriptor.last_masks.any()


def test_exact_cross_identity_tie_and_feature_mutation(frozen_v4):
    service = module.RecognitionSystemV4(frozen_v4[0], "skin_only")
    service.register_person(frame(), "a", "Alice")
    service.register_person(frame(), "b", "Bob")
    assert service.recognize(frame())[0] is None
    service.descriptor.feature_space_id = "changed"
    with pytest.raises(ValueError, match="Descriptor changed"):
        service.recognize(frame())


def test_api_v4_factory_full_frame_http_and_reload(frozen_v4, tmp_path, monkeypatch):
    api = importlib.import_module("src.api.app")
    path, _ = frozen_v4
    monkeypatch.setattr(api, "BACKEND", "v4")
    monkeypatch.setattr(api, "PROFILE", {})
    monkeypatch.setattr(api, "RECOGNITION_SYSTEM", None)
    monkeypatch.setattr(api, "DATABASE_PATH", str(tmp_path / "http_gallery"))
    monkeypatch.setenv("SKIN_V4_FREEZE_PATH", str(path))
    monkeypatch.setenv("SKIN_V4_METHOD", "skin_only")
    def forbidden_crop(image): raise AssertionError("V4 must receive the complete frame")
    monkeypatch.setattr(api, "detect_face", forbidden_crop)
    data = io.BytesIO()
    Image.fromarray(frame()).save(data, format="PNG")
    upload = {"image": ("synthetic.png", data.getvalue(), "image/png")}
    with TestClient(api.app) as client:
        assert client.get("/health").json()["database"]["version"] == "4.1"
        response = client.post("/api/v1/register", files=upload, data={"person_id": "a", "person_name": "Alice"})
        assert response.status_code == 200, response.text
        response = client.post("/api/v1/recognize", files=upload)
        assert response.status_code == 200, response.text
        assert response.json()["matched"] and response.json()["person_name"] == "Alice"
        monkeypatch.setattr(api, "RECOGNITION_SYSTEM", None)
        assert client.get("/api/v1/stats").json()["total_persons"] == 1
        assert client.delete("/api/v1/person/a").json()["deleted_embeddings"] == 1
        monkeypatch.setattr(api, "RECOGNITION_SYSTEM", None)
        assert client.get("/api/v1/stats").json()["total_persons"] == 0


def test_api_factory_refuses_missing_profile(frozen_v4, monkeypatch):
    api = importlib.import_module("src.api.app")
    monkeypatch.setattr(api, "BACKEND", "v4")
    monkeypatch.setattr(api, "PROFILE", {})
    monkeypatch.setattr(api, "RECOGNITION_SYSTEM", None)
    monkeypatch.delenv("SKIN_V4_FREEZE_PATH", raising=False)
    monkeypatch.delenv("SKIN_V4_METHOD", raising=False)
    with pytest.raises(ValueError, match="freeze_path and method_id"):
        api.init_recognition_system()
