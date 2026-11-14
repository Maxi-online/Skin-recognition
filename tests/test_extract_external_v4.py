"""Frozen extraction integration on synthetic arrays and stub model sessions only."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import experiments.extract_external_v4 as extractor
import experiments.development_v3 as development
import src.descriptors.hybrid_v4 as hybrid
import src.preprocessing_v3 as preprocessing
import src.utils.data_loader as data_loader
from experiments.external_evaluate_v4 import validate_freeze_bundle


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    actions = {1: "accept", 2: "strict_reject", 3: "all_reject"}
    reads = []
    monkeypatch.setattr(extractor, "ROOT", tmp_path)
    kps = np.array([[20., 20.], [60., 20.], [40., 40.], [25., 60.], [55., 60.]])

    class Processor:
        preprocessing_id = "fixture"
        runtime_providers = {"detector": ["CPUExecutionProvider"], "landmarks": ["CPUExecutionProvider"]}
        def __init__(self, *args, **kwargs):
            pass
        def process(self, image):
            identifier = int(image[0, 0, 0])
            if actions[identifier] != "accept":
                raise preprocessing.QualityRejection([actions[identifier]])
            return SimpleNamespace(identifier=identifier, kps5=kps, transform=np.array([[1., 0., 0.], [0., 1., 0.]]))

    class Skin:
        descriptor_dim = 3
        preprocessing_id = "fixture:strict"
        feature_space_id = "fixture_skin"
        def extract_features(self, packed, preprocessing_id):
            assert preprocessing_id == self.preprocessing_id
            return hybrid.unit_vector(np.array([1., 2., 3.])).astype(np.float32)

    class Global:
        descriptor_dim = 512
        runtime_providers = ["CPUExecutionProvider"]
        feature_space_id = "fixture_global"
        def __init__(self, path, *args, **kwargs):
            self.model_path = Path(path).resolve()
            self.model_sha256 = extractor.sha256(path)
        def extract_features(self, image, points):
            assert isinstance(points, np.ndarray) and points.shape == (5, 2), "Detector helper tuple was not unpacked"
            return hybrid.unit_vector(np.arange(1, 513, dtype=np.float64))

    def detector_only(processor, image):
        if actions[int(image[0, 0, 0])] == "all_reject":
            raise preprocessing.QualityRejection(["no_face"])
        return kps, {"quality_gate": "synthetic_detector_only"}

    def read_image(path):
        identifier = int(Path(path).stem)
        reads.append(identifier)
        return np.full((80, 80, 3), identifier, np.uint8)

    monkeypatch.setattr(preprocessing, "FacePreprocessorV3", Processor)
    monkeypatch.setattr(development.ClassicalTextureDescriptorV3, "load", lambda path: Skin())
    monkeypatch.setattr(hybrid, "prepared_tuple_from_face_v4", lambda face: ((face.identifier,), "fixture:strict"))
    monkeypatch.setattr(hybrid, "ArcFaceDescriptorV4", Global)
    monkeypatch.setattr(hybrid, "source_landmarks_for_single_face", detector_only)
    monkeypatch.setattr(data_loader, "read_image", read_image)
    for filename in ("skin.npz", "global.onnx"):
        (tmp_path / filename).write_bytes(b"synthetic model bytes; never parsed")
    hybrid_model = hybrid.HybridFaceSkinDescriptorV4(tmp_path / "global.onnx", tmp_path / "skin.npz", .1)
    methods = [
        {"method_id": "skin", "scope": "skin_only", "skin_model_path": "skin.npz", "feature_space_id": "fixture_skin"},
        {"method_id": "global", "scope": "whole_face", "global_model_path": "global.onnx", "feature_space_id": "fixture_global"},
        {"method_id": "hybrid", "scope": "hybrid", "skin_model_path": "skin.npz", "global_model_path": "global.onnx",
         "skin_weight": .1, "feature_space_id": hybrid_model.feature_space_id},
        {"method_id": "detector", "scope": "whole_face_detector_only", "global_model_path": "global.onnx", "feature_space_id": "fixture_global"},
    ]
    for method in methods:
        method.update(preprocessing_id="fixture" + (":detector_only" if method["method_id"] == "detector" else ":strict"),
                      score_threshold=-.5, search_score_threshold=-.4)
    rows = []
    for identifier in actions:
        path = tmp_path / f"{identifier}.synthetic"
        path.write_bytes(bytes([identifier]))
        rows.append({"path": path.as_posix(), "subject_id": "subject" + str(identifier), "sha256": extractor.sha256(path)})
    inventory = tmp_path / "inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = {"protocol_id": "orl_external_v4_1", "inventory_path": "inventory.csv", "inventory_sha256": extractor.sha256(inventory)}
    extractor.write_json(tmp_path / "protocol.json", protocol)
    sources = {}
    for name in ("experiments/extract_external_v4.py", "experiments/development_v3.py", "src/descriptors/hybrid_v4.py",
                 "src/preprocessing_v3.py", "experiments/external_evaluate_v4.py", "src/utils/data_loader.py",
                 "experiments/external_evaluate_v3.py", "experiments/scientific_metrics.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic frozen source", encoding="utf-8")
        sources[name] = extractor.sha256(path)
    freeze = {"schema_version": "study-v4.1", "timestamp": "2000-01-01T00:00:00+00:00", "source_hashes": sources,
              "model_artifacts": {name: extractor.sha256(tmp_path / name) for name in ("skin.npz", "global.onnx")},
              "methods": methods, "preprocessing_config": {}, "runtime": {"global_provider": "cpu", "preprocessing_provider": "cpu"},
              "protocols": {"orl": {"path": "protocol.json", "sha256": extractor.sha256(tmp_path / "protocol.json")}}}
    extractor.write_json(tmp_path / "freeze.json", freeze)
    return SimpleNamespace(root=tmp_path, actions=actions, reads=reads, freeze=freeze, methods=methods,
                           freeze_path=tmp_path / "freeze.json", output=tmp_path / "output")


def test_partial_quality_rejection_keeps_independent_detector_coverage_and_compatible_bundles(fixture):
    extractor.extract(fixture.freeze_path, "orl", fixture.output)
    for method in fixture.methods:
        directory = fixture.output / method["method_id"]
        validate_freeze_bundle(fixture.freeze_path, directory / "bundle.json", "orl", fixture.root)
        quality = json.loads((directory / "quality.json").read_text())
        assert len(quality) == 3
        expected = ["accepted", "accepted", "rejected"] if method["method_id"] == "detector" else ["accepted", "rejected", "rejected"]
        assert [r["status"] for r in quality] == expected
        assert [r["embedding_index"] for r in quality if r["status"] == "accepted"] == list(range(expected.count("accepted")))
        assert quality[-1]["rejection_codes"]


def test_all_rejected_exports_valid_empty_matrices_for_every_scope(fixture):
    fixture.actions.update({key: "all_reject" for key in fixture.actions})
    extractor.extract(fixture.freeze_path, "orl", fixture.output)
    for method_id, dimension in (("skin", 3), ("global", 512), ("hybrid", 515), ("detector", 512)):
        with np.load(fixture.output / method_id / "embeddings.npz", allow_pickle=False) as arrays:
            assert arrays["embeddings"].shape == (0, dimension)
        quality = json.loads((fixture.output / method_id / "quality.json").read_text())
        assert len(quality) == 3 and all(row["status"] == "rejected" for row in quality)


def test_canonical_float32_rounding_is_preserved_and_finished_resume_skips_decoding(fixture):
    extractor.extract(fixture.freeze_path, "orl", fixture.output)
    expected = hybrid.unit_vector(np.arange(1, 513, dtype=np.float64)).astype(np.float32).astype(np.float64)
    with np.load(fixture.output / "global/embeddings.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["embeddings"][0], expected)
        assert arrays["embeddings"].dtype == np.float64
    before = list(fixture.reads)
    extractor.extract(fixture.freeze_path, "orl", fixture.output)
    assert fixture.reads == before


def test_resume_refuses_changed_checkpoint_embedding_bytes(fixture):
    extractor.extract(fixture.freeze_path, "orl", fixture.output)
    (fixture.output / "skin.checkpoint.npz").write_bytes(b"corrupted numeric checkpoint")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        extractor.extract(fixture.freeze_path, "orl", fixture.output)


def test_resume_refuses_checkpoint_identity_or_embedding_index_corruption(fixture):
    extractor.extract(fixture.freeze_path, "orl", fixture.output)
    path = fixture.output / "checkpoint.json"
    original = json.loads(path.read_text())
    for field, value in (("subject_id", "another_person"), ("source_sha256", "f" * 64), ("embedding_index", 7)):
        changed = json.loads(json.dumps(original))
        changed["quality"]["skin"][0][field] = value
        extractor.write_json(path, changed)
        with pytest.raises(ValueError):
            extractor.extract(fixture.freeze_path, "orl", fixture.output)


def test_frozen_feature_identity_must_match_actual_model_before_input_decoding(fixture):
    fixture.freeze["methods"][0]["feature_space_id"] = "unrelated_model_feature_space"
    extractor.write_json(fixture.freeze_path, fixture.freeze)
    with pytest.raises(ValueError, match="[Ff]eature"):
        extractor.extract(fixture.freeze_path, "orl", fixture.output)
    assert fixture.reads == []
