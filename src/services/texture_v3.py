"""Frozen v3 skin-texture inference on full BGR frames.

The FEI development threshold is a pairwise verification threshold, not a
guarantee of 1:N identification error. No v2 gallery or live model is migrated.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
import threading

import numpy as np
from scipy.spatial.distance import cdist

from experiments.development_v3 import ClassicalTextureDescriptorV3, prepared_tuple_from_face
from src.database import SkinEmbeddingDatabase
from src.preprocessing_v3 import FacePreprocessorV3, PreprocessingConfigV3, QualityRejection

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


class SkinRecognitionSystemV3:
    """Full-frame detection/alignment, frozen strict skin descriptor and local gallery."""
    input_mode = "full_frame"
    profile = "strict_skin_dense_upright_RootSIFT_PCA128_whitened"

    def __init__(self, model_path, calibration_path, database_path=None, model_dir=None, providers=None,
                 search_operating_point_path=None):
        if not model_path or not calibration_path:
            raise ValueError("Both frozen model_path and calibration_path are required")
        self._lock = threading.RLock()
        self.model_path = Path(model_path).resolve()
        self.calibration_path = Path(calibration_path).resolve()
        self.database_path = database_path
        try:
            self.descriptor = ClassicalTextureDescriptorV3.load(self.model_path)
            self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
            raw = self.calibration_path.read_bytes()
            self.calibration_id = hashlib.sha256(raw).hexdigest()
            self.calibration = json.loads(raw.decode("utf-8-sig"))
            self._validate_frozen_profile()
            config = PreprocessingConfigV3(**self.calibration["preprocessing_config"])
            self.preprocessor = FacePreprocessorV3(model_dir=model_dir, config=config, providers=providers)
            if self.preprocessor.preprocessing_id != self.calibration["preprocessor_base_id"]:
                raise ValueError("Live preprocessing configuration differs from the frozen profile")
            if self.preprocessor.model_sha256 != self.calibration["preprocessing_model_sha256"]:
                raise ValueError("Live detector or landmark model differs from the frozen profile")
            self.preprocessing_id = self.preprocessor.preprocessing_id + ":strict"
            if self.preprocessing_id != self.descriptor.preprocessing_id:
                raise ValueError("Preprocessor and descriptor feature-space contracts differ")
            self._descriptor_feature_space_id = self.descriptor.feature_space_id
            composite = json.dumps({"descriptor": self._descriptor_feature_space_id,
                                    "preprocessing": self.preprocessing_id,
                                    "input_mode": self.input_mode}, sort_keys=True)
            self.feature_space_id = "skin-service-v3:" + hashlib.sha256(composite.encode()).hexdigest()
            self.descriptor_dim = int(self.descriptor.descriptor_dim)
            self.primary_score_threshold = float(self.calibration["score_threshold"])
            self.score_threshold = self.primary_score_threshold
            self.search_operating_point = None
            self.search_operating_point_id = None
            if search_operating_point_path:
                self._load_search_operating_point(search_operating_point_path)
            self.reject_all = self.score_threshold > 0
            self.recognition_threshold = (self.score_threshold ** 2) if not self.reject_all else None
            self.database = SkinEmbeddingDatabase(self.descriptor_dim, index_type="L2",
                                                  feature_space_id=self.feature_space_id)
            if database_path:
                path = self.database.storage_path(database_path)
                if path.exists():
                    self.database.load(database_path)
                elif Path(str(database_path)+".pkl").exists():
                    raise ValueError("Legacy gallery exists at this location; choose a new v3 gallery path")
            LOGGER.info("V3 skin profile loaded: candidate=%s dimensions=%d calibration=%s",
                        self.calibration["candidate_id"], self.descriptor_dim, self.calibration_id)
        except Exception:
            LOGGER.exception("Cannot initialize frozen v3 skin-texture service")
            raise

    @classmethod
    def from_env(cls):
        model = os.getenv("SKIN_V3_MODEL_PATH")
        calibration = os.getenv("SKIN_V3_CALIBRATION_PATH")
        if not model or not calibration:
            raise ValueError("Set SKIN_V3_MODEL_PATH and SKIN_V3_CALIBRATION_PATH explicitly")
        return cls(model, calibration, database_path=os.getenv("SKIN_V3_DATABASE_PATH"),
                   model_dir=os.getenv("SKIN_V3_PREPROCESSOR_MODEL_DIR"),
                   search_operating_point_path=os.getenv("SKIN_V3_SEARCH_POINT_PATH"))

    def _load_search_operating_point(self, path):
        raw = Path(path).read_bytes()
        point = json.loads(raw.decode("utf-8-sig"))
        if point.get("schema_version") != "skin-recognition-v3-search-operating-point-1":
            raise ValueError("Unsupported search operating-point schema")
        if point.get("source_freeze_sha256") != self.calibration_id:
            raise ValueError("Search operating point belongs to a different primary freeze")
        for key, expected in (("feature_space_id", self._descriptor_feature_space_id),
                              ("preprocessing_id", self.preprocessing_id),
                              ("model_sha256", self.model_sha256),
                              ("score_definition", "negative_euclidean_distance"),
                              ("decision_rule", "accept if score >= score_threshold")):
            if point.get(key) != expected:
                raise ValueError("Search operating point contract mismatch: " + key)
        threshold = float(point["score_threshold"])
        if not np.isfinite(threshold) or not -2 <= threshold <= 0:
            raise ValueError("Search threshold must be finite and in [-2, 0]")
        self.search_operating_point, self.score_threshold = point, threshold
        self.search_operating_point_id = hashlib.sha256(raw).hexdigest()

    def _validate_frozen_profile(self):
        record = self.calibration
        if record.get("schema_version") != "skin-recognition-v3-freeze-1":
            raise ValueError("Unsupported v3 frozen calibration schema")
        if record.get("model_sha256") != self.model_sha256:
            raise ValueError("Frozen model checksum does not match calibration")
        if record.get("feature_space_id") != self.descriptor.feature_space_id:
            raise ValueError("Frozen descriptor feature-space mismatch")
        if record.get("preprocessing_id") != self.descriptor.preprocessing_id:
            raise ValueError("Frozen preprocessing identity mismatch")
        if record.get("preprocessing_mode") != "landmark_strict":
            raise ValueError("This service requires landmark_strict skin preprocessing")
        if record.get("configuration") != self.descriptor.configuration:
            raise ValueError("Frozen feature configuration does not match the model")
        if self.descriptor.configuration.get("features", {}).get("scope") != "skin_regions":
            raise ValueError("This service accepts only strict skin-region descriptors")
        if record.get("score_definition") != "negative_euclidean_distance":
            raise ValueError("Unsupported calibration score definition")
        threshold = float(record["score_threshold"])
        if not np.isfinite(threshold) or threshold < -2:
            raise ValueError("Invalid threshold for unit-norm negative-Euclidean scores")
        if record.get("decision_rule") != "accept if score >= score_threshold":
            raise ValueError("Unsupported calibration decision rule")
        metrics_threshold = record.get("calibration_metrics", {}).get("threshold")
        if metrics_threshold is None or float(metrics_threshold) != threshold:
            raise ValueError("Frozen threshold and calibration metrics disagree")
        source_hashes = record.get("source_sha256", {})
        for required in ("experiments/development_v3.py", "src/preprocessing_v3.py"):
            if required not in source_hashes:
                raise ValueError("Frozen profile is missing an inference source checksum")
        for relative, expected in source_hashes.items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError("Frozen source path escapes the project")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen inference source changed: " + relative)

    def _assert_feature_space(self):
        if self.descriptor.feature_space_id != self._descriptor_feature_space_id:
            raise ValueError("Descriptor changed after service creation; new enrollment is required")
        if self.preprocessor.preprocessing_id + ":strict" != self.preprocessing_id:
            raise ValueError("Preprocessor changed after service creation; new enrollment is required")
        if self.database.feature_space_id != self.feature_space_id:
            raise ValueError("Gallery belongs to a different feature space")

    def _extract_vector(self, image):
        self._assert_feature_space()
        prepared_face = self.preprocessor.process(image)
        prepared, preprocessing_id = prepared_tuple_from_face(prepared_face, mode="strict")
        vector = np.asarray(self.descriptor.extract_features(prepared, preprocessing_id), dtype=np.float32)
        if vector.shape != (self.descriptor_dim,) or not np.isfinite(vector).all():
            raise ValueError("Frozen descriptor produced an invalid vector")
        if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5):
            raise ValueError("Frozen descriptor must produce a unit-norm vector")
        return vector

    def detect_face_info(self, image):
        """Detection-only overlay in original coordinates; no fake landmarks or fallback."""
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
            raise QualityRejection(["invalid_bgr_uint8_image"])
        with self.preprocessor._lock:
            boxes, keypoints = self.preprocessor.detector.detect(np.ascontiguousarray(image),
                                                                max_num=0, metric="default")
        if len(boxes) == 0:
            return None
        if len(boxes) != 1:
            raise QualityRejection(["multiple_faces"], {"detected_faces": int(len(boxes))})
        x1, y1, x2, y2 = boxes[0, :4]
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(image.shape[1], int(np.ceil(x2))), min(image.shape[0], int(np.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            raise QualityRejection(["invalid_detector_bbox"])
        information = {"bbox": (x1, y1, x2-x1, y2-y1), "score": float(boxes[0, 4]),
                       "detector": "SCRFD-local-frozen-v3"}
        if keypoints is not None:
            information["kps"] = np.asarray(keypoints[0], dtype=np.float32).tolist()
        return information

    def _snapshot(self):
        return (list(self.database.embeddings), list(self.database.metadata),
                dict(self.database.id_to_metadata), self.database.next_id, self.database.last_update)

    def _restore(self, snapshot):
        (self.database.embeddings, self.database.metadata, self.database.id_to_metadata,
         self.database.next_id, self.database.last_update) = snapshot

    def register_person(self, image, person_id, person_name, image_path=None, additional_metadata=None):
        with self._lock:
            vector = self._extract_vector(image)
            with self.database._lock:
                before = self._snapshot()
                try:
                    embedding_id = self.database.add_embedding(vector, person_id, person_name,
                                                                image_path, additional_metadata)
                    if self.database_path:
                        self.database.save(self.database_path)
                except Exception:
                    self._restore(before)
                    LOGGER.exception("V3 enrollment failed; live gallery transaction was rolled back")
                    raise
            return embedding_id

    def recognize(self, image, k=5, return_distances=False):
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= 100:
            raise ValueError("k must be an integer between 1 and 100")
        with self._lock:
            vector = self._extract_vector(image)
            # Match the frozen scientific scorer's float64 accumulation over
            # float32 embeddings, including scores exactly at the cutoff.
            with self.database._lock:
                if self.database.embeddings:
                    matrix = np.asarray(self.database.embeddings, dtype=np.float64)
                    scores = -cdist(vector.astype(np.float64)[None, :], matrix, metric="euclidean")[0]
                    order = np.argsort(-scores, kind="stable")
                    results = [(self.database.metadata[row]["id"], float(scores[row]),
                                dict(self.database.metadata[row])) for row in order]
                else:
                    results = []
            # Rank identities, not templates, and reject exact cross-identity ties.
            by_identity = {}
            for embedding_id, score, metadata in results:
                if metadata["person_id"] not in by_identity:
                    by_identity[metadata["person_id"]] = dict(metadata, squared_l2=float(score**2),
                                                               verification_score=score)
            ranked = list(by_identity.values())
            if not ranked:
                result = (None, 0.0, [])
                return result + ([],) if return_distances else result
            distance = ranked[0]["squared_l2"]
            similarity = float(np.clip(1.0 - distance/4.0, 0.0, 1.0))
            tied = len(ranked) > 1 and ranked[1]["squared_l2"] == distance
            # Compare the canonical frozen score directly: a nextafter threshold
            # must not lose its strict boundary when it is squared for display.
            accepted = not self.reject_all and not tied and ranked[0]["verification_score"] >= self.score_threshold
            result = (ranked[0]["person_name"] if accepted else None, similarity, ranked[:k])
            return result + ([item["squared_l2"] for item in ranked[:k]],) if return_distances else result

    def remove_person(self, person_id):
        with self._lock, self.database._lock:
            self._assert_feature_space()
            before = self._snapshot()
            try:
                identifiers = [index for index, _ in self.database.get_person_embeddings(person_id)]
                for identifier in identifiers:
                    self.database.remove_embedding(identifier)
                if identifiers and self.database_path:
                    self.database.save(self.database_path)
            except Exception:
                self._restore(before)
                LOGGER.exception("V3 deletion failed; live gallery transaction was rolled back")
                raise
            return len(identifiers)

    def save_database(self, filepath):
        with self._lock:
            self._assert_feature_space()
            self.database.save(filepath)

    def load_database(self, filepath):
        with self._lock:
            self._assert_feature_space()
            self.database.load(filepath)
            self.database_path = filepath

    def get_database_stats(self):
        with self._lock:
            self._assert_feature_space()
            return dict(self.database.get_stats(), prototype=True, version="3.1",
                        input_mode=self.input_mode, profile=self.calibration["primary_method"],
                        candidate_id=self.calibration["candidate_id"],
                        descriptor_feature_space_id=self._descriptor_feature_space_id,
                        preprocessing_id=self.preprocessing_id, model_sha256=self.model_sha256,
                        calibration_id=self.calibration_id,
                        calibration_dataset="FEI development calibration",
                        calibration_scope="pairwise_verification_development",
                        threshold_status=("development_zero_false_match_search_point" if self.search_operating_point else
                                          "FEI_development_pairwise_only"),
                        score_definition="negative_euclidean_distance",
                        score_threshold=self.score_threshold, recognition_threshold=self.recognition_threshold,
                        primary_score_threshold=self.primary_score_threshold,
                        search_operating_point_id=self.search_operating_point_id,
                        reject_all=self.reject_all, score_type="normalized_squared_l2_similarity_not_probability",
                        identification_error_guarantee=False,
                        note="Pairwise FEI calibration does not establish 1:N false-identification risk",
                        tie_policy="reject_exact_cross_identity_ties",
                        model_license="InsightFace pretrained preprocessing weights: non-commercial research")

