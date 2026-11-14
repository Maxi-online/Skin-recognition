"""Versioned inference for the native-small-frame geometry repair.

Model parameters and recognition thresholds are inherited unchanged from the
frozen V4.1 study. The effective preprocessing contract and gallery space are new.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.database import SkinEmbeddingDatabase
from src.descriptors.hybrid_v4 import fuse_embeddings, source_landmarks_from_cache
from src.preprocessing_v42 import FacePreprocessorV42, prepared_tuple_from_face_v42, source_landmarks_for_target_face_v42
from src.services.recognition_v4 import RecognitionSystemV4, resolve, digest, ROOT


class RecognitionSystemV42(RecognitionSystemV4):
    """Same learned model and thresholds, explicit new input geometry and gallery."""
    def __init__(self, freeze_path, method_id, database_path=None):
        calibration_path = resolve(freeze_path)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
        if calibration.get("schema_version") != "study-v4.2":
            raise ValueError("Expected study-v4.2 frozen profile")
        base_path = resolve(calibration["base_freeze_path"])
        if digest(base_path) != calibration["base_freeze_sha256"]:
            raise ValueError("V4.1 base freeze checksum mismatch")
        sources = calibration.get("source_hashes", {})
        if not {"src/preprocessing_v42.py", "src/services/recognition_v42.py"}.issubset(sources):
            raise ValueError("V4.2 freeze omits its adapter/service source checksums")
        for relative, expected in sources.items():
            path = resolve(relative)
            if not path.is_relative_to(ROOT) or digest(path) != expected:
                raise ValueError("Frozen V4.2 implementation changed: "+relative)
        matches = [method for method in calibration["methods"] if method["method_id"] == method_id]
        if len(matches) != 1:
            raise ValueError("Requested method is absent or ambiguous in the V4.2 freeze")
        method = matches[0]
        super().__init__(base_path, method_id, database_path=None)
        inherited_method = dict(self.method)
        base_runtime = self.calibration.get("runtime", {})
        for key in ("runtime", "preprocessing_config", "preprocessing_model_dir", "model_artifacts"):
            if key in calibration and calibration[key] != self.calibration.get(key):
                raise ValueError("V4.2 geometry repair cannot change the base runtime, preprocessing gates or models")
        immutable = ("scope", "skin_weight", "feature_space_id", "descriptor_dim", "score_threshold",
                     "search_score_threshold", "skin_model_path", "global_model_path", "score_definition")
        if any(method.get(key) != inherited_method.get(key) for key in immutable):
            raise ValueError("V4.2 geometry repair must preserve the V4.1 model, fusion and thresholds")
        self.base_calibration_id = self.calibration_id
        self.base_calibration_path = self.calibration_path
        self.calibration_path = calibration_path
        self.calibration = calibration
        self.calibration_id = digest(calibration_path)
        self.method = method
        self.preprocessor = FacePreprocessorV42(base_preprocessor=self.preprocessor)
        self.legacy_preprocessing_id = self.preprocessor.legacy_preprocessing_id+self.preprocessing_suffix
        self.preprocessing_id = self.preprocessor.preprocessing_id+self.preprocessing_suffix
        if self.preprocessing_id != method["preprocessing_id"]:
            raise ValueError("Effective V4.2 preprocessing identity mismatch")
        if method.get("legacy_preprocessing_id", self.legacy_preprocessing_id) != self.legacy_preprocessing_id:
            raise ValueError("Explicit legacy descriptor input mapping differs from the V4.1 contract")
        contract = {"descriptor": self._descriptor_feature_space_id, "preprocessing": self.preprocessing_id,
                    "runtime": base_runtime,
                    "storage": "float32", "scoring": "negative_scipy_float64_euclidean"}
        self.feature_space_id = "recognition-service-v4.2:"+hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        self.search_operating_point_id = self.calibration_id+":"+method_id
        self.database_path = database_path
        self.database = SkinEmbeddingDatabase(self.descriptor_dim, index_type="L2", feature_space_id=self.feature_space_id)
        if database_path:
            if self.database.storage_path(database_path).exists():
                self.database.load(database_path)
            elif Path(str(database_path)+".pkl").exists():
                raise ValueError("Use a new V4.2 gallery; legacy pickle galleries are incompatible")

    def _extract_vector(self, image):
        self._assert_feature_space()
        if self.scope == "whole_face_detector_only":
            points, _ = source_landmarks_for_target_face_v42(self.preprocessor, image)
            vector = self.descriptor.extract_features(image, points)
        else:
            prepared = self.preprocessor.process(image)
            if self.scope == "whole_face":
                points = source_landmarks_from_cache(prepared.kps5, prepared.transform)
                vector = self.descriptor.extract_features(image, points)
            else:
                packed, legacy_pid, effective_pid = prepared_tuple_from_face_v42(prepared)
                if effective_pid != self.preprocessing_id:
                    raise ValueError("Runtime input contract differs from the gallery")
                if self.scope == "skin_only":
                    vector = self.descriptor.extract_features(packed, legacy_pid)
                elif self.scope == "hybrid":
                    skin = self.descriptor.skin_model.extract_features(packed, legacy_pid)
                    points = source_landmarks_from_cache(prepared.kps5, prepared.transform)
                    global_feature = self.descriptor.global_model.extract_features(image, points)
                    vector = fuse_embeddings(global_feature, skin, self.descriptor.skin_weight)
                else:
                    raise ValueError("Unknown frozen recognition scope")
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.descriptor_dim,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
            raise ValueError("Invalid frozen recognition embedding")
        return vector

    def get_database_stats(self):
        stats = super().get_database_stats()
        stats.update(version="4.2", effective_preprocessing_id=self.preprocessing_id,
                     legacy_descriptor_preprocessing_id=self.legacy_preprocessing_id,
                     base_calibration_id=self.base_calibration_id,
                     input_geometry_adapter="native_small_frame_centered_zero_padding_no_upscale",
                     target_selection="single_face_or_unique_bbox_containing_original_frame_center",
                     thresholds="unchanged_from_v4.1_FEI_calibration")
        return stats
