"""Frozen skin-only, whole-face and hybrid inference with a safe local gallery."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from experiments.development_v3 import ClassicalTextureDescriptorV3
from src.database import SkinEmbeddingDatabase
from src.descriptors.hybrid_v4 import (ArcFaceDescriptorV4, HybridFaceSkinDescriptorV4,
                                     fuse_embeddings, source_landmarks_from_cache,
                                     source_landmarks_for_single_face, prepared_tuple_from_face_v4)
from src.preprocessing_v3 import FacePreprocessorV3, PreprocessingConfigV3
from src.services.texture_v3 import SkinRecognitionSystemV3

ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecognitionSystemV4(SkinRecognitionSystemV3):
    """Reuse transactional enrollment/search while validating the v4 contract."""
    input_mode = 'full_frame'

    def __init__(self, freeze_path, method_id, database_path=None):
        self._lock = threading.RLock()
        self.calibration_path = resolve(freeze_path)
        self.calibration_id = digest(self.calibration_path)
        self.calibration = json.loads(self.calibration_path.read_text(encoding='utf-8-sig'))
        if self.calibration.get('schema_version') != 'study-v4.1':
            raise ValueError('Expected study-v4.1 frozen profile')
        matches = [m for m in self.calibration['methods'] if m['method_id'] == method_id]
        if len(matches) != 1:
            raise ValueError('Requested method is absent or ambiguous in the freeze')
        self.method = matches[0]
        self.scope = self.method['scope']
        self.profile = method_id
        required_sources = {'src/preprocessing_v3.py', 'src/descriptors/hybrid_v4.py',
                            'experiments/development_v3.py', 'src/services/recognition_v4.py',
                            'src/services/texture_v3.py', 'src/database/database.py'}
        if not required_sources.issubset(self.calibration['source_hashes']):
            raise ValueError('Frozen profile omits required inference source hashes')
        for key in ('skin_model_path', 'global_model_path'):
            if self.method.get(key) and self.method[key] not in self.calibration['model_artifacts']:
                raise ValueError('Selected inference model is absent from frozen model_artifacts')
        for relative, expected in self.calibration['source_hashes'].items():
            path = resolve(relative)
            if not path.is_relative_to(ROOT) or digest(path) != expected:
                raise ValueError('Frozen implementation changed: ' + relative)
        for path, expected in self.calibration['model_artifacts'].items():
            if digest(resolve(path)) != expected:
                raise ValueError('Frozen model changed: ' + path)
        runtime = self.calibration.get('runtime', {})
        providers = {}
        if 'cuda' in (runtime.get('global_provider'), runtime.get('preprocessing_provider')):
            import torch
            import onnxruntime as ort
            ort.preload_dlls()
            if not torch.cuda.is_available():
                raise RuntimeError('This frozen profile requires its CUDA environment')
        for component in ('global', 'preprocessing'):
            providers[component] = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                                    if runtime.get(component + '_provider', 'cpu') == 'cuda'
                                    else ['CPUExecutionProvider'])
        self.preprocessor = FacePreprocessorV3(
            model_dir=self.calibration.get('preprocessing_model_dir'),
            config=PreprocessingConfigV3(**self.calibration.get('preprocessing_config', {})),
            providers=providers['preprocessing'])
        self.preprocessing_suffix = ':detector_only' if self.scope == 'whole_face_detector_only' else ':strict'
        self.preprocessing_id = self.preprocessor.preprocessing_id + self.preprocessing_suffix
        if self.preprocessing_id != self.method['preprocessing_id']:
            raise ValueError('Frozen preprocessing identity mismatch')
        if providers['preprocessing'][0] == 'CUDAExecutionProvider' and any(
                p[0] != 'CUDAExecutionProvider' for p in self.preprocessor.runtime_providers.values()):
            raise RuntimeError('Preprocessor silently fell back from CUDA')
        if self.scope == 'skin_only':
            self.descriptor = ClassicalTextureDescriptorV3.load(resolve(self.method['skin_model_path']))
        elif self.scope in ('whole_face', 'whole_face_detector_only'):
            self.descriptor = ArcFaceDescriptorV4(resolve(self.method['global_model_path']), providers['global'])
        elif self.scope == 'hybrid':
            self.descriptor = HybridFaceSkinDescriptorV4(
                resolve(self.method['global_model_path']), resolve(self.method['skin_model_path']),
                skin_weight=self.method['skin_weight'], preprocessor=self.preprocessor, providers=providers['global'])
        else:
            raise ValueError('Unknown frozen recognition scope')
        if self.scope in ('whole_face', 'whole_face_detector_only', 'hybrid'):
            model = self.descriptor if self.scope.startswith('whole_face') else self.descriptor.global_model
            if providers['global'][0] == 'CUDAExecutionProvider' and model.runtime_providers[0] != 'CUDAExecutionProvider':
                raise RuntimeError('Recognition silently fell back from CUDA')
        self._descriptor_feature_space_id = self.descriptor.feature_space_id
        if self._descriptor_feature_space_id != self.method['feature_space_id']:
            raise ValueError('Frozen descriptor feature space mismatch')
        self.descriptor_dim = int(self.descriptor.descriptor_dim)
        contract = dict(descriptor=self._descriptor_feature_space_id, preprocessing=self.preprocessing_id,
                        runtime=runtime, storage='float32', scoring='negative_scipy_float64_euclidean')
        self.feature_space_id = 'recognition-service-v4:' + hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()).hexdigest()
        self.primary_score_threshold = float(self.method['score_threshold'])
        self.score_threshold = float(self.method.get('search_score_threshold', self.primary_score_threshold))
        if any(not np.isfinite(x) or not -2 <= x <= 0 for x in (self.primary_score_threshold, self.score_threshold)):
            raise ValueError('Invalid unit-vector distance threshold')
        self.reject_all = False
        self.recognition_threshold = self.score_threshold ** 2
        self.search_operating_point_id = self.calibration_id + ':' + method_id
        self.model_sha256 = hashlib.sha256(json.dumps(self.calibration['model_artifacts'], sort_keys=True).encode()).hexdigest()
        self.database_path = database_path
        self.database = SkinEmbeddingDatabase(self.descriptor_dim, index_type='L2', feature_space_id=self.feature_space_id)
        if database_path:
            if self.database.storage_path(database_path).exists():
                self.database.load(database_path)
            elif Path(str(database_path) + '.pkl').exists():
                raise ValueError('Use a new v4 gallery location; legacy pickle galleries are incompatible')

    def _extract_vector(self, image):
        self._assert_feature_space()
        if self.scope == 'whole_face_detector_only':
            kps, _ = source_landmarks_for_single_face(self.preprocessor, image)
            vector = self.descriptor.extract_features(image, kps)
        else:
            prepared = self.preprocessor.process(image)
        if self.scope == 'skin_only':
            packed, preproc_id = prepared_tuple_from_face_v4(prepared)
            vector = self.descriptor.extract_features(packed, preproc_id)
        elif self.scope == 'whole_face':
            kps = source_landmarks_from_cache(prepared.kps5, prepared.transform)
            vector = self.descriptor.extract_features(image, kps)
        elif self.scope == 'hybrid':
            global_vector, skin_vector = self.descriptor.extract_components(image, prepared_face=prepared)
            vector = fuse_embeddings(global_vector, skin_vector, self.descriptor.skin_weight)
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.descriptor_dim,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
            raise ValueError('Invalid frozen recognition embedding')
        return vector

    def _assert_feature_space(self):
        if self.descriptor.feature_space_id != self._descriptor_feature_space_id:
            raise ValueError('Descriptor changed after service creation; new enrollment is required')
        if self.preprocessor.preprocessing_id + self.preprocessing_suffix != self.preprocessing_id:
            raise ValueError('Preprocessor changed after service creation; new enrollment is required')
        if self.database.feature_space_id != self.feature_space_id:
            raise ValueError('Gallery belongs to a different feature space')

    def get_database_stats(self):
        with self._lock:
            self._assert_feature_space()
            return dict(self.database.get_stats(), version='4.1', input_mode=self.input_mode,
                        profile=self.profile, recognition_scope=self.scope,
                        descriptor_feature_space_id=self._descriptor_feature_space_id,
                        preprocessing_id=self.preprocessing_id, calibration_id=self.calibration_id,
                        calibration_dataset='FEI development identities',
                        calibration_scope='frozen_development_operating_point',
                        threshold_status=self.method.get('search_threshold_rule', 'development_zero_false_match_search_point'),
                        identification_error_guarantee=False,
                        score_definition='negative_euclidean_distance', score_threshold=self.score_threshold,
                        primary_score_threshold=self.primary_score_threshold,
                        recognition_threshold=self.recognition_threshold,
                        score_type='normalized_squared_l2_similarity_not_probability',
                        tie_policy='reject_exact_cross_identity_ties',
                        model_license='InsightFace pretrained weights: non-commercial research')
