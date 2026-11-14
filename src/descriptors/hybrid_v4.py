"""Explicit whole-face ArcFace baseline and metric fusion with frozen skin texture.

The global component sees the whole aligned face and must never be described as
skin-only recognition. Local pretrained InsightFace weights are research-only.
No estimator is fitted by this module; the skin projection is loaded frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import cv2
import numpy as np

VERSION = "whole-face-skin-hybrid-v4.0"
ALIGNMENT_CONTRACT = "original_BGR_image;inverse_cached_affine_5kps;insightface_norm_crop112;RGB_NCHW_(x-127.5)/127.5"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_vector(value):
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError("A finite, nonempty feature vector is required")
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Zero-norm biometric feature")
    return vector / norm


def fuse_embeddings(global_feature, skin_feature, skin_weight):
    """Squared distance is (1-w)*global_d2 + w*skin_d2, exactly in real arithmetic."""
    if not np.isfinite(skin_weight) or not 0 <= skin_weight <= 1:
        raise ValueError("skin_weight must be finite and in [0,1]")
    global_feature, skin_feature = unit_vector(global_feature), unit_vector(skin_feature)
    return np.concatenate((np.sqrt(1-skin_weight)*global_feature,
                           np.sqrt(skin_weight)*skin_feature)).astype(np.float32)


def validate_bgr(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or
            image.shape[2] != 3 or min(image.shape[:2]) < 32):
        raise ValueError("Expected full-frame uint8 BGR HWC image, at least 32 pixels per side")


def source_landmarks_from_cache(aligned_kps5, source_to_aligned):
    points = np.asarray(aligned_kps5, dtype=np.float64)
    matrix = np.asarray(source_to_aligned, dtype=np.float64)
    if points.shape != (5, 2) or matrix.shape != (2, 3) or not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise ValueError("Invalid cached five landmarks or affine transform")
    if abs(np.linalg.det(matrix[:, :2])) < 1e-12:
        raise ValueError("Singular cached affine transform")
    inverse = cv2.invertAffineTransform(matrix)
    return points @ inverse[:, :2].T + inverse[:, 2]


def source_landmarks_for_single_face(processor, image):
    """Detector-only baseline: return original 5kps and explicit quality metadata.

    No 106-landmark, eye-size, pose or skin-region quality condition is imposed.
    """
    from src.preprocessing_v3 import QualityRejection
    validate_bgr(image)
    with processor._lock:
        boxes, points = processor.detector.detect(np.ascontiguousarray(image), max_num=0, metric="default")
    count = len(boxes)
    if count != 1:
        raise QualityRejection(["no_face" if count == 0 else "multiple_faces"], {"detected_faces": count})
    if (points is None or np.shape(points) != (1, 5, 2) or not np.isfinite(points).all() or
            not np.isfinite(boxes[0]).all()):
        raise QualityRejection(["invalid_five_landmarks"])
    points = np.asarray(points[0], dtype=np.float64)
    if np.linalg.norm(points[0]-points[1]) < 1:
        raise QualityRejection(["degenerate_five_landmarks"])
    metadata = {"preprocessing_id": processor.preprocessing_id+":detector_only", "detected_faces": 1,
                "bbox_xyxy": np.asarray(boxes[0, :4]).tolist(), "detection_score": float(boxes[0, 4]),
                "source_shape": list(image.shape), "quality_gate": "exactly_one_face_finite_nondegenerate_5kps_only"}
    return points, metadata


def arcface_crop(image, kps5):
    """One resampling from the original image using the upstream 112px contract."""
    validate_bgr(image)
    points = np.asarray(kps5, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all() or np.linalg.norm(points[0]-points[1]) < 1:
        raise ValueError("Expected five finite source-coordinate landmarks")
    from insightface.utils.face_align import norm_crop
    return norm_crop(image, points, image_size=112)


def prepared_tuple_from_face_v4(prepared_face):
    """Frozen strict packing, with an absent usability flag treated as rejected.

    The frozen preprocessor omits that flag for an empty ROI; it must contribute
    a zero feature block, rather than raising KeyError or treating fill as skin.
    """
    arrays = prepared_face.numeric_arrays()
    names = [str(name) for name in arrays["patch_names"]]
    patches = np.stack([arrays["skin_only_"+name] for name in names])
    quality = prepared_face.metadata["patch_quality"]
    masks = np.stack([np.full(patch.shape[:2], bool(quality.get(name, {}).get("strict_skin_usable", False)), np.uint8)
                      for name, patch in zip(names, patches)])
    return ((arrays["aligned_bgr"], patches, masks, arrays["skin_mask"]),
            prepared_face.metadata["preprocessing_id"]+":strict")


class ArcFaceDescriptorV4:
    """Fixed WebFace600k R50 whole-face baseline, with normalized 512-D output."""
    descriptor_dim = 512

    def __init__(self, model_path, providers=None, intra_op_threads=2):
        import onnxruntime as ort
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.model_sha256 = sha256(self.model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(intra_op_threads)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                           providers=providers or ["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape[1:] != [3, 112, 112] or len(outputs) != 1 or outputs[0].shape[-1] != 512:
            raise ValueError("Expected the fixed R50 112x112 RGB to 512-D ONNX contract")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.runtime_providers = self.session.get_providers()
        if providers and providers[0] != self.runtime_providers[0]:
            raise RuntimeError("Requested inference provider is unavailable; refusing an unfrozen runtime fallback")

    @property
    def feature_space_id(self):
        contract = {"version": VERSION, "component": "whole_face_ArcFace_R50_WebFace600k",
                    "model_sha256": self.model_sha256, "alignment": ALIGNMENT_CONTRACT,
                    "normalization": "L2", "output_dtype": "float32"}
        return "arcface-v4:"+hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()

    def extract_aligned_batch(self, images):
        if not images:
            raise ValueError("At least one aligned image is required")
        if any(not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (112, 112, 3) for image in images):
            raise ValueError("Aligned images must be uint8 BGR 112x112")
        blob = cv2.dnn.blobFromImages(images, 1/127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        result = self.session.run([self.output_name], {self.input_name: blob})[0]
        if result.shape != (len(images), 512):
            raise ValueError("Unexpected recognition model output")
        return np.stack([unit_vector(row) for row in result]).astype(np.float32)

    def extract_features(self, full_bgr_image, kps5):
        return self.extract_aligned_batch([arcface_crop(full_bgr_image, kps5)])[0]


class HybridFaceSkinDescriptorV4:
    """Full-frame inference; explicit global-face and local-skin components."""
    input_mode = "full_frame"

    def __init__(self, global_model_path, skin_model_path, skin_weight=0.02,
                 preprocessor=None, providers=None, intra_op_threads=2):
        from experiments.development_v3 import ClassicalTextureDescriptorV3
        if not np.isfinite(skin_weight) or not 0 <= skin_weight <= 1:
            raise ValueError("skin_weight must be finite and in [0,1]")
        self.skin_weight = float(skin_weight)
        self.global_model = ArcFaceDescriptorV4(global_model_path, providers, intra_op_threads)
        self.skin_model_path = Path(skin_model_path).resolve()
        self.skin_model_sha256 = sha256(self.skin_model_path)
        self.skin_model = ClassicalTextureDescriptorV3.load(self.skin_model_path)
        if self.skin_model.descriptor_dim < 1 or not self.skin_model.preprocessing_id.endswith(":strict"):
            raise ValueError("Expected a fitted strict-skin model")
        self.descriptor_dim = 512 + self.skin_model.descriptor_dim
        self.preprocessor = preprocessor
        self._lock = threading.RLock()

    @property
    def configuration(self):
        return {"schema_version": VERSION, "global_model_path": str(self.global_model.model_path),
                "global_model_sha256": self.global_model.model_sha256,
                "global_feature_space_id": self.global_model.feature_space_id,
                "skin_model_path": str(self.skin_model_path), "skin_model_sha256": self.skin_model_sha256,
                "skin_feature_space_id": self.skin_model.feature_space_id,
                "preprocessing_id": self.skin_model.preprocessing_id, "skin_weight": self.skin_weight,
                "alignment_contract": ALIGNMENT_CONTRACT,
                "output_dtype": "float32",
                "fusion": "concat(sqrt(1-w)*unit(global),sqrt(w)*unit(skin))",
                "scope": "whole_face_plus_local_skin_texture_not_skin_only",
                "license_scope": "InsightFace pretrained models: non-commercial research"}

    @property
    def feature_space_id(self):
        contract = {key: value for key, value in self.configuration.items() if not key.endswith("_path")}
        return VERSION+":"+hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()

    def extract_components(self, full_bgr_image, prepared_face=None):
        from src.preprocessing_v3 import FacePreprocessorV3
        validate_bgr(full_bgr_image)
        if prepared_face is None:
            with self._lock:
                if self.preprocessor is None:
                    self.preprocessor = FacePreprocessorV3()
                prepared_face = self.preprocessor.process(full_bgr_image)
        packed, preprocessing_id = prepared_tuple_from_face_v4(prepared_face)
        skin = np.asarray(self.skin_model.extract_features(packed, preprocessing_id), dtype=np.float32)
        if skin.shape != (self.skin_model.descriptor_dim,) or not np.isfinite(skin).all() or np.linalg.norm(skin) < 1e-12:
            raise ValueError("Invalid frozen skin model output")
        kps5 = source_landmarks_from_cache(prepared_face.kps5, prepared_face.transform)
        global_feature = self.global_model.extract_features(full_bgr_image, kps5)
        return global_feature, skin

    def extract_features(self, full_bgr_image):
        global_feature, skin = self.extract_components(full_bgr_image)
        return fuse_embeddings(global_feature, skin, self.skin_weight)

    def save(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        document = {**self.configuration, "feature_space_id": self.feature_space_id}
        output.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path, preprocessor=None, providers=None):
        path = Path(path).resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("schema_version") != VERSION:
            raise ValueError("Unsupported hybrid descriptor manifest")
        resolved = {}
        for component in ("global", "skin"):
            model_path = Path(config[component+"_model_path"])
            if not model_path.is_absolute():
                model_path = path.parent / model_path
            if sha256(model_path) != config[component+"_model_sha256"]:
                raise ValueError(f"{component} model checksum mismatch")
            resolved[component+"_model_path"] = model_path
        descriptor = cls(**resolved, skin_weight=config["skin_weight"], preprocessor=preprocessor, providers=providers)
        if descriptor.feature_space_id != config["feature_space_id"]:
            raise ValueError("Hybrid feature space/configuration mismatch")
        return descriptor
