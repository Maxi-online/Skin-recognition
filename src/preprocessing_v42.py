"""V4.2 input geometry repair; frozen V3/V4.1 implementations remain untouched.

Small images are centered on a zero-filled detector canvas without upscaling.
Detector outputs are mapped to the original image before 106-landmark inference,
alignment and skin extraction. With multiple detections the target must be the
unique box containing the original frame center. Skin/score thresholds are unchanged.
"""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np

from src.descriptors.hybrid_v4 import validate_bgr, prepared_tuple_from_face_v4
from src.preprocessing_v3 import FacePreprocessorV3, QualityRejection

ADAPTER_VERSION = "native-small-frame-detector-v4.2"
TARGET_SELECTION_POLICY = "single_face_or_unique_bbox_containing_original_frame_center"


def detector_canvas(image, detection_size=640):
    """Return canvas and translation; large/full-size frames are exact identities."""
    validate_bgr(image)
    if not isinstance(detection_size, int) or detection_size < 32:
        raise ValueError("Detection size must be an integer >=32")
    height, width = image.shape[:2]
    if max(height, width) >= detection_size:
        return image, (0, 0)
    x = (detection_size-width)//2
    y = (detection_size-height)//2
    canvas = np.zeros((detection_size, detection_size, 3), dtype=np.uint8)
    canvas[y:y+height, x:x+width] = image
    return canvas, (x, y)


class NativeResolutionDetector:
    """Adapter preserving the upstream detect interface and source coordinates."""
    def __init__(self, detector, detection_size=640):
        self.raw_detector = detector
        self.detection_size = int(detection_size)

    def __getattr__(self, name):
        return getattr(self.raw_detector, name)

    def detect(self, image, input_size=None, max_num=0, metric="default"):
        boxes, keypoints, _ = self.detect_with_metadata(image, input_size, max_num, metric)
        return boxes, keypoints

    def detect_with_metadata(self, image, input_size=None, max_num=0, metric="default"):
        if input_size is not None and tuple(input_size) != (self.detection_size, self.detection_size):
            raise ValueError("Changing input_size would violate the frozen detector adapter contract")
        if max_num != 0:
            raise ValueError("Preselecting detections would violate the explicit target selection policy")
        canvas, (x, y) = detector_canvas(image, self.detection_size)
        boxes, keypoints = self.raw_detector.detect(canvas, input_size=input_size,
                                                   max_num=max_num, metric=metric)
        if boxes.ndim != 2 or boxes.shape[1] < 5 or not np.isfinite(boxes).all():
            raise QualityRejection(["invalid_detector_outputs"], {"target_selection": {
                "policy": TARGET_SELECTION_POLICY, "original_detection_count": len(boxes) if boxes.ndim else None,
                "center_containing_count": None, "selected_index": None, "reason": "invalid_detector_outputs",
                "source_center_xy": [image.shape[1]/2., image.shape[0]/2.], "detector_padding_xy": [x, y]}})
        if canvas is not image:
            boxes = boxes.copy()
            boxes[:, :4] -= np.asarray([x, y, x, y], dtype=boxes.dtype)
            if keypoints is not None:
                keypoints = keypoints.copy()-np.asarray([x, y], dtype=keypoints.dtype)
        center = (image.shape[1]/2., image.shape[0]/2.)
        contains_center = ((boxes[:, 0] <= center[0]) & (center[0] <= boxes[:, 2]) &
                           (boxes[:, 1] <= center[1]) & (center[1] <= boxes[:, 3]))
        containing = np.flatnonzero(contains_center)
        selected = 0 if len(boxes) == 1 else int(containing[0]) if len(containing) == 1 else None
        reason = ("single_detection" if len(boxes) == 1 else "unique_center_bbox" if selected is not None else
                  "no_detection" if len(boxes) == 0 else "ambiguous_multiple_faces")
        selection = {"policy": TARGET_SELECTION_POLICY, "original_detection_count": len(boxes),
                     "center_containing_count": len(containing), "selected_index": selected,
                     "reason": reason, "source_center_xy": list(center), "detector_padding_xy": [x, y]}
        if selected is not None and len(boxes) > 1:
            boxes = boxes[selected:selected+1]
            if keypoints is not None:
                keypoints = keypoints[selected:selected+1]
        return boxes, keypoints, selection


class FacePreprocessorV42:
    """Original-image skin preprocessing with an explicit effective input contract."""
    version = "landmark-skin-preprocess-v4.2"

    def __init__(self, model_dir=None, config=None, providers=None, base_preprocessor=None):
        if base_preprocessor is not None and any(value is not None for value in (model_dir, config, providers)):
            raise ValueError("Pass either an existing base preprocessor or its constructor arguments")
        self.base = base_preprocessor or FacePreprocessorV3(model_dir=model_dir, config=config, providers=providers)
        self._lock = self.base._lock
        self.detector = NativeResolutionDetector(self.base.detector, self.base.config.detection_size)
        self.landmark_model = self.base.landmark_model

    @property
    def config(self):
        return self.base.config

    @property
    def runtime_providers(self):
        return self.base.runtime_providers

    @property
    def model_sha256(self):
        return self.base.model_sha256

    @property
    def legacy_preprocessing_id(self):
        return self.base.preprocessing_id

    @property
    def preprocessing_id(self):
        contract = {"version": ADAPTER_VERSION, "base_preprocessing_id": self.legacy_preprocessing_id,
                    "apply_if": "max(height,width)<detection_size", "detection_size": self.config.detection_size,
                    "canvas": "centered_uint8_zero_padding_no_resize", "detector_output": "original_image_coordinates",
                    "landmarks_and_skin_input": "original_image", "quality_gates": "unchanged",
                    "target_selection": TARGET_SELECTION_POLICY, "center_bbox_boundary": "inclusive"}
        return self.version+":"+hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()

    def adaptation_metadata(self, image):
        height, width = image.shape[:2]
        active = max(height, width) < self.config.detection_size
        x = (self.config.detection_size-width)//2 if active else 0
        y = (self.config.detection_size-height)//2 if active else 0
        return {"version": ADAPTER_VERSION, "applied": active, "source_shape": list(image.shape),
                "detector_padding_xy": [x, y], "detector_canvas_size": self.config.detection_size,
                "detector_input_scale": 1.0 if active else "unchanged_upstream_resize",
                "landmarks_and_skin_input": "original_image"}

    def _metadata(self, metadata, image):
        return {**metadata, "version": self.version, "preprocessing_id": self.preprocessing_id,
                "effective_preprocessing_id": self.preprocessing_id,
                "legacy_preprocessing_id": self.legacy_preprocessing_id,
                "input_geometry_adapter": self.adaptation_metadata(image),
                "contract_mapping": "legacy skin model consumes the unchanged aligned-region representation; effective input preprocessing is v4.2"}

    def process(self, image):
        try:
            validate_bgr(image)
        except ValueError as error:
            raise QualityRejection(["invalid_bgr_uint8_image"]) from error
        from insightface.app.common import Face
        selection = None
        try:
            with self._lock:
                boxes, keypoints, selection = self.detector.detect_with_metadata(image, max_num=0, metric="default")
                if len(boxes) != 1:
                    raise QualityRejection(["no_face" if len(boxes) == 0 else "multiple_faces"],
                                           {"detected_faces": int(len(boxes)), "target_selection": selection})
                if (keypoints is None or keypoints.shape != (1, 5, 2) or not np.isfinite(keypoints).all() or
                        not np.isfinite(boxes[0]).all()):
                    raise QualityRejection(["invalid_five_landmarks"])
                face = Face(bbox=boxes[0, :4], kps=keypoints[0])
                landmarks = self.landmark_model.get(image, face)
            prepared = self.base.process_observation(image, boxes[0, :4], keypoints[0], landmarks, float(boxes[0, 4]))
        except QualityRejection as error:
            metadata = dict(error.metadata)
            if selection is not None:
                metadata["target_selection"] = selection
            raise QualityRejection(error.codes, self._metadata(metadata, image)) from error
        prepared.metadata = self._metadata(prepared.metadata, image)
        prepared.metadata["target_selection"] = selection
        return prepared


def source_landmarks_for_target_face_v42(processor, image):
    """Same target policy as strict processing, without the 106/skin quality gates."""
    try:
        validate_bgr(image)
    except ValueError as error:
        raise QualityRejection(["invalid_bgr_uint8_image"]) from error
    try:
        with processor._lock:
            boxes, points, selection = processor.detector.detect_with_metadata(image, max_num=0, metric="default")
    except QualityRejection as error:
        raise QualityRejection(error.codes, processor._metadata(error.metadata, image)) from error
    if len(boxes) != 1:
        raise QualityRejection(["no_face" if not len(boxes) else "multiple_faces"],
                               processor._metadata({"target_selection": selection}, image))
    if points is None or points.shape != (1, 5, 2) or not np.isfinite(points).all():
        raise QualityRejection(["invalid_five_landmarks"], processor._metadata({"target_selection": selection}, image))
    points = np.asarray(points[0], dtype=np.float64)
    if np.linalg.norm(points[0]-points[1]) < 1:
        raise QualityRejection(["degenerate_five_landmarks"], processor._metadata({"target_selection": selection}, image))
    quality = processor._metadata({"bbox_xyxy": boxes[0, :4].tolist(), "detection_score": float(boxes[0, 4]),
                                   "target_selection": selection, "detected_faces": 1,
                                   "quality_gate": "unique_target_finite_nondegenerate_5kps_only"}, image)
    quality["preprocessing_id"] += ":detector_only"
    quality["effective_preprocessing_id"] += ":detector_only"
    quality["legacy_preprocessing_id"] += ":detector_only"
    return points, quality


def prepared_tuple_from_face_v42(prepared_face):
    """Return (strict inputs, explicit legacy model contract, effective contract)."""
    legacy = prepared_face.metadata.get("legacy_preprocessing_id")
    effective = prepared_face.metadata.get("effective_preprocessing_id")
    if (not legacy or not effective or effective != prepared_face.metadata.get("preprocessing_id") or
            prepared_face.metadata.get("input_geometry_adapter", {}).get("version") != ADAPTER_VERSION):
        raise ValueError("V4.2 effective/legacy preprocessing mapping is missing or inconsistent")
    packed, packed_id = prepared_tuple_from_face_v4(prepared_face)
    if packed_id != effective+":strict":
        raise ValueError("Prepared-region contract mismatch")
    return packed, legacy+":strict", effective+":strict"
