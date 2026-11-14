"""Landmark-based facial alignment and conservative skin-region preprocessing.

This module is independent of frozen v2. Anatomical/appearance masks are
heuristics, not validated semantic skin segmentation. No recognition embedding,
age or gender model is loaded. Missing/ambiguous faces cause explicit rejection.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import threading

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

PATCH_NAMES = ("forehead", "left_cheek", "right_cheek", "chin")
ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                       [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
# Coordinate-regression 2d106det ordering, visually verified on development images.
EYES = (tuple(range(33, 43)), tuple(range(87, 97)))
BROWS = (tuple(range(43, 52)), tuple(range(97, 106)))
MOUTH = tuple(range(52, 72))
NOSE = tuple(range(72, 87))


@dataclass(frozen=True)
class PreprocessingConfigV3:
    image_size: int = 224
    patch_size: int = 64
    detection_size: int = 640
    detection_threshold: float = 0.5
    min_eye_distance: float = 25.0
    max_alignment_error: float = 0.25
    min_patch_native_side: float = 8.0
    min_patch_coverage: float = 0.30
    min_usable_patches: int = 2
    dark_relative_threshold: float = 0.40
    appearance_chroma_limit: float = 5.0


class QualityRejection(ValueError):
    def __init__(self, codes, metadata=None):
        self.codes = list(codes)
        self.metadata = metadata or {}
        super().__init__(", ".join(self.codes))


@dataclass
class PreparedFace:
    aligned_bgr: np.ndarray
    landmarks106: np.ndarray
    kps5: np.ndarray
    skin_mask: np.ndarray
    patches: dict
    patch_masks: dict
    transform: np.ndarray
    metadata: dict

    def numeric_arrays(self):
        arrays = dict(aligned_bgr=self.aligned_bgr, landmarks106=self.landmarks106,
                    kps5=self.kps5, skin_mask=self.skin_mask,
                    patches_bgr=np.stack([self.patches[n] for n in PATCH_NAMES]),
                    patch_masks=np.stack([self.patch_masks[n] for n in PATCH_NAMES]),
                    patch_names=np.asarray(PATCH_NAMES), transform=self.transform,
                    metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)))
        for name in PATCH_NAMES:
            arrays[name] = self.patches[name]
            arrays[name + "_mask"] = self.patch_masks[name]
            arrays["masked_" + name] = np.where(self.patch_masks[name][..., None] > 0,
                                              self.patches[name], np.uint8(128)).astype(np.uint8)
            # A rectangular crop wholly inside the eroded valid mask avoids
            # exposing eyes/mouth or the person-specific mask silhouette to CNNs.
            eroded = cv2.erode(self.patch_masks[name], np.ones((3, 3), np.uint8))
            box = largest_valid_rectangle(eroded)
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                arrays["skin_only_" + name] = cv2.resize(self.patches[name][y1:y2, x1:x2],
                                                        self.patches[name].shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
            else:
                arrays["skin_only_" + name] = np.full_like(self.patches[name], 128)
            arrays["skin_rect_" + name] = np.asarray(box, dtype=np.int32)
        return arrays

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **self.numeric_arrays())


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:, :2].T + matrix[:, 2]


def largest_valid_rectangle(mask):
    """Largest all-valid axis-aligned rectangle; deterministic top/left tie break."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional mask")
    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best_area, best = 0, (0, 0, 0, 0)
    for y in range(mask.shape[0]):
        heights = np.where(mask[y] > 0, heights + 1, 0)
        stack = []
        for x in range(mask.shape[1] + 1):
            height = int(heights[x]) if x < mask.shape[1] else 0
            left = x
            while stack and stack[-1][1] > height:
                start, previous_height = stack.pop()
                area = previous_height * (x - start)
                candidate = (start, y - previous_height + 1, x, y + 1)
                if area > best_area or (area == best_area and (candidate[1], candidate[0]) < (best[1], best[0])):
                    best_area, best = area, candidate
                left = start
            if not stack or stack[-1][1] < height:
                stack.append((left, height))
    return best


def similarity_alignment(kps5, image_size=224):
    points = np.asarray(kps5, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise QualityRejection(["invalid_five_landmarks"])
    eye_distance = float(np.linalg.norm(points[1] - points[0]))
    if eye_distance < 1e-6:
        raise QualityRejection(["degenerate_eye_landmarks"])
    destination = ARCFACE_112 * (image_size / 112.0)
    if hasattr(SimilarityTransform, "from_estimate"):
        transform = SimilarityTransform.from_estimate(points, destination)
    else:
        transform = SimilarityTransform()
        if not transform.estimate(points, destination):
            raise QualityRejection(["similarity_estimation_failed"])
    matrix = transform.params[:2, :].copy()
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:, :2]) <= 0:
        raise QualityRejection(["invalid_similarity_transform"])
    residual = float(np.sqrt(np.mean(np.sum((transform_points(points, matrix) - destination) ** 2, axis=1))))
    return matrix, residual / float(np.linalg.norm(destination[1] - destination[0]))


def _hull_mask(points, size):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise QualityRejection(["invalid_polygon"])
    mask = np.zeros((size, size), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(points).astype(np.int32)), 1)
    return mask


def _expanded_hull(points, size, radius):
    mask = _hull_mask(points, size)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))


def build_skin_regions(aligned_bgr, landmarks106, aligned_kps5, config, valid_pixels=None):
    """Return actual-landmark ROI masks plus appearance rejection diagnostics."""
    size = config.image_size
    points = np.asarray(landmarks106, dtype=np.float64)
    keypoints = np.asarray(aligned_kps5, dtype=np.float64)
    if points.shape != (106, 2) or not np.isfinite(points).all():
        raise QualityRejection(["invalid_106_landmarks"])
    iod = float(np.linalg.norm(keypoints[1] - keypoints[0]))
    if iod < 1:
        raise QualityRejection(["degenerate_aligned_eyes"])
    brow_left, brow_right = points[list(BROWS[0])], points[list(BROWS[1])]
    brow_top = min(float(brow_left[:, 1].min()), float(brow_right[:, 1].min()))
    center_x = float((brow_left[:, 0].mean() + brow_right[:, 0].mean()) / 2)
    forehead = np.array([[center_x - .48*iod, brow_top - .50*iod],
                         [center_x + .48*iod, brow_top - .50*iod],
                         [center_x + .43*iod, brow_top - .10*iod],
                         [center_x - .43*iod, brow_top - .10*iod]])
    contour = np.concatenate([points[:33], forehead[:2]], axis=0)
    anatomical = _hull_mask(contour, size)
    edge = max(1, int(round(.045 * iod)))
    anatomical = cv2.erode(anatomical, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*edge+1, 2*edge+1)))
    exclusions = np.zeros_like(anatomical)
    for group in (*EYES, *BROWS, MOUTH, NOSE):
        exclusions |= _expanded_hull(points[list(group)], size, max(2, int(round(.055*iod))))
    anatomical &= 1 - exclusions
    if valid_pixels is not None:
        anatomical &= np.asarray(valid_pixels, dtype=np.uint8)

    mouth_bottom = float(points[list(MOUTH), 1].max())
    chin_bottom = float(points[0, 1])
    chin_x = float(points[0, 0])
    chin = np.array([[chin_x-.32*iod, mouth_bottom+.09*iod],
                     [chin_x+.32*iod, mouth_bottom+.09*iod],
                     [chin_x+.24*iod, chin_bottom-.07*iod],
                     [chin_x-.24*iod, chin_bottom-.07*iod]])
    polygons = {
        "forehead": forehead,
        "left_cheek": points[[9, 10, 11, 12, 13, 14, 35, 36, 37, 76, 77, 78, 55]],
        "right_cheek": points[[25, 26, 27, 28, 29, 30, 89, 90, 91, 82, 83, 84, 61]],
        "chin": chin,
    }
    region_masks = {name: _hull_mask(poly, size) & anatomical for name, poly in polygons.items()}

    # Face-adaptive broad appearance guard: no hardcoded ethnicity-specific skin box.
    # Seed only interior candidate cheeks. This rejects dark hair/very different
    # chroma but does NOT prove semantic skin purity or absence of facial hair.
    seed = region_masks["left_cheek"] | region_masks["right_cheek"]
    if np.count_nonzero(seed) < 25:
        raise QualityRejection(["insufficient_cheek_seed"])
    ycrcb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    seed_values = ycrcb[seed > 0]
    median = np.median(seed_values, axis=0)
    mad = np.median(np.abs(seed_values - median), axis=0)
    chroma_scale = np.maximum(1.4826*mad[1:], 8.0)
    chroma_distance = np.max(np.abs(ycrcb[..., 1:] - median[None, None, 1:]) / chroma_scale, axis=-1)
    appearance = ((ycrcb[..., 0] >= max(8.0, config.dark_relative_threshold*median[0])) &
                  (ycrcb[..., 0] < 252) & (chroma_distance <= config.appearance_chroma_limit)).astype(np.uint8)
    skin_mask = anatomical & appearance
    cleaned = {name: mask & appearance for name, mask in region_masks.items()}
    metadata = dict(mask_semantics="landmark_and_face_adaptive_appearance_heuristic_not_ground_truth_skin",
                    seed_pixels=int(np.count_nonzero(seed)), seed_ycrcb_median=median.tolist(),
                    seed_ycrcb_mad=mad.tolist(), geometric_mask_pixels=int(np.count_nonzero(anatomical)),
                    skin_mask_pixels=int(np.count_nonzero(skin_mask)),
                    appearance_rejected_fraction=float(np.count_nonzero(anatomical & (1-appearance))/max(1,np.count_nonzero(anatomical))))
    return skin_mask, cleaned, polygons, metadata


class FacePreprocessorV3:
    version = "landmark-skin-preprocess-v3.1"

    def __init__(self, model_dir=None, config=None, providers=None):
        self.config = config or PreprocessingConfigV3()
        if self.config.image_size < 112 or self.config.patch_size < 16:
            raise ValueError("image_size >=112 and patch_size >=16 are required")
        self.model_dir = Path(model_dir or Path.home()/".insightface/models/buffalo_l")
        filenames = {"detector": "det_10g.onnx", "landmarks": "2d106det.onnx"}
        self.model_paths = {name: self.model_dir/filename for name, filename in filenames.items()}
        for path in self.model_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Required local research model is missing: {path}")
        self.model_sha256 = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in self.model_paths.items()}
        self.providers = providers or ["CPUExecutionProvider"]
        self._lock = threading.RLock()
        from insightface import model_zoo
        self.detector = model_zoo.get_model(str(self.model_paths["detector"]), providers=self.providers)
        self.landmark_model = model_zoo.get_model(str(self.model_paths["landmarks"]), providers=self.providers)
        context_id = -1 if self.providers == ["CPUExecutionProvider"] else 0
        self.detector.prepare(context_id, input_size=(self.config.detection_size, self.config.detection_size),
                              det_thresh=self.config.detection_threshold)
        self.landmark_model.prepare(context_id)
        self.runtime_providers = {"detector": self.detector.session.get_providers(),
                                  "landmarks": self.landmark_model.session.get_providers()}

    @property
    def preprocessing_id(self):
        contract = dict(version=self.version, config=asdict(self.config), model_sha256=self.model_sha256)
        return self.version + ":" + hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()

    def process(self, image):
        if not isinstance(image,np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
            raise QualityRejection(["invalid_bgr_uint8_image"])
        from insightface.app.common import Face
        with self._lock:
            boxes, kps = self.detector.detect(np.ascontiguousarray(image), max_num=0, metric="default")
            if len(boxes) != 1:
                raise QualityRejection(["no_face" if len(boxes)==0 else "multiple_faces"],
                                       {"detected_faces":int(len(boxes))})
            if kps is None or kps.shape[1:] != (5,2):
                raise QualityRejection(["five_landmarks_missing"])
            face = Face(bbox=boxes[0,:4],kps=kps[0])
            landmarks = self.landmark_model.get(image,face)
        return self.process_observation(image, boxes[0,:4], kps[0], landmarks, float(boxes[0,4]))

    def process_observation(self,image,bbox,kps5,landmarks106,det_score):
        """Deterministic transformation of measured/predicted landmarks; useful for audit tests."""
        config=self.config
        points=np.asarray(landmarks106,dtype=np.float64)
        keypoints=np.asarray(kps5,dtype=np.float64)
        if points.shape!=(106,2) or not np.isfinite(points).all():
            raise QualityRejection(["invalid_106_landmarks"])
        eye_distance=float(np.linalg.norm(keypoints[1]-keypoints[0])) if keypoints.shape==(5,2) else 0
        if eye_distance < config.min_eye_distance:
            raise QualityRejection(["insufficient_source_eye_distance"],{"eye_distance_pixels":eye_distance})
        matrix,error=similarity_alignment(keypoints,config.image_size)
        if error > config.max_alignment_error:
            raise QualityRejection(["excessive_alignment_residual"],{"normalized_alignment_error":error})
        aligned=cv2.warpAffine(image,matrix,(config.image_size,config.image_size),
                               flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        source_valid=np.ones(image.shape[:2],np.uint8)
        valid=cv2.warpAffine(source_valid,matrix,(config.image_size,config.image_size),flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        aligned_points=transform_points(points,matrix).astype(np.float32)
        aligned_kps=transform_points(keypoints,matrix).astype(np.float32)
        skin_mask,region_masks,polygons,mask_quality=build_skin_regions(aligned,aligned_points,aligned_kps,config,valid)
        scale=float(np.sqrt(np.linalg.det(matrix[:,:2])))
        patches,masks,quality={}, {}, {}
        for name in PATCH_NAMES:
            polygon=np.asarray(polygons[name])
            x1,y1=np.floor(polygon.min(axis=0)).astype(int)
            x2,y2=np.ceil(polygon.max(axis=0)).astype(int)
            x1,y1=max(0,x1),max(0,y1)
            x2,y2=min(config.image_size,x2),min(config.image_size,y2)
            reason=[]
            if x2<=x1 or y2<=y1:
                patches[name]=np.zeros((config.patch_size,config.patch_size,3),np.uint8)
                masks[name]=np.zeros((config.patch_size,config.patch_size),np.uint8)
                quality[name]={"usable":False,"rejection_codes":["empty_roi"],"valid_fraction":0.0}
                continue
            crop=aligned[y1:y2,x1:x2]
            mask=region_masks[name][y1:y2,x1:x2]
            coverage=float(mask.mean())
            native_side=min(x2-x1,y2-y1)/scale
            if coverage < config.min_patch_coverage: reason.append("low_skin_coverage")
            if native_side < config.min_patch_native_side: reason.append("insufficient_native_resolution")
            if np.count_nonzero(mask) < 25: reason.append("too_few_skin_pixels")
            interpolation=cv2.INTER_AREA if min(crop.shape[:2]) >= config.patch_size else cv2.INTER_LINEAR
            patches[name]=cv2.resize(crop,(config.patch_size,config.patch_size),interpolation=interpolation)
            masks[name]=cv2.resize(mask,(config.patch_size,config.patch_size),interpolation=cv2.INTER_NEAREST)
            values=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)[mask>0]
            quality[name]={"usable":not reason,"rejection_codes":reason,"bbox_aligned_xyxy":list(map(int,[x1,y1,x2,y2])),
                           "native_min_side_pixels":float(native_side),"valid_fraction":coverage,
                           "valid_pixels":int(mask.sum()),"gray_std":float(values.std()) if len(values) else 0.0}
            rectangle=largest_valid_rectangle(cv2.erode(masks[name],np.ones((3,3),np.uint8)))
            rx1,ry1,rx2,ry2=rectangle
            strict_native_side=min((rx2-rx1)*(x2-x1),(ry2-ry1)*(y2-y1))/config.patch_size/scale
            strict_reason=list(reason)
            if strict_native_side < config.min_patch_native_side:
                strict_reason.append("insufficient_strict_skin_native_resolution")
            quality[name].update(strict_skin_usable=not strict_reason,
                                 strict_skin_rejection_codes=strict_reason,
                                 strict_skin_rectangle=list(map(int,rectangle)),
                                 strict_skin_native_min_side_pixels=float(strict_native_side))
        usable=sum(int(record["usable"]) for record in quality.values())
        strict_usable=sum(int(record.get("strict_skin_usable",False)) for record in quality.values())
        metadata=dict(preprocessing_id=self.preprocessing_id,version=self.version,config=asdict(config),
                      model_sha256=self.model_sha256,runtime_providers=getattr(self,"runtime_providers",{}),
                      source_shape=list(image.shape),bbox_xyxy=np.asarray(bbox).tolist(),
                      detection_score=det_score,source_eye_distance=eye_distance,
                      normalized_alignment_error=error,source_to_aligned_scale=scale,
                      usable_patches=usable,strict_usable_patches=strict_usable,
                      quality_warnings=["elevated_five_point_residual_possible_expression"] if error>0.15 else [],
                      patch_quality=quality,mask_quality=mask_quality,
                      license_scope="InsightFace pretrained models: non-commercial research")
        if strict_usable<config.min_usable_patches:
            raise QualityRejection(["insufficient_usable_skin_regions"],metadata)
        return PreparedFace(aligned,aligned_points,aligned_kps,skin_mask,patches,masks,matrix,metadata)

