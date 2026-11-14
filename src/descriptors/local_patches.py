"""Geometric facial ROIs on an already cropped, approximately frontal face.

No detector, landmark alignment, skin segmentation, or pore localization is
implied. ROI names describe intended regions, not verified anatomical masks.
"""
import cv2
import numpy as np
from ._image import as_gray_uint8

class LocalSkinPatchExtractor:
    def __init__(self, use_dlib=False, target_size=64):
        if use_dlib:
            raise ValueError("Landmark-based extraction is not implemented in this protocol; provide aligned face crops")
        if int(target_size) != target_size or target_size < 8:
            raise ValueError("target_size must be an integer >= 8")
        self.use_dlib = False
        self.target_size = int(target_size)

    def extract_patches(self, face_image, face_bbox=None):
        gray = as_gray_uint8(face_image, minimum_size=16)
        if face_bbox is not None:
            if len(face_bbox) != 4 or not np.isfinite(face_bbox).all():
                raise ValueError("face_bbox must contain four finite coordinates (x, y, width, height)")
            x, y, width, height = map(int, face_bbox)
            if x < 0 or y < 0 or width < 16 or height < 16 or x + width > gray.shape[1] or y + height > gray.shape[0]:
                raise ValueError("face_bbox must lie inside the image and have size >= 16")
            gray = gray[y:y + height, x:x + width]
        return self._extract_simple(gray)

    def _extract_simple(self, gray_image):
        height, width = gray_image.shape
        # Fixed protocol fractions; no claim of precise skin-only anatomical masks.
        regions = {
            "forehead": (0.25, 0.05, 0.75, 0.30),
            "left_cheek": (0.08, 0.45, 0.35, 0.68),
            "right_cheek": (0.65, 0.45, 0.92, 0.68),
            "chin": (0.35, 0.78, 0.65, 0.95),
        }
        patches = {}
        for name, (x1, y1, x2, y2) in regions.items():
            patch = gray_image[int(y1 * height):int(y2 * height), int(x1 * width):int(x2 * width)]
            if patch.size == 0 or min(patch.shape) < 2:
                raise ValueError(f"ROI {name} is too small; acquire a larger face crop")
            interpolation = cv2.INTER_AREA if min(patch.shape) >= self.target_size else cv2.INTER_LINEAR
            patches[name] = cv2.resize(patch, (self.target_size, self.target_size), interpolation=interpolation)
        return patches

    def get_patch_names(self):
        return ["forehead", "left_cheek", "right_cheek", "chin"]

