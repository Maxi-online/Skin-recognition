"""Gray-level co-occurrence texture statistics with explicit quantization."""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from ._image import as_gray_uint8

class GLCMTextureAnalyzer:
    def __init__(self, distances=None, angles=None, levels=32, symmetric=True, normed=True):
        self.distances = tuple([1, 2, 3] if distances is None else distances)
        self.angles = tuple([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4] if angles is None else angles)
        if not self.distances or any(int(v) != v or v < 1 for v in self.distances):
            raise ValueError("GLCM distances must be positive integers")
        if not self.angles or not np.isfinite(self.angles).all():
            raise ValueError("GLCM angles must be finite")
        if int(levels) != levels or not 2 <= levels <= 256:
            raise ValueError("GLCM levels must be an integer in [2, 256]")
        self.levels, self.symmetric, self.normed = int(levels), bool(symmetric), bool(normed)

    def extract_features(self, image):
        image = as_gray_uint8(image)
        # Multiplication quantization also supports levels that do not divide 256.
        quantized = ((image.astype(np.uint16) * self.levels) // 256).astype(np.uint8)
        glcm = graycomatrix(quantized, distances=self.distances, angles=self.angles,
                           levels=self.levels, symmetric=self.symmetric, normed=self.normed)
        blocks = [graycoprops(glcm, prop).ravel() for prop in
                  ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")]
        blocks.append(self._compute_entropy(glcm))
        return np.concatenate(blocks).astype(np.float32)

    def _compute_entropy(self, glcm):
        probabilities = glcm.astype(np.float64)
        sums = probabilities.sum(axis=(0, 1), keepdims=True)
        probabilities = np.divide(probabilities, sums, out=np.zeros_like(probabilities), where=sums > 0)
        logged = np.zeros_like(probabilities)
        np.log(probabilities, out=logged, where=probabilities > 0)
        return -np.sum(probabilities * logged, axis=(0, 1)).ravel()

    def extract_local_features(self, image, patch_size=32, stride=16):
        image = as_gray_uint8(image)
        if int(patch_size) != patch_size or patch_size < 3 or int(stride) != stride or stride < 1:
            raise ValueError("patch_size >= 3 and stride >= 1 must be integers")
        if min(image.shape) < patch_size:
            raise ValueError("Image is smaller than a requested patch")
        return np.stack([self.extract_features(image[y:y + patch_size, x:x + patch_size])
                         for y in range(0, image.shape[0] - patch_size + 1, stride)
                         for x in range(0, image.shape[1] - patch_size + 1, stride)])

    def extract_multiscale_features(self, image, scales=None):
        image = as_gray_uint8(image)
        scales = [1, 2, 4] if scales is None else scales
        if not scales or any(int(s) != s or s < 1 or min(image.shape) // s < 3 for s in scales):
            raise ValueError("Scales must be positive integers preserving at least 3 pixels")
        return np.concatenate([self.extract_features(image if s == 1 else cv2.resize(
            image, (image.shape[1] // s, image.shape[0] // s), interpolation=cv2.INTER_AREA)) for s in scales])

