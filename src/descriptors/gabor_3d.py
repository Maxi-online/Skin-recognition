"""Two-dimensional complex Gabor magnitude statistics.

Historical names remain for imports. Phase copies of a 2D kernel do not
measure depth, and response maxima are not validated skin pores.
"""
import cv2
import numpy as np
from scipy import ndimage
from ._image import as_gray_uint8

class Gabor2DFilterBank:
    """A fixed bank of zero-DC, unit-energy complex 2D Gabor kernels."""
    def __init__(self, frequencies=None, orientations=None, scales=None,
                 sigma_x=2.0, sigma_y=2.0):
        self.frequencies = tuple([0.05, 0.1, 0.15, 0.2, 0.25] if frequencies is None else frequencies)
        self.orientations = tuple(np.arange(6) * np.pi / 6 if orientations is None else orientations)
        self.scales = tuple([1.0, 1.5, 2.0, 2.5] if scales is None else scales)
        self.sigma_x, self.sigma_y = float(sigma_x), float(sigma_y)
        if not self.frequencies or not self.orientations or not self.scales:
            raise ValueError("A Gabor bank cannot be empty")
        if not all(np.isfinite(v) for v in (*self.frequencies, *self.orientations, *self.scales, self.sigma_x, self.sigma_y)):
            raise ValueError("Gabor parameters must be finite")
        if not all(0 < f < 0.5 for f in self.frequencies) or min(self.scales) <= 0 or min(self.sigma_x, self.sigma_y) <= 0:
            raise ValueError("Invalid Gabor frequency, scale, or sigma")
        self.filters = self._build_filter_bank()

    def _create_2d_gabor(self, frequency, theta, scale):
        radius = max(1, int(np.ceil(2 * scale * max(self.sigma_x, self.sigma_y))))
        coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
        x, y = np.meshgrid(coordinates, coordinates)
        u, v = x * np.cos(theta) + y * np.sin(theta), -x * np.sin(theta) + y * np.cos(theta)
        envelope = np.exp(-0.5 * ((u / (scale * self.sigma_x)) ** 2 + (v / (scale * self.sigma_y)) ** 2))
        kernel = envelope * np.exp(2j * np.pi * frequency * u)
        kernel -= kernel.mean()
        kernel /= np.linalg.norm(kernel)
        return kernel.astype(np.complex64)

    def _build_filter_bank(self):
        return [{"filter": self._create_2d_gabor(f, t, s), "freq": f, "theta": t, "scale": s}
                for f in self.frequencies for t in self.orientations for s in self.scales]

    def responses(self, image):
        gray = as_gray_uint8(image).astype(np.float32) / 255.0
        results = []
        for entry in self.filters:
            # filter2D is correlation; flip both axes for convolution.
            kernel = entry["filter"][::-1, ::-1]
            real = cv2.filter2D(gray, cv2.CV_32F, kernel.real, borderType=cv2.BORDER_REFLECT_101)
            imag = cv2.filter2D(gray, cv2.CV_32F, kernel.imag, borderType=cv2.BORDER_REFLECT_101)
            results.append(cv2.magnitude(real, imag))
        return np.stack(results, axis=-1)

    def extract_second_order_features(self, image):
        """Historical API: response statistics, not exclusively second-order features."""
        return self.features_from_responses(self.responses(image))

    def features_from_responses(self, responses):
        responses = np.asarray(responses, dtype=np.float32)
        if responses.ndim != 3 or min(responses.shape[:2]) < 3 or not np.isfinite(responses).all():
            raise ValueError("Expected finite response maps [H >= 3, W >= 3, filters]")
        features = []
        for i in range(responses.shape[-1]):
            response = responses[..., i]
            features.extend([float(response.mean()), float(response.std()), float(np.mean(response ** 2)),
                             *self._detect_response_peaks(response), *self._compute_texture_features(response)])
        return np.asarray(features, dtype=np.float32)

    def _detect_response_peaks(self, response_map, threshold_factor=0.7):
        maxima = ((response_map == ndimage.maximum_filter(response_map, size=3)) &
                  (response_map > threshold_factor * np.max(response_map)))
        return np.asarray([maxima.mean(), response_map[maxima].mean() if maxima.any() else 0.0], dtype=np.float32)

    def _detect_pores(self, response_map, threshold_factor=0.7):
        """Deprecated name: peak density/intensity, with no pore interpretation."""
        return self._detect_response_peaks(response_map, threshold_factor)

    def _compute_texture_features(self, response_map):
        grad_y, grad_x = np.gradient(response_map)
        magnitude = np.hypot(grad_x, grad_y)
        histogram, _ = np.histogram(response_map, bins=32)
        probabilities = histogram[histogram > 0] / histogram.sum()
        return [float(magnitude.mean()), float(magnitude.std()),
                *self._compute_lbp_like_features(response_map),
                float(-np.sum(probabilities * np.log(probabilities)))]

    def _compute_lbp_like_features(self, response_map):
        center = response_map[1:-1, 1:-1]
        neighbors = [response_map[:-2, 1:-1], response_map[2:, 1:-1], response_map[1:-1, :-2],
                     response_map[1:-1, 2:], response_map[:-2, :-2], response_map[:-2, 2:],
                     response_map[2:, :-2], response_map[2:, 2:]]
        return [float(np.mean(neighbor > center)) for neighbor in neighbors]

# Import compatibility only; this class does not process 3D observations.
Gabor3DFilterBank = Gabor2DFilterBank

def extract_gabor_features(image):
    return Gabor2DFilterBank().extract_second_order_features(image)

