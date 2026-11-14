"""Shared image contract: BGR/gray, uint8 or finite floats in [0, 1]/[0, 255]."""
import cv2
import numpy as np

def as_gray_uint8(image, minimum_size=3):
    array = np.asarray(image)
    if array.ndim not in (2, 3) or array.size == 0:
        raise ValueError("Expected a nonempty grayscale or BGR image")
    if array.ndim == 3 and array.shape[2] != 3:
        raise ValueError("Color images must have exactly three BGR channels")
    if min(array.shape[:2]) < minimum_size:
        raise ValueError(f"Image dimensions must be at least {minimum_size} pixels")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Image must contain finite numeric pixels")
    if np.min(array) < 0 or np.max(array) > 255:
        raise ValueError("Pixels must be in [0, 1] or [0, 255]")
    if np.issubdtype(array.dtype, np.floating) and np.max(array) <= 1:
        array = array * 255.0
    array = np.rint(array).astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY) if array.ndim == 3 else array

