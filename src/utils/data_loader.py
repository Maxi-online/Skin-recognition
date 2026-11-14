"""Deterministic image loading. Images remain BGR and may have different sizes."""

from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


def read_image(path):
    """Decode an image, including Windows paths containing non-ASCII characters."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def load_data(data_dir):
    """Load subject/image or subject/session/image folders without defining a split.

    A list, rather than a ragged NumPy array, preserves variable-size images.
    Scientific evaluations use the checked manifest in experiments.protocol.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    images, labels, label_dict = [], [], {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        paths = sorted(p for p in person_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
        if not paths:
            continue
        label = len(label_dict)
        label_dict[person_dir.name] = label
        for path in paths:
            images.append(read_image(path))
            labels.append(label)
    if not images:
        raise ValueError(f"No images found in dataset: {root}")
    return images, np.asarray(labels, dtype=np.int32), label_dict

