"""
Система распознавания по коже лица.
Исследовательский 2D текстурный дескриптор локальных областей лица.
Работает напрямую с векторами признаков без дополнительного обучения.
"""

import numpy as np
import cv2
import os
import threading
from datetime import datetime
import warnings
from typing import Optional, Tuple, List, Dict, Any
from ..database import SkinEmbeddingDatabase
from ..descriptors import HybridSkinDescriptor

# Современный детектор лиц: InsightFace (SCRFD) + ONNXRuntime
try:
    from insightface.app import FaceAnalysis  # type: ignore

    _INSIGHTFACE_AVAILABLE = True
except Exception:
    FaceAnalysis = None  # type: ignore
    _INSIGHTFACE_AVAILABLE = False

_FACE_APP = None

# Убираем известное предупреждение из insightface (оно у них внутри и не влияет на работу)
warnings.filterwarnings(
    "ignore",
    message=r".*`estimate` is deprecated.*SimilarityTransform\.from_estimate.*",
    category=FutureWarning,
    module=r"insightface\.utils\.face_align",
)


def _get_face_app():
    """
    Ленивый инициализатор InsightFace FaceAnalysis (SCRFD детектор).
    На Windows/CPU используем ONNXRuntime CPUExecutionProvider.
    """
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP

    if not _INSIGHTFACE_AVAILABLE or os.getenv("FACE_DETECTOR", "haar") != "insightface":
        return None
    # No network downloads at inference time. Optional model must be installed separately.
    from pathlib import Path
    model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    if not model_dir.exists() or not list(model_dir.glob("*.onnx")):
        return None

    try:
        providers = ["CPUExecutionProvider"]
        _FACE_APP = FaceAnalysis(name="buffalo_l", providers=providers)
        # ctx_id=-1 => CPU
        _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
        return _FACE_APP
    except Exception:
        # Любая проблема с загрузкой/весами/провайдерами — просто выключаем, чтобы не ломать сервер
        _FACE_APP = None
        return None


def detect_face_info(image: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Детект лица + доп.информация для UI (3D enrollment/оверлей):
    - bbox: (x, y, w, h)
    - kps: 5 keypoints (если доступны) [[x,y],...]
    - lmk106: 106 landmarks (если доступны) [[x,y],...]
    - lmk68_3d: 68 3D landmarks (если доступны) [[x,y,z],...]
    - score: confidence (если доступно)
    """
    # 1) Optional local InsightFace model; Haar is the reproducible default.
    app = _get_face_app()
    if app is not None and len(image.shape) == 3:
        try:
            img = np.ascontiguousarray(image)
            faces = app.get(img)
            if len(faces) > 1:
                raise ValueError("Exactly one face is required")
            if faces:
                best = max(
                    faces,
                    key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
                )
                x1, y1, x2, y2 = [int(v) for v in best.bbox]
                h, w = img.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    return None

                info: Dict[str, Any] = {"bbox": (x1, y1, x2 - x1, y2 - y1)}

                if hasattr(best, "kps") and best.kps is not None:
                    try:
                        info["kps"] = np.asarray(best.kps, dtype=np.float32).tolist()
                    except Exception:
                        pass

                # Плотные landmarks для UI (facemesh-эффект)
                # buffalo_l обычно содержит landmark_2d_106 и landmark_3d_68 модели.
                if hasattr(best, "landmark_2d_106") and best.landmark_2d_106 is not None:
                    try:
                        lmk = np.asarray(best.landmark_2d_106, dtype=np.float32)
                        if lmk.ndim == 2 and lmk.shape[1] == 2 and lmk.shape[0] >= 50:
                            info["lmk106"] = lmk.tolist()
                    except Exception:
                        pass

                if hasattr(best, "landmark_3d_68") and best.landmark_3d_68 is not None:
                    try:
                        lmk3 = np.asarray(best.landmark_3d_68, dtype=np.float32)
                        if lmk3.ndim == 2 and lmk3.shape[1] == 3 and lmk3.shape[0] >= 50:
                            info["lmk68_3d"] = lmk3.tolist()
                    except Exception:
                        pass

                if hasattr(best, "det_score") and best.det_score is not None:
                    try:
                        info["score"] = float(best.det_score)
                    except Exception:
                        pass

                return info
        except ValueError:
            raise
        except Exception:
            pass

    # 2) Fallback: Haar (bbox only)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) > 1:
        raise ValueError("Exactly one face is required")
    if len(faces) > 0:
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return {"bbox": tuple(largest_face)}

    return None


class SkinRecognitionSystem:
    """Research prototype. Input is a single face crop in BGR/gray uint8.

    The acceptance threshold is squared L2. The returned similarity is a
    normalized geometric score, NOT a probability or a calibrated confidence.
    """
    def __init__(self, database_path=None, descriptor_dim=512,
                 recognition_threshold=0.7, descriptor_path=None):
        if not np.isfinite(recognition_threshold) or not 0 <= recognition_threshold <= 4:
            raise ValueError("Squared-L2 threshold must lie in [0, 4]")
        self.descriptor = (HybridSkinDescriptor.load(descriptor_path) if descriptor_path
                           else HybridSkinDescriptor(descriptor_dim=descriptor_dim))
        self.descriptor_dim = self.descriptor.descriptor_dim
        self.recognition_threshold = float(recognition_threshold)
        self.database_path = database_path
        self._lock = threading.RLock()
        self.database = SkinEmbeddingDatabase(
            embedding_dim=self.descriptor_dim, index_type="L2",
            feature_space_id=self.descriptor.feature_space_id)
        if database_path:
            path = self.database.storage_path(database_path)
            if path.exists():
                self.database.load(database_path)
            elif os.path.exists(str(database_path) + ".pkl"):
                raise ValueError("Legacy embeddings use an incompatible descriptor; choose a new database path")

    def _preprocess_image(self, image):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
            raise ValueError("Expected a nonempty uint8 BGR/gray face crop")
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
            raise ValueError("Expected one, three, or four image channels")
        if min(image.shape[:2]) < 32:
            raise ValueError("Face crop is too small: minimum side is 32 pixels")
        interpolation = cv2.INTER_AREA if max(image.shape[:2]) > 224 else cv2.INTER_LINEAR
        return np.ascontiguousarray(cv2.resize(image, (224, 224), interpolation=interpolation))

    def _extract_vector(self, image):
        if self.database.feature_space_id != self.descriptor.feature_space_id:
            raise ValueError("Descriptor changed after database creation; re-enrollment is required")
        return self.descriptor.extract_features(self._preprocess_image(image))

    def register_person(self, image, person_id, person_name, image_path=None, additional_metadata=None):
        with self._lock:
            vector = self._extract_vector(image)
            embedding_id = self.database.add_embedding(
                vector, person_id, person_name, image_path, additional_metadata)
            if self.database_path:
                try:
                    self.database.save(self.database_path)
                except Exception:
                    self.database.remove_embedding(embedding_id)
                    raise
            return embedding_id

    def recognize(self, image, k=5, return_distances=False):
        with self._lock:
            results = self.database.search(self._extract_vector(image), k=k)
            if not results:
                result = (None, 0.0, [])
                return result + ([],) if return_distances else result
            distance = results[0][1]
            similarity = float(np.clip(1.0 - distance / 4.0, 0, 1))
            person_name = results[0][2]["person_name"] if distance <= self.recognition_threshold else None
            candidates = [dict(m, squared_l2=float(d)) for _, d, m in results]
            result = (person_name, similarity, candidates)
            return result + ([float(r[1]) for r in results],) if return_distances else result

    def save_database(self, filepath):
        with self._lock:
            self.database.save(filepath)

    def load_database(self, filepath):
        with self._lock:
            self.database.load(filepath)
            self.database_path = filepath

    def get_database_stats(self):
        return dict(self.database.get_stats(), recognition_threshold=self.recognition_threshold,
                    threshold_status="uncalibrated" if not hasattr(self, "calibration_id") else "calibrated",
                    score_type="normalized_squared_l2_similarity_not_probability",
                    prototype=True)

    def remove_person(self, person_id):
        with self._lock:
            # Preserve both the live gallery and durable state on an I/O failure.
            with self.database._lock:
                ids = [i for i, _ in self.database.get_person_embeddings(person_id)]
                before = (list(self.database.embeddings), list(self.database.metadata),
                          dict(self.database.id_to_metadata), self.database.last_update)
                try:
                    for embedding_id in ids:
                        self.database.remove_embedding(embedding_id)
                    if self.database_path:
                        self.database.save(self.database_path)
                except Exception:
                    (self.database.embeddings, self.database.metadata,
                     self.database.id_to_metadata, self.database.last_update) = before
                    raise
                return len(ids)


def detect_face(image):
    """Return bbox for one face, None for no face; reject ambiguous multi-face inputs."""
    info = detect_face_info(image)
    return info.get("bbox") if info else None
