"""Deterministic geometric-ROI Gabor/GLCM descriptor with optional train-only PCA.

This research implementation makes no claim of 3D reconstruction, validated pore
recognition, or equivalence to the published LGLG-WPCA method. Legacy flags remain
for compatibility. A fitted feature space must be frozen before enrollment.
"""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import cv2
import numpy as np
from ._image import as_gray_uint8
from .gabor_3d import Gabor2DFilterBank
from .lgjg_wpca import GaborStatisticsPCA, GaborPCAICA, moment_statistics, summary_statistics
from .glcm import GLCMTextureAnalyzer
from .local_patches import LocalSkinPatchExtractor


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64).ravel()
    if not np.isfinite(vector).all():
        raise ValueError("Descriptor contains nonfinite statistics")
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


class HybridSkinDescriptor:
    version = "skin-texture-v2"

    def __init__(self, use_lgwg_wpca=True, use_multifold_pca=True, use_glcm=True,
                 use_local_patches=True, descriptor_dim=512):
        if int(descriptor_dim) != descriptor_dim or not 1 <= descriptor_dim <= 65536:
            raise ValueError("descriptor_dim must be an integer in [1, 65536]")
        self.use_lgwg_wpca = bool(use_lgwg_wpca)
        self.use_multifold_pca = bool(use_multifold_pca)
        self.use_glcm = bool(use_glcm)
        self.use_local_patches = bool(use_local_patches)
        self.descriptor_dim = int(descriptor_dim)
        self.gabor_bank = Gabor2DFilterBank()
        self.patch_extractor = LocalSkinPatchExtractor()
        self.glcml_analyzer = GLCMTextureAnalyzer()
        self.lgwg_wpca = GaborStatisticsPCA(n_components=128)
        self.multifold_pca = GaborPCAICA(n_pca_components=64, n_ica_components=32)
        self.lgwg_wpca_trained = False
        self.multifold_pca_trained = False

    def _patches(self, face_image, face_bbox=None):
        gray = as_gray_uint8(face_image, minimum_size=16)
        if self.use_local_patches:
            return list(self.patch_extractor.extract_patches(gray, face_bbox).values())
        if face_bbox is not None:
            if len(face_bbox) != 4 or not np.isfinite(face_bbox).all():
                raise ValueError("face_bbox must be (x, y, width, height)")
            x, y, width, height = map(int, face_bbox)
            if x < 0 or y < 0 or width < 16 or height < 16 or x + width > gray.shape[1] or y + height > gray.shape[0]:
                raise ValueError("face_bbox must be inside the image and have size >= 16")
            gray = gray[y:y + height, x:x + width]
        size = self.patch_extractor.target_size
        interpolation = cv2.INTER_AREA if min(gray.shape) >= size else cv2.INTER_LINEAR
        return [cv2.resize(gray, (size, size), interpolation=interpolation)]

    def extract_features(self, face_image, face_bbox=None):
        """Extract a fixed vector from an already cropped BGR/gray face.

        Optional bbox uses (x, y, width, height), in source-image coordinates.
        Inputs must already meet the acquisition/cropping protocol; this method
        does not detect whether the image is a face.
        """
        combined = np.concatenate([_unit(self._extract_patch_features(patch))
                                   for patch in self._patches(face_image, face_bbox)])
        if combined.size > self.descriptor_dim:
            # Balanced, deterministic contiguous-bin means: no sample-wise fit.
            projected = np.asarray([chunk.mean() for chunk in np.array_split(combined, self.descriptor_dim)])
        else:
            projected = np.pad(combined, (0, self.descriptor_dim - combined.size))
        descriptor = _unit(projected).astype(np.float32)
        if np.linalg.norm(descriptor) < 1e-12:
            raise ValueError("Image produced a zero descriptor")
        return descriptor

    def _extract_patch_features(self, patch):
        responses = self._get_gabor_responses(patch)
        features = [_unit(self.gabor_bank.features_from_responses(responses))]
        if self.use_lgwg_wpca:
            weighted = self.lgwg_wpca.transform(responses) if self.lgwg_wpca_trained else moment_statistics(responses)
            features.append(_unit(weighted))
        if self.use_multifold_pca:
            multifold = self.multifold_pca.transform(responses) if self.multifold_pca_trained else summary_statistics(responses)
            features.append(_unit(multifold))
        if self.use_glcm:
            features.append(_unit(self.glcml_analyzer.extract_features(patch)))
        return np.concatenate(features)

    def _get_gabor_responses(self, image):
        return self.gabor_bank.responses(image)

    def _extract_basic_gabor_stats(self, gabor_responses):
        return summary_statistics(gabor_responses)

    def fit(self, training_images):
        """Fit projections using ONLY training faces, with the inference ROI path.

        Patch summary vectors are retained instead of the full response maps.
        Call before creating an embedding database; fitting changes feature_space_id.
        """
        moments, summaries = [], []
        image_count = 0
        for image in training_images:
            image_count += 1
            for patch in self._patches(image):
                if not self.use_lgwg_wpca and not self.use_multifold_pca:
                    continue
                responses = self._get_gabor_responses(patch)
                if self.use_lgwg_wpca:
                    moments.append(moment_statistics(responses))
                if self.use_multifold_pca:
                    summaries.append(summary_statistics(responses))
        if image_count < 2:
            raise ValueError("At least two training images are required")
        # Commit both models only after both fits succeed.
        weighted = GaborStatisticsPCA(self.lgwg_wpca.n_components, self.lgwg_wpca.alpha)
        multifold = GaborPCAICA(self.multifold_pca.n_pca_components, self.multifold_pca.n_ica_components)
        if self.use_lgwg_wpca:
            weighted.fit_statistics(np.asarray(moments))
        if self.use_multifold_pca:
            multifold.fit_statistics(np.asarray(summaries))
        self.lgwg_wpca, self.multifold_pca = weighted, multifold
        self.lgwg_wpca_trained = self.use_lgwg_wpca
        self.multifold_pca_trained = self.use_multifold_pca
        return self

    def get_config(self):
        return {
            "use_lgwg_wpca": self.use_lgwg_wpca,
            "use_multifold_pca": self.use_multifold_pca,
            "use_glcm": self.use_glcm,
            "use_local_patches": self.use_local_patches,
            "descriptor_dim": self.descriptor_dim,
            "patch_size": self.patch_extractor.target_size,
            "gabor": {"frequencies": self.gabor_bank.frequencies,
                      "orientations": self.gabor_bank.orientations,
                      "scales": self.gabor_bank.scales,
                      "sigma_x": self.gabor_bank.sigma_x, "sigma_y": self.gabor_bank.sigma_y},
            "glcm": {"distances": self.glcml_analyzer.distances, "angles": self.glcml_analyzer.angles,
                     "levels": self.glcml_analyzer.levels, "symmetric": self.glcml_analyzer.symmetric,
                     "normed": self.glcml_analyzer.normed},
            "weighted_pca": {"n_components": self.lgwg_wpca.n_components, "alpha": self.lgwg_wpca.alpha},
            "pca_ica": {"n_pca_components": self.multifold_pca.n_pca_components,
                        "n_ica_components": self.multifold_pca.n_ica_components}
        }

    def _learned_arrays(self):
        arrays = {}
        if self.lgwg_wpca_trained:
            arrays.update(wpca_weights=self.lgwg_wpca.weights,
                          wpca_mean=self.lgwg_wpca.pca.mean_,
                          wpca_components=self.lgwg_wpca.pca.components_)
        if self.multifold_pca_trained:
            arrays.update(pca_mean=self.multifold_pca.pca.mean_,
                          pca_components=self.multifold_pca.pca.components_,
                          ica_mean=self.multifold_pca.ica.mean_,
                          ica_components=self.multifold_pca.ica.components_)
        return {name: np.asarray(value, dtype="<f8") for name, value in arrays.items()}

    @property
    def feature_space_id(self):
        metadata = {"version": self.version, "config": self.get_config(),
                    "weighted_fitted": self.lgwg_wpca_trained,
                    "pca_ica_fitted": self.multifold_pca_trained}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        for name, value in sorted(self._learned_arrays().items()):
            digest.update(name.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        return self.version + ":" + digest.hexdigest()

    def save(self, path):
        """Atomically store JSON and numeric arrays, never executable pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"version": self.version, "config": self.get_config(),
                    "weighted_fitted": self.lgwg_wpca_trained,
                    "pca_ica_fitted": self.multifold_pca_trained,
                    "feature_space_id": self.feature_space_id}
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **self._learned_arrays())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return str(path)

    @classmethod
    def load(cls, path):
        """Load the numeric artifact and validate dimensions plus feature-space digest."""
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata["version"] != cls.version:
                raise ValueError("Unsupported descriptor version")
            config = dict(metadata["config"])
            patch_size, gabor, glcm = config.pop("patch_size"), config.pop("gabor"), config.pop("glcm")
            weighted, pca_ica = config.pop("weighted_pca"), config.pop("pca_ica")
            descriptor = cls(**config)
            descriptor.patch_extractor = LocalSkinPatchExtractor(target_size=patch_size)
            descriptor.gabor_bank = Gabor2DFilterBank(**gabor)
            descriptor.glcml_analyzer = GLCMTextureAnalyzer(**glcm)
            descriptor.lgwg_wpca = GaborStatisticsPCA(**weighted)
            descriptor.multifold_pca = GaborPCAICA(**pca_ica)
            descriptor.lgwg_wpca_trained = bool(metadata["weighted_fitted"])
            descriptor.multifold_pca_trained = bool(metadata["pca_ica_fitted"])
            n_filters = len(descriptor.gabor_bank.filters)
            def get(name):
                value = np.asarray(archive[name], dtype=np.float64)
                if value.size == 0 or not np.isfinite(value).all():
                    raise ValueError("Invalid learned descriptor array: " + name)
                return value.copy()
            if descriptor.lgwg_wpca_trained:
                if not descriptor.use_lgwg_wpca:
                    raise ValueError("Fitted weighted PCA is disabled in config")
                model = descriptor.lgwg_wpca
                weights, mean, components = get("wpca_weights"), get("wpca_mean"), get("wpca_components")
                if weights.shape != (4 * n_filters,) or mean.shape != weights.shape or components.ndim != 2 or components.shape[1] != weights.size or not 1 <= components.shape[0] <= model.n_components or np.any(weights <= 0):
                    raise ValueError("Invalid weighted PCA dimensions")
                model.weights = weights
                model.pca = SimpleNamespace(mean_=mean, components_=components)
                model.is_fitted = True
            if descriptor.multifold_pca_trained:
                if not descriptor.use_multifold_pca:
                    raise ValueError("Fitted PCA/ICA is disabled in config")
                model = descriptor.multifold_pca
                mean, components, ica_mean, ica_components = (get(name) for name in ("pca_mean", "pca_components", "ica_mean", "ica_components"))
                if mean.shape != (5 * n_filters,) or components.ndim != 2 or components.shape[1] != mean.size or not 1 <= components.shape[0] <= model.n_pca_components or ica_mean.ndim != 1 or not 1 <= ica_mean.size <= components.shape[0] or ica_components.ndim != 2 or ica_components.shape[1] != ica_mean.size or not 1 <= ica_components.shape[0] <= model.n_ica_components:
                    raise ValueError("Invalid PCA/ICA dimensions")
                model.pca = SimpleNamespace(mean_=mean, components_=components)
                model.ica = SimpleNamespace(mean_=ica_mean, components_=ica_components)
                model.ica_input_dim = ica_mean.size
                model.is_fitted = True
            if descriptor.feature_space_id != metadata["feature_space_id"]:
                raise ValueError("Descriptor metadata or weights failed the feature-space digest check")
        return descriptor


def extract_gabor_features(image):
    return HybridSkinDescriptor().extract_features(image)

