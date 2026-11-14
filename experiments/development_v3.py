"""Development-only classical texture experiments (v3).

This program filters the manifest to train/calibration BEFORE opening images or
feature files. It never evaluates the v2 test cohort. Calibration is used for
model development, so these numbers are not final generalization estimates.
The untouched external dataset must evaluate the eventual frozen pipeline.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np
from scipy.fft import dctn
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scientific_metrics import (
    cmc_from_gallery, cross_session_pairs, select_threshold_at_far, verification_from_scores,
)
from src.descriptors.local_patches import LocalSkinPatchExtractor
from src.utils.data_loader import read_image

ALLOWED_SPLITS = frozenset({"train", "calibration"})


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 1e-12)


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def prepared_tuple_from_face(prepared_face, mode="strict"):
    """Use the exact cache packing for live preprocessing and frozen evaluation."""
    if mode not in {"strict", "masked"}:
        raise ValueError("Unknown prepared-face packing mode")
    arrays = prepared_face.numeric_arrays()
    names = [str(name) for name in arrays["patch_names"]]
    if mode == "strict":
        patches = np.stack([arrays["skin_only_"+name] for name in names])
        quality = prepared_face.metadata["patch_quality"]
        masks = np.stack([np.full(patch.shape[:2], bool(quality[name]["strict_skin_usable"]), np.uint8)
                          for name, patch in zip(names, patches)])
    else:
        patches, masks = arrays["patches_bgr"], arrays["patch_masks"]
    return ((arrays["aligned_bgr"], patches, masks, arrays["skin_mask"]),
            prepared_face.metadata["preprocessing_id"]+":"+mode)


def development_rows(manifest_path):
    """Do not open any image while filtering; forbidden test images stay unread."""
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    required = {"path", "subject_id", "session_id", "split", "role"}
    if not all_rows or not required.issubset(all_rows[0]):
        raise ValueError("Invalid input manifest")
    rows = [row for row in all_rows if row["split"] in ALLOWED_SPLITS]
    if {row["split"] for row in rows} != ALLOWED_SPLITS:
        raise ValueError("Both train and calibration rows required")
    groups = {split: {row["subject_id"] for row in rows if row["split"] == split} for split in ALLOWED_SPLITS}
    if groups["train"] & groups["calibration"]:
        raise ValueError("Train/calibration identity leakage")
    for row in rows:
        expected = {"train"} if row["split"] == "train" else {"gallery", "probe"}
        if row["role"] not in expected:
            raise ValueError(f"Invalid development role: {row['role']}")
    for split, subjects in groups.items():
        if len(subjects) < 2:
            raise ValueError(f"At least two {split} identities required")
    for subject in groups["calibration"]:
        if {row["role"] for row in rows if row["subject_id"] == subject} != {"gallery", "probe"}:
            raise ValueError(f"Missing calibration gallery/probe: {subject}")
    return rows


@dataclass(frozen=True)
class FeatureConfig:
    descriptor: str
    scope: str
    photometric: str
    grid: int = 4


def prepare_inputs(rows, legacy_run_dir, prepared_manifest=None, prepared_mode="masked"):
    """Load only allowed development sources; reject missing preprocessed images."""
    patch_extractor = LocalSkinPatchExtractor(use_dlib=False)
    prepared, provenance = {}, []
    if prepared_manifest:
        cached = {}
        manifest_text = Path(prepared_manifest).read_text(encoding="utf-8")
        cache_rows = (json.loads(manifest_text) if manifest_text.lstrip().startswith("[") else
                      [json.loads(line) for line in manifest_text.splitlines() if line.strip()])
        for row in cache_rows:
            if row.get("split") in ALLOWED_SPLITS:
                cached[str(Path(row["source_path"]).resolve())] = row
        missing = []
        for row in rows:
            path = str(Path(row["path"]).resolve())
            metadata = cached.get(path)
            if metadata is None or metadata.get("status") != "accepted":
                # "ok"/"success" are accepted legacy spellings; all other states fail closed.
                if metadata is None or metadata.get("status") not in {"ok", "success"}:
                    missing.append({"source_path": path, "metadata": metadata})
                    continue
            if metadata["split"] != row["split"] or metadata["subject_id"] != row["subject_id"]:
                raise ValueError("Prepared cache identity/split does not match development manifest")
            cache_path = Path(metadata["cache_path"])
            if not cache_path.is_absolute():
                cache_path = Path(prepared_manifest).parent / cache_path
            with np.load(cache_path, allow_pickle=False) as archive:
                image = archive["aligned_bgr"].copy()
                patches = archive["patches_bgr"].copy()
                masks = archive["patch_masks"].copy().astype(np.uint8)
                skin_mask = archive["skin_mask"].copy().astype(np.uint8)
                if prepared_mode == "strict":
                    names = [str(name) for name in archive["patch_names"]]
                    patches = np.stack([archive["skin_only_"+name] for name in names])
                    quality = metadata.get("quality", {}).get("patch_quality", {})
                    masks = np.stack([np.full(patch.shape[:2], bool(quality.get(name, {}).get("strict_skin_usable", False)),
                                              np.uint8) for name, patch in zip(names, patches)])
            prepared[path] = (image, patches, masks, skin_mask)
            provenance.append({"path": path, "split": row["split"], "subject_id": row["subject_id"],
                               "cache_path": str(cache_path), "preprocessing_id": metadata["preprocessing_id"]+":"+prepared_mode,
                               "source_sha256": metadata["source_sha256"]})
        if missing:
            raise ValueError("Prepared inputs missing/rejected for development cohort; no silent exclusions: " +
                             json.dumps(missing, ensure_ascii=False))
    else:
        boxes = {str(Path(record["path"]).resolve()): record["bbox_xywh"] for record in
                 json.loads((Path(legacy_run_dir) / "face_boxes.json").read_text(encoding="utf-8"))}
        for row in rows:
            path = str(Path(row["path"]).resolve())
            if row["split"] not in ALLOWED_SPLITS:
                raise AssertionError("Forbidden split reached image loading")
            image = read_image(path)
            if path not in boxes or boxes[path] is None:
                raise ValueError(f"Missing legacy face box: {path}")
            x, y, width, height = boxes[path]
            image = image[y:y+height, x:x+width]
            interpolation = cv2.INTER_AREA if max(image.shape[:2]) > 224 else cv2.INTER_LINEAR
            image = cv2.resize(image, (224, 224), interpolation=interpolation)
            gray_patches = list(patch_extractor.extract_patches(image).values())
            patches = np.stack([cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR) for patch in gray_patches])
            prepared[path] = (image, patches, np.ones(patches.shape[:3], dtype=np.uint8),
                              np.ones(image.shape[:2], dtype=np.uint8))
            provenance.append({"path": path, "split": row["split"], "subject_id": row["subject_id"],
                               "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                               "preprocessing_id": "v2_haar_exactly_one_min60_resize224_geometric4roi"})
    return prepared, provenance


def photometric(gray, mode):
    if mode == "raw":
        return gray
    if mode == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if mode == "gamma":
        return np.clip(np.sqrt(gray.astype(np.float32)/255) * 255, 0, 255).astype(np.uint8)
    if mode == "local_contrast":
        x = gray.astype(np.float32)/255
        mean = cv2.GaussianBlur(x, (0, 0), 5)
        variance = cv2.GaussianBlur((x-mean)**2, (0, 0), 5)
        z = np.clip((x-mean)/np.sqrt(variance+0.01), -3, 3)
        return np.rint((z+3)*255/6).astype(np.uint8)
    raise ValueError(f"Unknown photometric transform: {mode}")


def spatial_lbp(gray, mask, grid, kind):
    configurations = ((8, 1), (16, 2), (24, 3)) if kind == "mslbp" else ((8, 1), (8, 2), (8, 3))
    blocks = []
    height, width = gray.shape
    for points, radius in configurations:
        method = "uniform" if kind == "mslbp" else "nri_uniform"
        bins = points+2 if method == "uniform" else points*(points-1)+3
        codes = local_binary_pattern(gray, points, radius, method=method).astype(np.int32)
        valid = cv2.erode(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        for y0, y1 in zip(np.linspace(0, height, grid+1).astype(int)[:-1],
                          np.linspace(0, height, grid+1).astype(int)[1:]):
            for x0, x1 in zip(np.linspace(0, width, grid+1).astype(int)[:-1],
                              np.linspace(0, width, grid+1).astype(int)[1:]):
                selected = codes[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
                hist = np.bincount(selected, minlength=bins).astype(np.float64)
                if hist.sum():
                    hist = np.sqrt(hist/hist.sum())
                blocks.append(hist)
    return np.concatenate(blocks)


def region_features(gray, mask, descriptor, grid):
    if descriptor in {"mslbp", "lbp_nri"}:
        return spatial_lbp(gray, mask, grid, descriptor)
    if descriptor == "hog":
        # Outside-mask source pixels never enter the gradient descriptor. Mask
        # geometry can still be visible; strict rectangular crops avoid that cue.
        image = gray.astype(np.float32)/255
        if not mask.all():
            mean = float(np.median(image[mask.astype(bool)])) if mask.any() else 0.0
            image = np.where(mask, image, mean)
            soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2)
            image = image*soft + mean*(1-soft)
        return hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                   block_norm="L2-Hys", transform_sqrt=True, feature_vector=True).astype(np.float64)
    if descriptor == "dct":
        image = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float64)
        resized_mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST).astype(bool)
        selected = image[resized_mask]
        if selected.size < 16:
            return np.zeros(255)
        image = np.where(resized_mask, (image-selected.mean())/(selected.std()+1), 0)
        return dctn(image, norm="ortho")[:16, :16].ravel()[1:]
    if descriptor == "rootsift":
        if not mask.all():
            fill = int(np.median(gray[mask.astype(bool)])) if mask.any() else 0
            gray = np.where(mask, gray, fill).astype(np.uint8)
        sift = cv2.SIFT_create()
        # Explicit orientation=0 gives spatially consistent upright dense descriptors.
        points = [(x, y) for y in range(8, gray.shape[0]-7, 8) for x in range(8, gray.shape[1]-7, 8)]
        keypoints = [cv2.KeyPoint(float(x), float(y), 16.0, 0.0) for x, y in points]
        _, descriptors = sift.compute(gray, keypoints)
        if descriptors is None or len(descriptors) != len(keypoints):
            raise ValueError("Dense SIFT failed to return every requested location")
        valid = cv2.erode(mask.astype(np.uint8), np.ones((17, 17), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        descriptors = descriptors.astype(np.float64)
        descriptors /= np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12)
        descriptors = np.sqrt(descriptors)
        descriptors *= np.asarray([bool(valid[y, x]) for x, y in points])[:, None]
        return descriptors.ravel()
    raise ValueError(f"Unknown descriptor: {descriptor}")


def extract_features(prepared, config):
    face, patches, masks, skin_mask = prepared
    if config.scope == "skin_regions":
        regions = list(zip(patches, masks))
    elif config.scope == "whole_face_reference":
        image = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
        regions = [(image, np.ones(image.shape[:2], np.uint8))]
    elif config.scope == "skin_mask_global":
        image = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(skin_mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        regions = [(image, mask)]
    else:
        raise ValueError("Unknown scope")
    blocks = []
    for region, mask in regions:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
        gray = photometric(gray, config.photometric)
        vector = region_features(gray, mask, config.descriptor, config.grid)
        if not np.any(mask):
            vector = np.zeros_like(vector)
        blocks.append(normalize_rows(vector))
    features = normalize_rows(np.concatenate(blocks)).astype(np.float32)
    if not np.isfinite(features).all() or np.linalg.norm(features) < 1e-6:
        raise ValueError("Descriptor is non-finite or empty after skin masking")
    return features


def fit_pca_family(train, all_features, max_components, seed):
    n_components = min(max_components, len(train)-1, train.shape[1])
    if n_components < 2:
        raise ValueError("Too few training samples for PCA")
    model = PCA(n_components=n_components, whiten=False, svd_solver="randomized", random_state=seed)
    model.fit(train)
    transformed = model.transform(all_features)
    return model, transformed


def within_identity_whitener(train, train_ids, power=1.0):
    """Train-only shrinkage whitening of paired within-identity differences."""
    differences = []
    for identity in np.unique(train_ids):
        vectors = train[train_ids == identity]
        if len(vectors) < 2:
            raise ValueError("Within-identity whitening requires >=2 train images per identity")
        centered = vectors-vectors.mean(axis=0)
        differences.extend(centered)
    model = LedoitWolf(assume_centered=True).fit(np.asarray(differences))
    values, vectors = np.linalg.eigh(model.covariance_)
    floor = max(float(values.max())*1e-3, 1e-12)
    transform = (vectors * np.maximum(values, floor)[None, :]**(-power/2)) @ vectors.T
    return transform


def evaluate_calibration(features, rows, target_far):
    gallery = np.asarray([i for i, row in enumerate(rows) if row["split"] == "calibration" and row["role"] == "gallery"])
    probes = np.asarray([i for i, row in enumerate(rows) if row["split"] == "calibration" and row["role"] == "probe"])
    gallery_ids = np.asarray([rows[i]["subject_id"] for i in gallery])
    probe_ids = np.asarray([rows[i]["subject_id"] for i in probes])
    y, scores, _, _, matrix = cross_session_pairs(features[gallery], gallery_ids, features[probes], probe_ids)
    threshold = select_threshold_at_far(y, scores, target_far)
    metric = verification_from_scores(y, scores, threshold, threshold_source="development_calibration_tuned")
    cmc = cmc_from_gallery(matrix, gallery_ids, probe_ids, max_rank=min(10, len(gallery_ids)))
    return {**asdict(metric), "rank1": float(cmc[0]), "cmc": cmc.tolist()}


def feature_configs(descriptors, photometrics, scopes):
    return [FeatureConfig(descriptor, scope, photo, grid=4 if scope == "skin_regions" else 8)
            for scope in scopes for descriptor in descriptors for photo in photometrics]


class ClassicalTextureDescriptorV3:
    """Frozen, safely serializable descriptor with an explicit preprocessing ID.

    Input is the prepared tuple ``(face_bgr, patches_bgr, patch_masks, skin_mask)``.
    A caller must pass the matching preprocessing ID on every extraction. The
    fitted descriptor never auto-fits from an inference image or changes space.
    """
    version = "classical-texture-v3.1"

    def __init__(self, configuration, preprocessing_id, max_pca_components=128, seed=42):
        self.configuration = json.loads(json.dumps(configuration))
        self.feature_config = FeatureConfig(**self.configuration["features"])
        self.preprocessing_id = str(preprocessing_id)
        self.max_pca_components = int(max_pca_components)
        self.seed = int(seed)
        self.pca_mean = np.empty(0, np.float32)
        self.pca_components = np.empty((0, 0), np.float32)
        self.post_transform = np.empty((0, 0), np.float64)
        self.fitted = False

    def fit(self, training_prepared, training_ids=None):
        raw = np.stack([extract_features(face, self.feature_config) for face in training_prepared])
        transform = self.configuration["transform"]
        kind = transform["kind"]
        if kind == "identity":
            self.raw_dimension = raw.shape[1]
            self.descriptor_dim = self.raw_dimension
        else:
            dimension = int(transform["dimension"])
            model, projected = fit_pca_family(raw, raw, self.max_pca_components, self.seed)
            if dimension > projected.shape[1]:
                raise ValueError("Requested dimension exceeds training PCA rank")
            self.pca_mean = model.mean_.copy()
            self.pca_components = model.components_[:dimension].copy()
            if kind == "pca":
                scale = np.maximum(model.explained_variance_[:dimension], model.explained_variance_[0]*
                                   transform["variance_floor_fraction"])**(-transform["whitening_power"]/2)
                self.post_transform = np.diag(scale.astype(np.float64))
            elif kind == "pca_within_identity_whitening":
                if training_ids is None or len(training_ids) != len(raw):
                    raise ValueError("Paired training identity labels required")
                self.post_transform = within_identity_whitener(projected[:, :dimension], np.asarray(training_ids), transform["power"])
            else:
                raise ValueError("Unknown fitted transform")
            self.raw_dimension = raw.shape[1]
            self.descriptor_dim = dimension
        self.fitted = True
        return self

    @classmethod
    def from_development(cls, run_dir, candidate_id):
        """Import exact train-fitted matrices from a recorded development candidate."""
        run_dir = Path(run_dir)
        result = json.loads((run_dir/"results.json").read_text(encoding="utf-8"))
        selected = next((item for item in result["results"] if item["id"] == candidate_id), None)
        if selected is None:
            raise ValueError("Candidate ID is not present in development results")
        provenance = json.loads((run_dir/"provenance.json").read_text(encoding="utf-8"))
        preprocessing_ids = {item["preprocessing_id"] for item in provenance}
        if len(preprocessing_ids) != 1:
            raise ValueError("A model requires a single fixed preprocessing contract")
        config = json.loads((run_dir/"run_config.json").read_text(encoding="utf-8"))
        descriptor = cls(selected["configuration"], preprocessing_ids.pop(), config["max_pca_components"], config["seed"])
        identifier = selected["feature_id"]
        with np.load(run_dir/"features"/(identifier+".npz"), allow_pickle=False) as archive:
            descriptor.raw_dimension = int(archive["features"].shape[1])
        transform = descriptor.configuration["transform"]
        if transform["kind"] == "identity":
            descriptor.descriptor_dim = descriptor.raw_dimension
        else:
            dimension = int(transform["dimension"])
            with np.load(run_dir/"features"/(identifier+"_pca.npz"), allow_pickle=False) as archive:
                descriptor.pca_mean = archive["mean"].copy()
                descriptor.pca_components = archive["components"][:dimension].copy()
                variance = archive["explained_variance"]
                if transform["kind"] == "pca":
                    scale = np.maximum(variance[:dimension], variance[0]*transform["variance_floor_fraction"])**(-transform["whitening_power"]/2)
                    descriptor.post_transform = np.diag(scale.astype(np.float64))
            if transform["kind"] == "pca_within_identity_whitening":
                filename = identifier+f"_within_{dimension}_{transform['power']}.npz"
                with np.load(run_dir/"features"/filename, allow_pickle=False) as archive:
                    descriptor.post_transform = archive["transform"].copy()
            descriptor.descriptor_dim = dimension
        descriptor.fitted = True
        descriptor._validate()
        return descriptor

    def transform_raw(self, raw):
        if not self.fitted:
            raise ValueError("Descriptor must be fitted on training data first")
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape[-1] != self.raw_dimension or not np.isfinite(raw).all():
            raise ValueError("Invalid raw feature vector")
        if self.configuration["transform"]["kind"] == "identity":
            return normalize_rows(raw).astype(np.float32)
        projected = (raw-self.pca_mean) @ self.pca_components.T
        return normalize_rows(projected @ self.post_transform).astype(np.float32)

    def extract_features(self, prepared, preprocessing_id):
        if preprocessing_id != self.preprocessing_id:
            raise ValueError("Preprocessing contract mismatch; embeddings would be incomparable")
        return self.transform_raw(extract_features(prepared, self.feature_config))

    def _metadata(self):
        return {"version": self.version, "configuration": self.configuration,
                "preprocessing_id": self.preprocessing_id, "max_pca_components": self.max_pca_components,
                "seed": self.seed, "fitted": self.fitted, "raw_dimension": self.raw_dimension,
                "descriptor_dim": self.descriptor_dim}

    @property
    def feature_space_id(self):
        if not self.fitted:
            raise ValueError("Unfitted descriptor has no deployable feature space")
        digest = hashlib.sha256(json.dumps(self._metadata(), sort_keys=True, separators=(",", ":")).encode())
        for array in (self.pca_mean, self.pca_components, self.post_transform):
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        return self.version+":"+digest.hexdigest()

    def _validate(self):
        if not self.fitted or self.raw_dimension < 1 or self.descriptor_dim < 1:
            raise ValueError("Invalid descriptor metadata")
        if not all(np.isfinite(value).all() for value in (self.pca_mean, self.pca_components, self.post_transform)):
            raise ValueError("Nonfinite descriptor weights")
        if self.configuration["transform"]["kind"] != "identity":
            if (self.pca_mean.shape != (self.raw_dimension,) or
                self.pca_components.shape != (self.descriptor_dim, self.raw_dimension) or
                self.post_transform.shape != (self.descriptor_dim, self.descriptor_dim)):
                raise ValueError("Invalid fitted projection dimensions")

    def save(self, path):
        self._validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
        metadata = {**self._metadata(), "feature_space_id": self.feature_space_id}
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                                    pca_mean=self.pca_mean, pca_components=self.pca_components,
                                    post_transform=self.post_transform)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return str(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata["version"] != cls.version:
                raise ValueError("Unsupported classical descriptor version")
            descriptor = cls(metadata["configuration"], metadata["preprocessing_id"], metadata["max_pca_components"], metadata["seed"])
            for name in ("raw_dimension", "descriptor_dim", "fitted"):
                setattr(descriptor, name, metadata[name])
            for name in ("pca_mean", "pca_components", "post_transform"):
                setattr(descriptor, name, archive[name].copy())
            descriptor._validate()
            if descriptor.feature_space_id != metadata["feature_space_id"]:
                raise ValueError("Descriptor checksum does not match metadata/weights")
        return descriptor


def run(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Development output must be a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    def save(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save("status.json", {"status": "running", "stage": "development_only_no_test_access"})
    try:
        rows = development_rows(args.manifest)
        train_index = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train"])
        train_ids = np.asarray([rows[i]["subject_id"] for i in train_index])
        prepared, provenance = prepare_inputs(rows, args.legacy_run_dir, args.prepared_manifest, args.prepared_mode)
        save("provenance.json", provenance)
        configs = feature_configs(args.descriptors.split(","), args.photometrics.split(","), args.scopes.split(","))
        candidate_configs = None
        if args.candidate_configs:
            candidate_configs = json.loads(Path(args.candidate_configs).read_text(encoding="utf-8"))
            if not isinstance(candidate_configs, list) or not candidate_configs:
                raise ValueError("Candidate configs must be a nonempty JSON list")
            configs = list(dict.fromkeys(FeatureConfig(**item["features"]) for item in candidate_configs))
        run_config = vars(args).copy()
        run_config.update(
            stage="development_only", forbidden_split="test",
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            image_counts={split: sum(row["split"] == split for row in rows) for split in sorted(ALLOWED_SPLITS)},
            identity_counts={split: len({row["subject_id"] for row in rows if row["split"] == split})
                             for split in sorted(ALLOWED_SPLITS)},
            selection_rule="maximum calibration rank1, then minimum calibration EER",
            external_holdout_required=True,
        )
        save("run_config.json", run_config)
        all_results, failures = [], []
        matrices = output / "features"
        matrices.mkdir()
        for feature_config in configs:
            specification = asdict(feature_config)
            identifier = config_id(specification)
            print("FEATURE", identifier, json.dumps(specification), flush=True)
            feature_started = time.perf_counter()
            try:
                features = np.stack([extract_features(prepared[str(Path(row["path"]).resolve())], feature_config) for row in rows])
                elapsed = time.perf_counter()-feature_started
                np.savez_compressed(matrices/(identifier+".npz"), features=features,
                                    subject_ids=np.asarray([row["subject_id"] for row in rows]),
                                    splits=np.asarray([row["split"] for row in rows]),
                                    paths=np.asarray([row["path"] for row in rows]))
                method_results = []
                def record(transformed, transform_config):
                    specification_full = {"features": specification, "transform": transform_config}
                    if candidate_configs is not None and specification_full not in candidate_configs:
                        return
                    metrics = evaluate_calibration(transformed, rows, args.target_far)
                    item = {"id": config_id(specification_full), "feature_id": identifier,
                            "configuration": specification_full,
                            "feature_extraction_seconds": elapsed,
                            "mean_seconds_per_image": elapsed/len(rows),
                            "dimension": int(transformed.shape[1]), "calibration": metrics,
                            "engineering_target_met_on_development_only": metrics["eer"] <= .05 and metrics["rank1"] >= .95}
                    all_results.append(item)
                    method_results.append(item)
                record(features, {"kind": "identity"})
                pca_started = time.perf_counter()
                model, projected = fit_pca_family(features[train_index], features, args.max_pca_components, args.seed)
                np.savez_compressed(matrices/(identifier+"_pca.npz"), mean=model.mean_, components=model.components_,
                                    explained_variance=model.explained_variance_)
                for dimension in (16, 32, 64, 96, 128):
                    if dimension > projected.shape[1]:
                        continue
                    for whitening_power in (0.0, 0.5, 1.0):
                        scale = np.maximum(model.explained_variance_[:dimension],
                                           model.explained_variance_[0]*1e-3)**(-whitening_power/2)
                        transformed = normalize_rows(projected[:, :dimension]*scale)
                        record(transformed, {"kind": "pca", "dimension": dimension,
                                             "whitening_power": whitening_power, "variance_floor_fraction": 1e-3})
                    if args.within_identity_whitening:
                        for power in (0.5, 1.0):
                            transform = within_identity_whitener(projected[train_index, :dimension], train_ids, power)
                            transformed = normalize_rows(projected[:, :dimension] @ transform)
                            record(transformed, {"kind": "pca_within_identity_whitening", "dimension": dimension,
                                                 "power": power, "shrinkage": "LedoitWolf_train_only"})
                            np.savez_compressed(matrices/(identifier+f"_within_{dimension}_{power}.npz"), transform=transform)
                best = min(method_results, key=lambda item: (-item["calibration"]["rank1"], item["calibration"]["eer"]))
                print("BEST FEATURE", identifier, "rank1", round(best["calibration"]["rank1"], 4),
                      "EER", round(best["calibration"]["eer"], 4), "transform", best["configuration"]["transform"],
                      "feature_seconds", round(elapsed, 2), "fit_eval_seconds", round(time.perf_counter()-pca_started, 2), flush=True)
            except (ValueError, RuntimeError, cv2.error) as exc:
                failures.append({"feature_id": identifier, "configuration": specification,
                                 "error_type": type(exc).__name__, "error": str(exc)})
                print("FEATURE FAILED", identifier, str(exc), flush=True)
            save("partial_results.json", {"stage": "development_only", "results": all_results, "failures": failures})
        if not all_results:
            raise ValueError("Every development configuration failed")
        ranked = sorted(all_results, key=lambda item: (-item["calibration"]["rank1"], item["calibration"]["eer"]))
        best_skin = [item for item in ranked if item["configuration"]["features"]["scope"] != "whole_face_reference"]
        report = {"stage": "development_only_not_external_validation", "tested_configurations": len(all_results),
                  "selection_rule": run_config["selection_rule"], "best": ranked[0],
                  "best_skin": best_skin[0] if best_skin else None,
                  "results": ranked, "failures": failures, "elapsed_seconds": time.perf_counter()-started,
                  "warnings": [
                      "Calibration is repeatedly used for development; reported maxima are selection-biased.",
                      "No test images or test labels were evaluated by this script.",
                      "Whole-face references must not be described as skin-only identification.",
                      "Skin masks, when supplied, are heuristic and not ground-truth skin segmentation.",
                      "An untouched external cohort is required for a frozen-method accuracy claim.",
                  ]}
        save("results.json", report)
        with (output/"results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "descriptor", "scope", "photometric", "transform", "dimension", "eer", "rank1", "auc", "far", "frr"])
            for item in ranked:
                feature = item["configuration"]["features"]
                metric = item["calibration"]
                writer.writerow([item["id"], feature["descriptor"], feature["scope"], feature["photometric"],
                                 json.dumps(item["configuration"]["transform"], sort_keys=True), item["dimension"],
                                 metric["eer"], metric["rank1"], metric["auc"], metric["far"], metric["frr"]])
        save("status.json", {"status": "completed", "stage": report["stage"],
                             "elapsed_seconds": report["elapsed_seconds"]})
        print("DEVELOPMENT COMPLETE", json.dumps({"best": report["best"], "best_skin": report["best_skin"]}), flush=True)
        return report
    except Exception as exc:
        save("status.json", {"status": "failed", "error": str(exc), "error_type": type(exc).__name__})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", default=str(ROOT/"research/results/fei_frontal_seed42_conditional/evaluated_manifest.csv"))
    result.add_argument("--legacy-run-dir", default=str(ROOT/"research/results/fei_frontal_seed42_conditional"))
    result.add_argument("--prepared-manifest")
    result.add_argument("--prepared-mode", choices=("masked", "strict"), default="masked")
    result.add_argument("--candidate-configs", help="JSON list of previously selected fixed configurations; no new hyperparameter search")
    result.add_argument("--output", required=True)
    result.add_argument("--descriptors", default="lbp_nri,mslbp,hog,dct,rootsift")
    result.add_argument("--photometrics", default="raw,clahe")
    result.add_argument("--scopes", default="skin_regions,whole_face_reference")
    result.add_argument("--max-pca-components", type=int, default=128)
    result.add_argument("--within-identity-whitening", action="store_true")
    result.add_argument("--target-far", type=float, default=.01)
    result.add_argument("--seed", type=int, default=42)
    return result


if __name__ == "__main__":
    run(parser().parse_args())

