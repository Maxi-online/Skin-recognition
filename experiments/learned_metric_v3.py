"""Train-only linear metric learning; selection uses development calibration only."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from experiments.scientific_metrics import compute_eer, select_threshold_at_far, verification_from_scores


def unit(x):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def fit_projection(features, labels, dimension, shrinkage=1.0):
    """PCA then regularized within-class covariance normalization.

    All inputs MUST be from train identities. Shrinkage=1 retains PCA geometry;
    smaller shrinkage penalizes directions that vary within a train identity.
    """
    x = unit(features)
    if len(x) != len(labels) or len(set(labels)) < 2:
        raise ValueError("Training labels must match features and include multiple identities")
    dimension = min(int(dimension), x.shape[1], len(x) - 1)
    pca = PCA(n_components=dimension, svd_solver="full").fit(x)
    projected = pca.transform(x)
    residuals = np.empty_like(projected)
    labels = np.asarray(labels)
    for subject in np.unique(labels):
        selected = labels == subject
        residuals[selected] = projected[selected] - projected[selected].mean(axis=0)
    dof = max(1, len(x) - len(np.unique(labels)))
    within = residuals.T @ residuals / dof
    scale = np.trace(within) / dimension
    regularized = (1 - shrinkage) * within + shrinkage * max(scale, 1e-8) * np.eye(dimension)
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    inverse_root = (eigenvectors * np.maximum(eigenvalues, 1e-8) ** -0.5) @ eigenvectors.T
    return pca.mean_, pca.components_.T @ inverse_root


def transform(features, mean, projection):
    return unit((unit(features) - mean) @ projection).astype(np.float32)


def development_metrics(features, records, target_far=.01):
    if any(record["split"] not in ("train", "calibration") for record in records):
        raise ValueError("Development must never receive test rows")
    gallery = [i for i, r in enumerate(records) if r["split"] == "calibration" and r["role"] == "gallery"]
    probes = [i for i, r in enumerate(records) if r["split"] == "calibration" and r["role"] == "probe"]
    if not gallery or not probes:
        raise ValueError("Calibration gallery and probe are required")
    gl = np.array([records[i]["subject_id"] for i in gallery])
    pl = np.array([records[i]["subject_id"] for i in probes])
    scores = -cdist(features[probes], features[gallery])
    labels = (pl[:, None] == gl).astype(np.int8)
    if not labels.any(axis=1).all():
        raise ValueError("Each calibration probe requires a gallery identity")
    threshold = select_threshold_at_far(labels.ravel(), scores.ravel(), target_far)
    result = asdict(verification_from_scores(labels.ravel(), scores.ravel(), threshold,
                                            threshold_source="development_calibration"))
    best_genuine = np.max(np.where(labels, scores, -np.inf), axis=1)
    # Conservative tie policy, consistent with the frozen v2 protocol.
    best_impostor = np.max(np.where(1 - labels, scores, -np.inf), axis=1)
    result["rank1"] = float(np.mean(best_genuine > best_impostor))
    result["calibration_subjects"] = len(np.unique(pl))
    return result


def select_metric(features, records, output, prefix="features"):
    """Finite prespecified development grid. No unseen evaluation occurs here."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(r["split"] not in ("train", "calibration") for r in records):
        raise ValueError("Attempted test access in development")
    train = np.array([r["split"] == "train" for r in records])
    labels = np.array([r["subject_id"] for r in records])[train]
    trial_results = []
    best_key, best_state = None, None
    variants = [("raw", None, None)] + [(f"pca{d}_wccn{a}", d, a)
                     for d in (32, 64, 128) for a in (0.1, 0.3, 0.7, 1.0)]
    for name, dimension, alpha in variants:
        if dimension is None:
            mean, projection = np.zeros(features.shape[1]), None
            candidate = unit(features).astype(np.float32)
        else:
            mean, projection = fit_projection(features[train], labels, dimension, alpha)
            candidate = transform(features, mean, projection)
        metrics = development_metrics(candidate, records)
        result = dict(variant=name, dimension=dimension, shrinkage=alpha, metrics=metrics)
        trial_results.append(result)
        key = (metrics["eer"], -metrics["rank1"], metrics["frr"])
        if best_key is None or key < best_key:
            best_key, best_state = key, (result, mean, projection, candidate)
        print(f"{prefix}/{name}: dev EER={metrics['eer']:.4f}, rank1={metrics['rank1']:.4f}", flush=True)
    result, mean, projection, features_out = best_state
    payload = dict(stage="development_only", selection="lowest_calibration_EER_then_rank1_then_FRR",
                   test_images_loaded=0, feature_prefix=prefix, best=result, trials=trial_results)
    (output / f"{prefix}_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    config = dict(feature_prefix=prefix, raw=projection is None, variant=result["variant"],
                  input_dimension=features.shape[1])
    np.savez_compressed(output / f"{prefix}_metric.npz", mean=mean,
                        projection=projection if projection is not None else np.empty((0, 0)),
                        metadata=np.array(json.dumps(config)), features=features_out)
    return payload
