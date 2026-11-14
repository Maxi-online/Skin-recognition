"""Run a reproducible biometric pilot or cross-session evaluation.

Only train identities fit descriptor transforms. Calibration identities select
operating thresholds. Test identities supply descriptive EER/AUC and fixed-
threshold FAR/FRR. No failure is silently turned into a successful experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol import load_manifest, validate_protocol, write_sample_inventory
from experiments.scientific_metrics import (
    bootstrap_identity_intervals, cmc_from_gallery, cross_session_pairs,
    select_threshold_at_far, verification_from_scores,
)
from src.descriptors import HybridSkinDescriptor
from src.descriptors.glcm import GLCMTextureAnalyzer
from src.descriptors.local_patches import LocalSkinPatchExtractor
from src.utils.data_loader import read_image

METHOD_CONFIGS = {
    "gabor_only": dict(use_lgwg_wpca=False, use_multifold_pca=False, use_glcm=False),
    "gabor_glcm": dict(use_lgwg_wpca=False, use_multifold_pca=False, use_glcm=True),
    "hybrid": dict(use_lgwg_wpca=True, use_multifold_pca=True, use_glcm=True),
    "no_wpca": dict(use_lgwg_wpca=False, use_multifold_pca=True, use_glcm=True),
    "no_multifold": dict(use_lgwg_wpca=True, use_multifold_pca=False, use_glcm=True),
    "global": dict(use_lgwg_wpca=True, use_multifold_pca=True, use_glcm=True, use_local_patches=False),
    "hybrid_global": dict(use_lgwg_wpca=True, use_multifold_pca=True, use_glcm=True, use_local_patches=False),
}


class TextureBaseline:
    """Independent handcrafted baselines on exactly the same local ROI geometry."""
    def __init__(self, method):
        self.method = method
        self.patches = LocalSkinPatchExtractor(use_dlib=False)
        self.glcm = GLCMTextureAnalyzer()
        self.feature_space_id = "texture_baseline_v1_" + method

    def fit(self, images):
        return self

    def extract_features(self, image):
        from skimage.feature import local_binary_pattern
        blocks = []
        for patch in self.patches.extract_patches(image).values():
            if self.method == "lbp":
                values = local_binary_pattern(patch, 8, 1, method="uniform")
                vector = np.bincount(values.astype(np.int32).ravel(), minlength=10).astype(np.float64)
            else:
                vector = self.glcm.extract_features(patch).astype(np.float64)
            norm = np.linalg.norm(vector)
            blocks.append(vector / norm if norm else vector)
        features = np.concatenate(blocks)
        norm = np.linalg.norm(features)
        return (features / norm if norm else features).astype(np.float32)


def build_descriptor(method, descriptor_dim):
    if method in {"lbp", "glcm_only"}:
        return TextureBaseline(method)
    if method not in METHOD_CONFIGS:
        raise ValueError(f"Unknown method {method}")
    return HybridSkinDescriptor(descriptor_dim=descriptor_dim, **METHOD_CONFIGS[method])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment_record():
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "scikit-image", "opencv-python", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    files = sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "experiments").glob("*.py")))
    return {
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }


def preprocess(rows, crop_mode, image_size, allow_exclusions, output, allow_unknown_sessions):
    detector = None
    if crop_mode == "haar":
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            raise RuntimeError("OpenCV Haar face detector could not be loaded")
        detector_hash = hashlib.sha256(cascade_path.read_bytes()).hexdigest()
    else:
        detector_hash = None
    prepared, failures, bounding_boxes = {}, [], []
    for row in rows:
        image = read_image(row.path)
        bbox = None
        if detector is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) != 1:
                failures.append({"path": str(row.path), "subject_id": row.subject_id, "split": row.split,
                                 "role": row.role, "reason": f"expected_one_face_detected_{len(faces)}"})
                continue
            x, y, w, h = map(int, faces[0])
            bbox = [x, y, w, h]
            image = image[y:y+h, x:x+w]
        interpolation = cv2.INTER_AREA if max(image.shape[:2]) > image_size else cv2.INTER_LINEAR
        prepared[row.path] = cv2.resize(image, (image_size, image_size), interpolation=interpolation)
        bounding_boxes.append({"path": str(row.path), "bbox_xywh": bbox})
    write_json(output / "preprocessing_failures.json", failures)
    write_json(output / "face_boxes.json", bounding_boxes)
    excluded_subjects = sorted({record["subject_id"] for record in failures})
    report = {
        "mode": crop_mode, "resize": [image_size, image_size], "color_order": "BGR",
        "landmark_alignment": False, "photometric_normalization": "none",
        "haar_parameters": {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": [60, 60]} if detector else None,
        "detector_sha256": detector_hash, "input_images": len(rows), "detected_images": len(prepared),
        "failed_images": len(failures), "detection_coverage": len(prepared) / len(rows),
        "exclusion_policy": "entire_identity_if_any_image_fails",
        "excluded_subjects": excluded_subjects,
    }
    write_json(output / "preprocessing.json", report)
    if failures and not allow_exclusions:
        raise ValueError(f"{len(failures)} images failed face preprocessing. See preprocessing_failures.json. "
                         "No biometric result was generated. To report a conditional pilot explicitly use "
                         "--allow-preprocessing-exclusions.")
    retained = [row for row in rows if row.subject_id not in excluded_subjects]
    report["retained_images"] = len(retained)
    report["retained_protocol"] = validate_protocol(retained, allow_unknown_sessions)
    report["biometric_metrics_conditional_on_preprocessing_success"] = bool(failures)
    write_json(output / "preprocessing.json", report)
    return retained, prepared, report


def split_embeddings(rows, features, split):
    gallery_index = [i for i, row in enumerate(rows) if row.split == split and row.role == "gallery"]
    probe_index = [i for i, row in enumerate(rows) if row.split == split and row.role == "probe"]
    gallery_labels = np.asarray([rows[i].subject_id for i in gallery_index])
    probe_labels = np.asarray([rows[i].subject_id for i in probe_index])
    return features[gallery_index], gallery_labels, features[probe_index], probe_labels


def export_curves(output, y, scores, cmc):
    far, tar, thresholds = roc_curve(y, scores, drop_intermediate=False)
    with (output / "roc_det.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["false_accept_rate", "true_accept_rate", "false_reject_rate", "score_threshold"])
        for f, t, threshold in zip(far, tar, thresholds):
            writer.writerow([f, t, 1 - t, "reject_all" if not np.isfinite(threshold) else threshold])
    with (output / "cmc.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "identification_rate"])
        writer.writerows(enumerate(cmc, 1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    axes[0].plot(far, tar)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    axes[0].set(xlabel="False accept rate", ylabel="True accept rate", title="Test ROC", xlim=(0, 1), ylim=(0, 1))
    # Normal-deviate axes are a true DET plot; clip endpoints for finite rendering.
    clipped_far = np.clip(far, 1e-4, 1-1e-4)
    clipped_frr = np.clip(1-tar, 1e-4, 1-1e-4)
    axes[1].plot(norm.ppf(clipped_far), norm.ppf(clipped_frr))
    ticks = np.array([0.001, 0.01, 0.1, 0.5, 0.9, 0.99])
    axes[1].set_xticks(norm.ppf(ticks), ["0.1", "1", "10", "50", "90", "99"])
    axes[1].set_yticks(norm.ppf(ticks), ["0.1", "1", "10", "50", "90", "99"])
    axes[1].set(xlabel="False accept rate (%)", ylabel="False reject rate (%)", title="Test DET")
    axes[2].plot(np.arange(1, len(cmc)+1), cmc, marker="o")
    axes[2].set(xlabel="Identity rank", ylabel="Identification rate", title="Test CMC", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "curves.png", dpi=180)
    fig.savefig(output / "curves.pdf")
    plt.close(fig)


def run(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    try:
        rows, audit = load_manifest(args.manifest, args.allow_unknown_sessions)
        write_json(output / "dataset_audit.json", audit)
        write_sample_inventory(rows, output / "input_manifest.csv")
        config = vars(args).copy()
        config["environment"] = environment_record()
        config["score_definition"] = "negative_euclidean_distance"
        config["service_distance_conversion"] = "score = -sqrt(database_squared_l2)"
        config["threshold_rule"] = "accept if score >= calibration_threshold"
        config["primary_selection_rule"] = "lowest_calibration_EER; ties resolved by method order"
        write_json(output / "run_config.json", config)
        if args.audit_only:
            status.update(status="audit_passed", elapsed_seconds=time.perf_counter()-started)
            write_json(output / "status.json", status)
            return {"audit": audit}
        methods = args.methods.split(",")
        if not methods or len(set(methods)) != len(methods):
            raise ValueError("Methods must be a nonempty comma-separated list without duplicates")
        if any(method not in set(METHOD_CONFIGS) | {"lbp", "glcm_only"} for method in methods):
            raise ValueError("Unknown method in --methods")
        rows, images, prep = preprocess(rows, args.face_crop, args.image_size, args.allow_preprocessing_exclusions,
                                       output, args.allow_unknown_sessions)
        write_sample_inventory(rows, output / "evaluated_manifest.csv")
        train_images = [images[row.path] for row in rows if row.split == "train"]
        evaluation_rows = [row for row in rows if row.split != "train"]
        summary = {
            "status": "completed", "protocol": audit["protocol_kind"],
            "cross_session_established": audit["cross_session_established"],
            "warnings": audit["warnings"] + [
                "No pore-level, 3D, liveness, adversarial or demographic robustness claim is tested.",
                "Approximate geometric skin ROIs are not landmark-aligned or pixel-level skin segmented.",
                "Confidence intervals exclude model-training and threshold-calibration uncertainty.",
                "Zero empirical FAR and degenerate bootstrap intervals are not population risk guarantees.",
            ],
            "preprocessing": prep, "methods": {},
        }
        for method in methods:
            print(f"[{method}] fit on {len(train_images)} training images", flush=True)
            method_output = output / method
            method_output.mkdir()
            descriptor = build_descriptor(method, args.descriptor_dim)
            fit_started = time.perf_counter()
            descriptor.fit(train_images)
            fit_seconds = time.perf_counter() - fit_started
            features, latency = [], []
            for index, row in enumerate(evaluation_rows):
                extract_started = time.perf_counter()
                features.append(descriptor.extract_features(images[row.path]))
                latency.append(time.perf_counter() - extract_started)
                if (index + 1) % 40 == 0:
                    print(f"[{method}] extracted {index+1}/{len(evaluation_rows)} images", flush=True)
            features = np.stack(features)
            if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) == 0):
                raise ValueError(f"{method}: non-finite or zero embeddings")
            feature_space_id = getattr(descriptor, "feature_space_id", None)
            if callable(feature_space_id):
                feature_space_id = feature_space_id()
            np.savez_compressed(method_output / "embeddings.npz", embeddings=features,
                                paths=np.asarray([str(row.path) for row in evaluation_rows]),
                                subject_ids=np.asarray([row.subject_id for row in evaluation_rows]),
                                split=np.asarray([row.split for row in evaluation_rows]),
                                role=np.asarray([row.role for row in evaluation_rows]))
            if hasattr(descriptor, "save"):
                descriptor_path = method_output / "descriptor.npz"
                descriptor.save(descriptor_path)
            else:
                descriptor_path = method_output / "descriptor_config.json"
                write_json(descriptor_path, {"method": method, "feature_space_id": feature_space_id,
                                             "fitted_parameters": False})
            calibration = split_embeddings(evaluation_rows, features, "calibration")
            cy, cs, _, _, _ = cross_session_pairs(*calibration, max_pairs=args.max_pairs)
            threshold = select_threshold_at_far(cy, cs, args.target_far)
            calibration_metrics = verification_from_scores(cy, cs, threshold, threshold_source="calibration")
            test = split_embeddings(evaluation_rows, features, "test")
            ty, ts, tpi, tgi, score_matrix = cross_session_pairs(*test, max_pairs=args.max_pairs)
            test_metrics = verification_from_scores(ty, ts, threshold, threshold_source="independent_calibration")
            cmc = cmc_from_gallery(score_matrix, test[1], test[3], max_rank=min(10, len(np.unique(test[1]))))
            confidence_intervals = bootstrap_identity_intervals(
                ty, ts, tpi, tgi, threshold, n_bootstrap=args.bootstrap, seed=args.seed)
            result = {
                "method": method, "feature_space_id": feature_space_id,
                "feature_dimension": int(features.shape[1]), "fitted_images": len(train_images),
                "fit_seconds": fit_seconds,
                "extraction_seconds": {"median": float(np.median(latency)), "p95": float(np.percentile(latency, 95)),
                                       "total": float(np.sum(latency)), "images": len(latency)},
                "calibration": asdict(calibration_metrics), "test": asdict(test_metrics),
                "target_calibration_far": args.target_far,
                "calibration_empirical_far_resolution": 1.0/calibration_metrics.impostor_pairs,
                "test_intervals": confidence_intervals, "test_cmc": cmc.tolist(),
                "test_rank1": float(cmc[0]), "random_rank1_reference": 1/len(np.unique(test[1])),
                "cmc_tie_policy": "conservative_identity_rank",
                "descriptor_artifact": str(descriptor_path),
                "descriptor_sha256": hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),
            }
            with (method_output / "test_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["probe_subject_id", "gallery_subject_id", "genuine", "score", "accepted"])
                writer.writerows(zip(tpi, tgi, ty, ts, (ts >= threshold).astype(int)))
            export_curves(method_output, ty, ts, cmc)
            write_json(method_output / "metrics.json", result)
            summary["methods"][method] = result
            write_json(output / "partial_results.json", summary)
            print(f"[{method}] test EER={test_metrics.eer:.4f}, AUC={test_metrics.auc:.4f}, "
                  f"rank1={cmc[0]:.4f}; FAR={test_metrics.far:.4f}, FRR={test_metrics.frr:.4f}", flush=True)
        selected = min(methods, key=lambda method: summary["methods"][method]["calibration"]["eer"])
        summary["calibration_selected_method"] = selected
        summary["elapsed_seconds"] = time.perf_counter()-started
        write_json(output / "results.json", summary)
        with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["method", "calibration_eer", "test_eer", "test_auc", "test_far", "test_frr",
                             "test_rank1", "median_extraction_seconds", "threshold"])
            for method, result in summary["methods"].items():
                writer.writerow([method, result["calibration"]["eer"], result["test"]["eer"], result["test"]["auc"],
                                 result["test"]["far"], result["test"]["frr"], result["test_rank1"],
                                 result["extraction_seconds"]["median"], result["test"]["threshold"]])
        status.update(status="completed", elapsed_seconds=summary["elapsed_seconds"])
        write_json(output / "status.json", status)
        return summary
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      elapsed_seconds=time.perf_counter()-started)
        write_json(output / "status.json", status)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True, help="CSV with explicit identity/session/split/role metadata")
    result.add_argument("--output", required=True, help="New empty run directory")
    result.add_argument("--methods", default="gabor_only,gabor_glcm,hybrid")
    result.add_argument("--descriptor-dim", type=int, default=512)
    result.add_argument("--face-crop", choices=("haar", "already-cropped"), default="haar")
    result.add_argument("--image-size", type=int, default=224)
    result.add_argument("--allow-unknown-sessions", action="store_true")
    result.add_argument("--allow-preprocessing-exclusions", action="store_true")
    result.add_argument("--audit-only", action="store_true")
    result.add_argument("--target-far", type=float, default=0.01)
    result.add_argument("--bootstrap", type=int, default=1000)
    result.add_argument("--max-pairs", type=int, default=2_000_000)
    result.add_argument("--seed", type=int, default=42)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.image_size < 32 or args.descriptor_dim < 1 or args.bootstrap < 1 or args.max_pairs < 1:
        raise SystemExit("Image size >=32; descriptor dimension, bootstrap count and pair budget must be positive")
    if not 0 <= args.target_far < 1:
        raise SystemExit("--target-far must be in [0, 1)")
    try:
        run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"EXPERIMENT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

