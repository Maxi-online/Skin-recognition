"""Freeze a conservative 1:N operating point using FEI development scores only.

This supplements, and never overwrites, the primary pairwise freeze. Zero
empirical development false matches does not imply a population FPIR bound.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.development_v3 import ClassicalTextureDescriptorV3
from experiments.scientific_metrics import cross_session_pairs, compute_far_frr


def conservative_cutoff(genuine, scores):
    genuine = np.asarray(genuine, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if genuine.shape != scores.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration pair arrays")
    if not np.any(genuine) or not np.any(~genuine):
        raise ValueError("Both genuine and impostor calibration pairs are required")
    return float(np.nextafter(np.max(scores[~genuine]), np.inf))


def open_set_metrics(score_matrix, gallery_ids, probe_ids, threshold, enrolled_ids):
    scores = np.asarray(score_matrix, dtype=np.float64)
    gallery_ids, probe_ids = np.asarray(gallery_ids), np.asarray(probe_ids)
    enrolled_ids = set(enrolled_ids)
    select_gallery = np.asarray([subject in enrolled_ids for subject in gallery_ids])
    if not select_gallery.any() or scores.shape != (len(probe_ids), len(gallery_ids)):
        raise ValueError("Invalid open-set gallery")
    reduced = scores[:, select_gallery]
    selected_ids = gallery_ids[select_gallery]
    maximum = reduced.max(axis=1)
    known = np.asarray([subject in enrolled_ids for subject in probe_ids])
    if not np.any(known) or not np.any(~known):
        raise ValueError("Both known and unknown probes are required")
    correct_accepted = []
    for i, identity in enumerate(probe_ids):
        if not known[i]:
            continue
        true = reduced[i, selected_ids == identity]
        if not len(true):
            raise ValueError("Enrolled identity has no gallery image")
        # Tied competing identity counts as an identification failure.
        other = reduced[i, selected_ids != identity]
        correct_accepted.append(bool(true.max() >= threshold and np.all(true.max() > other)))
    false_identifications = int(np.sum(maximum[~known] >= threshold))
    return {
        "known_probe_count": int(known.sum()), "unknown_probe_count": int((~known).sum()),
        "enrolled_identity_count": len(enrolled_ids),
        "false_positive_identifications": false_identifications,
        "false_negative_identifications": int(len(correct_accepted)-sum(correct_accepted)),
        "fpir": false_identifications/int((~known).sum()),
        "fnir": 1-float(np.mean(correct_accepted)),
        "correct_accept_identification_rate": float(np.mean(correct_accepted)),
        "tie_policy": "a_tied_competing_identity_is_a_failure",
    }


def create_point(freeze_path, output_path, seed=42):
    freeze_path, output_path = Path(freeze_path).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise ValueError("Refuse to overwrite an operating-point freeze")
    freeze_bytes = freeze_path.read_bytes()
    freeze = json.loads(freeze_bytes)
    model_path = Path(freeze["model_path"])
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != freeze["model_sha256"]:
        raise ValueError("Frozen model checksum mismatch")
    model = ClassicalTextureDescriptorV3.load(model_path)
    if model.feature_space_id != freeze["feature_space_id"]:
        raise ValueError("Frozen feature-space mismatch")
    run_dir = freeze_path.parent
    results = json.loads((run_dir/"results.json").read_text(encoding="utf-8"))
    candidate = next(item for item in results["results"] if item["id"] == freeze["candidate_id"])
    raw_path = run_dir/"features"/(candidate["feature_id"]+".npz")
    with np.load(raw_path, allow_pickle=False) as archive:
        select = archive["splits"] == "calibration"
        # Only FEI calibration features are loaded for the score calculation.
        raw = archive["features"][select]
        identities = archive["subject_ids"][select]
        paths = archive["paths"][select]
    role_by_path = {}
    import csv
    source_manifest = ROOT/"research/results/fei_frontal_seed42_conditional/evaluated_manifest.csv"
    with source_manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "calibration":
                role_by_path[str(Path(row["path"]).resolve())] = row["role"]
    roles = np.asarray([role_by_path[str(Path(path).resolve())] for path in paths])
    features = model.transform_raw(raw)
    g, p = roles == "gallery", roles == "probe"
    gallery_ids, probe_ids = identities[g], identities[p]
    y, scores, pair_probe_ids, pair_gallery_ids, matrix = cross_session_pairs(
        features[g], gallery_ids, features[p], probe_ids)
    cutoff = conservative_cutoff(y, scores)
    far, frr = compute_far_frr(y, scores, cutoff)
    identities_sorted = np.unique(identities)
    order = np.random.default_rng(seed).permutation(identities_sorted)
    enrolled, unknown = order[:len(order)//2], order[len(order)//2:]
    score_digest = hashlib.sha256()
    score_digest.update(np.asarray(matrix, dtype="<f8").tobytes())
    score_digest.update(json.dumps({"shape": list(matrix.shape), "gallery_ids": gallery_ids.tolist(),
                                   "probe_ids": probe_ids.tolist()}, sort_keys=True).encode())
    score_artifact = output_path.with_name("search_calibration_scores.npz")
    if score_artifact.exists():
        raise ValueError("Refuse to overwrite saved calibration scores")
    np.savez_compressed(score_artifact, scores=matrix, gallery_ids=gallery_ids, probe_ids=probe_ids)
    report = {
        "schema_version": "skin-recognition-v3-search-operating-point-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_freeze_path": str(freeze_path),
        "source_freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "model_sha256": freeze["model_sha256"], "feature_space_id": model.feature_space_id,
        "preprocessing_id": model.preprocessing_id,
        "score_definition": "negative_euclidean_distance",
        "score_threshold": cutoff,
        "selection_rule": "nextafter(maximum_of_all_FEI_calibration_impostor_scores, positive_infinity)",
        "maximum_calibration_impostor_score": float(np.max(scores[y == 0])),
        "decision_rule": "accept if score >= score_threshold",
        "calibration_dataset_id": "FEI_frontal_seed42_retained_calibration_36_identities",
        "calibration_identity_count": len(identities_sorted),
        "calibration_image_count": len(identities),
        "calibration_genuine_pairs": int(np.sum(y == 1)),
        "calibration_impostor_pairs": int(np.sum(y == 0)),
        "calibration_far": far, "calibration_frr": frr,
        "calibration_score_hash": score_digest.hexdigest(),
        "calibration_score_hash_definition": "float64_little_endian_score_matrix_C_order_then_sorted_JSON_shape_and_ordered_identity_arrays",
        "calibration_scores_path": str(score_artifact),
        "calibration_scores_file_sha256": hashlib.sha256(score_artifact.read_bytes()).hexdigest(),
        "raw_feature_artifact_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "open_set_development_protocol": {
            "seed": seed, "enrolled_identity_ids": enrolled.tolist(), "unknown_identity_ids": unknown.tolist(),
            "gallery": "one neutral frontal image per enrolled identity",
            "probes": "one smiling frontal image per calibration identity",
            "secondary_point": open_set_metrics(matrix, gallery_ids, probe_ids, cutoff, enrolled),
            "primary_pairwise_point": open_set_metrics(matrix, gallery_ids, probe_ids, freeze["score_threshold"], enrolled),
            "primary_pairwise_score_threshold": freeze["score_threshold"],
        },
        "external_data_used": False, "v2_test_data_used": False, "model_changed": False,
        "primary_freeze_changed": False,
        "limitations": [
            "The open-set development partition reuses the data that define the cutoff and is descriptive.",
            "Zero empirical false matches is not a population FPIR guarantee or a statistical upper bound.",
            "The primary pairwise threshold remains unchanged and must be reported separately.",
            "Both operating points must be evaluated unchanged on the untouched external protocol.",
        ],
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = create_point(args.freeze, args.output, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))

