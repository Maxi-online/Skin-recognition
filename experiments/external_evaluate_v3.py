"""Evaluate one frozen descriptor on the preregistered external corpus.

No fitting, candidate selection or threshold calibration is implemented here.
Running the CLI opens external images; importing it and synthetic unit tests do
not. All file hashes and the preprocessing contract are checked first.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scientific_metrics import bootstrap_identity_intervals, verification_from_scores
from src.utils.data_loader import read_image

LEGACY_ID = "v2_haar_exactly_one_min60_resize224_geometric4roi"
HAAR_SHA256 = "0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0"
DEFAULT_PROTOCOL = "research/data/georgia_tech/external_protocol_v1.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def project_path(path, root=ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def _check_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"Invalid SHA256 for {label}")
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {label}")


def load_freeze(freeze_path, root=ROOT):
    """Validate the entire manifest before a model or external image is opened."""
    freeze_path = Path(freeze_path).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {"model_path", "model_sha256", "preprocessing_id", "preprocessing_mode",
                "score_threshold", "protocol_sha256", "source_hashes", "timestamp"}
    if not required.issubset(freeze):
        raise ValueError("Incomplete freeze manifest: " + ", ".join(sorted(required - set(freeze))))
    if freeze["preprocessing_mode"] not in {"legacy_haar", "landmark_strict"}:
        raise ValueError("Unsupported frozen preprocessing mode")
    if not isinstance(freeze["preprocessing_id"], str) or not freeze["preprocessing_id"]:
        raise ValueError("A nonempty preprocessing_id is required")
    if isinstance(freeze["score_threshold"], bool) or not np.isfinite(float(freeze["score_threshold"])):
        raise ValueError("A finite frozen score_threshold is required")
    stamp = datetime.fromisoformat(freeze["timestamp"].replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("Freeze timestamp must include UTC timezone")
    if stamp > datetime.now(timezone.utc):
        raise ValueError("Freeze timestamp cannot be in the future")
    sources = freeze["source_hashes"]
    if not isinstance(sources, dict):
        raise ValueError("source_hashes must map project-relative paths to SHA256")
    required_sources = {"experiments/external_evaluate_v3.py", "experiments/development_v3.py",
                        "experiments/scientific_metrics.py", "src/utils/data_loader.py", "src/preprocessing_v3.py"}
    if freeze["preprocessing_mode"] == "legacy_haar":
        required_sources |= {"src/descriptors/local_patches.py", "src/descriptors/_image.py"}
    else:
        required_sources.add("src/preprocessing_v3.py")
    if not required_sources.issubset(sources):
        raise ValueError("Freeze must hash every required implementation source: " + ", ".join(sorted(required_sources - set(sources))))
    for path, expected in sources.items():
        _check_hash(project_path(path, root), expected, path)
    model_path = project_path(freeze["model_path"], root)
    protocol_path = project_path(freeze.get("protocol_path", DEFAULT_PROTOCOL), root)
    _check_hash(model_path, freeze["model_sha256"], "model")
    _check_hash(protocol_path, freeze["protocol_sha256"], "external protocol")
    if "search_operating_point_path" in freeze:
        _check_hash(project_path(freeze["search_operating_point_path"], root),
                    freeze["search_operating_point_sha256"], "secondary frozen search operating point")
    return freeze, model_path, protocol_path


def load_protocol(protocol_path, root=ROOT):
    """Read inventory metadata, never external image contents."""
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "georgia_tech_external_v1":
        raise ValueError("This evaluator requires georgia_tech_external_v1")
    info = protocol["dataset_inventory"]
    inventory_path = project_path(info["path"], root)
    _check_hash(inventory_path, info["sha256"], "inventory")
    acquisition = protocol.get("dataset_acquisition")
    if acquisition:
        _check_hash(project_path(acquisition["path"], root), acquisition["sha256"], "acquisition provenance")
    with inventory_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_path = {row["path"]: row for row in rows}
    if len(by_path) != len(rows):
        raise ValueError("Duplicate paths in inventory")
    closed, opened = protocol["closed_set"], protocol["open_set"]
    identities = sorted({row["subject_id"] for row in rows})
    if identities != closed["identities"] or len(identities) != 50 or len(rows) != 750:
        raise ValueError("Inventory does not match the frozen 50-identity, 750-image protocol")
    for subject in identities:
        subset = [row for row in rows if row["subject_id"] == subject]
        if sorted(row["image_id"] for row in subset) != [f"{i:02d}" for i in range(1, 16)]:
            raise ValueError("Every identity must retain image IDs 01 through 15")
    expected_gallery = [row["path"] for row in rows if row["image_id"] == "01"]
    expected_probes = [row["path"] for row in rows if row["image_id"] != "01"]
    if closed["gallery_paths"] != expected_gallery or closed["probe_paths"] != expected_probes:
        raise ValueError("Closed-set lists differ from the frozen image-ID rule")
    permutation = np.random.default_rng(42).permutation(identities).tolist()
    enrolled, unknown = permutation[:25], permutation[25:]
    if opened["enrolled_identities"] != enrolled or opened["unknown_identities"] != unknown:
        raise ValueError("Open-set identity assignment differs from the frozen seed42 permutation")
    expected_open = {
        "gallery_paths": [row["path"] for row in rows if row["subject_id"] in enrolled and row["image_id"] == "01"],
        "known_probe_paths": [row["path"] for row in rows if row["subject_id"] in enrolled and row["image_id"] != "01"],
        "unknown_probe_paths": [row["path"] for row in rows if row["subject_id"] in unknown and row["image_id"] != "01"],
    }
    if any(opened[key] != expected for key, expected in expected_open.items()):
        raise ValueError("Open-set image lists do not match the frozen assignment")
    if any(row.get("session_id") != "unknown" for row in rows):
        raise ValueError("Session metadata changed: this protocol specifies unknown sessions")
    return protocol, rows, by_path


class InputPreparer:
    def __init__(self, freeze, root=ROOT):
        self.mode = freeze["preprocessing_mode"]
        self.preprocessing_id = freeze["preprocessing_id"]
        if self.mode == "legacy_haar":
            if self.preprocessing_id != LEGACY_ID:
                raise ValueError("Legacy preprocessing contract mismatch")
            from src.descriptors.local_patches import LocalSkinPatchExtractor
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            _check_hash(cascade_path, HAAR_SHA256, "frozen legacy Haar cascade")
            self.detector = cv2.CascadeClassifier(str(cascade_path))
            if self.detector.empty():
                raise ValueError("Haar cascade failed to load")
            self.patch_extractor = LocalSkinPatchExtractor(use_dlib=False, target_size=64)
        else:
            from src.preprocessing_v3 import FacePreprocessorV3, PreprocessingConfigV3
            directory = freeze.get("preprocessing_model_dir")
            directory = project_path(directory, root) if directory else None
            config = PreprocessingConfigV3(**freeze.get("preprocessing_config", {}))
            self.processor = FacePreprocessorV3(model_dir=directory, config=config, providers=["CPUExecutionProvider"])
            if self.processor.preprocessing_id + ":strict" != self.preprocessing_id:
                raise ValueError("Landmark preprocessing/weights/config contract mismatch")

    def __call__(self, image):
        from src.preprocessing_v3 import QualityRejection
        if self.mode == "landmark_strict":
            from experiments.development_v3 import prepared_tuple_from_face
            result = self.processor.process(image)
            prepared, preprocessing_id = prepared_tuple_from_face(result, mode="strict")
            if preprocessing_id != self.preprocessing_id:
                raise ValueError("Prepared face preprocessing contract mismatch")
            return prepared, result.metadata
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise QualityRejection(["invalid_bgr_uint8_image"])
        boxes = self.detector.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                             scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(boxes) != 1:
            raise QualityRejection(["no_face" if len(boxes) == 0 else "multiple_faces"], {"detected_faces": len(boxes)})
        x, y, width, height = map(int, boxes[0])
        crop = image[y:y + height, x:x + width]
        interpolation = cv2.INTER_AREA if max(crop.shape[:2]) > 224 else cv2.INTER_LINEAR
        aligned = cv2.resize(crop, (224, 224), interpolation=interpolation)
        patches = np.stack([cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR) for patch in self.patch_extractor.extract_patches(aligned).values()])
        return (aligned, patches, np.ones(patches.shape[:3], np.uint8), np.ones(aligned.shape[:2], np.uint8)), {
            "bbox_xywh": [x, y, width, height], "detected_faces": 1, "roi_semantics": "fixed_geometric_regions_not_verified_skin_only"}


def score_matrix(features, gallery_paths, probe_paths):
    """Invalid comparisons are explicit mask entries, never fabricated scores."""
    gallery_valid = np.asarray([path in features for path in gallery_paths], dtype=bool)
    probe_valid = np.asarray([path in features for path in probe_paths], dtype=bool)
    scores = np.zeros((len(probe_paths), len(gallery_paths)), dtype=np.float64)
    valid = probe_valid[:, None] & gallery_valid[None, :]
    if gallery_valid.any() and probe_valid.any():
        gallery = np.stack([features[path] for path in gallery_paths if path in features])
        probes = np.stack([features[path] for path in probe_paths if path in features])
        scores[np.ix_(probe_valid, gallery_valid)] = -cdist(probes, gallery, metric="euclidean")
    return scores, valid, gallery_valid, probe_valid


def conservative_ranks(scores, valid, gallery_ids, probe_ids):
    """Return zero for missing genuine comparisons; exact ties count against truth."""
    scores, valid = np.asarray(scores), np.asarray(valid, dtype=bool)
    gallery_ids, probe_ids = np.asarray(gallery_ids), np.asarray(probe_ids)
    if scores.ndim != 2 or scores.shape != valid.shape or scores.shape != (len(probe_ids), len(gallery_ids)):
        raise ValueError("Score/mask/identity dimensions differ")
    if not np.isfinite(scores[valid]).all() or len(np.unique(gallery_ids)) != len(gallery_ids):
        raise ValueError("Valid scores must be finite and gallery identities unique")
    ranks = np.zeros(len(probe_ids), dtype=np.int32)
    for i, identity in enumerate(probe_ids):
        genuine = (gallery_ids == identity) & valid[i]
        if genuine.any():
            truth = scores[i, genuine][0]
            ranks[i] = 1 + np.count_nonzero(valid[i] & (gallery_ids != identity) & (scores[i] >= truth))
    return ranks


def open_set_decisions(scores, valid, gallery_ids, probe_ids, enrolled_ids, threshold):
    """Operational FPIR/FNIR inputs include every intended probe and missing template."""
    gallery_ids, probe_ids = np.asarray(gallery_ids), np.asarray(probe_ids)
    scores, valid = np.asarray(scores), np.asarray(valid, dtype=bool)
    if scores.shape != valid.shape or scores.shape != (len(probe_ids), len(gallery_ids)):
        raise ValueError("Score/mask/identity dimensions differ")
    if not np.isfinite(threshold) or not np.isfinite(scores[valid]).all():
        raise ValueError("Finite scores and threshold required")
    known = np.isin(probe_ids, np.asarray(enrolled_ids))
    accepted = np.zeros(len(probe_ids), dtype=bool)
    correct = np.zeros(len(probe_ids), dtype=bool)
    winners = np.full(len(probe_ids), "", dtype=f"<U{max([1] + [len(str(x)) for x in gallery_ids])}")
    reasons = []
    for i, identity in enumerate(probe_ids):
        available = np.flatnonzero(valid[i])
        if not len(available):
            reasons.append("no_available_comparisons")
            continue
        best = float(scores[i, available].max())
        best_indices = available[scores[i, available] == best]
        if len(best_indices) != 1:
            reasons.append("tied_maximum")
            continue
        winner = gallery_ids[best_indices[0]]
        winners[i] = winner
        if best < threshold:
            reasons.append("below_frozen_threshold")
            continue
        accepted[i] = True
        correct[i] = bool(known[i] and winner == identity)
        reasons.append("correct_accept" if correct[i] else "wrong_identity_accept" if known[i] else "unknown_false_identification")
    return {"known": known, "accepted": accepted, "correct": correct, "winner": winners, "reason": reasons}


def _mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def rate_identity_bootstrap(values, identity_ids, n_bootstrap=1000, seed=42):
    """Probe-identity percentile CI conditional on the fixed enrolled gallery."""
    values, identity_ids = np.asarray(values, dtype=np.float64), np.asarray(identity_ids)
    if values.ndim != 1 or values.shape != identity_ids.shape or not np.isfinite(values).all():
        raise ValueError("Finite per-probe outcomes and equally sized identity IDs required")
    identities, inverse = np.unique(identity_ids, return_inverse=True)
    if len(identities) < 2:
        return {"status": "unavailable", "reason": "fewer_than_two_probe_identities"}
    if n_bootstrap < 1:
        raise ValueError("Bootstrap replicates must be positive")
    numerator = np.bincount(inverse, weights=values)
    denominator = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        draw = rng.integers(len(identities), size=len(identities))
        samples.append(float(numerator[draw].sum() / denominator[draw].sum()))
    return {"status": "ok", "method": "probe_identity_resampling_percentile_95",
            "conditional_on": "frozen_model_threshold_and_enrolled_gallery", "replicates": n_bootstrap,
            "seed": seed, "identity_count": len(identities), "interval": np.percentile(samples, [2.5, 97.5]).tolist()}


def load_search_operating_point(freeze, root=ROOT):
    if "search_operating_point_path" not in freeze:
        return None
    path = project_path(freeze["search_operating_point_path"], root)
    _check_hash(path, freeze["search_operating_point_sha256"], "secondary frozen search operating point")
    point = json.loads(path.read_text(encoding="utf-8"))
    _check_hash(project_path(point["source_freeze_path"], root), point["source_freeze_sha256"], "secondary point's original model freeze")
    for key in ("model_sha256", "preprocessing_id", "feature_space_id"):
        if point[key] != freeze[key]:
            raise ValueError("Secondary operating point belongs to another model/preprocessing: " + key)
    if point["score_definition"] != "negative_euclidean_distance" or not np.isfinite(float(point["score_threshold"])):
        raise ValueError("Invalid secondary operating point score contract")
    return point


def load_prepared_cache(manifest_path, rows, preprocessing_id):
    """Reuse only fully accounted, hashed preprocessing from the same image list."""
    manifest_path = Path(manifest_path).resolve()
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    cached = {record["path"]: record for record in records}
    if len(cached) != len(records) or set(cached) != {row["path"] for row in rows}:
        raise ValueError("Prepared cache must account for every frozen input exactly once")
    for row in rows:
        record = cached[row["path"]]
        if record.get("preprocessing_id") != preprocessing_id or record["sha256"] != row["sha256"]:
            raise ValueError("Prepared cache preprocessing or source hash mismatch")
        preprocessing_status = record.get("preprocessing_status", record["status"])
        if preprocessing_status not in {"accepted", "rejected"}:
            raise ValueError("Prepared cache contains an incomplete input")
        record = {**record, "preprocessing_status": preprocessing_status}
        if preprocessing_status == "accepted":
            cache_path = Path(record["cache_path"])
            if not cache_path.is_absolute():
                cache_path = manifest_path.parent / cache_path
            _check_hash(cache_path, record["cache_sha256"], "prepared cache")
            record = {**record, "resolved_cache_path": str(cache_path)}
        cached[row["path"]] = record
    return cached


def summarize_protocol(protocol, by_path, features, threshold, n_bootstrap=1000, seed=42):
    closed, opened = protocol["closed_set"], protocol["open_set"]
    galleries, probes = closed["gallery_paths"], closed["probe_paths"]
    gallery_ids = np.asarray([by_path[path]["subject_id"] for path in galleries])
    probe_ids = np.asarray([by_path[path]["subject_id"] for path in probes])
    scores, valid, gv, pv = score_matrix(features, galleries, probes)
    y = (probe_ids[:, None] == gallery_ids[None, :]).astype(np.int8)
    ranks = conservative_ranks(scores, valid, gallery_ids, probe_ids)
    max_rank = len(galleries)
    cmc = [float(np.mean((ranks > 0) & (ranks <= rank))) for rank in range(1, max_rank + 1)]
    eligible = ranks > 0
    conditional_cmc = [float(np.mean(ranks[eligible] <= rank)) for rank in range(1, max_rank + 1)] if eligible.any() else []
    pair_y, pair_scores = y[valid], scores[valid]
    pair_probe = np.broadcast_to(probe_ids[:, None], scores.shape)[valid]
    pair_gallery = np.broadcast_to(gallery_ids[None, :], scores.shape)[valid]
    counts = {"genuine_pairs": int(np.count_nonzero(pair_y == 1)), "impostor_pairs": int(np.count_nonzero(pair_y == 0))}
    verification = {"status": "unavailable", "reason": "both_pair_classes_required", **counts}
    if counts["genuine_pairs"] and counts["impostor_pairs"]:
        verification = {"status": "ok", **asdict(verification_from_scores(pair_y, pair_scores, threshold, threshold_source="frozen_FEI_calibration"))}
        verification["tar"] = 1 - verification["frr"]
        verification["confidence_intervals"] = bootstrap_identity_intervals(pair_y, pair_scores, pair_probe, pair_gallery, threshold, n_bootstrap, seed)
        fpr, tpr, roc_thresholds = roc_curve(pair_y, pair_scores, drop_intermediate=False)
    else:
        fpr, tpr, roc_thresholds = np.array([]), np.array([]), np.array([])
    open_gallery_indices = np.asarray([galleries.index(path) for path in opened["gallery_paths"]])
    open_probes = opened["known_probe_paths"] + opened["unknown_probe_paths"]
    open_probe_indices = np.asarray([probes.index(path) for path in open_probes])
    open_scores = scores[np.ix_(open_probe_indices, open_gallery_indices)]
    open_valid = valid[np.ix_(open_probe_indices, open_gallery_indices)]
    open_ids = probe_ids[open_probe_indices]
    decisions = open_set_decisions(open_scores, open_valid, gallery_ids[open_gallery_indices], open_ids, opened["enrolled_identities"], threshold)
    known = decisions["known"]
    fp = decisions["accepted"][~known]
    fn = ~decisions["correct"][known]
    # Eligible known probes must themselves succeed and have the true template.
    known_genuine_valid = np.any(open_valid & (open_ids[:, None] == gallery_ids[open_gallery_indices][None, :]), axis=1)
    unknown_valid = pv[open_probe_indices] & np.any(gv[open_gallery_indices]) & ~known
    report = {
        "score_definition": "negative_Euclidean_distance_L2_normalized_embeddings; larger_is_more_similar",
        "decision_rule": "accept_if_score_greater_than_or_equal_to_frozen_threshold; open_set_tied_maxima_rejected",
        "score_threshold": float(threshold),
        "closed_set": {"intended_gallery_images": len(galleries), "successful_gallery_images": int(gv.sum()),
                       "intended_probe_images": len(probes), "successful_probe_images": int(pv.sum()),
                       "genuine_template_available_probe_count": int(eligible.sum()),
                       "rank1_operational": cmc[0], "rank5_operational": cmc[min(4, max_rank - 1)],
                       "cmc_operational": cmc, "rank1_conditional": conditional_cmc[0] if conditional_cmc else None,
                       "cmc_conditional": conditional_cmc,
                       "operational_denominator": len(probes), "conditional_denominator": int(eligible.sum()),
                       "rank1_operational_ci": rate_identity_bootstrap((ranks > 0) & (ranks <= 1), probe_ids, n_bootstrap, seed),
                       "rank1_conditional_ci": rate_identity_bootstrap(ranks[eligible] <= 1, probe_ids[eligible], n_bootstrap, seed)},
        "verification_conditional": verification,
        "open_set": {"intended_enrolled_identities": len(opened["enrolled_identities"]),
                     "successful_enrolled_templates": int(gv[open_gallery_indices].sum()),
                     "known_probe_count": int(known.sum()), "unknown_probe_count": int((~known).sum()),
                     "known_probe_success_count": int(pv[open_probe_indices][known].sum()),
                     "unknown_probe_success_count": int(pv[open_probe_indices][~known].sum()),
                     "fpir_operational": _mean_or_none(fp), "fnir_operational": _mean_or_none(fn),
                     "false_identifications": int(fp.sum()), "known_identification_failures": int(fn.sum()),
                     "fpir_conditional": _mean_or_none(decisions["accepted"][unknown_valid]),
                     "fnir_conditional": _mean_or_none(~decisions["correct"][known_genuine_valid]),
                     "fpir_conditional_denominator": int(unknown_valid.sum()),
                     "fnir_conditional_denominator": int(known_genuine_valid.sum()),
                     "fpir_operational_ci": rate_identity_bootstrap(fp, open_ids[~known], n_bootstrap, seed),
                     "fnir_operational_ci": rate_identity_bootstrap(fn, open_ids[known], n_bootstrap, seed)},
        "limitations": ["All session IDs are unknown; no cross-session guarantee.",
                        "Conditional pair metrics omit failed processing and must be read with coverage.",
                        "Missing gallery templates reduce the effective distractor gallery; coverage and operational denominators are explicit.",
                        "Confidence intervals condition on the frozen method and threshold; they do not include development-selection uncertainty.",
                        "Probe-rate intervals also condition on the enrolled gallery; zero-error percentile intervals do not establish a population upper error bound.",
                        "Transferred verification threshold does not guarantee the same open-set FPIR.",
                        "No resolved-pore, measured-depth, real-person dataset non-overlap or commercial-readiness claim follows from this evaluation."],
    }
    numeric = {"scores": scores, "valid_comparisons": valid, "pair_labels": y, "ranks": ranks,
               "gallery_valid": gv, "probe_valid": pv, "roc_fpr": fpr, "roc_tpr": tpr,
               "roc_thresholds": roc_thresholds, "open_probe_indices": open_probe_indices,
               "open_gallery_indices": open_gallery_indices, "open_known": known,
               "open_accepted": decisions["accepted"], "open_correct": decisions["correct"]}
    return report, numeric, decisions


def save_artifacts(output, protocol, by_path, features, report, numeric, decisions):
    """No portraits are exported. NPZ contains numeric arrays with no pickle."""
    output = Path(output)
    galleries, probes = protocol["closed_set"]["gallery_paths"], protocol["closed_set"]["probe_paths"]
    valid_paths = sorted(features)
    vectors = np.stack([features[path] for path in valid_paths]) if valid_paths else np.empty((0, 0), np.float32)
    np.savez_compressed(output / "embeddings.npz", embeddings=vectors, sample_index=np.arange(len(valid_paths), dtype=np.int32))
    write_json(output / "embedding_index.json", [{"sample_index": i, "path": path, "subject_id": by_path[path]["subject_id"]} for i, path in enumerate(valid_paths)])
    np.savez_compressed(output / "scores.npz", **numeric)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["probe_path", "gallery_path", "probe_identity", "gallery_identity", "genuine", "comparison_available", "score_negative_L2"])
        for i, probe in enumerate(probes):
            for j, gallery in enumerate(galleries):
                available = bool(numeric["valid_comparisons"][i, j])
                writer.writerow([probe, gallery, by_path[probe]["subject_id"], by_path[gallery]["subject_id"],
                                 int(numeric["pair_labels"][i, j]), int(available), float(numeric["scores"][i, j]) if available else ""])
    open_paths = protocol["open_set"]["known_probe_paths"] + protocol["open_set"]["unknown_probe_paths"]
    with (output / "decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "probe_path", "identity", "known", "conservative_rank", "accepted", "correct", "winner", "reason"])
        for path, rank in zip(probes, numeric["ranks"]):
            writer.writerow(["closed_set", path, by_path[path]["subject_id"], 1, int(rank) if rank else "", "", int(rank == 1), "", "ranked" if rank else "genuine_comparison_unavailable"])
        for i, path in enumerate(open_paths):
            writer.writerow(["open_set", path, by_path[path]["subject_id"], int(decisions["known"][i]), "",
                             int(decisions["accepted"][i]), int(decisions["correct"][i]), decisions["winner"][i], decisions["reason"][i]])
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    if numeric["roc_fpr"].size:
        axes[0].plot(numeric["roc_fpr"], numeric["roc_tpr"])
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="FAR", ylabel="TAR", title="Conditional verification ROC", xlim=(0, 1), ylim=(0, 1))
    cmc = report["closed_set"]["cmc_operational"]
    axes[1].plot(range(1, len(cmc) + 1), cmc, label="All intended probes")
    conditional = report["closed_set"]["cmc_conditional"]
    if conditional:
        axes[1].plot(range(1, len(conditional) + 1), conditional, label="Valid genuine comparison")
    axes[1].set(xlabel="Rank", ylabel="Identification rate", title="Closed-set CMC", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    opened = report["open_set"]
    axes[2].bar(["FPIR", "FNIR"], [opened["fpir_operational"] or 0, opened["fnir_operational"] or 0])
    axes[2].set(ylabel="Operational error rate", title="Open-set: frozen FEI threshold", ylim=(0, 1))
    fig.savefig(output / "external_evaluation.png", dpi=200)
    fig.savefig(output / "external_evaluation.pdf")
    plt.close(fig)


def run(freeze_path, output, root=ROOT, prepared_cache_manifest=None):
    # All metadata/model/source validation precedes external image reading.
    freeze, model_path, protocol_path = load_freeze(freeze_path, root)
    protocol, rows, by_path = load_protocol(protocol_path, root)
    from experiments.development_v3 import ClassicalTextureDescriptorV3
    from src.preprocessing_v3 import QualityRejection
    descriptor = ClassicalTextureDescriptorV3.load(model_path)
    if descriptor.preprocessing_id != freeze["preprocessing_id"]:
        raise ValueError("Frozen model and preprocessing contract differ")
    if "feature_space_id" in freeze and descriptor.feature_space_id != freeze["feature_space_id"]:
        raise ValueError("Frozen feature-space identifier does not match model")
    search_point = load_search_operating_point(freeze, root)
    cached = load_prepared_cache(prepared_cache_manifest, rows, freeze["preprocessing_id"]) if prepared_cache_manifest else None
    preparer = None if cached is not None else InputPreparer(freeze, root)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output must be a new empty directory; do not overwrite a previous external run")
    output.mkdir(parents=True, exist_ok=True)
    (output / "prepared_cache").mkdir()
    write_json(output / "freeze_used.json", freeze)
    write_json(output / "protocol_used.json", protocol)
    start, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    records, features = [], {}
    write_json(output / "status.json", {"status": "running", "started_utc": started_utc, "external_access_started": False})
    try:
        for index, row in enumerate(rows):
            path = project_path(row["path"], root)
            record = {**row, "status": "pending", "preprocessing_status": "pending", "rejection_codes": [], "preprocessing_id": freeze["preprocessing_id"]}
            # Tampering/missing source aborts instead of being laundered as a quality rejection.
            _check_hash(path, row["sha256"], row["path"])
            cached_record = cached.get(row["path"]) if cached is not None else None
            if cached_record and cached_record["preprocessing_status"] == "rejected":
                record.update(status="rejected", preprocessing_status="rejected", rejection_codes=cached_record["rejection_codes"],
                              quality=cached_record.get("quality", {}), reused_preprocessing_rejection=True)
                records.append(record)
                continue
            try:
                if cached_record:
                    cache_path = Path(cached_record["resolved_cache_path"])
                    with np.load(cache_path, allow_pickle=False) as archive:
                        prepared = tuple(archive[name].copy() for name in ("aligned_bgr", "patches_bgr", "patch_masks", "skin_mask"))
                    quality = cached_record.get("quality", {})
                    record.update(cache_path=str(cache_path), cache_sha256=cached_record["cache_sha256"], reused_preprocessing=True)
                else:
                    try:
                        image = read_image(path)
                    except (ValueError, OSError) as error:
                        raise QualityRejection(["image_decode_failure"], {"error": str(error)}) from error
                    prepared, quality = preparer(image)
                    cache_path = output / "prepared_cache" / f"{index:04d}.npz"
                    np.savez_compressed(cache_path, **dict(zip(("aligned_bgr", "patches_bgr", "patch_masks", "skin_mask"), prepared)))
                    record.update(cache_path=cache_path.relative_to(output).as_posix(), cache_sha256=sha256(cache_path))
                record.update(preprocessing_status="accepted", quality=quality)
                vector = np.asarray(descriptor.extract_features(prepared, freeze["preprocessing_id"]), dtype=np.float32)
                if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
                    raise QualityRejection(["invalid_or_zero_embedding"])
                features[row["path"]] = vector
                record.update(status="accepted", quality=quality)
            except QualityRejection as error:
                record.update(status="rejected", rejection_codes=error.codes, quality=error.metadata)
                if record["preprocessing_status"] != "accepted":
                    record["preprocessing_status"] = "rejected"
            records.append(record)
            if (index + 1) % 25 == 0:
                write_json(output / "image_quality.json", records)
                write_json(output / "status.json", {"status": "running", "started_utc": started_utc, "external_access_started": True,
                                                      "processed": len(records), "accepted": len(features)})
                print(f"External evaluation: {index + 1}/{len(rows)} inputs; accepted={len(features)}", flush=True)
        report, numeric, decisions = summarize_protocol(protocol, by_path, features, float(freeze["score_threshold"]))
        if search_point:
            secondary, secondary_numeric, secondary_decisions = summarize_protocol(protocol, by_path, features, float(search_point["score_threshold"]))
            report["additional_operating_points"] = {
                "conservative_search": {"source_path": freeze["search_operating_point_path"],
                                        "source_sha256": freeze["search_operating_point_sha256"],
                                        "score_threshold": float(search_point["score_threshold"]),
                                        "verification_conditional": secondary["verification_conditional"],
                                        "open_set": secondary["open_set"]}}
            secondary_dir = output / "conservative_search"
            secondary_dir.mkdir()
            save_artifacts(secondary_dir, protocol, by_path, features, secondary, secondary_numeric, secondary_decisions)
            write_json(secondary_dir / "results.json", secondary)
        report.update(protocol_id=protocol["protocol_id"], protocol_sha256=freeze["protocol_sha256"],
                      freeze_sha256=sha256(freeze_path), model_sha256=freeze["model_sha256"],
                      feature_space_id=descriptor.feature_space_id, preprocessing_id=freeze["preprocessing_id"],
                      configuration=descriptor.configuration, started_utc=started_utc,
                      elapsed_seconds=time.perf_counter() - start,
                      coverage={"intended_images": len(rows), "accepted_images": len(features),
                                "image_success_rate": len(features) / len(rows),
                                "identities_with_any_success": len({by_path[path]["subject_id"] for path in features}),
                                "identities_with_all_15_images": sum(all(row["path"] in features for row in rows if row["subject_id"] == identity)
                                                                      for identity in protocol["closed_set"]["identities"]),
                                "rejection_code_counts": dict(Counter(code for record in records for code in record["rejection_codes"]))})
        if prepared_cache_manifest:
            report["prepared_cache_reuse"] = {"manifest_path": str(Path(prepared_cache_manifest).resolve()),
                                              "manifest_sha256": sha256(prepared_cache_manifest),
                                              "contract": "same frozen preprocessing and source-image hashes; no repeated detector evaluation"}
        # Detect a changed manifest, implementation or weights during the run.
        current_freeze, _, _ = load_freeze(freeze_path, root)
        if current_freeze != freeze:
            raise ValueError("Freeze manifest changed during external evaluation")
        save_artifacts(output, protocol, by_path, features, report, numeric, decisions)
        write_json(output / "image_quality.json", records)
        write_json(output / "results.json", report)
        write_json(output / "artifact_sha256.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "status.json"})
        write_json(output / "status.json", {"status": "completed", "external_access_started": True,
                                             "started_utc": started_utc, "elapsed_seconds": time.perf_counter() - start})
        return report
    except Exception as error:
        write_json(output / "image_quality.json", records)
        write_json(output / "status.json", {"status": "failed", "external_access_started": True,
                                             "processed": len(records), "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, help="Pre-external frozen model/threshold/source manifest JSON")
    parser.add_argument("--output", required=True, help="New empty result directory")
    parser.add_argument("--prepared-cache-manifest", help="Hashed image_quality.json from the same frozen preprocessing, for a preregistered comparison")
    args = parser.parse_args()
    result = run(args.freeze, args.output, prepared_cache_manifest=args.prepared_cache_manifest)
    print(json.dumps({"status": "completed", "rank1_operational": result["closed_set"]["rank1_operational"],
                      "FPIR": result["open_set"]["fpir_operational"], "FNIR": result["open_set"]["fnir_operational"]}, indent=2))
