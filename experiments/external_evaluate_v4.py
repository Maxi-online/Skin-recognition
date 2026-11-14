"""Score frozen v4 embedding bundles; never decode images or fit model weights.

CLI: --freeze study.json --bundle method_bundle.json --dataset lfw|orl|fei_holdout --output DIR
Bundles identify their frozen method and hash the embeddings and quality log.
The sole fitted quantity is the separately labelled LFW 9-fold threshold used
for the secondary conventional 10-fold threshold-CV protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.external_evaluate_v3 import project_path, sha256, summarize_protocol, write_json
from experiments.scientific_metrics import verification_from_scores


def check_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Invalid SHA256: " + label)
    if sha256(path) != expected:
        raise ValueError("SHA256 mismatch: " + label)


def utc(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Timestamp needs a timezone")
    return stamp.astimezone(timezone.utc)


def canonical(path, root=ROOT):
    return project_path(path, root).as_posix()


def validate_freeze_bundle(freeze_path, bundle_path, dataset, root=ROOT):
    """Verify provenance before loading embeddings. No external images are read."""
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "study-v4.1":
        raise ValueError("Expected study-v4.1 freeze")
    check_hash(freeze_path, bundle["freeze_sha256"], "model study freeze")
    if utc(bundle["extraction_started_utc"]) < utc(freeze["timestamp"]):
        raise ValueError("External extraction predates model freeze")
    if utc(freeze["timestamp"]) > datetime.now(timezone.utc):
        raise ValueError("Freeze timestamp is in the future")
    required_sources = {"experiments/external_evaluate_v4.py", "experiments/external_evaluate_v3.py", "experiments/scientific_metrics.py"}
    if not required_sources.issubset(freeze["source_hashes"]):
        raise ValueError("Freeze omits required evaluator source hashes")
    for path, digest in freeze["source_hashes"].items():
        check_hash(project_path(path, root), digest, path)
    if not freeze.get("model_artifacts"):
        raise ValueError("Freeze needs hashed model artifacts")
    for path, digest in freeze["model_artifacts"].items():
        check_hash(project_path(path, root), digest, path)
    methods = [method for method in freeze["methods"] if method["method_id"] == bundle["method_id"]]
    if len(methods) != 1:
        raise ValueError("Bundle must identify exactly one frozen method")
    method = methods[0]
    for name in ("feature_space_id", "preprocessing_id", "scope"):
        if bundle[name] != method[name]:
            raise ValueError("Bundle method contract mismatch: " + name)
    for name in ("score_threshold", "search_score_threshold"):
        if name in method and (isinstance(method[name], bool) or not np.isfinite(float(method[name]))):
            raise ValueError("Frozen thresholds must be finite")
    if "score_threshold" not in method:
        raise ValueError("Missing primary frozen score threshold")
    if method.get("score_definition", "negative_euclidean_distance") != "negative_euclidean_distance":
        raise ValueError("Only negative Euclidean scores are supported")
    info = freeze["protocols"][dataset]
    protocol_path = project_path(info["path"], root)
    check_hash(protocol_path, info["sha256"], "frozen protocol")
    if bundle["protocol_sha256"] != info["sha256"]:
        raise ValueError("Bundle used another external protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != dataset + "_external_v4_1" or bundle["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Wrong external protocol ID")
    for field in ("embeddings", "quality"):
        check_hash(project_path(bundle[field + "_path"], root), bundle[field + "_sha256"], field + " artifact")
    return freeze, bundle, method, protocol


def load_protocol_data(protocol, dataset, root=ROOT):
    if dataset not in {"lfw", "orl", "fei_holdout"}:
        raise ValueError("Unsupported protocol dataset")
    inventory_path = project_path(protocol["inventory_path"], root)
    check_hash(inventory_path, protocol["inventory_sha256"], "dataset inventory")
    with inventory_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    inventory = {canonical(row["path"], root): row for row in rows}
    if len(inventory) != len(rows):
        raise ValueError("Duplicate inventory paths")
    pairs = None
    if dataset == "lfw":
        manifest = project_path(protocol["pair_manifest_path"], root)
        check_hash(manifest, protocol["pair_manifest_sha256"], "official pair manifest")
        official = project_path(protocol["official_pairs_path"], root)
        check_hash(official, protocol["official_pairs_sha256"], "official pairs.txt")
        with manifest.open(encoding="utf-8", newline="") as stream:
            pairs = list(csv.DictReader(stream))
        original = official.read_text(encoding="utf-8").splitlines()
        if [int(x) for x in original[0].split()] != [10, 300] or len(pairs) != 6000 or len(original) != 6001:
            raise ValueError("Original LFW6000 pair protocol is required")
        for index, (pair, line) in enumerate(zip(pairs, original[1:])):
            fields = line.split()
            same = len(fields) == 3
            if len(fields) not in (3, 4):
                raise ValueError("Invalid official pair line")
            names = (fields[0], fields[0] if same else fields[2])
            numbers = (int(fields[1]), int(fields[2] if same else fields[3]))
            if (int(pair["pair_id"]) != index or int(pair["fold_id"]) != index // 600 + 1 or
                    int(pair["is_same"]) != int(same) or same != (index % 600 < 300)):
                raise ValueError("Pair order, fold or label differs from official pairs.txt")
            for side, name, number in zip(("a", "b"), names, numbers):
                path = canonical(pair["path_" + side], root)
                if path not in inventory or inventory[path]["subject_id"] != name or Path(path).name != f"{name}_{number:04d}.jpg":
                    raise ValueError("Pair endpoint differs from official identity/image index")
                pair["path_" + side] = path
            if (pair["subject_id_a"], pair["subject_id_b"]) != names:
                raise ValueError("Pair identity labels disagree with official file")
        required = {pair[key] for pair in pairs for key in ("path_a", "path_b")}
    else:
        closed, opened = protocol["closed_set"], protocol["open_set"]
        expected = {canonical(row["path"], root) for row in rows}
        if len(rows) != 400 or len({r["subject_id"] for r in rows}) != 40:
            raise ValueError("Expected frozen 40-identity x10-image inventory")
        gallery_index = 1 if dataset == "orl" else 11
        image_indices = set(range(1, 11)) if dataset == "orl" else {3, 4, 5, 6, 7, 8, 11, 12, 13, 14}
        if dataset == "fei_holdout" and protocol.get("evaluation_scope") != "within_domain_heldout_identities_prior_v2_frontal_exposure":
            raise ValueError("FEI heldout must disclose prior v2 frontal benchmark exposure")
        for section, keys in ((closed, ("gallery_paths", "probe_paths")),
                              (opened, ("gallery_paths", "known_probe_paths", "unknown_probe_paths"))):
            for key in keys:
                section[key] = [canonical(path, root) for path in section[key]]
                if len(section[key]) != len(set(section[key])) or not set(section[key]).issubset(expected):
                    raise ValueError("Duplicate/unknown protocol input")
        if set(closed["gallery_paths"]) & set(closed["probe_paths"]) or set(closed["gallery_paths"] + closed["probe_paths"]) != expected:
            raise ValueError("Gallery/probes must partition all400 inputs")
        for row in rows:
            is_gallery = canonical(row["path"], root) in closed["gallery_paths"]
            if is_gallery != (int(row["image_id"]) == gallery_index):
                raise ValueError("Wrong frozen gallery image index")
        subjects = sorted({row["subject_id"] for row in rows})
        for subject in subjects:
            samples = [row for row in rows if row["subject_id"] == subject]
            if len(samples) != 10 or {int(row["image_id"]) for row in samples} != image_indices:
                raise ValueError("Wrong image indices or count for an identity")
        permutation = np.random.default_rng(42).permutation(subjects).tolist()
        if opened["enrolled_identities"] != permutation[:20] or opened["unknown_identities"] != permutation[20:]:
            raise ValueError("Open-set identity assignment changed")
        enrolled = set(opened["enrolled_identities"])
        expected_lists = {
            "gallery_paths": {path for path in closed["gallery_paths"] if inventory[path]["subject_id"] in enrolled},
            "known_probe_paths": {path for path in closed["probe_paths"] if inventory[path]["subject_id"] in enrolled},
            "unknown_probe_paths": {path for path in closed["probe_paths"] if inventory[path]["subject_id"] not in enrolled},
        }
        if any(set(opened[key]) != expected_paths for key, expected_paths in expected_lists.items()):
            raise ValueError("Open-set gallery/probes differ from frozen identity roles")
        required = expected
    return inventory, pairs, required


def load_embeddings(bundle, inventory, required_paths, root=ROOT):
    quality = json.loads(project_path(bundle["quality_path"], root).read_text(encoding="utf-8"))
    if isinstance(quality, dict):
        quality = quality["records"]
    records = {}
    for row in quality:
        path = canonical(row.get("path", row.get("source_path", "")), root)
        if path in records or path not in inventory:
            raise ValueError("Quality log contains duplicate or unknown input path")
        if row.get("subject_id") != inventory[path]["subject_id"] or row.get("source_sha256") != inventory[path]["sha256"]:
            raise ValueError("Quality source identity/hash differs from inventory")
        if row.get("status") not in {"accepted", "rejected"}:
            raise ValueError("Quality log contains incomplete input")
        codes = row.get("rejection_codes", [])
        if not isinstance(codes, list) or any(not isinstance(code, str) or not code for code in codes):
            raise ValueError("Rejection codes must be a list of nonempty strings")
        if row["status"] == "rejected" and not codes:
            raise ValueError("Rejected input requires explicit reason codes")
        records[path] = row
    if not required_paths.issubset(records):
        raise ValueError("Quality log omits prescribed external inputs")
    with np.load(project_path(bundle["embeddings_path"], root), allow_pickle=False) as archive:
        embeddings = archive["embeddings"].copy()
    if embeddings.ndim != 2 or embeddings.dtype.kind not in "fiu" or embeddings.shape[1] < 1:
        raise ValueError("Embeddings must be numeric [N,D]")
    features, used = {}, set()
    for path, row in records.items():
        if row["status"] != "accepted":
            continue
        index = row.get("embedding_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(embeddings) or index in used:
            raise ValueError("Accepted inputs require unique valid embedding_index")
        vector = embeddings[index].astype(np.float64)
        if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
            raise ValueError("Accepted embeddings must be finite L2-unit vectors")
        used.add(index)
        features[path] = vector
    if len(used) != len(embeddings):
        raise ValueError("Embedding artifact contains unused rows; export accepted rows only")
    selected = [records[path] for path in required_paths]
    coverage = {"required_images": len(required_paths), "accepted_images": sum(row["status"] == "accepted" for row in selected),
                "quality_records_total": len(records), "extra_inventory_images_processed": len(records) - len(required_paths),
                "rejection_codes": dict(Counter(code for row in selected if row["status"] == "rejected" for code in row["rejection_codes"]))}
    coverage["image_success_rate"] = coverage["accepted_images"] / max(1, coverage["required_images"])
    return features, records, coverage


def pair_arrays(pairs, features):
    n = len(pairs)
    scores = np.zeros(n, np.float64)
    valid = np.zeros(n, bool)
    for i, pair in enumerate(pairs):
        if pair["path_a"] in features and pair["path_b"] in features:
            scores[i] = -cdist(np.asarray(features[pair["path_a"]], np.float64)[None, :],
                               np.asarray(features[pair["path_b"]], np.float64)[None, :], metric="euclidean")[0, 0]
            valid[i] = True
    return np.asarray([int(p["is_same"]) for p in pairs], np.int8), scores, valid


def operational_rates(labels, scores, available, threshold, weights=None):
    y, scores, available = np.asarray(labels), np.asarray(scores), np.asarray(available, bool)
    if y.ndim != 1 or scores.shape != y.shape or available.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid pair arrays")
    if not np.isfinite(scores[available]).all() or not np.isfinite(threshold):
        raise ValueError("Valid scores and threshold must be finite")
    weight = np.ones(len(y)) if weights is None else np.asarray(weights, np.float64)
    if weight.shape != y.shape or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("Invalid pair weights")
    accept = available & (scores >= threshold)
    positive, negative = y == 1, y == 0
    ng, ni = weight[positive].sum(), weight[negative].sum()
    return {"far": float(weight[negative & accept].sum() / ni) if ni else None,
            "frr": float(weight[positive & ~accept].sum() / ng) if ng else None,
            "accuracy": float(weight[accept == positive].sum() / weight.sum()),
            "genuine_pairs": int(positive.sum()), "impostor_pairs": int(negative.sum()),
            "available_pairs": int(available.sum()), "intended_pairs": len(y)}


def fit_accuracy_threshold(labels, scores, available):
    """Training-fold-only exact threshold sweep; conservative tie breaking."""
    y, scores, valid = np.asarray(labels), np.asarray(scores, np.float64), np.asarray(available, bool)
    operational_rates(y, scores, valid, 0)
    if not valid.any():
        raise ValueError("No valid training comparisons for threshold selection")
    values = np.unique(scores[valid])[::-1]
    best_threshold = float(np.nextafter(values[0], np.inf))
    correct = int(np.count_nonzero(y == 0))  # Initially reject every pair, including failed endpoints.
    best_correct = correct
    for value in values:
        group = valid & (scores == value)
        correct += int(np.sum(y[group] == 1)) - int(np.sum(y[group] == 0))
        if correct > best_correct:
            best_correct, best_threshold = correct, float(value)
    return best_threshold


def threshold_cross_validation(labels, scores, available, folds):
    y, scores, valid, folds = np.asarray(labels), np.asarray(scores), np.asarray(available, bool), np.asarray(folds)
    if folds.shape != y.shape or len(np.unique(folds)) < 2:
        raise ValueError("At least two predeclared folds required")
    records = []
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        try:
            threshold = fit_accuracy_threshold(y[train], scores[train], valid[train])
        except ValueError as error:
            if str(error) != "No valid training comparisons for threshold selection":
                raise
            records.append({"fold": int(fold), "status": "unavailable", "reason": str(error)})
            continue
        rates = operational_rates(y[test], scores[test], valid[test], threshold)
        conditional = operational_rates(y[test][valid[test]], scores[test][valid[test]], np.ones(valid[test].sum(), bool), threshold) if valid[test].any() else None
        records.append({"fold": int(fold), "status": "ok", "threshold_from_other_folds": threshold,
                        "train_pair_count": int(train.sum()), "operational": rates, "conditional": conditional})
    accuracies = [r["operational"]["accuracy"] for r in records if r["status"] == "ok"]
    complete = len(accuracies) == len(records)
    std = float(np.std(accuracies, ddof=1)) if complete and len(accuracies) > 1 else None
    return {"status": "ok" if complete else "incomplete", "folds": records,
            "mean_accuracy": float(np.mean(accuracies)) if complete else None,
            "sample_standard_deviation": std, "standard_error": std / np.sqrt(len(accuracies)) if std is not None else None,
            "scope": "Secondary threshold-CV only; model weights/preprocessing are fixed. Training folds select each threshold; heldout fold scores never select it.",
            "uncertainty_note": "Fold standard error is descriptive, not a pair-independent binomial confidence interval."}


def pair_identity_bootstrap(labels, scores, available, first_ids, second_ids, threshold, replicates=1000, seed=42):
    y, scores, valid = np.asarray(labels), np.asarray(scores), np.asarray(available, bool)
    a, b = np.asarray(first_ids), np.asarray(second_ids)
    operational_rates(y, scores, valid, threshold)
    if a.shape != y.shape or b.shape != y.shape or not np.array_equal(y.astype(bool), a == b):
        raise ValueError("Pair identity equality must match genuine/impostor labels")
    ids = np.unique(np.concatenate([a, b]))
    if len(ids) < 2 or replicates < 1:
        raise ValueError("At least two identities and positive bootstrap replicates required")
    ai, bi = np.searchsorted(ids, a), np.searchsorted(ids, b)
    rng = np.random.default_rng(seed)
    samples, conditional, attempts = [], [], 0
    while len(samples) < replicates and attempts < replicates * 20:
        attempts += 1
        counts = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
        weights = np.where(y == 1, counts[ai], counts[ai] * counts[bi])
        if not all(weights[y == cls].sum() > 0 for cls in (0, 1)):
            continue
        rate = operational_rates(y, scores, valid, threshold, weights)
        samples.append([rate["far"], rate["frr"], rate["accuracy"]])
        if all(weights[valid & (y == cls)].sum() > 0 for cls in (0, 1)):
            result = verification_from_scores(y[valid], scores[valid], threshold, sample_weight=weights[valid])
            conditional.append([result.eer, result.auc, result.far, result.frr])
    if len(samples) != replicates:
        raise ValueError("Insufficient valid identity-bootstrap resamples")
    bounds = np.percentile(samples, [2.5, 97.5], axis=0)
    result = {"method": "identity_resampling_percentile_95", "replicates": replicates, "seed": seed,
              "identity_count": len(ids), "conditional_on": "frozen_model_and_operating_threshold",
              "pair_dependence": "Same-person weight=multiplicity; different-person weight=product of endpoint multiplicities. Pairs are not bootstrapped independently.",
              "operational_intervals": {name: bounds[:, i].tolist() for i, name in enumerate(("far", "frr", "accuracy"))},
              "conditional_pair_replicates": len(conditional)}
    if len(conditional) >= 2:
        cb = np.percentile(conditional, [2.5, 97.5], axis=0)
        result["conditional_pair_intervals"] = {name: cb[:, i].tolist() for i, name in enumerate(("eer", "auc", "far", "frr"))}
    return result


def evaluate_lfw(pairs, features, threshold, bootstrap_replicates=1000):
    y, scores, valid = pair_arrays(pairs, features)
    operational = operational_rates(y, scores, valid, threshold)
    operational["tar"] = 1 - operational["frr"] if operational["frr"] is not None else None
    conditional = {"status": "unavailable", "reason": "both_valid_pair_classes_required"}
    if all(np.any(valid & (y == cls)) for cls in (0, 1)):
        conditional = {"status": "ok", **asdict(verification_from_scores(y[valid], scores[valid], threshold, "frozen_FEI_development"))}
        conditional["accuracy"] = operational_rates(y[valid], scores[valid], np.ones(valid.sum(), bool), threshold)["accuracy"]
        fpr, tpr, roc_thresholds = roc_curve(y[valid], scores[valid], drop_intermediate=False)
    else:
        fpr, tpr, roc_thresholds = np.array([]), np.array([]), np.array([])
    bootstrap = pair_identity_bootstrap(y, scores, valid, [p["subject_id_a"] for p in pairs],
                                        [p["subject_id_b"] for p in pairs], threshold, bootstrap_replicates)
    report = {"score_threshold": float(threshold), "fixed_threshold_operational": operational,
              "fixed_threshold_conditional": conditional, "confidence_intervals": bootstrap,
              "secondary_threshold_cross_validation": threshold_cross_validation(y, scores, valid, [int(p["fold_id"]) for p in pairs]),
              "limitations": ["Official pair verification is not open-set identification; pair FAR is not search FPIR.",
                              "Failed genuine pairs count as false rejection; failed impostor pairs count as rejection. Read operational accuracy together with coverage.",
                              "External pretraining may contain LFW identities. No identity-disjoint-pretraining claim is made.",
                              "Session mapping is unknown. No pore-resolution or depth claim follows from 250px images.",
                              "Zero-error percentile intervals do not establish a population upper bound on error."]}
    return report, {"labels": y, "scores": scores, "valid_pairs": valid,
                    "fold_ids": np.asarray([int(p["fold_id"]) for p in pairs], np.int16),
                    "roc_fpr": fpr, "roc_tpr": tpr, "roc_thresholds": roc_thresholds}


def evaluate_orl(protocol, inventory, features, threshold, bootstrap_replicates=1000):
    """Shared closed/open evaluation for ORL and the declared FEI heldout IDs."""
    report, numeric, decisions = summarize_protocol(protocol, inventory, features, threshold, bootstrap_replicates)
    operational = operational_rates(numeric["pair_labels"].ravel(), numeric["scores"].ravel(), numeric["valid_comparisons"].ravel(), threshold)
    operational["tar"] = 1 - operational["frr"] if operational["frr"] is not None else None
    report["verification_operational"] = operational
    report["evaluation_scope"] = protocol.get("evaluation_scope", "external_general_biometrics_low_resolution_control")
    if protocol.get("evaluation_scope") == "within_domain_heldout_identities_prior_v2_frontal_exposure":
        report["limitations"].append("FEI heldout identities are excluded from v4 fitting; their frontal11/12 images were evaluated in v2. This is not a new unseen external dataset.")
    return report, numeric, decisions


def write_results(output, dataset, report, numeric, pairs=None, protocol=None, inventory=None, decisions=None):
    write_json(output / "results.json", report)
    np.savez_compressed(output / "scores.npz", **numeric)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if dataset == "lfw":
            writer.writerow(["pair_id", "fold_id", "path_a", "path_b", "is_same", "available", "negative_L2"])
            for i, pair in enumerate(pairs):
                writer.writerow([pair["pair_id"], pair["fold_id"], pair["path_a"], pair["path_b"], pair["is_same"],
                                 int(numeric["valid_pairs"][i]), float(numeric["scores"][i]) if numeric["valid_pairs"][i] else ""])
        else:
            writer.writerow(["probe_path", "gallery_path", "genuine", "available", "negative_L2"])
            for i, probe in enumerate(protocol["closed_set"]["probe_paths"]):
                for j, gallery in enumerate(protocol["closed_set"]["gallery_paths"]):
                    valid = numeric["valid_comparisons"][i, j]
                    writer.writerow([probe, gallery, int(numeric["pair_labels"][i, j]), int(valid), float(numeric["scores"][i, j]) if valid else ""])
    if dataset != "lfw":
        write_json(output / "probe_decisions.json", {"closed_conservative_ranks": numeric["ranks"].tolist(),
                                                     "open_probe_order": protocol["open_set"]["known_probe_paths"] + protocol["open_set"]["unknown_probe_paths"],
                                                     "open_known": decisions["known"].tolist(), "open_accepted": decisions["accepted"].tolist(),
                                                     "open_correct": decisions["correct"].tolist(), "open_winner": decisions["winner"].tolist(),
                                                     "open_reason": decisions["reason"]})
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    if numeric["roc_fpr"].size:
        ax.plot(numeric["roc_fpr"], numeric["roc_tpr"])
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="FAR", ylabel="TAR", title=dataset.upper() + ": conditional verification ROC", xlim=(0, 1), ylim=(0, 1))
    fig.savefig(output / "roc.png", dpi=200)
    fig.savefig(output / "roc.pdf")
    plt.close(fig)


def run(freeze_path, bundle_path, dataset, output, root=ROOT):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty evaluation directory")
    freeze, bundle, method, protocol = validate_freeze_bundle(freeze_path, bundle_path, dataset, root)
    inventory, pairs, required = load_protocol_data(protocol, dataset, root)
    features, records, coverage = load_embeddings(bundle, inventory, required, root)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"status": "running", "dataset": dataset, "method_id": method["method_id"]})
    try:
        threshold = float(method["score_threshold"])
        if dataset == "lfw":
            report, numeric = evaluate_lfw(pairs, features, threshold)
            decisions = None
        else:
            report, numeric, decisions = evaluate_orl(protocol, inventory, features, threshold)
        report.update(dataset=dataset, method_id=method["method_id"], scope=method["scope"],
                      feature_space_id=method["feature_space_id"], preprocessing_id=method["preprocessing_id"],
                      image_coverage=coverage, freeze_sha256=bundle["freeze_sha256"], bundle_sha256=sha256(bundle_path),
                      protocol_sha256=bundle["protocol_sha256"], source_hashes=freeze["source_hashes"],
                      score_definition="negative_euclidean_distance_of_frozen_L2_unit_embeddings",
                      evaluation_utc=datetime.now(timezone.utc).isoformat())
        if "search_score_threshold" in method and dataset != "lfw":
            second, second_numeric, second_decisions = evaluate_orl(protocol, inventory, features, float(method["search_score_threshold"]))
            second_dir = output / "search_operating_point"
            second_dir.mkdir()
            write_results(second_dir, dataset, second, second_numeric, protocol=protocol, inventory=inventory, decisions=second_decisions)
            report["additional_frozen_search_operating_point"] = {"score_threshold": float(method["search_score_threshold"]),
                                                                 "open_set": second["open_set"], "verification_operational": second["verification_operational"],
                                                                 "verification_conditional": second["verification_conditional"]}
        validate_freeze_bundle(freeze_path, bundle_path, dataset, root)
        write_results(output, dataset, report, numeric, pairs, protocol, inventory, decisions)
        write_json(output / "input_provenance.json", {"freeze": freeze, "bundle": bundle, "protocol": protocol})
        write_json(output / "quality_summary.json", coverage)
        write_json(output / "artifact_sha256.json", {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*"))
                                                      if p.is_file() and p.name != "status.json"})
        write_json(output / "status.json", {"status": "completed", "dataset": dataset, "method_id": method["method_id"]})
        return report
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--dataset", choices=("lfw", "orl", "fei_holdout"), required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    run(arguments.freeze, arguments.bundle, arguments.dataset, arguments.output)
