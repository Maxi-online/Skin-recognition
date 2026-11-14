"""Frozen CFP holdout and known-benchmark repair audits for study-v4.2.

No image decoding, feature extraction or model fitting is performed here.
The original study-v4.1 evaluators remain unchanged. Pure metric functions are
reused, including fixed-threshold operations and separately labelled threshold-CV.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.external_evaluate_v4 import (canonical, check_hash, evaluate_lfw, evaluate_orl,
                                             load_embeddings, load_protocol_data as load_legacy_protocol,
                                             utc, write_results as write_legacy_results)
from experiments.external_evaluate_v3 import project_path, sha256, write_json


def validate_freeze_bundle(freeze_path, bundle_path, dataset, root=ROOT):
    if dataset not in {"cfp_ff", "cfp_fp", "lfw", "orl", "fei_holdout"}:
        raise ValueError("Unsupported study-v4.2 dataset")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8-sig"))
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8-sig"))
    if freeze.get("schema_version") != "study-v4.2":
        raise ValueError("Expected study-v4.2 freeze")
    base_path = project_path(freeze["base_freeze_path"], root)
    check_hash(base_path, freeze["base_freeze_sha256"], "original study-v4.1 freeze")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("schema_version") != "study-v4.1" or utc(base["timestamp"]) > utc(freeze["timestamp"]):
        raise ValueError("Invalid parent study freeze")
    preserved = ("scope", "skin_weight", "feature_space_id", "score_threshold", "search_score_threshold",
                 "skin_model_path", "global_model_path", "score_definition")
    old_methods = {method["method_id"]: method for method in base["methods"]}
    new_methods = {method["method_id"]: method for method in freeze["methods"]}
    if len(old_methods) != len(base["methods"]) or len(new_methods) != len(freeze["methods"]) or old_methods.keys() != new_methods.keys():
        raise ValueError("Geometry repair must preserve every unique frozen method")
    for name, method in new_methods.items():
        if any(method.get(key) != old_methods[name].get(key) for key in preserved):
            raise ValueError("Geometry repair changed a model, fusion weight or decision threshold")
    if freeze["model_artifacts"] != base["model_artifacts"]:
        raise ValueError("Geometry repair changed frozen model artifact hashes")
    check_hash(freeze_path, bundle["freeze_sha256"], "v4.2 model freeze")
    if utc(bundle["extraction_started_utc"]) < utc(freeze["timestamp"]):
        raise ValueError("External extraction predates model freeze")
    if utc(freeze["timestamp"]) > datetime.now(timezone.utc):
        raise ValueError("Freeze timestamp is in the future")
    required = {"experiments/external_evaluate_v42.py", "experiments/external_evaluate_v4.py",
                "experiments/external_evaluate_v3.py", "experiments/scientific_metrics.py"}
    if not required.issubset(freeze["source_hashes"]):
        raise ValueError("Freeze omits required evaluator/dependency hashes")
    for filename, digest in freeze["source_hashes"].items():
        check_hash(project_path(filename, root), digest, filename)
    if not freeze.get("model_artifacts"):
        raise ValueError("Freeze needs hashed model artifacts")
    for filename, digest in freeze["model_artifacts"].items():
        check_hash(project_path(filename, root), digest, filename)
    methods = [m for m in freeze["methods"] if m["method_id"] == bundle["method_id"]]
    if len(methods) != 1:
        raise ValueError("Bundle must identify exactly one frozen method")
    method = methods[0]
    for key in ("scope", "preprocessing_id", "feature_space_id"):
        if method[key] != bundle[key]:
            raise ValueError("Frozen method contract mismatch: " + key)
    for key in ("score_threshold", "search_score_threshold"):
        if key in method and (isinstance(method[key], bool) or not np.isfinite(float(method[key]))):
            raise ValueError("Frozen thresholds must be finite")
    if "score_threshold" not in method or method.get("score_definition", "negative_euclidean_distance") != "negative_euclidean_distance":
        raise ValueError("Frozen negative-L2 primary threshold required")
    info = freeze["protocols"][dataset]
    path = project_path(info["path"], root)
    check_hash(path, info["sha256"], "evaluation protocol")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected_id = dataset + ("_external_v4_2" if dataset.startswith("cfp_") else "_external_v4_1")
    if bundle["protocol_sha256"] != info["sha256"] or protocol["protocol_id"] != expected_id or bundle["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Bundle/evaluation protocol mismatch")
    if not dataset.startswith("cfp_"):
        original_info = base["protocols"][dataset]
        original_path = project_path(original_info["path"], root)
        check_hash(original_path, original_info["sha256"], "original benchmark protocol")
        original_protocol = json.loads(original_path.read_text(encoding="utf-8"))
        if protocol.get("parent_protocol_sha256") != original_info["sha256"]:
            raise ValueError("Known repair audit must name its original protocol hash")
        if protocol.get("evaluation_scope") != "known_benchmark_geometry_repair_audit_v4_2":
            raise ValueError("Previously observed input cannot become an untouched holdout")
        allowed = {"repair_protocol_created_utc", "parent_protocol_sha256", "parent_protocol_path", "prior_evaluation_scope",
                   "evaluation_scope", "status_at_freeze"}
        if {k: v for k, v in original_protocol.items() if k not in allowed} != {k: v for k, v in protocol.items() if k not in allowed}:
            raise ValueError("Geometry audit changed the original pair/gallery/open-set protocol")
    for key in ("embeddings", "quality"):
        check_hash(project_path(bundle[key + "_path"], root), bundle[key + "_sha256"], key + " artifact")
    return freeze, bundle, method, protocol


def load_protocol_data(protocol, dataset, root=ROOT):
    """Validate the unchanged original40split files and pair endpoint mappings."""
    if dataset in {"lfw", "orl", "fei_holdout"}:
        if protocol.get("evaluation_scope") != "known_benchmark_geometry_repair_audit_v4_2":
            raise ValueError("Known benchmark requires geometry-repair scope")
        validation_copy = copy.deepcopy(protocol)
        if dataset == "fei_holdout":
            validation_copy["evaluation_scope"] = "within_domain_heldout_identities_prior_v2_frontal_exposure"
        inventory, pairs, required = load_legacy_protocol(validation_copy, dataset, root)
        # Reuse the same canonicalized, verified gallery/probe paths for scoring.
        for name in ("closed_set", "open_set"):
            if name in validation_copy:
                protocol[name] = validation_copy[name]
        return inventory, pairs, required
    if dataset not in {"cfp_ff", "cfp_fp"}:
        raise ValueError("Expected cfp_ff or cfp_fp")
    kind = dataset[-2:].upper()
    role = "primary_frontal_uncontrolled_holdout" if kind == "FF" else "secondary_profile_pose_stress_test"
    if protocol.get("comparison") != kind or protocol.get("evaluation_role") != role:
        raise ValueError("CFP comparison/primary-stress role changed")
    inventory_path = project_path(protocol["inventory_path"], root)
    check_hash(inventory_path, protocol["inventory_sha256"], "CFP inventory")
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    inventory = {canonical(row["path"], root): row for row in rows}
    if len(rows) != 7000 or len(inventory) != len(rows) or len({row["subject_id"] for row in rows}) != 500:
        raise ValueError("Expected CFP7000uniqueimages/500identities")
    for subject in {row["subject_id"] for row in rows}:
        for pose, count in (("frontal", 10), ("profile", 4)):
            selected = [row for row in rows if row["subject_id"] == subject and row["pose"] == pose]
            if len(selected) != count or {int(row["image_id"]) for row in selected} != set(range(1, count+1)):
                raise ValueError("CFP identity/pose image indices changed")
    official = protocol["official_protocol_hashes"]
    lists = [path for path in official if path.endswith("/Protocol/Pair_list_F.txt")]
    if len(lists) != 1:
        raise ValueError("Official frontal lookup missing or ambiguous")
    directory = project_path(lists[0], root).parent
    expected_files = {directory / ("Pair_list_" + view + ".txt") for view in ("F", "P")}
    expected_files.update(directory / "Split" / comparison / f"{fold:02d}" / filename
                          for comparison in ("FF", "FP") for fold in range(1, 11) for filename in ("same.txt", "diff.txt"))
    if {project_path(path, root) for path in official} != expected_files:
        raise ValueError("Expected both original CFP lookups and40splitfiles")
    for path, digest in official.items():
        check_hash(project_path(path, root), digest, "official CFP protocol input")
    lookups = {}
    for view, pose, count in (("F", "frontal", 5000), ("P", "profile", 2000)):
        mapping = {}
        for line in (directory / ("Pair_list_" + view + ".txt")).read_text().splitlines():
            index, relative = line.split()
            path = (directory / relative).resolve().as_posix()
            if int(index) in mapping or path not in inventory or inventory[path]["pose"] != pose:
                raise ValueError("Invalid original CFP lookup")
            mapping[int(index)] = path
        if set(mapping) != set(range(1, count+1)) or len(set(mapping.values())) != count:
            raise ValueError("Original CFP lookup range changed")
        lookups[view] = mapping
    manifest = project_path(protocol["pair_manifest_path"], root)
    check_hash(manifest, protocol["pair_manifest_sha256"], "CFP pair manifest")
    with manifest.open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    if len(pairs) != 7000:
        raise ValueError("Original CFP7000pairs required")
    cursor, fold_ids = 0, []
    for fold in range(1, 11):
        identities = set()
        for label, filename in ((1, "same.txt"), (0, "diff.txt")):
            source = directory / "Split" / kind / f"{fold:02d}" / filename
            original = source.read_text().splitlines()
            if len(original) != 350:
                raise ValueError("Original CFP fold/class count changed")
            for line_number, line in enumerate(original, 1):
                a, b = [int(value.strip()) for value in line.split(",")]
                first, second = lookups["F"][a], lookups["F" if kind == "FF" else "P"][b]
                ida, idb = inventory[first]["subject_id"], inventory[second]["subject_id"]
                pair = pairs[cursor]
                if (int(pair["pair_id"]) != cursor or int(pair["fold_id"]) != fold or int(pair["is_same"]) != label or
                    canonical(pair["path_a"], root) != first or canonical(pair["path_b"], root) != second or
                    pair["subject_id_a"] != ida or pair["subject_id_b"] != idb or int(ida == idb) != label or first == second or
                    canonical(pair["source_pair_file"], root) != source.resolve().as_posix() or int(pair["source_line"]) != line_number or
                    int(pair["official_index_a"]) != a or int(pair["official_index_b"]) != b):
                    raise ValueError("CFP pair manifest differs from original ordered endpoint/label/fold")
                pair["path_a"], pair["path_b"] = first, second
                identities.update((ida, idb))
                cursor += 1
        if len(identities) != 50 or any(identities & previous for previous in fold_ids):
            raise ValueError("CFP folds must contain50disjointidentities")
        fold_ids.append(identities)
    if protocol.get("acquisition_path"):
        check_hash(project_path(protocol["acquisition_path"], root), protocol["acquisition_sha256"], "CFP acquisition")
    required = {pair[key] for pair in pairs for key in ("path_a", "path_b")}
    return inventory, pairs, required


def evaluate_pairs(pairs, features, threshold, comparison, bootstrap_replicates=1000):
    report, numeric = evaluate_lfw(pairs, features, threshold, bootstrap_replicates)
    report["comparison"] = comparison
    report["limitations"] = [
        "CFP pair verification is not open-set identification; pair FAR does not measure1:N FPIR.",
        "All prescribed comparisons are retained operationally. Failed genuine pairs are false rejections; failed impostor pairs are rejections. Read metrics with image and pair coverage.",
        "CFP is an untouched project input holdout at protocol creation; celebrity identities can overlap LFW and external pretraining.",
        "Folds are identity-disjoint within CFP; no per-image session mapping, pore-resolution or measured-depth validation is supplied.",
        "The profile-stress experiment is reported separately from the primary frontal-frontal experiment.",
        "Percentile intervals condition on frozen weights and threshold; zero observed errors do not establish a zero population error rate."]
    return report, numeric


def write_pair_results(output, report, numeric, pairs, dataset):
    write_json(output / "results.json", report)
    np.savez_compressed(output / "scores.npz", **numeric)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair_id", "fold_id", "path_a", "path_b", "is_same", "available", "negative_L2"])
        for i, pair in enumerate(pairs):
            writer.writerow([pair["pair_id"], pair["fold_id"], pair["path_a"], pair["path_b"], pair["is_same"],
                             int(numeric["valid_pairs"][i]), float(numeric["scores"][i]) if numeric["valid_pairs"][i] else ""])
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    if numeric["roc_fpr"].size:
        ax.plot(numeric["roc_fpr"], numeric["roc_tpr"])
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="FAR", ylabel="TAR", title=dataset.upper().replace("_", "-") + ": conditional verification ROC", xlim=(0, 1), ylim=(0, 1))
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
        decisions = None
        if dataset.startswith("cfp_"):
            report, numeric = evaluate_pairs(pairs, features, threshold, protocol["comparison"])
        elif dataset == "lfw":
            report, numeric = evaluate_lfw(pairs, features, threshold)
        else:
            report, numeric, decisions = evaluate_orl(protocol, inventory, features, threshold)
        if not dataset.startswith("cfp_"):
            report.setdefault("limitations", []).append("This is a geometry-repair audit on previously observed benchmark inputs, not a new untouched holdout. Recognition models and thresholds are unchanged from study-v4.1.")
        if dataset == "fei_holdout":
            report["limitations"].append("FEI heldout identities were excluded from v4 fitting, but frontal11/12 inputs were benchmarked in v2 and this inventory was evaluated in v4.1.")
        report.update(dataset=dataset, method_id=method["method_id"], scope=method["scope"],
                      evaluation_role=protocol.get("evaluation_role", protocol.get("evaluation_scope")), image_coverage=coverage,
                      preprocessing_id=method["preprocessing_id"], feature_space_id=method["feature_space_id"],
                      freeze_sha256=bundle["freeze_sha256"], bundle_sha256=sha256(bundle_path), protocol_sha256=bundle["protocol_sha256"],
                      source_hashes=freeze["source_hashes"], score_definition="negative_euclidean_distance_of_frozen_L2_unit_embeddings",
                      evaluation_utc=datetime.now(timezone.utc).isoformat())
        if "search_score_threshold" in method and dataset.startswith("cfp_"):
            second, _ = evaluate_pairs(pairs, features, float(method["search_score_threshold"]), protocol["comparison"])
            report["additional_frozen_conservative_pair_operating_point"] = {
                "score_threshold": float(method["search_score_threshold"]),
                "operational": second["fixed_threshold_operational"], "conditional": second["fixed_threshold_conditional"],
                "confidence_intervals": second["confidence_intervals"], "scope": "Verification at a pre-frozen search-derived cutoff; CFP does not measure open-set search risk."}
        elif "search_score_threshold" in method and dataset != "lfw":
            second, second_numeric, second_decisions = evaluate_orl(protocol, inventory, features, float(method["search_score_threshold"]))
            second_dir = output / "search_operating_point"
            second_dir.mkdir()
            write_legacy_results(second_dir, dataset, second, second_numeric, protocol=protocol, inventory=inventory, decisions=second_decisions)
            report["additional_frozen_search_operating_point"] = {
                "score_threshold": float(method["search_score_threshold"]), "open_set": second["open_set"],
                "verification_operational": second["verification_operational"], "verification_conditional": second["verification_conditional"]}
        validate_freeze_bundle(freeze_path, bundle_path, dataset, root)
        if dataset.startswith("cfp_"):
            write_pair_results(output, report, numeric, pairs, dataset)
        else:
            write_legacy_results(output, dataset, report, numeric, pairs, protocol, inventory, decisions)
        write_json(output / "input_provenance.json", {"freeze": freeze, "bundle": bundle, "protocol": protocol})
        write_json(output / "quality_summary.json", coverage)
        write_json(output / "artifact_sha256.json", {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "status.json"})
        write_json(output / "status.json", {"status": "completed", "dataset": dataset, "method_id": method["method_id"]})
        return report
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--dataset", choices=("cfp_ff", "cfp_fp", "lfw", "orl", "fei_holdout"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.freeze, args.bundle, args.dataset, args.output)
