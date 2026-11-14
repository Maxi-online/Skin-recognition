"""FEI-only fusion selection, followed by explicitly known GT diagnostics.

No unseen dataset is accepted by this runner. Development calibration metrics
are selection statistics, not independent performance evidence.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.descriptors.hybrid_v4 import (
    HybridFaceSkinDescriptorV4, arcface_crop, fuse_embeddings, sha256,
    source_landmarks_from_cache,
    source_landmarks_for_single_face,
    prepared_tuple_from_face_v4,
)
from experiments.scientific_metrics import (
    cmc_from_gallery, cross_session_pairs, select_threshold_at_far, verification_from_scores,
)

SKIN_WEIGHTS = (0.0, 0.02, 0.05, 0.1, 0.2)
DEFAULT_SKIN = ROOT/"research/development/v3_classical_landmark_strict/models/skin_rootsift_pca128_frozen.npz"
DEFAULT_GLOBAL = Path.home()/".insightface/models/buffalo_l/w600k_r50.onnx"


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def path_from(value, base=ROOT):
    path = Path(value)
    return path if path.is_absolute() else base/path


def load_development_rows(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("split") in {"train", "calibration"}]
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Missing/duplicate FEI development rows")
    if any(not row["subject_id"].startswith("fei_") for row in rows):
        raise ValueError("Only FEI development subjects are permitted")
    ids = {split: {row["subject_id"] for row in rows if row["split"] == split} for split in ("train", "calibration")}
    if not all(ids.values()) or ids["train"] & ids["calibration"]:
        raise ValueError("Invalid train/calibration identity split")
    return rows


def packed_cache(archive, quality):
    if "skin_only_forehead" in archive:
        names = [str(item) for item in archive["patch_names"]]
        patches = np.stack([archive["skin_only_"+name] for name in names])
        masks = np.stack([np.full(patch.shape[:2], bool(quality["patch_quality"].get(name, {}).get("strict_skin_usable", False)), np.uint8)
                          for name, patch in zip(names, patches)])
    else:
        patches, masks = archive["patches_bgr"], archive["patch_masks"]
    return archive["aligned_bgr"], patches, masks, archive["skin_mask"]


def extract_fei(rows, model, output, gpu_reprocess=False):
    """Use CPU frozen caches for development, or rerun the full CUDA pipeline."""
    from src.preprocessing_v3 import FacePreprocessorV3, QualityRejection
    processor = FacePreprocessorV3(providers=["CUDAExecutionProvider", "CPUExecutionProvider"]) if gpu_reprocess else None
    if processor is not None and any(providers[0] != "CUDAExecutionProvider" for providers in processor.runtime_providers.values()):
        raise RuntimeError("GPU calibration requires both preprocessing sessions on CUDA")
    global_features, skin_features, records, align_audit = [], [], [], []
    detector_features, detector_records = [], []
    started = time.monotonic()
    for index, row in enumerate(rows):
        record = {**row, "embedding_index": -1}
        if row.get("status") != "accepted" and not gpu_reprocess:
            records.append(record)
            continue
        source = path_from(row["source_path"])
        if sha256(source) != row["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {source}")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {source}")
        if processor is not None:
            detector_record = {**row, "embedding_index": -1}
            try:
                detector_kps, detector_quality = source_landmarks_for_single_face(processor, image)
                detector_features.append(model.global_model.extract_features(image, detector_kps))
                detector_record.update(status="accepted", embedding_index=len(detector_features)-1,
                                       quality=detector_quality, preprocessing_id=detector_quality["preprocessing_id"])
            except QualityRejection as error:
                detector_record.update(status="rejected", rejection_codes=error.codes, quality=error.metadata)
            detector_records.append(detector_record)
        try:
            if processor is not None:
                prepared = processor.process(image)
                packed, pid = prepared_tuple_from_face_v4(prepared)
                kps = source_landmarks_from_cache(prepared.kps5, prepared.transform)
                record.update(quality=prepared.metadata, status="accepted", preprocessing_id=prepared.metadata["preprocessing_id"])
                gpu_cache = Path(output)/"gpu_prepared"/(row["sample_id"]+".npz")
                prepared.save(gpu_cache)
                record.update(cache_path=str(gpu_cache.resolve()), cache_sha256=sha256(gpu_cache))
                aligned224 = prepared.aligned_bgr
            else:
                cache_path = path_from(row["cache_path"], Path(output).parent)
                if sha256(cache_path) != row["cache_sha256"]:
                    raise ValueError(f"Cache hash mismatch: {cache_path}")
                with np.load(cache_path, allow_pickle=False) as archive:
                    packed = packed_cache(archive, row["quality"])
                    kps = source_landmarks_from_cache(archive["kps5"], archive["transform"])
                    aligned224 = archive["aligned_bgr"]
                pid = row["preprocessing_id"]+":strict"
            crop = arcface_crop(image, kps)
            if len(align_audit) < 10:
                shortcut = cv2.resize(aligned224, (112, 112), interpolation=cv2.INTER_LINEAR)
                align_audit.append({"sample_id": row["sample_id"],
                                    "resized224_vs_direct112_mean_abs_pixel_error": float(np.abs(shortcut.astype(float)-crop).mean()),
                                    "used": "direct_source_norm_crop112"})
            global_feature = model.global_model.extract_aligned_batch([crop])[0]
            skin = model.skin_model.extract_features(packed, pid)
            record["embedding_index"] = len(global_features)
            record["source_kps5"] = kps.tolist()
            global_features.append(global_feature)
            skin_features.append(skin)
        except QualityRejection as error:
            record.update(status="rejected", rejection_codes=error.codes, quality=error.metadata)
        records.append(record)
        if (index+1) % 20 == 0:
            print(json.dumps({"processed": index+1, "total": len(rows), "accepted": len(global_features),
                              "elapsed_seconds": round(time.monotonic()-started, 2)}), flush=True)
    if not global_features:
        raise ValueError("No accepted FEI images")
    accepted = [row for row in records if row["embedding_index"] >= 0]
    components = {"global_embeddings": np.stack(global_features).astype(np.float32), "skin_embeddings": np.stack(skin_features).astype(np.float32),
                  **{key: np.asarray([row[source_key] for row in accepted]) for key, source_key in
                     (("sample_ids", "sample_id"), ("subject_ids", "subject_id"), ("splits", "split"),
                      ("roles", "role"), ("paths", "source_path"))}}
    np.savez_compressed(Path(output)/"components.npz", **components)
    write_json(Path(output)/"manifest.json", records)
    write_json(Path(output)/"alignment_audit.json", align_audit)
    if detector_records:
        detector_accepted = [row for row in detector_records if row["embedding_index"] >= 0]
        np.savez_compressed(Path(output)/"detector_only_components.npz",
                            embeddings=np.stack(detector_features).astype(np.float32),
                            **{key: np.asarray([row[source_key] for row in detector_accepted]) for key, source_key in
                               (("sample_ids", "sample_id"), ("subject_ids", "subject_id"), ("splits", "split"),
                                ("roles", "role"), ("paths", "source_path"))})
        write_json(Path(output)/"detector_only_manifest.json", detector_records)
    return components, records


def candidate_vectors(components, weight):
    if weight == 0:
        return components["global_embeddings"].astype(np.float32)
    if weight == 1:
        return components["skin_embeddings"].astype(np.float32)
    return np.stack([fuse_embeddings(g, s, weight) for g, s in
                     zip(components["global_embeddings"], components["skin_embeddings"])])


def evaluate_calibration(vectors, components, intended_probes=None):
    gallery = (components["splits"] == "calibration") & (components["roles"] == "gallery")
    probe = (components["splits"] == "calibration") & (components["roles"] == "probe")
    ids = components["subject_ids"]
    if not gallery.any() or not probe.any():
        raise ValueError("No accepted calibration gallery/probe images")
    labels, scores, _, _, matrix = cross_session_pairs(vectors[gallery], ids[gallery], vectors[probe], ids[probe])
    threshold = select_threshold_at_far(labels, scores, 0.01)
    strict = float(np.nextafter(scores[labels == 0].max(), np.inf))
    metric = asdict(verification_from_scores(labels, scores, threshold, "FEI_development_calibration"))
    eligible = np.isin(ids[probe], ids[gallery])
    conditional_rank1 = float(cmc_from_gallery(matrix[eligible], ids[gallery], ids[probe][eligible], 5)[0]) if eligible.any() else 0.0
    metric.update(rank1=conditional_rank1*int(eligible.sum())/(intended_probes or int(probe.sum())),
                  rank1_conditional=conditional_rank1,
                  intended_probe_images=intended_probes or int(probe.sum()),
                  genuine_template_available_probes=int(eligible.sum()),
                  gallery_images=int(gallery.sum()), probe_images=int(probe.sum()),
                  search_score_threshold=strict,
                  search_metrics=asdict(verification_from_scores(labels, scores, strict, "FEI_development_zero_false_match")))
    return metric


def run_fei(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = load_development_rows(args.manifest)
    if args.calibration_only:
        rows = [row for row in rows if row["split"] == "calibration"]
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]
    model = HybridFaceSkinDescriptorV4(args.global_model, args.skin_model, providers=providers)
    plan = {"schema_version": "hybrid-fei-development-v4.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "development_only_not_independent_test", "weights": list(SKIN_WEIGHTS),
            "selection_rule": "maximize_rank1_then_minimize_EER_then_minimize_FRR_at_FAR1pct_then_minimize_skin_weight",
            "best_positive_hybrid_rule": "same_order_restricted_to_w_greater_than_zero",
            "model_configuration": model.configuration,
            "source_manifest": str(Path(args.manifest).resolve()), "source_manifest_sha256": sha256(args.manifest),
            "runtime": {"global": providers, "preprocessing": "CUDA_full_reprocess" if args.gpu_reprocess else "frozen_CPU_cache",
                        "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
            "score_numeric_contract": "final_embedding_float32_before_scipy_cdist_float64_accumulation",
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                              (ROOT/"src/descriptors/hybrid_v4.py", Path(__file__).resolve())}}
    write_json(output/"plan.json", plan)
    if args.reuse_components or args.reuse_global:
        prior = Path(args.reuse_components or args.reuse_global).resolve()
        prior_plan = json.loads((prior/"plan.json").read_text(encoding="utf-8"))
        if (prior_plan["source_manifest_sha256"] != plan["source_manifest_sha256"] or
                prior_plan["runtime"] != plan["runtime"]):
            raise ValueError("Reusable components differ in cohort/runtime")
        for key in ("global_model_sha256", "skin_model_sha256"):
            if key == "skin_model_sha256" and args.reuse_global:
                continue
            if prior_plan["model_configuration"][key] != model.configuration[key]:
                raise ValueError("Reusable components were computed by a different model")
        with np.load(prior/"components.npz", allow_pickle=False) as archive:
            components = {key: archive[key].astype(np.float32) if key.endswith("embeddings") else archive[key] for key in archive.files}
        records = json.loads((prior/"manifest.json").read_text(encoding="utf-8"))
        if [row["sample_id"] for row in records] != [row["sample_id"] for row in rows]:
            raise ValueError("Reusable component rows differ from the requested cohort/order")
        if args.reuse_global:
            skin_features = []
            for row in records:
                if row["embedding_index"] < 0:
                    continue
                cache = path_from(row["cache_path"])
                if sha256(cache) != row["cache_sha256"]:
                    raise ValueError("Reused GPU prepared-cache checksum mismatch")
                with np.load(cache, allow_pickle=False) as archive:
                    packed = packed_cache(archive, row["quality"])
                    skin_features.append(model.skin_model.extract_features(packed, row["preprocessing_id"]+":strict"))
            components["skin_embeddings"] = np.stack(skin_features).astype(np.float32)
        np.savez_compressed(output/"components.npz", **components)
        write_json(output/"manifest.json", records)
        if (prior/"detector_only_components.npz").exists():
            with np.load(prior/"detector_only_components.npz", allow_pickle=False) as archive:
                np.savez_compressed(output/"detector_only_components.npz", **{key: archive[key] for key in archive.files})
            write_json(output/"detector_only_manifest.json", json.loads((prior/"detector_only_manifest.json").read_text(encoding="utf-8")))
        plan["reused_component_artifact"] = {"path": str(prior/"components.npz"), "sha256": sha256(prior/"components.npz")}
        write_json(output/"plan.json", plan)
    else:
        components, records = extract_fei(rows, model, output, args.gpu_reprocess)
    intended_probes = sum(row["split"] == "calibration" and row["role"] == "probe" for row in rows)
    candidates = []
    for weight in (*SKIN_WEIGHTS, 1.0):
        method = "global_only" if weight == 0 else "skin_only" if weight == 1 else f"hybrid_skin_{weight:g}"
        vectors = candidate_vectors(components, weight)
        metric = evaluate_calibration(vectors, components, intended_probes)
        model.skin_weight = weight
        feature_id = model.global_model.feature_space_id if weight == 0 else model.skin_model.feature_space_id if weight == 1 else model.feature_space_id
        candidate = {"method_id": method, "scope": "whole_face" if weight == 0 else "skin_only" if weight == 1 else "whole_face_plus_skin",
                     "skin_weight": weight, "feature_space_id": feature_id, "descriptor_dim": vectors.shape[1],
                     "score_definition": "negative_euclidean_distance", "score_threshold": metric["threshold"],
                     "search_score_threshold": metric["search_score_threshold"], "calibration_metrics": metric}
        candidates.append(candidate)
        if 0 < weight < 1:
            model.save(output/f"{method}_model.json")
        print(json.dumps({"method": method, "rank1": metric["rank1"], "eer": metric["eer"],
                          "far": metric["far"], "frr": metric["frr"]}), flush=True)
    if (output/"detector_only_components.npz").is_file():
        with np.load(output/"detector_only_components.npz", allow_pickle=False) as archive:
            detector_components = {key: archive[key] for key in archive.files}
        metric = evaluate_calibration(detector_components["embeddings"], detector_components, intended_probes)
        candidates.append({"method_id": "whole_face_detector_only", "scope": "whole_face_detector_only",
                           "skin_weight": 0.0, "feature_space_id": model.global_model.feature_space_id,
                           "descriptor_dim": 512, "preprocessing_id": model.skin_model.preprocessing_id.removesuffix(":strict")+":detector_only",
                           "score_definition": "negative_euclidean_distance", "score_threshold": metric["threshold"],
                           "search_score_threshold": metric["search_score_threshold"], "calibration_metrics": metric})
        print(json.dumps({"method": "whole_face_detector_only", "rank1": metric["rank1"], "eer": metric["eer"]}), flush=True)
    key = lambda item: (-item["calibration_metrics"]["rank1"], item["calibration_metrics"]["eer"],
                        item["calibration_metrics"]["frr"], item["skin_weight"])
    selected = min((item for item in candidates if item["skin_weight"] in SKIN_WEIGHTS and item["method_id"] != "whole_face_detector_only"), key=key)
    positive = min((item for item in candidates if 0 < item["skin_weight"] < 1), key=key)
    freeze = {**plan, "methods": candidates, "selected_method_id": selected["method_id"],
              "best_positive_hybrid_method_id": positive["method_id"],
              "accepted_images": int(len(components["sample_ids"])), "intended_images": len(records),
              "components_sha256": sha256(output/"components.npz"),
              "quality_manifest_sha256": sha256(output/"manifest.json"),
              "limitations": ["Global component uses the complete aligned face, not skin-only evidence.",
                              "FEI calibration selects weights and thresholds; metrics are development estimates.",
                              "Pretraining identity overlap with public evaluation datasets is not established.",
                              "Commercial use requires appropriate pretrained-weight licensing."]}
    write_json(output/"freeze.json", freeze)
    return freeze


def run_gt(args):
    # This is a previously observed diagnostic set, never labelled an unseen test.
    from experiments.external_evaluate_v3 import load_protocol, summarize_protocol
    from src.preprocessing_v3 import FacePreprocessorV3
    freeze_path = Path(args.fei_freeze).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "hybrid-fei-development-v4.0":
        raise ValueError("A completed FEI fusion freeze is required before GT diagnostics")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol, rows, by_path = load_protocol(ROOT/"research/data/georgia_tech/external_protocol_v1.json")
    quality = json.loads((ROOT/"research/results/georgia_tech_v3_primary/image_quality.json").read_text(encoding="utf-8"))
    quality = {row["path"]: row for row in quality}
    config = freeze["model_configuration"]
    model = HybridFaceSkinDescriptorV4(config["global_model_path"], config["skin_model_path"])
    detector = FacePreprocessorV3().detector
    global_features, skin_features, paths = [], [], []
    start = time.monotonic()
    for index, row in enumerate(rows):
        source = ROOT/row["path"]
        record = quality[row["path"]]
        if record["status"] != "accepted":
            raise ValueError("GT diagnostic input rejected by prior fixed preprocessing; define explicit coverage handling")
        if sha256(source) != row["sha256"]:
            raise ValueError("GT source checksum mismatch")
        image = cv2.imread(str(source))
        boxes, kps = detector.detect(image, max_num=0, metric="default")
        if len(boxes) != 1 or kps is None:
            raise ValueError("GT global detector disagrees with cached accepted cohort")
        global_features.append(model.global_model.extract_features(image, kps[0]))
        cache = path_from(record["cache_path"], ROOT/"research/results/georgia_tech_v3_primary")
        if sha256(cache) != record["cache_sha256"]:
            raise ValueError("GT cache checksum mismatch")
        with np.load(cache, allow_pickle=False) as archive:
            skin_features.append(model.skin_model.extract_features(packed_cache(archive, record["quality"]), record["preprocessing_id"]))
        paths.append(row["path"])
        if (index+1) % 50 == 0:
            print(json.dumps({"known_GT_processed": index+1, "total": len(rows), "seconds": round(time.monotonic()-start, 1)}), flush=True)
    components = {"global_embeddings": np.stack(global_features).astype(np.float32), "skin_embeddings": np.stack(skin_features).astype(np.float32), "paths": np.asarray(paths)}
    np.savez_compressed(output/"components.npz", **components)
    reports = []
    for method in freeze["methods"]:
        vectors = candidate_vectors(components, method["skin_weight"])
        features = dict(zip(paths, vectors))
        report, numeric, _ = summarize_protocol(protocol, by_path, features, method["score_threshold"], n_bootstrap=200)
        report.update(method_id=method["method_id"], scope=method["scope"],
                      evaluation_status="previously_observed_GT_diagnostic_not_independent_final_test")
        reports.append(report)
        np.savez_compressed(output/(method["method_id"]+"_scores.npz"), **numeric)
        print(json.dumps({"method": method["method_id"], "rank1": report["closed_set"]["rank1_operational"],
                          "eer": report["verification_conditional"]["eer"], "fpir": report["open_set"]["fpir_operational"]}), flush=True)
    result = {"source_FEI_freeze": str(freeze_path), "source_FEI_freeze_sha256": sha256(freeze_path),
              "status": "known_diagnostic_only_no_candidate_reselection", "methods": reports}
    write_json(output/"results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fei", "gt-diagnostic"), default="fei")
    parser.add_argument("--manifest", default=str(ROOT/"research/development/v3_faces_v31/manifest.json"))
    parser.add_argument("--global-model", default=str(DEFAULT_GLOBAL))
    parser.add_argument("--skin-model", default=str(DEFAULT_SKIN))
    parser.add_argument("--output", required=True)
    parser.add_argument("--fei-freeze")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--gpu-reprocess", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--reuse-components", help="Re-score matching earlier FEI component arrays using the final float32 contract")
    parser.add_argument("--reuse-global", help="Reuse matching global/GPU-prepared inputs, recomputing skin with a new train-fitted model")
    args = parser.parse_args()
    if args.gpu_reprocess and not args.gpu:
        parser.error("--gpu-reprocess requires --gpu")
    if args.gpu:
        import torch  # Load CUDA/cuDNN DLLs from the dedicated GPU environment first.
        import onnxruntime as ort
        ort.preload_dlls()
    if args.stage == "fei":
        run_fei(args)
    else:
        if not args.fei_freeze:
            parser.error("GT diagnostics require --fei-freeze")
        run_gt(args)


if __name__ == "__main__":
    main()
