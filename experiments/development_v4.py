"""Multi-condition FEI development for v4; former test identities stay excluded.

The preprocessing implementation is imported unchanged from v3.1. New image
conditions expand training/calibration, while public external holdouts are never
opened by this program. Two preparation workers write disjoint cache paths.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INDICES = (3, 4, 5, 6, 7, 8, 11, 12, 13, 14)
CONDITIONS = {
    3: ("pose_variant_unspecified", "image_index_only"),
    4: ("pose_variant_unspecified", "image_index_only"),
    5: ("approximately_10_degrees_right", "acquisition_report_figure5_explicit"),
    6: ("approximately_10_degrees_left", "acquisition_report_figure5_explicit"),
    7: ("pose_variant_unspecified", "image_index_only"),
    8: ("pose_variant_unspecified", "image_index_only"),
    11: ("neutral_frontal", "acquisition_report_figure5_explicit"),
    12: ("smiling_frontal", "acquisition_report_figure5_explicit"),
    13: ("illumination_variant_unquantified", "inferred_from_acquisition_figure4"),
    14: ("illumination_variant_unquantified", "inferred_from_acquisition_figure4"),
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def make_rows(source_manifest):
    source_manifest = Path(source_manifest).resolve()
    with source_manifest.open(encoding="utf-8-sig", newline="") as handle:
        original = list(csv.DictReader(handle))
    identities = {}
    for row in original:
        if row["split"] not in {"train", "calibration"}:
            continue
        if row["subject_id"] in identities and identities[row["subject_id"]]["split"] != row["split"]:
            raise ValueError("Development identity leakage")
        source = Path(row["path"])
        if not source.is_absolute():
            source = (source_manifest.parent/source).resolve()
        identities[row["subject_id"]] = {"split": row["split"], "parent": source.parent,
                                         "original_subject_number": source.stem.split("-")[0]}
    result = []
    for identity, details in sorted(identities.items()):
        for index in INDICES:
            path = details["parent"]/f"{details['original_subject_number']}-{index:02d}.jpg"
            if not path.is_file():
                raise ValueError(f"Missing prescribed image: {path}")
            condition, evidence = CONDITIONS[index]
            result.append({
                "sample_id": f"{identity}_{path.stem}", "subject_id": identity,
                "split": details["split"],
                "role": "train" if details["split"] == "train" else "gallery" if index == 11 else "probe",
                "session_id": "unknown", "condition_index": index, "condition": condition,
                "condition_evidence": evidence, "source_path": str(path.resolve()),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    if len(result) != 1600 or Counter(r["split"] for r in result) != Counter(train=1200, calibration=400):
        raise ValueError("Expected original FEI development split: 120 training +40 calibration identities, ten images each")
    return result


def worker(output, worker_index, workers, model_dir=None, resume=False):
    from src.preprocessing_v3 import FacePreprocessorV3, QualityRejection
    from src.utils.data_loader import read_image
    output = Path(output).resolve()
    contract = json.loads((output/"contract.json").read_text(encoding="utf-8"))
    current_source = hashlib.sha256((ROOT/"src/preprocessing_v3.py").read_bytes()).hexdigest()
    if current_source != contract["preprocessing_source_sha256"]:
        raise ValueError("Frozen v3.1 preprocessor source has changed")
    rows = json.loads((output/"source_manifest.json").read_text(encoding="utf-8"))
    assigned = rows[worker_index::workers]
    previous = json.loads(Path(contract["reused_manifest"]).read_text(encoding="utf-8"))
    reused = {str(Path(row["source_path"]).resolve()): row for row in previous if row["split"] in {"train", "calibration"}}
    processor = FacePreprocessorV3(model_dir=model_dir)
    if processor.preprocessing_id != contract["preprocessing_id"]:
        raise ValueError("Preprocessing/model contract mismatch")
    cache = output/"cache"/f"worker_{worker_index}"
    cache.mkdir(parents=True, exist_ok=resume)
    manifest_path = output/f"worker_{worker_index}.json"
    status_path = output/f"worker_{worker_index}_status.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8")) if resume and manifest_path.exists() else []
    if [r["sample_id"] for r in records] != [r["sample_id"] for r in assigned[:len(records)]]:
        raise ValueError("Resume checkpoint does not match assigned source manifest")
    for record in records:
        if record["status"] == "accepted" and not Path(record["cache_path"]).is_file():
            raise ValueError("A committed cache artifact is missing")
    started = time.perf_counter()
    write_json(status_path, {"status": "running", "assigned_images": len(assigned)})
    try:
        for index in range(len(records), len(assigned)):
            source = assigned[index]
            if source["split"] not in {"train", "calibration"}:
                raise AssertionError("Forbidden test image reached preparation")
            record = dict(source, preprocessing_id=processor.preprocessing_id, status="pending", cache_path=None)
            old = reused.get(source["source_path"])
            if old and old["status"] == "accepted" and old["preprocessing_id"] == processor.preprocessing_id:
                if old["source_sha256"] != source["source_sha256"]:
                    raise ValueError("Reused source image hash mismatch")
                old_cache = Path(old["cache_path"])
                if hashlib.sha256(old_cache.read_bytes()).hexdigest() != old["cache_sha256"]:
                    raise ValueError("Reused prepared cache hash mismatch")
                record.update(status="accepted", cache_path=str(old_cache), cache_sha256=old["cache_sha256"],
                              quality=old["quality"], rejection_codes=[], cache_reused=True)
            else:
                try:
                    prepared = processor.process(read_image(source["source_path"]))
                    cache_path = cache/(source["sample_id"]+".npz")
                    prepared.save(cache_path)
                    record.update(status="accepted", cache_path=str(cache_path),
                                  cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                                  quality=prepared.metadata, rejection_codes=[], cache_reused=False)
                except QualityRejection as exc:
                    record.update(status="rejected", rejection_codes=exc.codes, quality=exc.metadata, cache_reused=False)
            records.append(record)
            if (index+1)%20 == 0:
                write_json(manifest_path, records)
                write_json(status_path, {"status": "running", "processed": len(records), "assigned_images": len(assigned),
                                         "accepted": sum(r["status"]=="accepted" for r in records),
                                         "elapsed_seconds": time.perf_counter()-started})
                print(f"worker{worker_index}: {index+1}/{len(assigned)} accepted={sum(r['status']=='accepted' for r in records)}", flush=True)
        write_json(manifest_path, records)
        write_json(status_path, {"status": "completed", "processed": len(records), "assigned_images": len(assigned),
                                 "accepted": sum(r["status"]=="accepted" for r in records),
                                 "elapsed_seconds": time.perf_counter()-started})
    except Exception as exc:
        write_json(manifest_path, records)
        write_json(status_path, {"status": "failed", "error": str(exc), "processed": len(records)})
        raise


def prepare(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Use a new empty v4 preparation directory")
    output.mkdir(parents=True, exist_ok=True)
    rows = (json.loads((output/"source_manifest.json").read_text(encoding="utf-8")) if args.resume
            else make_rows(args.source_manifest))
    reused_manifest = Path(args.reused_manifest).resolve()
    old_contract = json.loads((reused_manifest.parent/"contract.json").read_text(encoding="utf-8"))
    source_hash = hashlib.sha256((ROOT/"src/preprocessing_v3.py").read_bytes()).hexdigest()
    if old_contract["source_module_sha256"] != source_hash:
        raise ValueError("The v3.1 preprocessing source changed after its cache contract")
    if not args.resume:
        write_json(output/"source_manifest.json", rows)
        with (output/"source_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer=csv.writer(handle)
            writer.writerow(["path","subject_id","session_id","split","role","condition","condition_evidence"])
            for row in rows:
                writer.writerow([row["source_path"],row["subject_id"],row["session_id"],row["split"],row["role"],
                                 row["condition"],row["condition_evidence"]])
    contract = {
        "version": "v4_multicondition_development_preparation_1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing_id": old_contract["preprocessing_id"], "preprocessing_source_sha256": source_hash,
        "model_sha256": old_contract["model_sha256"], "reused_manifest": str(reused_manifest),
        "source_identity_manifest": str(Path(args.source_manifest).resolve()),
        "source_identity_manifest_sha256": hashlib.sha256(Path(args.source_manifest).read_bytes()).hexdigest(),
        "indices": list(INDICES), "conditions": CONDITIONS, "source_images": len(rows),
        "training_images": 1200, "calibration_images": 400, "train_identities": 120, "calibration_identities": 40,
        "gallery_index": 11, "probe_indices": [index for index in INDICES if index != 11],
        "test_identities_used": False, "external_images_used": False, "session_metadata": "unknown",
        "workers": args.workers, "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.resume:
        contract = json.loads((output/"contract.json").read_text(encoding="utf-8"))
        if contract["workers"] != args.workers:
            raise ValueError("Resumption must preserve original worker assignments")
        write_json(output/"resume_record.json", {"resumed_utc":datetime.now(timezone.utc).isoformat(),
                    "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "reason":"worker processes disappeared; resume from persisted completed prefix"})
    else:
        write_json(output/"contract.json", contract)
    write_json(output/"status.json", {"status": "running", "images": len(rows), "workers": args.workers})
    started = time.perf_counter()
    children = []
    for index in range(args.workers):
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
                   "--worker-index", str(index), "--workers", str(args.workers)]
        if args.model_dir:
            command += ["--model-dir", args.model_dir]
        if args.resume:
            command += ["--resume"]
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "4"
        environment["OPENBLAS_NUM_THREADS"] = "4"
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        children.append(subprocess.Popen(command, env=environment, creationflags=flags))
    codes = [child.wait() for child in children]
    if any(code != 0 for code in codes):
        write_json(output/"status.json", {"status": "failed", "worker_exit_codes": codes})
        raise RuntimeError(f"Preparation worker failed: {codes}")
    all_records = []
    for index in range(args.workers):
        all_records.extend(json.loads((output/f"worker_{index}.json").read_text(encoding="utf-8")))
    order = {row["sample_id"]: i for i,row in enumerate(rows)}
    all_records.sort(key=lambda row:order[row["sample_id"]])
    if len(all_records) != len(rows) or len({r["sample_id"] for r in all_records}) != len(rows):
        raise ValueError("Missing or duplicate worker outputs")
    write_json(output/"manifest.json", all_records)
    with (output/"manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    status = {
        "status": "completed", "images": len(all_records),
        "accepted": sum(row["status"]=="accepted" for row in all_records),
        "rejected": sum(row["status"]=="rejected" for row in all_records),
        "reused_images": sum(row.get("cache_reused",False) for row in all_records),
        "counts_by_split": {split: dict(Counter(row["status"] for row in all_records if row["split"]==split))
                            for split in ("train","calibration")},
        "counts_by_condition": {str(index): dict(Counter(row["status"] for row in all_records if row["condition_index"]==index))
                                for index in INDICES},
        "rejection_reasons": dict(Counter(reason for row in all_records for reason in row.get("rejection_codes",[]))),
        "elapsed_seconds": time.perf_counter()-started, "preprocessing_id": contract["preprocessing_id"],
        "old_test_used": False, "external_data_used": False,
    }
    write_json(output/"status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return status


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    actions=result.add_subparsers(dest="action",required=True)
    prep=actions.add_parser("prepare")
    prep.add_argument("--source-manifest",default=str(ROOT/"data/fei/manifest_frontal_seed42.csv"))
    prep.add_argument("--reused-manifest",default=str(ROOT/"research/development/v3_faces_v31/manifest.json"))
    prep.add_argument("--output",required=True)
    prep.add_argument("--workers",type=int,choices=(1,2),default=2)
    prep.add_argument("--model-dir")
    prep.add_argument("--resume",action="store_true")
    job=actions.add_parser("worker")
    job.add_argument("--output",required=True)
    job.add_argument("--worker-index",type=int,required=True)
    job.add_argument("--workers",type=int,required=True)
    job.add_argument("--model-dir")
    job.add_argument("--resume",action="store_true")
    return result


if __name__=="__main__":
    arguments=parser().parse_args()
    if arguments.action=="prepare":
        prepare(arguments)
    else:
        worker(arguments.output,arguments.worker_index,arguments.workers,arguments.model_dir,arguments.resume)

