"""Prepare v3 landmark/skin caches using development identities only.

The old v2 test partition is deliberately excluded, even when the source CSV
contains it. New external evaluation must use a separately frozen protocol.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing_v3 import FacePreprocessorV3, QualityRejection, PATCH_NAMES
from src.utils.data_loader import read_image


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")


def prepare(manifest_path, output, model_dir=None):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new empty directory")
    with manifest_path.open(encoding="utf-8-sig",newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [r for r in all_rows if r.get("split") in ("train","calibration")]
    if not rows or {r["split"] for r in rows}!={"train","calibration"}:
        raise ValueError("Development manifest needs train and calibration rows")
    train_ids={r["subject_id"] for r in rows if r["split"]=="train"}
    calibration_ids={r["subject_id"] for r in rows if r["split"]=="calibration"}
    if train_ids & calibration_ids:
        raise ValueError("Identity leakage between development training and calibration")
    output.mkdir(parents=True,exist_ok=True)
    (output/"cache").mkdir()
    started=time.perf_counter()
    write_json(output/"status.json",{"status":"running","started_utc":datetime.now(timezone.utc).isoformat()})
    processor=FacePreprocessorV3(model_dir=model_dir)
    contract=dict(version=processor.version,preprocessing_id=processor.preprocessing_id,
                  model_sha256=processor.model_sha256,runtime_providers=processor.runtime_providers,
                  source_manifest=str(manifest_path),source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  source_module_sha256=hashlib.sha256((ROOT/"src/preprocessing_v3.py").read_bytes()).hexdigest(),
                  development_splits=["train","calibration"],excluded_test_rows=sum(r.get("split")=="test" for r in all_rows),
                  old_test_used=False,source_images=len(rows),train_identities=len(train_ids),
                  calibration_identities=len(calibration_ids),patch_names=list(PATCH_NAMES),
                  color_order="BGR",mask_values=[0,1],
                  strict_deep_patch_keys=["skin_only_"+name for name in PATCH_NAMES],
                  mask_limitations="Anatomical and adaptive-appearance heuristics; not ground-truth skin/hair segmentation.",
                  model_license="InsightFace supplied pretrained models: non-commercial research only.",
                  model_license_source="https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md")
    write_json(output/"contract.json",contract)
    records,seen,previews,preview_subjects=[],set(),[],set()
    try:
        for index,row in enumerate(rows):
            source=Path(row["path"])
            if not source.is_absolute():source=(manifest_path.parent/source).resolve()
            sample_id=re.sub(r"[^A-Za-z0-9_.-]+","_",row["subject_id"]+"_"+source.stem)
            if sample_id in seen:raise ValueError("Duplicate sample ID: "+sample_id)
            seen.add(sample_id)
            record=dict(sample_id=sample_id,subject_id=row["subject_id"],split=row["split"],role=row["role"],
                        session_id=row.get("session_id","unknown"),condition=row.get("condition","unspecified"),
                        source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        preprocessing_id=processor.preprocessing_id,status="pending",cache_path=None)
            try:
                prepared=processor.process(read_image(source))
                cache=output/"cache"/(sample_id+".npz")
                prepared.save(cache)
                record.update(status="accepted",cache_path=str(cache),
                              cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                              quality=prepared.metadata,rejection_codes=[])
                if row["split"]=="train" and row["subject_id"] not in preview_subjects and len(previews)<8:
                    arrays=prepared.numeric_arrays()
                    cells=[cv2.resize(prepared.aligned_bgr,(128,128))]
                    for name in PATCH_NAMES:
                        patch=cv2.resize(arrays["skin_only_"+name],(128,128))
                        cv2.putText(patch,name,(3,14),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,255,255),1)
                        cells.append(patch)
                    previews.append(np.concatenate(cells,axis=1))
                    preview_subjects.add(row["subject_id"])
            except QualityRejection as error:
                record.update(status="rejected",rejection_codes=error.codes,quality=error.metadata)
            records.append(record)
            if (index+1)%20==0:
                write_json(output/"manifest.json",records)
                print(f"Prepared {index+1}/{len(rows)}; accepted={sum(r['status']=='accepted' for r in records)}",flush=True)
        write_json(output/"manifest.json",records)
        with (output/"manifest.jsonl").open("w",encoding="utf-8") as stream:
            for record in records:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
        if previews:cv2.imwrite(str(output/"development_skin_crops.jpg"),np.concatenate(previews,axis=0))
        reasons=Counter(code for r in records for code in r.get("rejection_codes",[]))
        summary=dict(status="completed",elapsed_seconds=time.perf_counter()-started,
                     images=len(records),accepted=sum(r["status"]=="accepted" for r in records),
                     rejected=sum(r["status"]=="rejected" for r in records),rejection_reasons=dict(reasons),
                     counts_by_split={split:dict(Counter(r["status"] for r in records if r["split"]==split))
                                      for split in ("train","calibration")},
                     old_test_used=False,preprocessing_id=processor.preprocessing_id)
        write_json(output/"status.json",summary)
        return summary
    except Exception as error:
        write_json(output/"manifest.json",records)
        write_json(output/"status.json",dict(status="failed",error_type=type(error).__name__,error=str(error),
                                             completed_images=len(records),elapsed_seconds=time.perf_counter()-started))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",default=str(ROOT/"data/fei/manifest_frontal_seed42.csv"))
    parser.add_argument("--output",default=str(ROOT/"research/development/v3_faces"))
    parser.add_argument("--model-dir")
    args=parser.parse_args()
    print(json.dumps(prepare(args.manifest,args.output,args.model_dir),indent=2))

