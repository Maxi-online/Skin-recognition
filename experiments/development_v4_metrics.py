"""Finite 16-configuration pure-texture development on multi-condition FEI.

Training/calibration identities only. No former test, Georgia Tech, LFW, or ORL
pixels or embeddings are read by this program.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from experiments.development_v3 import (
    ClassicalTextureDescriptorV3, FeatureConfig, config_id, extract_features,
    fit_pca_family, normalize_rows, within_identity_whitener,
)
from experiments.scientific_metrics import cross_session_pairs, select_threshold_at_far, verification_from_scores
from experiments.calibrate_search_v3 import conservative_cutoff


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def prepared_input(record):
    if record["split"] not in {"train","calibration"}:
        raise ValueError("Forbidden identity split")
    with np.load(record["cache_path"],allow_pickle=False) as archive:
        names=[str(n) for n in archive["patch_names"]]
        patches=np.stack([archive["skin_only_"+name] for name in names])
        quality=record["quality"]["patch_quality"]
        masks=np.stack([np.full(patch.shape[:2],bool(quality.get(name,{}).get("strict_skin_usable",False)),np.uint8)
                        for name,patch in zip(names,patches)])
        return archive["aligned_bgr"].copy(),patches,masks,archive["skin_mask"].copy()


def measure(features,rows,source_records,target_far=.01):
    gallery=np.asarray([i for i,r in enumerate(rows) if r["split"]=="calibration" and r["role"]=="gallery"])
    probes=np.asarray([i for i,r in enumerate(rows) if r["split"]=="calibration" and r["role"]=="probe"])
    gallery_ids=np.asarray([rows[i]["subject_id"] for i in gallery])
    probe_ids=np.asarray([rows[i]["subject_id"] for i in probes])
    y,scores,pair_probe,pair_gallery,matrix=cross_session_pairs(features[gallery],gallery_ids,features[probes],probe_ids)
    threshold=select_threshold_at_far(y,scores,target_far)
    metric=verification_from_scores(y,scores,threshold,"development_calibration_tuned")
    ranks=[]
    for row,identity in zip(matrix,probe_ids):
        same=gallery_ids==identity
        ranks.append(1+int(np.sum(row[~same]>=row[same].max())) if same.any() else 0)
    ranks=np.asarray(ranks)
    intended=[r for r in source_records if r["split"]=="calibration" and r["role"]=="probe"]
    successes=(ranks>0)&(ranks<=1)
    conservative=conservative_cutoff(y,scores)
    conservative_metric=verification_from_scores(y,scores,conservative,"development_calibration_max_impostor")
    conditions={}
    for condition in sorted({r["condition_index"] for r in intended}):
        denominator=sum(r["condition_index"]==condition for r in intended)
        mask=np.asarray([rows[i]["condition_index"]==condition for i in probes])
        conditions[str(condition)]={"intended_probes":denominator,"processed_probes":int(mask.sum()),
                                     "rank1_correct":int(successes[mask].sum()),
                                     "rank1_operational":float(successes[mask].sum()/denominator)}
    result={**asdict(metric),"rank1":float(successes.sum()/len(intended)),
            "rank1_operational":float(successes.sum()/len(intended)),
            "rank1_conditional":float(successes[ranks>0].mean()) if np.any(ranks>0) else None,
            "rank5_operational":float(np.sum((ranks>0)&(ranks<=5))/len(intended)),
            "intended_probe_count":len(intended),"successful_probe_count":len(probes),
            "successful_gallery_count":len(gallery),"rank1_correct":int(successes.sum()),
            "quality_rejected_probes":len(intended)-len(probes),"by_condition":conditions,
            "conservative_search":asdict(conservative_metric)}
    return result,{"scores":matrix,"gallery_ids":gallery_ids,"probe_ids":probe_ids,"ranks":ranks,
                   "gallery_indices":gallery,"probe_indices":probes}


def run(args):
    cache_dir=Path(args.prepared_dir).resolve();output=Path(args.output).resolve()
    if json.loads((cache_dir/"status.json").read_text())["status"]!="completed":
        raise ValueError("Preparation must be completed")
    if output.exists() and any(output.iterdir()):raise ValueError("Output must be new and empty")
    output.mkdir(parents=True,exist_ok=True);(output/"features").mkdir()
    started=time.perf_counter();write_json(output/"status.json",{"status":"running","stage":"development_only"})
    records=json.loads((cache_dir/"manifest.json").read_text(encoding="utf-8"))
    if any(r["split"] not in {"train","calibration"} for r in records):raise ValueError("Forbidden test data")
    rows=[r for r in records if r["status"]=="accepted"]
    train=np.asarray([i for i,r in enumerate(rows) if r["split"]=="train"])
    train_ids=np.asarray([rows[i]["subject_id"] for i in train])
    contract=json.loads((cache_dir/"contract.json").read_text())
    configuration={"descriptor":"rootsift","scope":"skin_regions","photometric":"raw","grid":4}
    feature_id=config_id(configuration)
    run_config={
        "stage":"multi_condition_FEI_development_only","max_pca_components":256,"seed":42,
        "prepared_dir":str(cache_dir),"prepared_manifest_sha256":hashlib.sha256((cache_dir/"manifest.json").read_bytes()).hexdigest(),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preprocessing_id":contract["preprocessing_id"]+":strict","configuration_count":16,
        "selection_rule":"highest operational calibration rank1, then lowest conditional calibration EER",
        "training_images":len(train),"training_identities":len(np.unique(train_ids)),
        "calibration_images":sum(r["split"]=="calibration" for r in rows),
        "calibration_identities":len({r["subject_id"] for r in rows if r["split"]=="calibration"}),
        "expected_calibration_probes":360,"previous_test_or_external_data_used":False,
    }
    write_json(output/"run_config.json",run_config)
    provenance=[{"path":r["source_path"],"subject_id":r["subject_id"],"split":r["split"],
                 "preprocessing_id":contract["preprocessing_id"]+":strict","source_sha256":r["source_sha256"],
                 "cache_path":r["cache_path"],"cache_sha256":r["cache_sha256"]} for r in rows]
    write_json(output/"provenance.json",provenance)
    features=[]
    feature_started=time.perf_counter()
    for i,row in enumerate(rows):
        features.append(extract_features(prepared_input(row),FeatureConfig(**configuration)))
        if (i+1)%200==0:print(f"Raw strict RootSIFT {i+1}/{len(rows)}",flush=True)
    raw=np.stack(features);del features
    extraction_seconds=time.perf_counter()-feature_started
    feature_path=output/"features"/(feature_id+".npz")
    np.savez_compressed(feature_path,features=raw,subject_ids=np.asarray([r["subject_id"] for r in rows]),
                        splits=np.asarray([r["split"] for r in rows]),roles=np.asarray([r["role"] for r in rows]),
                        paths=np.asarray([r["source_path"] for r in rows]),
                        condition_indices=np.asarray([r["condition_index"] for r in rows]))
    all_results=[]
    def record(transformed,transform_config):
        full={"features":configuration,"transform":transform_config}
        metrics,_=measure(transformed,rows,records)
        item={"id":config_id(full),"feature_id":feature_id,"configuration":full,
              "dimension":int(transformed.shape[1]),"calibration":metrics,
              "feature_extraction_seconds":extraction_seconds,
              "engineering_target_met_on_development_only":metrics["rank1"]>=.95 and metrics["eer"]<=.05}
        all_results.append(item)
        write_json(output/"partial_results.json",{"stage":"development_only","results":all_results})
        print("DEV",item["id"],json.dumps(transform_config),"rank1_operational",round(metrics["rank1"],4),
              "EER",round(metrics["eer"],4),"FAR",round(metrics["far"],4),"FRR",round(metrics["frr"],4),flush=True)
    record(raw,{"kind":"identity"})
    pca_started=time.perf_counter()
    with threadpool_limits(limits=8):
        model,projected=fit_pca_family(raw[train],raw,256,42)
    np.savez_compressed(output/"features"/(feature_id+"_pca.npz"),mean=model.mean_,components=model.components_,
                        explained_variance=model.explained_variance_)
    print("PCA fit complete",round(time.perf_counter()-pca_started,2),"seconds",flush=True)
    for dimension in (64,128,256):
        for power in (0.0,0.5,1.0):
            scale=np.maximum(model.explained_variance_[:dimension],model.explained_variance_[0]*1e-3)**(-power/2)
            record(normalize_rows(projected[:,:dimension]*scale),
                   {"kind":"pca","dimension":dimension,"whitening_power":power,"variance_floor_fraction":.001})
        for power in (.5,1.0):
            with threadpool_limits(limits=8):
                transform=within_identity_whitener(projected[train,:dimension],train_ids,power)
            np.savez_compressed(output/"features"/(feature_id+f"_within_{dimension}_{power}.npz"),transform=transform)
            record(normalize_rows(projected[:,:dimension]@transform),
                   {"kind":"pca_within_identity_whitening","dimension":dimension,"power":power,
                    "shrinkage":"LedoitWolf_train_only"})
    ranked=sorted(all_results,key=lambda r:(-r["calibration"]["rank1"],r["calibration"]["eer"]))
    result={"stage":"development_only_not_external_validation","tested_configurations":len(ranked),
            "best":ranked[0],"best_skin":ranked[0],"results":ranked,"failures":[],
            "elapsed_seconds":time.perf_counter()-started,"training_images":len(train),
            "calibration_intended_images":400,"calibration_successful_images":sum(r["split"]=="calibration" for r in rows),
            "quality_rejections":{split:[{"sample_id":r["sample_id"],"rejection_codes":r["rejection_codes"]}
                                        for r in records if r["split"]==split and r["status"]!="accepted"]
                                  for split in ("train","calibration")}}
    write_json(output/"results.json",result)
    descriptor=ClassicalTextureDescriptorV3.from_development(output,ranked[0]["id"])
    artifact=output/"models"/"pure_skin_multicondition_best.npz";descriptor.save(artifact)
    transformed=descriptor.transform_raw(raw)
    replay,numeric=measure(transformed,rows,records)
    if replay["rank1"]!=ranked[0]["calibration"]["rank1"]:
        raise ValueError("Saved model replay rank1 changed")
    np.savez_compressed(output/"selected_embeddings.npz",embeddings=transformed,
                        paths=np.asarray([r["source_path"] for r in rows]),
                        identities=np.asarray([r["subject_id"] for r in rows]),
                        splits=np.asarray([r["split"] for r in rows]),roles=np.asarray([r["role"] for r in rows]),
                        condition_indices=np.asarray([r["condition_index"] for r in rows]))
    np.savez_compressed(output/"calibration_scores.npz",**numeric)
    manifest={
        "stage":"FEI_multicondition_development_candidate","created_utc":datetime.now(timezone.utc).isoformat(),
        "model_path":str(artifact),"model_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "feature_space_id":descriptor.feature_space_id,"preprocessing_id":descriptor.preprocessing_id,
        "candidate_id":ranked[0]["id"],"configuration":ranked[0]["configuration"],
        "score_definition":"negative_euclidean_distance","score_threshold":replay["threshold"],
        "conservative_search_threshold":replay["conservative_search"]["threshold"],
        "calibration_metrics":replay,"training_images":len(train),
        "training_identities":len(np.unique(train_ids)),"calibration_intended_images":400,
        "calibration_successful_images":sum(r["split"]=="calibration" for r in rows),
        "calibration_identity_count":40,"external_data_used":False,"v2_test_identities_used":False,
        "source_sha256":{"experiments/development_v4_metrics.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "experiments/development_v3.py":hashlib.sha256((ROOT/"experiments/development_v3.py").read_bytes()).hexdigest(),
                         "src/preprocessing_v3.py":hashlib.sha256((ROOT/"src/preprocessing_v3.py").read_bytes()).hexdigest()},
        "results_sha256":hashlib.sha256((output/"results.json").read_bytes()).hexdigest(),
        "prepared_manifest_sha256":run_config["prepared_manifest_sha256"],
        "selection_bias_note":"Finite16grid selected on FEI calibration; final unseen evaluation is still required.",
    }
    write_json(output/"candidate_manifest.json",manifest)
    write_json(output/"status.json",{"status":"completed","elapsed_seconds":time.perf_counter()-started,
                                    "best_candidate":ranked[0]["id"]})
    print("V4 PURE SKIN COMPLETE",json.dumps(manifest),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir",default=str(ROOT/"research/development/v4_faces"))
    parser.add_argument("--output",required=True)
    run(parser.parse_args())

