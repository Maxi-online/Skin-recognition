"""Acquire/hash external LFW and AT&T/ORL; never run a recognition model.

LFW bytes must match SHA256 values published by scikit-learn's official loader.
Original archives, source snapshots, full-resolution images and official pairs
are preserved. Neither extraction nor auditing selects images by model success.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image
from sklearn.datasets import _lfw

ROOT = Path(__file__).resolve().parents[2]
STAMP = "2026-11-14"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def download(url, path, expected_sha256=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + ".download")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Skin-recognition-research-acquisition/1.0"}), timeout=120) as source:
            with temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        if expected_sha256 and sha256(temporary) != expected_sha256:
            raise ValueError("Downloaded SHA256 differs from official loader: " + url)
        temporary.rename(path)
    digest = sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Existing file differs from official loader SHA256: " + str(path))
    return {"url": url, "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest}


def destination(root, name):
    member = PurePosixPath(name.replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("Unsafe archive member: " + name)
    path = (root / Path(*member.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Archive member escapes destination")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def extract_zip(path, root):
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip():
            raise ValueError("Corrupt ZIP archive")
        for item in bundle.infolist():
            if item.is_dir():
                continue
            target = destination(root, item.filename)
            if target.exists():
                if target.stat().st_size != item.file_size:
                    raise ValueError("Existing extraction differs: " + str(target))
                continue
            with bundle.open(item) as source, target.open("xb") as stream:
                shutil.copyfileobj(source, stream)


def extract_tar(path, root):
    with tarfile.open(path, "r:gz") as bundle:
        for item in bundle:
            if item.isdir():
                continue
            if not item.isfile():
                raise ValueError("Nonregular TAR member rejected: " + item.name)
            target = destination(root, item.name)
            if target.exists():
                if target.stat().st_size != item.size:
                    raise ValueError("Existing extraction differs: " + str(target))
                continue
            with bundle.extractfile(item) as source, target.open("xb") as stream:
                shutil.copyfileobj(source, stream)


def audit(dataset, originals, suffix, out):
    rows, failures, hashes = [], [], defaultdict(list)
    for index, path in enumerate(sorted(originals.rglob("*" + suffix))):
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
            digest = sha256(path)
            relative = path.relative_to(ROOT).as_posix()
            hashes[digest].append(relative)
            rows.append({"path": relative, "subject_id": path.parent.name, "image_id": path.stem,
                         "session_id": "unknown", "width": width, "height": height, "mode": mode, "sha256": digest})
        except Exception as error:
            failures.append({"path": str(path), "error": str(error)})
        if (index + 1) % 2000 == 0:
            print(f"{dataset}: structurally read {index + 1} images; no model evaluation", flush=True)
    with (out / "inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "subject_id", "image_id", "session_id", "width", "height", "mode", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    overlap = {}
    for name, file, hash_column in [("FEI", ROOT / "research/data/fei_inventory.csv", "file_sha256"),
                                    ("GeorgiaTech", ROOT / "research/data/georgia_tech/inventory.csv", "sha256")]:
        if file.exists():
            with file.open(encoding="utf-8", newline="") as stream:
                other = {row[hash_column] for row in csv.DictReader(stream)}
            overlap[name] = [row["path"] for row in rows if row["sha256"] in other]
    identities = Counter(row["subject_id"] for row in rows)
    result = {"images": len(rows), "identities": len(identities),
              "images_per_identity_histogram": dict(Counter(identities.values())),
              "image_formats": dict(Counter(f"{row['width']}x{row['height']}:{row['mode']}" for row in rows)),
              "errors": failures, "exact_file_duplicates": [paths for paths in hashes.values() if len(paths) > 1],
              "exact_file_overlap": overlap,
              "inventory_sha256": sha256(out / "inventory.csv"),
              "evaluation_status": "No detector, alignment, feature extractor, recognition model, recognition score or visual identity matching was used.",
              "non_overlap_limitation": "Exact-file hashing cannot prove no real-person overlap, re-encoded duplicates or absence from pretrained-model training data."}
    return rows, result


def freeze(out, name, protocol):
    path = out / name
    if path.exists():
        raise FileExistsError("Refusing to overwrite the frozen external protocol: " + str(path))
    save(path, protocol)
    save(out / (path.stem + ".sha256.json"), {path.name: sha256(path), "inventory.csv": sha256(out / "inventory.csv")})
    return sha256(path)


def common_protocol(dataset, out):
    return {"dataset": dataset, "created_utc": datetime.now(timezone.utc).isoformat(),
            "status_at_freeze": "Acquired and structurally audited only; no model evaluation or feature extraction performed.",
            "inventory_path": (out / "inventory.csv").relative_to(ROOT).as_posix(), "inventory_sha256": sha256(out / "inventory.csv"),
            "session_provenance": "session_id=unknown. File index is not a documented session label. No cross-session guarantee is made.",
            "freeze_gate": "Freeze v4 model weights, preprocessing, scope (skin-only/global/hybrid), fusion rule, calibration thresholds, software hashes and all eligible methods before the first external feature extraction. No method/hyperparameter/fusion selection on these external results.",
            "pretrained_models": "Use of externally trained off-the-shelf models is an unrestricted/external-training setting. Identity overlap with pretraining cannot be excluded without an auditable training manifest. Disjoint project splits do not prove pretrained identity disjointness.",
            "failure_policy": "Retain every scheduled input in operational denominators. Quality failures and failed pair endpoints are explicit rejection/failure, with conditional metrics and exact coverage separately reported. Never silently discard harder inputs.",
            "claims": "General low-resolution facial biometrics control; neither resolved pores nor measured depth is established."}


def orl():
    data, out = ROOT / "data/orl", ROOT / "research/data/orl"
    out.mkdir(parents=True, exist_ok=True)
    source = download("https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html", out / "official_source.html")
    archive = download("https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip", data / "archives/att_faces.zip")
    extract_zip(data / "archives/att_faces.zip", data / "originals")
    rows, quality = audit("ORL", data / "originals", ".pgm", out)
    if quality["images"] != 400 or quality["identities"] != 40 or quality["errors"]:
        raise ValueError("Unexpected ORL inventory")
    if quality["image_formats"] != {"92x112:L": 400}:
        raise ValueError("Unexpected ORL image dimensions/mode")
    subjects = sorted({row["subject_id"] for row in rows})
    if any({row["image_id"] for row in rows if row["subject_id"] == subject} != {str(i) for i in range(1, 11)} for subject in subjects):
        raise ValueError("ORL must contain image numbers 1 through 10 for every identity")
    acquisition = {"dataset": "AT&T Database of Faces, formerly ORL", "checked_on": STAMP,
                   "source": source, "archive": archive, "audit": quality,
                   "license": "Official source requests credit to AT&T Laboratories Cambridge; no standard open license or explicit commercial/redistribution grant was found. Do not infer CC/MIT/BSD coverage for image data.",
                   "session_provenance": "Source says some subjects photographed at different times during April1992-April1994, but no per-image session map is supplied."}
    save(out / "acquisition.json", acquisition)
    p = common_protocol("AT&T/ORL", out)
    p["protocol_id"] = "orl_external_v4_1"
    p["protocol_origin"] = "Locally predefined evaluation; not claimed to be an official AT&T protocol. Source does not prescribe one train/test split."
    p["input_semantics"] = "Use supplied full 92x112 grayscale PGM face crops, convert gray to BGR if the frozen model requires it; any resizing/detection/alignment must be declared in the model freeze before evaluation. Do not use quality-guided image selection."
    gallery = [row["path"] for row in rows if row["image_id"] == "1"]
    probes = [row["path"] for row in rows if row["image_id"] != "1"]
    perm = np.random.default_rng(42).permutation(subjects).tolist()
    enrolled, unknown = perm[:20], perm[20:]
    p["closed_set"] = {"identities": subjects, "gallery_paths": gallery, "probe_paths": probes,
                       "gallery_image_id": "1", "probe_image_ids": [str(i) for i in range(2, 11)],
                       "expected_counts": {"gallery": 40, "probes": 360, "genuine_pairs": 360, "impostor_pairs": 14040},
                       "metrics": ["rank1", "rank5", "ROC_AUC", "descriptive_EER", "FAR_FRR_at_frozen_development_threshold"],
                       "tie_rule": "Conservative rank counts all tied impostor templates ahead of the genuine template."}
    p["open_set"] = {"assignment": "lexicographically sorted subject IDs, numpy.random.default_rng(42).permutation; first20 enrolled, last20 unknown",
                     "permutation": perm, "enrolled_identities": enrolled, "unknown_identities": unknown,
                     "gallery_paths": [r["path"] for r in rows if r["subject_id"] in enrolled and r["image_id"] == "1"],
                     "known_probe_paths": [r["path"] for r in rows if r["subject_id"] in enrolled and r["image_id"] != "1"],
                     "unknown_probe_paths": [r["path"] for r in rows if r["subject_id"] in unknown and r["image_id"] != "1"],
                     "expected_counts": {"gallery": 20, "known_probes": 180, "unknown_probes": 180},
                     "metrics": "FPIR (unknown accepted /180), FNIR (known not correctly accepted /180) at frozen development threshold; reject tied maxima; report all failures explicitly."}
    p["uncertainty"] = "1000 identity-resampling bootstrap replicates, seed42, conditional on frozen model/threshold. Probe-rate intervals condition on fixed gallery. Report small-sample limitations and all input denominators."
    protocol_sha = freeze(out, "external_protocol_v4_1.json", p)
    print(json.dumps({"dataset": "ORL", "images": 400, "identities": 40, "protocol_sha256": protocol_sha, "inventory_sha256": quality["inventory_sha256"], "no_model_evaluation": True}), flush=True)


def lfw():
    data, out = ROOT / "data/lfw", ROOT / "research/data/lfw"
    out.mkdir(parents=True, exist_ok=True)
    loader_path = Path(_lfw.__file__)
    snapshot = out / "sklearn_lfw_loader_snapshot.py"
    if not snapshot.exists():
        snapshot.write_bytes(loader_path.read_bytes())
    archive_meta = _lfw.ARCHIVE
    # Figshare's public file API avoids stale expiring CDN redirects. The
    # canonical scikit-learn URL and exact authoritative checksum remain logged.
    file_id = archive_meta.url.rstrip("/").split("/")[-1]
    archive = download("https://api.figshare.com/v2/file/download/" + file_id,
                       data / "archives" / archive_meta.filename, archive_meta.checksum)
    archive["canonical_loader_url"] = archive_meta.url
    archive["expected_sha256_from_official_loader"] = archive_meta.checksum
    targets = []
    for metadata in _lfw.TARGETS:
        target = download("https://api.figshare.com/v2/file/download/" + metadata.url.rstrip("/").split("/")[-1],
                          out / metadata.filename, metadata.checksum)
        target.update(canonical_loader_url=metadata.url, expected_sha256_from_official_loader=metadata.checksum)
        targets.append(target)
    extract_tar(data / "archives" / archive_meta.filename, data / "originals")
    rows, quality = audit("LFW", data / "originals", ".jpg", out)
    if quality["images"] != 13233 or quality["identities"] != 5749 or quality["errors"]:
        raise ValueError("Unexpected LFW inventory")
    if any((row["width"], row["height"]) != (250, 250) for row in rows):
        raise ValueError("LFW original image must remain250x250")
    by_subject_image = {(row["subject_id"], int(row["image_id"].rsplit("_", 1)[-1])): row["path"] for row in rows}
    lines = (out / "pairs.txt").read_text(encoding="utf-8").splitlines()
    if [int(x) for x in lines[0].split()] != [10, 300] or len(lines[1:]) != 6000:
        raise ValueError("Unexpected official LFW pair header/count")
    pairs = []
    for index, line in enumerate(lines[1:]):
        fields = line.split()
        if len(fields) == 3:
            first, second = (fields[0], int(fields[1])), (fields[0], int(fields[2]))
            same = 1
        elif len(fields) == 4:
            first, second = (fields[0], int(fields[1])), (fields[2], int(fields[3]))
            same = 0
        else:
            raise ValueError("Invalid official pair specification")
        if same != int(index % 600 < 300):
            raise ValueError("Official pair order must be300genuine then300impostor per fold")
        pairs.append({"pair_id": index, "fold_id": index // 600 + 1, "is_same": same,
                      "path_a": by_subject_image[first], "path_b": by_subject_image[second],
                      "subject_id_a": first[0], "subject_id_b": second[0]})
    with (out / "pairs_6000.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    acquisition = {"dataset": "LFW original unfunnelled 250x250", "checked_on": STAMP, "archive": archive, "official_pair_files": targets,
                   "official_loader": {"source_url": "https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py", "local_snapshot": snapshot.relative_to(ROOT).as_posix(), "sha256": sha256(snapshot)},
                   "dataset_original_source": "http://vis-www.cs.umass.edu/lfw/", "source_access_note": "Original UMass host unavailable in this session. Images and pairs are the exact checksum-verified copies referenced by the official scikit-learn loader.",
                   "original_technical_report": "https://people.cs.umass.edu/~elm/papers/lfw.pdf", "audit": quality,
                   "license": "No verified blanket open/commercial license for the underlying news photographs is asserted. scikit-learn BSD software license does not license dataset images. Preserve source attribution; do not redistribute portraits as project-owned or CC-licensed assets.",
                   "license_context_source": "https://www.microsoft.com/en-us/research/uploads/prod/2019/01/1803.09010.pdf",
                   "pretraining_overlap": "LFW contains public figures; pretrained face models can have identity overlap. Exact-file non-overlap with project data does not rule this out. No identity-disjoint-pretraining claim is made."}
    save(out / "acquisition.json", acquisition)
    p = common_protocol("LFW original unfunnelled", out)
    p.update(protocol_id="lfw_external_v4_1", image_variant="original250x250_unfunneled_no_sklearn_default_crop_or_resize",
             pair_manifest_path=(out / "pairs_6000.csv").relative_to(ROOT).as_posix(), pair_manifest_sha256=sha256(out / "pairs_6000.csv"),
             official_pairs_path=(out / "pairs.txt").relative_to(ROOT).as_posix(), official_pairs_sha256=sha256(out / "pairs.txt"),
             expected_counts={"pairs": 6000, "genuine_pairs": 3000, "impostor_pairs": 3000, "folds": 10, "genuine_per_fold": 300, "impostor_per_fold": 300},
             model_use="Only frozen v4 methods. No LFW image feature fitting, fusion tuning, fine tuning or candidate selection. External pretrained face models explicitly declared as unrestricted/external-training setting.",
             primary_fixed_threshold={"threshold_source": "Frozen FEI/declared development calibration before any LFW score", "metrics": ["FAR", "FRR", "TAR", "accuracy_all6000", "conditional_ROC_AUC", "descriptive_EER", "processing_coverage"],
                                      "accuracy": "Correct operational decisions /6000; failed genuine pair endpoints count as false rejections; failed impostor endpoints are rejection. Also report coverage and conditional metrics to expose this asymmetry."},
             secondary_official_10fold_threshold_cv={"scope": "Secondary conventional threshold-CV comparison, explicitly distinct from transferred fixed-threshold evaluation; classifier weights/preprocessing remain frozen.",
                                                     "folds": "Preserve original10folds. For heldout foldk, use ONLY scores/labels in the other9folds to choose one threshold, then report accuracy on foldk.",
                                                     "threshold_rule": "Maximize accuracy on all5400 training-fold pairs using accept iff score>=threshold. Candidate thresholds are sorted unique valid training scores plus nextafter(max_score,+inf); failures fixed as rejections. If several thresholds maximize accuracy, choose the largest (most conservative). No heldout scores choose threshold.",
                                                     "reporting": "Report10heldout accuracies, mean, sample standard deviation and standard error; no assumption that all pair labels/identities are independent. Also show per-fold coverage. This local deterministic threshold optimization is the declared implementation of standard fold separation, not a claim to reproduce every historical scoring convention."},
             confidence_intervals="For fixed-threshold pair metrics, identity-resampling1000replicates,seed42: same-person pairs weight=identity multiplicity; different-person pairs weight=product of endpoint multiplicities. Conditional on frozen model/threshold. Report sparse/reused identity and image limitations.",
             no_open_set_claim="Official LFW6000pair verification is not open-set1:N identification; do not report pairFAR as searchFPIR.")
    protocol_sha = freeze(out, "external_protocol_v4_1.json", p)
    save(out / "pair_protocol_sha256.json", {"pairs.txt": sha256(out / "pairs.txt"), "pairs_6000.csv": sha256(out / "pairs_6000.csv")})
    print(json.dumps({"dataset": "LFW", "images": 13233, "identities": 5749, "pairs": 6000, "protocol_sha256": protocol_sha, "inventory_sha256": quality["inventory_sha256"], "no_model_evaluation": True}), flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(orl), executor.submit(lfw)]
        for future in futures:
            future.result()
