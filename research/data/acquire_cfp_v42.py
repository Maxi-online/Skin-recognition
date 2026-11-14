"""Acquire CFP and freeze its original FF/FP pair lists without model inference.

Only JPEG decoding, dimensions, cryptographic hashes and supplied annotations are
inspected. No detector, embedding model, metric or recognition package is loaded.
The source archive and all official files are preserved byte-for-byte.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
import urllib.request
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "research/data/cfp"
ARCHIVE = ROOT / "data/cfp/archives/cfp-dataset.zip"
EXTRACTED = ROOT / "data/cfp/originals"
SOURCE = EXTRACTED / "cfp-dataset"
URL = "http://www.cfpw.io/cfp-dataset.zip"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def acquire(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".download")
        if partial.exists():
            raise ValueError("Existing partial download must be resolved before restarting: " + str(partial))
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("xb") as stream:
            count, last = 0, time.monotonic()
            while block := response.read(1024 * 1024):
                stream.write(block)
                count += len(block)
                if time.monotonic() - last > 15:
                    print("downloaded_bytes", count, flush=True)
                    last = time.monotonic()
        partial.rename(path)
    return {"url": url, "path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def safe_extract():
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt CFP ZIP")
        members = archive.infolist()
        if len({item.filename for item in members}) != len(members):
            raise ValueError("Repeated archive path")
        for item in members:
            member = PurePosixPath(item.filename.replace("\\", "/"))
            if member.is_absolute() or ".." in member.parts or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Unsafe archive member")
            destination = (EXTRACTED / Path(*member.parts)).resolve()
            if not destination.is_relative_to(EXTRACTED):
                raise ValueError("Archive path escapes destination")
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            payload = archive.read(item)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.read_bytes() != payload:
                    raise ValueError("Existing original differs from archive: " + str(destination))
            else:
                destination.write_bytes(payload)
        return len(members)


def inspect_image(path):
    parts = path.relative_to(SOURCE / "Data/Images").parts
    subject, pose, filename = parts
    with Image.open(path) as image:
        image.load()
        width, height, mode = image.width, image.height, image.mode
        rgb = image.convert("RGB")
        decoded = hashlib.sha256(f"{width}x{height}:RGB:".encode() + rgb.tobytes()).hexdigest()
    return {"path": rel(path), "subject_id": subject, "image_id": int(Path(filename).stem),
            "pose": pose, "session_id": "unknown", "width": width, "height": height,
            "mode": mode, "sha256": sha256(path), "rgb_sha256": decoded}


def pair_lookup(filename, by_path, expected_pose, expected_count):
    result = {}
    for line in (SOURCE / "Protocol" / filename).read_text().splitlines():
        number, name = line.split()
        path = (SOURCE / "Protocol" / name).resolve()
        if not path.is_relative_to(SOURCE / "Data/Images"):
            raise ValueError("Pair-list image escapes original image tree")
        relative = rel(path)
        if relative not in by_path or by_path[relative]["pose"] != expected_pose:
            raise ValueError("Unknown image/pose in official pair lookup")
        key = int(number)
        if key in result:
            raise ValueError("Repeated official image index")
        result[key] = relative
    if set(result) != set(range(1, expected_count + 1)) or len(set(result.values())) != expected_count:
        raise ValueError("Unexpected official lookup range")
    return result


def freeze_pairs(kind, lookup_f, lookup_p, by_path, acquisition_digest, official_hashes):
    rows, fold_identities = [], {}
    for fold in range(1, 11):
        identities = set()
        for label, filename in ((1, "same.txt"), (0, "diff.txt")):
            source = SOURCE / "Protocol/Split" / kind / f"{fold:02d}" / filename
            lines = source.read_text().splitlines()
            if len(lines) != 350:
                raise ValueError("Every official fold/class must contain350pairs")
            for line_no, line in enumerate(lines, 1):
                first, second = [int(value.strip()) for value in line.split(",")]
                a, b = lookup_f[first], (lookup_f if kind == "FF" else lookup_p)[second]
                ida, idb = by_path[a]["subject_id"], by_path[b]["subject_id"]
                if int(ida == idb) != label or a == b:
                    raise ValueError("Official pair identity/label/self-pair mismatch")
                identities.update((ida, idb))
                rows.append({"pair_id": len(rows), "fold_id": fold, "is_same": label, "path_a": a, "path_b": b,
                             "subject_id_a": ida, "subject_id_b": idb, "pose_a": by_path[a]["pose"], "pose_b": by_path[b]["pose"],
                             "source_pair_file": rel(source), "source_line": line_no,
                             "official_index_a": first, "official_index_b": second})
        fold_identities[str(fold)] = sorted(identities)
    if len(rows) != 7000 or len(set().union(*[set(ids) for ids in fold_identities.values()])) != 500:
        raise ValueError("Expected7000pairs/500identities")
    for i in range(1, 11):
        if len(fold_identities[str(i)]) != 50:
            raise ValueError("Expected50identities per original fold")
        if any(set(fold_identities[str(i)]) & set(fold_identities[str(j)]) for j in range(i+1, 11)):
            raise ValueError("Original CFP folds should be identity-disjoint")
    manifest = META / f"pairs_{kind.lower()}_7000.csv"
    csv_write(manifest, rows)
    unique = {r[key] for r in rows for key in ("path_a", "path_b")}
    undirected = Counter(tuple(sorted((r["path_a"], r["path_b"]))) for r in rows)
    protocol = {
        "protocol_id": f"cfp_{kind.lower()}_external_v4_2", "dataset": "CFP", "comparison": kind,
        "evaluation_role": "primary_frontal_uncontrolled_holdout" if kind == "FF" else "secondary_profile_pose_stress_test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status_at_freeze": "Acquired and metadata/decode audited only; no detector, embeddings, model inference or recognition metrics.",
        "inventory_path": rel(META / "inventory.csv"), "inventory_sha256": sha256(META / "inventory.csv"),
        "acquisition_path": rel(META / "acquisition.json"), "acquisition_sha256": acquisition_digest,
        "pair_manifest_path": rel(manifest), "pair_manifest_sha256": sha256(manifest),
        "official_protocol_hashes": official_hashes, "pair_count": 7000, "genuine_pairs": 3500, "impostor_pairs": 3500,
        "fold_count": 10, "pairs_per_fold": 700, "genuine_per_fold": 350, "impostor_per_fold": 350,
        "identity_count": 500, "identities_per_fold": fold_identities, "required_images": len(unique),
        "official_pair_preservation": "Original endpoints/labels/folds and within-file order are unchanged; serialization is fold1..10, same.txt then diff.txt. Repeated or reversed pairs are preserved.",
        "repeated_undirected_pair_groups": sum(count > 1 for count in undirected.values()),
        "duplicate_undirected_pair_rows_beyond_first": sum(count-1 for count in undirected.values()),
        "primary_operating_point": "Fixed FEI-development negative-L2 thresholds and weights from separately timestamped v4.2 aggregate freeze; no threshold/model/input-adapter selection using CFP scores or quality outcomes.",
        "secondary_protocol": "Original10-fold partitions, with threshold only selected on other9folds to maximize all-denominator accuracy, largest/conservative threshold on ties; frozen weights and preprocessing throughout.",
        "failure_policy": "Every7000scheduled pair retained. Failed genuine endpoint counts as false rejection; failed impostor endpoint counts as rejection. Separately report accepted-only ROC/AUC/EER/FAR/FRR and image/pair coverage.",
        "confidence_intervals": "Identity-resampling intervals conditional on frozen model/threshold; genuine-pair multiplicity once, impostor-pair product of endpoint identity multiplicities. Do not treat reused pairs as independent. Fold mean/std separately for secondaryCV.",
        "input_semantics": "Use original JPEG image files at supplied size with the frozen v4.2 automatic input adapter. Manual30-point fiducials are preserved but not used for primary inference. No selection by detection success.",
        "prior_exposure": "CFP has not been evaluated in this project at protocol creation. Public celebrity labels may overlap LFW or external pretraining; untouched input holdout is not a claim of identity-disjoint pretraining.",
        "session_provenance": "No per-image session metadata; frontal/profile labels are pose categories, not acquisition sessions.",
        "limits": "Pair verification, not open-set search. No resolved-pore, measured-depth or commercial-model-license claim.",
    }
    destination = META / f"external_protocol_{kind.lower()}_v4_2.json"
    save(destination, protocol)
    return {"path": rel(destination), "sha256": sha256(destination), "pair_manifest_sha256": sha256(manifest),
            "pair_count": len(rows), "required_images": len(unique), "repeated_undirected_pair_groups": protocol["repeated_undirected_pair_groups"]}


def main():
    META.mkdir(parents=True, exist_ok=True)
    if any(META.glob("external_protocol_*_v4_2.json")):
        raise ValueError("Frozen CFP protocols already exist; inspect instead of replacing them")
    archive = acquire(URL, ARCHIVE)
    source_records = [acquire(url, META / name) for name, url in (
        ("official_index.html", "http://www.cfpw.io/"),
        ("official_paper.pdf", "http://www.cfpw.io/paper.pdf"),
        ("official_submission_rules.html", "http://www.cfpw.io/submit.html"))]
    members = safe_extract()
    images = sorted((SOURCE / "Data/Images").rglob("*.jpg"))
    if len(images) != 7000:
        raise ValueError("Expected CFP7000originalJPEGs")
    with ThreadPoolExecutor(max_workers=6) as pool:
        inventory = list(pool.map(inspect_image, images))
    subjects = sorted({r["subject_id"] for r in inventory})
    if subjects != [f"{i:03d}" for i in range(1, 501)]:
        raise ValueError("Expected500sourceidentitydirectories")
    for subject in subjects:
        for pose, n in (("frontal", 10), ("profile", 4)):
            selected = [r for r in inventory if r["subject_id"] == subject and r["pose"] == pose]
            if len(selected) != n or {r["image_id"] for r in selected} != set(range(1, n+1)):
                raise ValueError("Unexpected subject pose image count")
    csv_write(META / "inventory.csv", inventory)
    by_path = {r["path"]: r for r in inventory}
    all_files = sorted(path for path in SOURCE.rglob("*") if path.is_file())
    hashes = {rel(path): sha256(path) for path in all_files}
    save(META / "archive_file_sha256.json", hashes)
    official_hashes = {path: digest for path, digest in hashes.items() if "/Protocol/" in path}
    duplicates = {}
    for field in ("sha256", "rgb_sha256"):
        groups = defaultdict(list)
        for row in inventory:
            groups[row[field]].append(row["path"])
        duplicates[field] = [paths for paths in groups.values() if len(paths) > 1]
    overlap = {}
    for name, filename in (("fei", "research/data/fei_inventory.csv"), ("georgia_tech", "research/data/georgia_tech/inventory.csv"),
                           ("lfw", "research/data/lfw/inventory.csv"), ("orl", "research/data/orl/inventory.csv")):
        path = ROOT / filename
        with path.open(encoding="utf-8-sig", newline="") as stream:
            previous = list(csv.DictReader(stream))
        lookup = defaultdict(list)
        for row in previous:
            lookup[row.get("sha256", row.get("file_sha256"))].append(row["path"])
        overlap[name] = [{"cfp_path": r["path"], "other_paths": lookup[r["sha256"]]} for r in inventory if r["sha256"] in lookup]
    names = (SOURCE / "Data/list_name.txt").read_text().splitlines()
    if len(names) != 500:
        raise ValueError("Expected500sourcepublic-name annotations")
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    with (ROOT / "research/data/lfw/inventory.csv").open(newline="", encoding="utf-8") as stream:
        lfw_ids = sorted({r["subject_id"] for r in csv.DictReader(stream)})
    lfw_names = defaultdict(list)
    for name in lfw_ids:
        lfw_names[normalize(name)].append(name)
    name_matches = [{"cfp_subject_id": f"{i:03d}", "cfp_source_name": name, "lfw_labels": lfw_names[normalize(name)]}
                    for i, name in enumerate(names, 1) if normalize(name) in lfw_names]
    save(META / "duplicate_and_overlap_audit.json", {"within_cfp": duplicates, "exact_file_overlap": overlap,
         "normalized_source_name_matches_lfw": name_matches,
         "limits": "Exact file/name equality only. No recognition, perceptual matching or assertion that unmatched names represent distinct people. No official pair removed."})
    report = {"dataset": "Celebrities in Frontal-Profile (CFP)", "created_utc": datetime.now(timezone.utc).isoformat(),
              "archive": archive, "source_records": source_records, "archive_members": members,
              "images": len(inventory), "identities": len(subjects), "poses": dict(Counter(r["pose"] for r in inventory)),
              "image_sizes": dict(Counter(f"{r['width']}x{r['height']}" for r in inventory)),
              "modes": dict(Counter(r["mode"] for r in inventory)), "decode_errors": 0,
              "inventory_sha256": sha256(META / "inventory.csv"), "archive_file_sha256": sha256(META / "archive_file_sha256.json"),
              "duplicate_file_groups": len(duplicates["sha256"]), "duplicate_decoded_RGB_groups": len(duplicates["rgb_sha256"]),
              "exact_file_overlap_counts": {name: len(rows) for name, rows in overlap.items()},
              "normalized_lfw_label_match_count": len(name_matches),
              "license_evidence": "Official page offers direct download and paper citation. Archive Readme describes data/protocol but contains no explicit standardized license. No blanket open/commercial or redistribution permission asserted. No registration, agreement submission, email or payment performed.",
              "transport_evidence": "Author-published HTTP endpoint used. HTTPS certificate verification failed; TLS verification was not disabled. SHA256 is a locally recorded integrity digest, not an independently published author checksum.",
              "session_provenance": "unknown; pose categories are supplied",
              "feature_extraction_performed": False, "model_inference_performed": False, "recognition_metrics_computed": False,
              "official_readme_path": rel(SOURCE / "Readme.txt"), "official_names_path": rel(SOURCE / "Data/list_name.txt"),
              "acquisition_script_sha256": sha256(Path(__file__))}
    save(META / "acquisition.json", report)
    lookup_f = pair_lookup("Pair_list_F.txt", by_path, "frontal", 5000)
    lookup_p = pair_lookup("Pair_list_P.txt", by_path, "profile", 2000)
    results = {kind: freeze_pairs(kind, lookup_f, lookup_p, by_path, sha256(META / "acquisition.json"), official_hashes) for kind in ("FF", "FP")}
    save(META / "frozen_protocol_sha256.json", results)
    print(json.dumps({"images": report["images"], "image_sizes": report["image_sizes"], "protocols": results,
                      "duplicates": report["duplicate_file_groups"], "overlap": report["exact_file_overlap_counts"],
                      "lfw_label_matches": len(name_matches)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
