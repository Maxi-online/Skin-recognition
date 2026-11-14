"""Acquire the author-hosted Georgia Tech face corpus without model evaluation.

No explicit redistribution license was found on the source page or README.
Keep the dataset local; do not infer an unrestricted or commercial license.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "georgia_tech"
PROVENANCE = ROOT / "research" / "data" / "georgia_tech"
BASE = "https://www.anefian.com/research/"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(name):
    path = DATA / "archives" / name
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".download")
        with urllib.request.urlopen(BASE + name, timeout=120) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip():
                raise ValueError(f"ZIP CRC mismatch: {name}")
        temporary.rename(path)
    target_root = (DATA / ("originals" if name == "gt_db.zip" else "labels")).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(path) as bundle:
        for entry in bundle.infolist():
            member = PurePosixPath(entry.filename.replace("\\", "/"))
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            if entry.is_dir():
                continue
            destination = (target_root / Path(*member.parts)).resolve()
            if not destination.is_relative_to(target_root):
                raise ValueError(f"Unsafe destination: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with bundle.open(entry) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
            elif destination.stat().st_size != entry.file_size:
                raise ValueError(f"Existing file differs: {destination}")
            names.append(str(destination.relative_to(DATA)))
    return {"url": BASE + name, "archive_sha256": sha256(path), "bytes": path.stat().st_size,
            "file_count": len(names), "files": names}


def main():
    (DATA / "archives").mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    source_records = []
    for name in ("face_reco.htm", "GTDB_README.txt"):
        destination = PROVENANCE / name
        if not destination.exists():
            destination.write_bytes(urllib.request.urlopen(BASE + name, timeout=60).read())
        source_records.append({"url": BASE + name, "local_path": str(destination.relative_to(ROOT)),
                               "sha256": sha256(destination)})
    with ThreadPoolExecutor(max_workers=2) as executor:
        archives = list(executor.map(download, ("gt_db.zip", "labels_gt.zip")))
    rows, errors = [], []
    duplicates = defaultdict(list)
    subjects, dimensions = Counter(), Counter()
    for image_path in sorted((DATA / "originals").rglob("*.jpg")):
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
            digest = sha256(image_path)
            subject = image_path.parent.name
            subjects[subject] += 1
            dimensions[f"{width}x{height}"] += 1
            duplicates[digest].append(str(image_path.relative_to(ROOT)))
            rows.append({"path": image_path.relative_to(ROOT).as_posix(), "subject_id": subject,
                         "image_id": image_path.stem, "session_id": "unknown",
                         "width": width, "height": height, "sha256": digest})
        except Exception as error:
            errors.append({"path": str(image_path), "error": str(error)})
    with (PROVENANCE / "inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "subject_id", "image_id", "session_id",
                                                  "width", "height", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    fei_inventory = ROOT / "research" / "data" / "fei_inventory.csv"
    overlap = []
    if fei_inventory.exists():
        with fei_inventory.open(encoding="utf-8", newline="") as stream:
            fei_hashes = {row["file_sha256"] for row in csv.DictReader(stream)}
        overlap = [row["path"] for row in rows if row["sha256"] in fei_hashes]
    result = {
        "dataset": "Georgia Tech face database", "checked_on": "2026-11-14",
        "source_records": source_records, "archives": archives,
        "images_read": len(rows), "identity_count": len(subjects),
        "images_per_identity_histogram": dict(Counter(subjects.values())),
        "dimensions": dict(dimensions), "errors": errors,
        "exact_file_duplicates": [group for group in duplicates.values() if len(group) > 1],
        "exact_file_overlap_with_fei": overlap,
        "session_provenance": "Author reports two or three sessions, but no per-image session mapping in README. Unknown in inventory.",
        "license_status": "No explicit license found in retrieved source page and README. No redistribution or commercial permission inferred.",
        "evaluation_status": "Untouched by model evaluation: acquisition, structural inventory and exact hashes only.",
        "limitations": "Anonymous identity labels do not prove absence of person overlap with another corpus. No near-duplicate or recognition evaluation performed.",
    }
    (PROVENANCE / "acquisition.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("archives", "source_records")}, indent=2))


if __name__ == "__main__":
    main()
