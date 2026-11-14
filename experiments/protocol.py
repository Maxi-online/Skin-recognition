"""Auditable, identity-disjoint train/calibration/test image manifest.

Required CSV columns: path,subject_id,session_id,split,role.
Paths are relative to the manifest. Splits are train/calibration/test; roles
are train for training images, gallery/probe otherwise. Real session identifiers
must come from dataset metadata. Missing sessions are "unknown", never invented.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path

import numpy as np

from src.utils.data_loader import read_image


@dataclass(frozen=True)
class Sample:
    path: Path
    subject_id: str
    session_id: str
    split: str
    role: str
    sha256: str
    pixel_sha256: str
    width: int
    height: int


def validate_protocol(rows, allow_unknown_sessions=False):
    if not rows:
        raise ValueError("Manifest is empty")
    splits = {"train", "calibration", "test"}
    if {row.split for row in rows} != splits:
        raise ValueError("Manifest must contain train, calibration and test splits")
    owners = {}
    by_subject = defaultdict(list)
    for row in rows:
        if not row.subject_id.strip() or not row.session_id.strip():
            raise ValueError("subject_id and session_id must be explicit; use unknown for absent session metadata")
        if row.split not in splits:
            raise ValueError(f"Unknown split: {row.split}")
        expected_roles = {"train"} if row.split == "train" else {"gallery", "probe"}
        if row.role not in expected_roles:
            raise ValueError(f"Invalid role {row.role} for {row.split}")
        previous_split = owners.setdefault(row.subject_id, row.split)
        if previous_split != row.split:
            raise ValueError(f"Identity leakage across splits: {row.subject_id}")
        by_subject[row.subject_id].append(row)
    for split in sorted(splits):
        count = len({r.subject_id for r in rows if r.split == split})
        if count < 2:
            raise ValueError(f"{split} needs at least two identities; got {count}")
    unknown = []
    for subject, samples in sorted(by_subject.items()):
        if samples[0].split == "train":
            continue
        gallery = [row for row in samples if row.role == "gallery"]
        probes = [row for row in samples if row.role == "probe"]
        if not gallery or not probes:
            raise ValueError(f"Identity {subject} needs both gallery and probe images")
        sessions = {row.session_id.lower() for row in samples}
        if "unknown" in sessions:
            unknown.append(subject)
            if not allow_unknown_sessions:
                raise ValueError("Unknown session metadata; cross-session evaluation is impossible. "
                                 "An explicitly labeled pilot requires --allow-unknown-sessions")
        # Even pilot mode does not permit known session overlap disguised as independence.
        known_g = {r.session_id for r in gallery if r.session_id.lower() != "unknown"}
        known_p = {r.session_id for r in probes if r.session_id.lower() != "unknown"}
        if known_g & known_p:
            raise ValueError(f"Gallery/probe session leakage for identity {subject}: {sorted(known_g & known_p)}")
    if len([r for r in rows if r.split == "train"]) < 4:
        raise ValueError("Training needs at least four images")
    counts = {}
    for split in ("train", "calibration", "test"):
        selected = [r for r in rows if r.split == split]
        counts[split] = {
            "identities": len({r.subject_id for r in selected}),
            "images": len(selected),
            "roles": dict(sorted(Counter(r.role for r in selected).items())),
        }
    return {
        "split_unit": "identity",
        "cross_session_established": not unknown,
        "protocol_kind": "image_disjoint_pilot_unknown_sessions" if unknown else "identity_disjoint_cross_session",
        "unknown_session_identities": unknown,
        "counts": counts,
        "warnings": (["Cross-session independence is NOT established. Same-session performance may be optimistic."]
                     if unknown else []),
    }


def load_manifest(manifest_path, allow_unknown_sessions=False, min_size=32):
    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise ValueError(f"Manifest does not exist: {manifest}")
    required = {"path", "subject_id", "session_id", "split", "role"}
    rows = []
    path_seen, content_seen, pixels_seen = set(), {}, {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {sorted(required)}")
        for line, record in enumerate(reader, 2):
            if any(record.get(field) is None or not record[field].strip() for field in required):
                raise ValueError(f"Manifest line {line}: missing required value")
            path = Path(record["path"].strip())
            path = (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()
            if path in path_seen:
                raise ValueError(f"Duplicate image path at line {line}: {path}")
            path_seen.add(path)
            image = read_image(path)
            height, width = image.shape[:2]
            if min(height, width) < min_size:
                raise ValueError(f"Image smaller than {min_size}px: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            pixel_digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if digest in content_seen or pixel_digest in pixels_seen:
                previous = content_seen.get(digest, pixels_seen.get(pixel_digest))
                raise ValueError(f"Duplicate image content: {path} and {previous}")
            content_seen[digest], pixels_seen[pixel_digest] = path, path
            rows.append(Sample(path, record["subject_id"].strip(), record["session_id"].strip(),
                               record["split"].strip(), record["role"].strip(), digest, pixel_digest,
                               width, height))
    audit = validate_protocol(rows, allow_unknown_sessions)
    audit.update({
        "manifest_path": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "duplicate_check": "absolute_path_file_sha256_decoded_pixels_sha256",
        "near_duplicates_checked": False,
        "input_size_range": {
            "width": [min(r.width for r in rows), max(r.width for r in rows)],
            "height": [min(r.height for r in rows), max(r.height for r in rows)],
        },
    })
    return rows, audit


def write_sample_inventory(rows, output_path):
    """Record image provenance without copying biometric images into the report."""
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "subject_id", "session_id", "split", "role", "sha256",
                         "pixel_sha256", "width", "height"])
        for row in rows:
            writer.writerow([str(row.path), row.subject_id, row.session_id, row.split, row.role,
                             row.sha256, row.pixel_sha256, row.width, row.height])


def identity_split(subject_ids, seed=42, train_fraction=0.6, calibration_fraction=0.2):
    """Freeze identity membership before any fitted transform or model selection."""
    subjects = np.asarray(sorted(set(subject_ids)))
    if len(subjects) < 6:
        raise ValueError("At least six identities are needed for three nontrivial splits")
    if not (0 < train_fraction < 1 and 0 < calibration_fraction < 1
            and train_fraction + calibration_fraction < 1):
        raise ValueError("Invalid split fractions")
    shuffled = np.random.default_rng(seed).permutation(subjects)
    n_train = max(2, int(len(subjects) * train_fraction))
    n_cal = max(2, int(len(subjects) * calibration_fraction))
    if len(subjects) - n_train - n_cal < 2:
        n_train = len(subjects) - n_cal - 2
    if n_train < 2:
        raise ValueError("Split fractions leave fewer than two training identities")
    return {str(subject): ("train" if index < n_train else
                          "calibration" if index < n_train + n_cal else "test")
            for index, subject in enumerate(shuffled)}

