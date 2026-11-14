"""Read-only FEI inventory: identifiers, dimensions, and exact duplicates."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    rows = []
    errors = []
    byte_groups = defaultdict(list)
    pixel_groups = defaultdict(list)
    counts = Counter()
    dimensions = Counter()
    for path in sorted((ROOT / "data" / "fei" / "originals").glob("*.jpg")):
        match = re.fullmatch(r"(\d+)-(\d+)\.jpg", path.name)
        if not match:
            errors.append({"path": str(path), "error": "Unexpected filename"})
            continue
        try:
            image = Image.open(path).convert("RGB")
            image.load()
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            pixel_hash = hashlib.sha256(
                f"{image.width},{image.height},RGB:".encode() + image.tobytes()).hexdigest()
            counts[match[1]] += 1
            dimensions[f"{image.width}x{image.height}"] += 1
            byte_groups[file_hash].append(path.name)
            pixel_groups[pixel_hash].append(path.name)
            rows.append({"path": path.relative_to(ROOT).as_posix(),
                         "subject_id": match[1], "image_index": int(match[2]),
                         "width": image.width, "height": image.height,
                         "file_sha256": file_hash, "rgb_sha256": pixel_hash})
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})
    target = ROOT / "research" / "data"
    with (target / "fei_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "subject_id", "image_index",
                                                    "width", "height", "file_sha256", "rgb_sha256"])
        writer.writeheader()
        writer.writerows(rows)
    result = {"dataset": "FEI originals", "checked_on": "2026-11-14",
              "images_read": len(rows), "subjects": len(counts),
              "images_per_subject_histogram": dict(Counter(counts.values())),
              "dimensions": dict(dimensions),
              "exact_file_duplicates": [names for names in byte_groups.values() if len(names) > 1],
              "exact_rgb_duplicates": [names for names in pixel_groups.values() if len(names) > 1],
              "errors": errors,
              "limitations": "No near-duplicate, identity-label, skin-purity or session validation is implied."}
    (target / "fei_quality_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
