"""Create a fixed, subject-disjoint FEI frontal pilot manifest.

FEI acquisition report, Figure 5 (PDF page 6), explicitly labels image 11 as
neutral frontal and 12 as smiling frontal. The corpus provides no independent
session identifier here. All sessions are therefore recorded as unknown.
Source: https://fei.edu.br/~cet/iniciacaocientifica_LeoOliveira_2006.pdf
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol import identity_split, load_manifest


def prepare_fei(data_dir, output, seed=42):
    data_dir = Path(data_dir).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Manifest already exists: {output}")
    if not data_dir.is_dir():
        raise ValueError(f"FEI originals directory missing: {data_dir}")
    subjects = [str(i) for i in range(1, 201)]
    # Require the complete selected subset; absence is not silently called a cohort.
    for subject in subjects:
        for frame in (11, 12):
            if not (data_dir / f"{subject}-{frame:02d}.jpg").is_file():
                raise ValueError(f"Missing FEI image: {subject}-{frame:02d}.jpg")
    splits = identity_split(subjects, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    import os
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "subject_id", "session_id", "split", "role", "condition"])
        for subject in subjects:
            for frame, condition in ((11, "neutral_frontal"), (12, "smiling_frontal")):
                image_path = data_dir / f"{subject}-{frame:02d}.jpg"
                role = "train" if splits[subject] == "train" else "gallery" if frame == 11 else "probe"
                writer.writerow([Path(os.path.relpath(image_path, output.parent)).as_posix(),
                                 f"fei_{int(subject):03d}", "unknown", splits[subject], role, condition])
    _, audit = load_manifest(output, allow_unknown_sessions=True)
    metadata = {
        "dataset": "FEI Face Database",
        "source": "https://fei.edu.br/~cet/facedatabase.html",
        "frontal_index_evidence": "https://fei.edu.br/~cet/iniciacaocientifica_LeoOliveira_2006.pdf, Figure 5, page 6",
        "seed": seed, "identity_sort_order": "lexicographic_original_numeric_ids",
        "train_fraction": 0.6, "calibration_fraction": 0.2, "test_fraction": 0.2,
        "image_indices": {"11": "neutral_frontal", "12": "smiling_frontal"},
        "gallery": "neutral_frontal", "probe": "smiling_frontal",
        "session_metadata_available": False,
        "scope": "image-disjoint frontal expression pilot; not cross-session or pore-resolution validation",
        "audit": audit,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "fei" / "originals"))
    parser.add_argument("--output", default=str(ROOT / "data" / "fei" / "manifest_frontal_seed42.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_fei(args.data_dir, args.output, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

