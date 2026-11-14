"""Paired exploratory EER differences on a completed fixed experiment.

Identities, not individual correlated pairs, are resampled. Differences below
zero favor the candidate. Intervals are unadjusted exploratory 95% intervals,
conditional on the fitted models and this dataset/protocol.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.scientific_metrics import compute_eer


def load_pairs(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([int(row["genuine"]) for row in rows]),
        np.asarray([float(row["score"]) for row in rows]),
        np.asarray([row["probe_subject_id"] for row in rows]),
        np.asarray([row["gallery_subject_id"] for row in rows]),
    )


def compare(candidate, reference, replicates=1000, seed=42):
    cy, cs, cp, cg = candidate
    ry, rs, rp, rg = reference
    if not all(np.array_equal(left, right) for left, right in ((cy, ry), (cp, rp), (cg, rg))):
        raise ValueError("Paired comparisons require identical pair ordering and identities")
    if not np.array_equal(cy.astype(bool), cp == cg):
        raise ValueError("Pair identity/label mismatch")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    identities = np.unique(np.concatenate([cp, cg]))
    pi, gi = np.searchsorted(identities, cp), np.searchsorted(identities, cg)
    if len(identities) < 2:
        raise ValueError("At least two identities required")
    rng = np.random.default_rng(seed)
    deltas, attempts = [], 0
    while len(deltas) < replicates and attempts < 20 * replicates:
        attempts += 1
        count = np.bincount(rng.integers(len(identities), size=len(identities)), minlength=len(identities))
        weight = np.where(cy == 1, count[pi], count[pi] * count[gi])
        if not all(weight[cy == cls].sum() > 0 for cls in (0, 1)):
            continue
        deltas.append(compute_eer(cy, cs, weight)[0] - compute_eer(ry, rs, weight)[0])
    if len(deltas) != replicates:
        raise ValueError("Insufficient valid bootstrap replicates")
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return {
        "metric": "candidate_EER_minus_reference_EER",
        "difference": compute_eer(cy, cs)[0] - compute_eer(ry, rs)[0],
        "percentile_95_interval": [float(lower), float(upper)],
        "zero_in_interval": bool(lower <= 0 <= upper),
        "replicates": replicates, "seed": seed, "identity_count": len(identities),
        "interpretation": "negative favors candidate; exploratory unadjusted interval",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate", default="hybrid")
    parser.add_argument("--references", default="gabor_only,lbp,hybrid_global")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if result["status"] != "completed":
        raise ValueError("Only completed runs can be compared")
    output = run_dir / ("paired_comparisons_" + args.candidate + ".json")
    if output.exists():
        raise ValueError(f"Comparison output already exists: {output}")
    candidate_path = run_dir / args.candidate / "test_pairs.csv"
    candidate = load_pairs(candidate_path)
    report = {
        "candidate": args.candidate,
        "method": "paired_identity_resampling",
        "status": "exploratory_not_multiple_comparison_adjusted",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {args.candidate: hashlib.sha256(candidate_path.read_bytes()).hexdigest()},
        "comparisons": {},
    }
    for reference in args.references.split(","):
        reference_path = run_dir / reference / "test_pairs.csv"
        report["input_sha256"][reference] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        report["comparisons"][reference] = compare(candidate, load_pairs(reference_path), args.bootstrap, args.seed)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

