"""Synthetic CFP protocol fixtures only; no external image/model inference."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.external_evaluate_v42 import evaluate_pairs, load_protocol_data, sha256


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def synthetic_cfp(tmp_path):
    rows, lookup = [], {"F": {}, "P": {}}
    directory = tmp_path / "synthetic/Protocol"
    directory.mkdir(parents=True)
    for view, pose, count in (("F", "frontal", 10), ("P", "profile", 4)):
        lines = []
        for subject in range(1, 501):
            for image in range(1, count + 1):
                index = (subject - 1) * count + image
                path = f"synthetic/Data/Images/{subject:03d}/{pose}/{image:02d}.jpg"
                rows.append(dict(path=path, subject_id=f"{subject:03d}", image_id=image,
                                 pose=pose, sha256="a" * 64))
                lookup[view][index] = rows[-1]
                lines.append(f"{index} ../Data/Images/{subject:03d}/{pose}/{image:02d}.jpg")
        (directory / f"Pair_list_{view}.txt").write_text("\n".join(lines) + "\n")
    inventory = tmp_path / "inventory.csv"
    write_csv(inventory, rows)
    manifests = {}
    for kind in ("FF", "FP"):
        pairs = []
        for fold in range(1, 11):
            for label, name in ((1, "same.txt"), (0, "diff.txt")):
                source = directory / "Split" / kind / f"{fold:02d}" / name
                source.parent.mkdir(parents=True, exist_ok=True)
                original = []
                for n in range(350):
                    sa = (fold - 1) * 50 + n % 50 + 1
                    sb = sa if label else (fold - 1) * 50 + (n + 1) % 50 + 1
                    ia, ib = n // 50 + 1, n // 50 + 2
                    view, count = ("F", 10) if kind == "FF" else ("P", 4)
                    if kind == "FP":
                        ib = n // 50 % 4 + 1
                    a, b = (sa - 1) * 10 + ia, (sb - 1) * count + ib
                    first, second = lookup["F"][a], lookup[view][b]
                    original.append(f"{a},{b}")
                    pairs.append(dict(pair_id=len(pairs), fold_id=fold, is_same=label,
                                      path_a=first["path"], path_b=second["path"],
                                      subject_id_a=first["subject_id"], subject_id_b=second["subject_id"],
                                      source_pair_file=source.relative_to(tmp_path).as_posix(),
                                      source_line=n+1, official_index_a=a, official_index_b=b))
                source.write_text("\n".join(original) + "\n")
        manifest = tmp_path / f"pairs_{kind}.csv"
        write_csv(manifest, pairs)
        manifests[kind] = manifest
    official = {p.relative_to(tmp_path).as_posix(): sha256(p) for p in directory.rglob("*.txt")}
    protocols = {}
    for kind in ("FF", "FP"):
        protocols[kind] = dict(
            comparison=kind,
            evaluation_role="primary_frontal_uncontrolled_holdout" if kind == "FF" else "secondary_profile_pose_stress_test",
            inventory_path="inventory.csv", inventory_sha256=sha256(inventory),
            official_protocol_hashes=official, pair_manifest_path=manifests[kind].name,
            pair_manifest_sha256=sha256(manifests[kind]))
    return tmp_path, protocols


@pytest.mark.parametrize("kind", ["FF", "FP"])
def test_original_pair_mappings_folds_and_views_are_validated(synthetic_cfp, kind):
    root, protocols = synthetic_cfp
    inventory, pairs, required = load_protocol_data(protocols[kind], "cfp_" + kind.lower(), root)
    assert len(inventory) == len(pairs) == 7000
    assert len({r["subject_id"] for r in inventory.values()}) == 500
    assert all(Path(path).is_absolute() for path in required)
    assert sum(int(p["is_same"]) for p in pairs) == 3500
    assert all("/profile/" in p["path_b"] for p in pairs) if kind == "FP" else all("/frontal/" in p["path_b"] for p in pairs)


def test_rehashed_pair_reordering_still_fails_against_official_original(synthetic_cfp):
    root, protocols = synthetic_cfp
    protocol = protocols["FF"]
    path = root / protocol["pair_manifest_path"]
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0], rows[1] = rows[1], rows[0]
    rows[0]["pair_id"], rows[1]["pair_id"] = 0, 1
    write_csv(path, rows)
    protocol["pair_manifest_sha256"] = sha256(path)
    with pytest.raises(ValueError, match="original ordered endpoint"):
        load_protocol_data(protocol, "cfp_ff", root)


def test_original_split_file_corruption_is_detected_before_scoring(synthetic_cfp):
    root, protocols = synthetic_cfp
    target = root / "synthetic/Protocol/Split/FF/01/same.txt"
    target.write_text(target.read_text() + "1,2\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_protocol_data(protocols["FF"], "cfp_ff", root)


def test_profile_stress_cannot_be_relabelled_as_primary_holdout(synthetic_cfp):
    root, protocols = synthetic_cfp
    protocols["FP"]["evaluation_role"] = "primary_frontal_uncontrolled_holdout"
    with pytest.raises(ValueError, match="primary-stress role"):
        load_protocol_data(protocols["FP"], "cfp_fp", root)


def tiny_pairs():
    pairs, features = [], {}
    for fold in range(1, 11):
        a, b = f"person{fold}a", f"person{fold}b"
        features[a+"1"] = features[a+"2"] = np.array([1., 0.])
        features[b+"1"] = np.array([0., 1.])
        for label, second, identity in ((1, a+"2", a), (0, b+"1", b)):
            pairs.append(dict(pair_id=len(pairs), fold_id=fold, is_same=label, path_a=a+"1", path_b=second,
                              subject_id_a=a, subject_id_b=identity))
    return pairs, features


def test_quality_failures_keep_operational_denominators_and_threshold_cv_is_secondary():
    pairs, features = tiny_pairs()
    del features["person1a2"]
    report, numeric = evaluate_pairs(pairs, features, -.5, "FF", bootstrap_replicates=30)
    assert report["fixed_threshold_operational"]["intended_pairs"] == 20
    assert report["fixed_threshold_operational"]["frr"] == .1
    assert report["fixed_threshold_conditional"]["frr"] == 0
    assert len(report["secondary_threshold_cross_validation"]["folds"]) == 10
    assert report["confidence_intervals"]["identity_count"] == 20
    assert numeric["valid_pairs"].sum() == 19
    assert not any("250px" in text or "LFW identities" in text for text in report["limitations"])


def test_all_rejected_is_unavailable_conditionally_not_perfect_verification():
    pairs, _ = tiny_pairs()
    report, numeric = evaluate_pairs(pairs, {}, -.5, "FP", bootstrap_replicates=30)
    assert report["fixed_threshold_conditional"]["status"] == "unavailable"
    assert report["fixed_threshold_operational"]["frr"] == 1
    assert report["fixed_threshold_operational"]["far"] == 0
    assert report["secondary_threshold_cross_validation"]["status"] == "incomplete"
    assert not numeric["valid_pairs"].any()
