"""Protocol tests use random image fixtures solely to check software behavior."""
import csv
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from experiments.protocol import Sample, identity_split, load_manifest, validate_protocol
from src.utils.data_loader import load_data


def rows(session_unknown=False):
    result = []
    for split in ("train", "calibration", "test"):
        for subject_index in range(2):
            subject = f"{split}_{subject_index}"
            for i in range(2):
                role = "train" if split == "train" else "gallery" if i == 0 else "probe"
                session = "unknown" if session_unknown else f"session_{i}"
                result.append(Sample(Path(f"{subject}_{i}.png"), subject, session, split, role,
                                     f"file_{subject}_{i}", f"pixels_{subject}_{i}", 40, 40))
    return result


class ProtocolTests(unittest.TestCase):
    def test_subject_disjoint_cross_session(self):
        audit = validate_protocol(rows())
        self.assertTrue(audit["cross_session_established"])
        self.assertEqual(audit["counts"]["test"]["identities"], 2)

    def test_identity_leakage_raises(self):
        fixture = rows()
        fixture[-1] = replace(fixture[-1], subject_id="train_0")
        with self.assertRaisesRegex(ValueError, "Identity leakage"):
            validate_protocol(fixture)

    def test_gallery_probe_session_leakage_raises(self):
        fixture = rows()
        fixture[-1] = replace(fixture[-1], session_id="session_0")
        with self.assertRaisesRegex(ValueError, "session leakage"):
            validate_protocol(fixture, allow_unknown_sessions=True)

    def test_unknown_sessions_require_explicit_pilot_flag(self):
        fixture = rows(session_unknown=True)
        with self.assertRaisesRegex(ValueError, "Unknown session"):
            validate_protocol(fixture)
        audit = validate_protocol(fixture, allow_unknown_sessions=True)
        self.assertFalse(audit["cross_session_established"])
        self.assertEqual(audit["protocol_kind"], "image_disjoint_pilot_unknown_sessions")

    def test_missing_gallery_is_not_silently_skipped(self):
        fixture = [row for row in rows() if row.path != Path("test_0_0.png")]
        with self.assertRaisesRegex(ValueError, "both gallery and probe"):
            validate_protocol(fixture)

    def test_identity_split_reproducible_and_complete(self):
        subjects = [str(i) for i in range(200)]
        a = identity_split(subjects, 42)
        self.assertEqual(a, identity_split(list(reversed(subjects)), 42))
        self.assertEqual([list(a.values()).count(split) for split in ("train", "calibration", "test")], [120, 40, 40])
        self.assertNotEqual(a, identity_split(subjects, 43))

    def make_manifest(self, root):
        fixture = rows()
        rng = np.random.default_rng(9)
        for row in fixture:
            image = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
            cv2.imencode(".png", image)[1].tofile(root / row.path)
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "subject_id", "session_id", "split", "role"])
            for row in fixture:
                writer.writerow([str(row.path), row.subject_id, row.session_id, row.split, row.role])
        return manifest, fixture

    def test_manifest_decodes_hashes_and_rejects_duplicate_content(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, fixture = self.make_manifest(root)
            loaded, audit = load_manifest(manifest)
            self.assertEqual(len(loaded), 12)
            self.assertEqual(len(loaded[0].sha256), 64)
            self.assertFalse(audit["near_duplicates_checked"])
            (root / fixture[-1].path).write_bytes((root / fixture[0].path).read_bytes())
            with self.assertRaisesRegex(ValueError, "Duplicate image content"):
                load_manifest(manifest)

    def test_manifest_rejects_corrupt_images(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, fixture = self.make_manifest(root)
            (root / fixture[-1].path).write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cannot decode"):
                load_manifest(manifest)

    def test_end_to_end_software_fixture_and_no_overwrite(self):
        from experiments.run_experiments import parser, run
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.make_manifest(root)
            output = root / "software_fixture_only"
            args = parser().parse_args(["--manifest", str(manifest), "--output", str(output),
                                       "--methods", "lbp", "--bootstrap", "10", "--face-crop", "already-cropped"])
            result = run(args)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["methods"]["lbp"]["fitted_images"], 4)
            self.assertEqual(result["methods"]["lbp"]["test"]["threshold_source"], "independent_calibration")
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "completed")
            self.assertTrue((output / "lbp" / "curves.pdf").is_file())
            with self.assertRaisesRegex(ValueError, "not empty"):
                run(args)

    def test_legacy_loader_supports_variable_size_unicode_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            person = root / "человек"
            person.mkdir()
            for i, shape in enumerate(((40, 50, 3), (60, 45, 3))):
                cv2.imencode(".png", np.zeros(shape, dtype=np.uint8))[1].tofile(person / f"{i}.png")
            images, labels, mapping = load_data(root)
            self.assertEqual([image.shape for image in images], [(40, 50, 3), (60, 45, 3)])
            self.assertEqual(labels.tolist(), [0, 0])
            self.assertEqual(mapping, {"человек": 0})


if __name__ == "__main__":
    unittest.main()

