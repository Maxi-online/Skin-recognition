"""Replay saved descriptor artifacts against recorded source images/crops."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_experiments import TextureBaseline
from src.descriptors import HybridSkinDescriptor
from src.utils.data_loader import read_image


def verify(run_dir, samples_per_method=3):
    run_dir = Path(run_dir).resolve()
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    boxes = {row["path"]: row["bbox_xywh"] for row in
             json.loads((run_dir / "face_boxes.json").read_text(encoding="utf-8"))}
    if result["status"] != "completed" or samples_per_method < 1:
        raise ValueError("Completed run and positive replay sample count required")
    report = {"status": "passed", "purpose": "artifact_roundtrip_reproducibility_not_independent_accuracy",
              "methods": {}}
    for method, metrics in result["methods"].items():
        artifact = run_dir / method / Path(metrics["descriptor_artifact"]).name
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != metrics["descriptor_sha256"]:
            raise ValueError(f"Saved descriptor checksum mismatch: {method}")
        descriptor = (HybridSkinDescriptor.load(artifact) if artifact.suffix == ".npz" else TextureBaseline(method))
        if descriptor.feature_space_id != metrics["feature_space_id"]:
            raise ValueError(f"Feature-space mismatch: {method}")
        with np.load(run_dir / method / "embeddings.npz", allow_pickle=False) as archive:
            paths, features = archive["paths"], archive["embeddings"]
            indices = np.unique(np.linspace(0, len(paths)-1, min(samples_per_method, len(paths))).astype(int))
            errors = []
            for index in indices:
                path = str(paths[index])
                image = read_image(path)
                bbox = boxes[path]
                if bbox is not None:
                    x, y, w, h = bbox
                    image = image[y:y+h, x:x+w]
                size = config["image_size"]
                interpolation = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_LINEAR
                image = cv2.resize(image, (size, size), interpolation=interpolation)
                actual = descriptor.extract_features(image)
                error = float(np.max(np.abs(actual - features[index])))
                if not np.allclose(actual, features[index], rtol=1e-6, atol=1e-7):
                    raise ValueError(f"Replay embedding mismatch: {method} {path}, error={error}")
                errors.append(error)
        report["methods"][method] = {"samples_replayed": len(indices), "max_absolute_error": max(errors),
                                     "feature_space_id": metrics["feature_space_id"]}
    output = run_dir / "artifact_replay_check.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--samples-per-method", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, args.samples_per_method), ensure_ascii=False, indent=2))

