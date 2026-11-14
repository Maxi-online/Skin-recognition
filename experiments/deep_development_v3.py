"""Frozen pretrained networks see only masked skin crops; no test data are loaded."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_crops(manifest):
    path = Path(manifest).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("records", payload.get("samples"))
    if records is None:
        raise ValueError("Missing preprocessing records")
    if any(r["split"] not in ("train", "calibration") for r in records):
        raise ValueError("Development manifest contains test identities")
    bad_subjects = {r["subject_id"] for r in records if not r.get("cache_path") or
                    not all(q.get("strict_skin_usable", False)
                            for q in r.get("quality", {}).get("patch_quality", {}).values())}
    kept, arrays = [], []
    for record in records:
        if record["subject_id"] in bad_subjects:
            continue
        cache = Path(record["cache_path"])
        if not cache.is_absolute():
            cache = path.parent / cache
        with np.load(cache, allow_pickle=False) as archive:
            names = ("forehead", "left_cheek", "right_cheek", "chin")
            # Rectangles wholly inside the eroded skin mask avoid exposing the
            # silhouette of eyes, lips, or a person-specific mask to the network.
            patches = np.stack([archive["skin_only_" + name] for name in names])
        arrays.append(patches[..., ::-1].copy())
        kept.append(record)
    if not arrays:
        raise ValueError("No successful development skin crops")
    return kept, np.stack(arrays), sorted(bad_subjects)


def extract(model_name, crops, output, batch_size=32):
    import torch
    import torch.nn.functional as F
    import timm
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA environment required; use .venv-gpu")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    model = timm.create_model(model_name, pretrained=True, num_classes=0,
                             **({"img_size": 224} if "dinov2" in model_name else {}))
    # Fingerprint every tensor, not just the architecture or download URL.
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    weights_sha = digest.hexdigest()
    model = model.cuda().eval()
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    flat = crops.reshape(-1, *crops.shape[2:])
    chunks = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(flat), batch_size):
            x = torch.from_numpy(flat[offset:offset+batch_size].transpose(0, 3, 1, 2)).cuda().float() / 255.
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
            x = (x - mean) / std
            features = model.forward_features(x)
            if features.ndim == 3:
                # CLS and spatial token pooling operate inside EACH skin crop.
                tokens = features[:, 1:]
                side = int(tokens.shape[1] ** .5)
                if side * side != tokens.shape[1]:
                    raise ValueError("Unexpected token geometry")
                grid = tokens.transpose(1, 2).reshape(len(x), -1, side, side)
                pooled = F.adaptive_avg_pool2d(grid, (2, 2)).flatten(1)
                vector = torch.cat([F.normalize(features[:, 0], dim=1), F.normalize(pooled, dim=1)], dim=1)
            else:
                pooled = F.adaptive_avg_pool2d(features, (2, 2)).flatten(1)
                vector = F.normalize(pooled, dim=1)
            chunks.append(F.normalize(vector, dim=1).cpu().numpy())
            print(f"{model_name}: skin patches {min(offset+batch_size, len(flat))}/{len(flat)}", flush=True)
    vectors = np.concatenate(chunks).reshape(len(crops), -1)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    metadata = dict(model=model_name, weights_sha256=weights_sha, network_input="masked_skin_crop_only",
                    crop_selection="maximal_rectangle_inside_eroded_skin_mask", input_color="RGB", resize=[224, 224],
                    trainable_backbone=False, seconds=time.perf_counter()-started,
                    device=torch.cuda.get_device_name(0), torch=torch.__version__, timm=timm.__version__)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=vectors, metadata=np.array(json.dumps(metadata)))
    return vectors, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "research/development/v3_faces/manifest.json"))
    parser.add_argument("--output", default=str(ROOT / "research/development/v3_deep"))
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = args.model.replace(".", "_")
    records, crops, excluded = load_crops(args.manifest)
    (output / "cohort.json").write_text(json.dumps(dict(records=records, excluded_subjects=excluded,
                                                        test_images_loaded=0), indent=2), encoding="utf-8")
    features, metadata = extract(args.model, crops, output / f"{prefix}_features.npz", args.batch_size)
    from experiments.learned_metric_v3 import select_metric
    report = select_metric(features, records, output, prefix)
    print(json.dumps(dict(backbone=metadata, best=report["best"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
