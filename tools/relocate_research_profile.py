"""Create an auditable path-only copy of the frozen profile on another machine.

Original manifests are never overwritten. Model bytes, descriptor identifiers,
fusion weights and decision thresholds remain unchanged and are verified.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def portable_name(path):
    return str(path).replace('\\', '/').rsplit('/', 1)[-1]


def rebase(document, model_dir):
    result = copy.deepcopy(document)
    mapping = {}
    artifacts = {}
    for path, expected in document['model_artifacts'].items():
        name = portable_name(path)
        candidate = model_dir / name if name.endswith('.onnx') else resolve(path)
        if sha(candidate) != expected:
            raise ValueError('Model bytes differ from the published experiment: ' + str(candidate))
        mapped = str(candidate.resolve()) if name.endswith('.onnx') else path
        mapping[path] = mapped
        artifacts[mapped] = expected
    result['model_artifacts'] = artifacts
    result['preprocessing_model_dir'] = str(model_dir.resolve())
    for method in result['methods']:
        for key in ('skin_model_path', 'global_model_path'):
            if key in method:
                method[key] = mapping[method[key]]
    for path, expected in result['source_hashes'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Inference/evaluation source differs from the published experiment: ' + path)
    return result


def write_new(path, document):
    if path.exists():
        raise FileExistsError('Path-only profile already exists: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path.home() / '.insightface/models/buffalo_l')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    originals = {version: ROOT / f'research/frozen/study_v4_{version}.json' for version in (1, 2)}
    documents = {version: json.loads(path.read_text(encoding='utf-8')) for version, path in originals.items()}
    if documents[2]['base_freeze_sha256'] != sha(originals[1]):
        raise ValueError('Original frozen profile lineage is inconsistent')
    rebased = {version: rebase(document, args.model_dir) for version, document in documents.items()}
    destinations = {version: ROOT / f'research/frozen/relocated_study_v4_{version}.json' for version in (1, 2)}
    provenance = dict(operation='path_only_relocation', created_utc=datetime.now(timezone.utc).isoformat(),
                      original_manifest_sha256={str(version): sha(path) for version, path in originals.items()},
                      model_bytes_and_recognition_parameters_changed=False)
    for version in (1, 2):
        rebased[version]['path_relocation'] = dict(provenance, original_version=version)
    if args.dry_run:
        print(json.dumps(dict(status='verified', model_dir=str(args.model_dir.resolve()),
                              original_manifests=provenance['original_manifest_sha256']), indent=2))
        return
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError('A relocated manifest already exists; originals and existing copies are preserved')
    write_new(destinations[1], rebased[1])
    rebased[2]['base_freeze_path'] = destinations[1].relative_to(ROOT).as_posix()
    rebased[2]['base_freeze_sha256'] = sha(destinations[1])
    write_new(destinations[2], rebased[2])
    print(json.dumps(dict(freeze_path=str(destinations[2]), method_id='hybrid', backend='v42',
                          model_and_source_hashes_verified=True), indent=2))


if __name__ == '__main__':
    main()
