"""Extract frozen v4.2 methods on a declared external inventory, without fitting.

The common preprocessing gate is identical for all ablation methods. Models
can execute on CUDA; the chosen runtime is recorded and never inferred from its
availability. Numeric checkpoints permit safe continuation of long evaluations.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def checked(path, expected):
    path = resolve(path)
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return path


def read_freeze(freeze_path, dataset):
    freeze_path = resolve(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    base_path = checked(freeze['base_freeze_path'], freeze['base_freeze_sha256'])
    base = json.loads(base_path.read_text(encoding='utf-8'))
    previous = {m['method_id']: m for m in base['methods']}
    if {m['method_id'] for m in freeze['methods']} != set(previous):
        raise ValueError('Geometry-only repair must preserve every frozen method')
    preserved_fields = ('scope', 'skin_weight', 'feature_space_id', 'score_threshold',
                        'search_score_threshold', 'skin_model_path', 'global_model_path')
    for method in freeze['methods']:
        if any(method.get(key) != previous[method['method_id']].get(key) for key in preserved_fields):
            raise ValueError('Geometry-only repair changed a frozen descriptor or decision rule')
    if freeze.get('schema_version') != 'study-v4.2':
        raise ValueError('Expected study-v4.2 freeze')
    when = datetime.fromisoformat(freeze['timestamp'].replace('Z', '+00:00'))
    if when.tzinfo is None or when > datetime.now(timezone.utc):
        raise ValueError('Freeze must predate extraction and specify timezone')
    required_sources = {'experiments/extract_external_v4.py', 'experiments/extract_external_v42.py',
                        'src/preprocessing_v42.py', 'experiments/external_evaluate_v42.py', 'experiments/development_v3.py',
                        'src/descriptors/hybrid_v4.py', 'src/preprocessing_v3.py',
                        'experiments/external_evaluate_v4.py', 'src/utils/data_loader.py'}
    if not required_sources.issubset(freeze['source_hashes']):
        raise ValueError('Freeze omits required implementation hashes')
    for group in ('source_hashes', 'model_artifacts'):
        for path, digest in freeze[group].items():
            checked(path, digest)
    info = freeze['protocols'][dataset]
    protocol_path = checked(info['path'], info['sha256'])
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    inventory = checked(protocol['inventory_path'], protocol['inventory_sha256'])
    with inventory.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate inventory path')
    if 'pair_manifest_path' in protocol:
        pairs_path = checked(protocol['pair_manifest_path'], protocol['pair_manifest_sha256'])
        with pairs_path.open(encoding='utf-8-sig', newline='') as handle:
            pairs = list(csv.DictReader(handle))
        needed = {p[key] for p in pairs for key in ('path_a', 'path_b')}
        rows = [r for r in rows if r['path'] in needed]
        if len(rows) != len(needed):
            raise ValueError('Pair endpoint missing from inventory')
    methods = freeze['methods']
    if not methods or len({m['method_id'] for m in methods}) != len(methods):
        raise ValueError('Unique nonempty methods required')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', m['method_id']) for m in methods):
        raise ValueError('Method IDs must be safe file names')
    return freeze, protocol, rows


def validate_checkpoint_records(records, rows, vectors, dimension):
    if len(records) > len(rows):
        raise ValueError('Checkpoint has too many inputs')
    accepted = 0
    for record, row in zip(records, rows):
        if any(record.get(key) != row[source] for key, source in
               [('path', 'path'), ('subject_id', 'subject_id'), ('source_sha256', 'sha256')]):
            raise ValueError('Checkpoint metadata differs from frozen inventory')
        codes = record.get('rejection_codes')
        if not isinstance(codes, list) or any(not isinstance(code, str) or not code for code in codes):
            raise ValueError('Malformed checkpoint rejection codes')
        if record.get('status') == 'accepted':
            if type(record.get('embedding_index')) is not int or record['embedding_index'] != accepted or codes:
                raise ValueError('Checkpoint embedding index/status mismatch')
            accepted += 1
        elif record.get('status') == 'rejected':
            if record.get('embedding_index') is not None or not codes:
                raise ValueError('Checkpoint rejection is incomplete')
        else:
            raise ValueError('Unknown checkpoint quality status')
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.shape != (accepted, dimension) or not np.isfinite(matrix).all():
        raise ValueError('Checkpoint embedding matrix shape/values invalid')
    if accepted and not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6):
        raise ValueError('Checkpoint embeddings are not unit vectors')


def extract(freeze_path, dataset, output, global_provider='cpu', preprocessing_provider='cpu'):
    from experiments.development_v3 import ClassicalTextureDescriptorV3
    from src.descriptors.hybrid_v4 import (ArcFaceDescriptorV4, HybridFaceSkinDescriptorV4,
                                         fuse_embeddings, source_landmarks_from_cache)
    from src.preprocessing_v3 import PreprocessingConfigV3, QualityRejection
    from src.preprocessing_v42 import (FacePreprocessorV42, prepared_tuple_from_face_v42,
                                       source_landmarks_for_target_face_v42)
    from src.utils.data_loader import read_image

    freeze_path = resolve(freeze_path)
    freeze, protocol, rows = read_freeze(freeze_path, dataset)
    frozen_sha = sha256(freeze_path)
    output = resolve(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'checkpoint.json'
    provider = ['CPUExecutionProvider']
    if global_provider == 'cuda' or preprocessing_provider == 'cuda':
        import torch  # Load matching CUDA/cuDNN before creating ONNX sessions.
        import onnxruntime as ort
        ort.preload_dlls()
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')
        if global_provider == 'cuda':
            provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if freeze.get('runtime', {}).get('global_provider', 'cpu') != global_provider or freeze.get('runtime', {}).get('preprocessing_provider', 'cpu') != preprocessing_provider:
        raise ValueError('Runtime provider differs from frozen evaluation plan')
    preprocessing_providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                               if preprocessing_provider == 'cuda' else ['CPUExecutionProvider'])
    processor = FacePreprocessorV42(model_dir=freeze.get('preprocessing_model_dir'),
                                  config=PreprocessingConfigV3(**freeze.get('preprocessing_config', {})),
                                  providers=preprocessing_providers)
    if preprocessing_provider == 'cuda' and any(value[0] != 'CUDAExecutionProvider' for value in processor.runtime_providers.values()):
        raise RuntimeError('Preprocessing silently fell back from CUDA')
    methods = freeze['methods']
    if any(m['preprocessing_id'] != processor.preprocessing_id +
           (':detector_only' if m['scope'] == 'whole_face_detector_only' else ':strict') for m in methods):
        raise ValueError('Common preprocessing contract mismatch')
    skin_models, global_models = {}, {}
    for method in methods:
        skin_path = method.get('skin_model_path')
        global_path = method.get('global_model_path')
        for path in (skin_path, global_path):
            if path and path not in freeze['model_artifacts']:
                raise ValueError('Method model is absent from frozen model_artifacts')
        if skin_path and skin_path not in skin_models:
            skin_models[skin_path] = ClassicalTextureDescriptorV3.load(resolve(skin_path))
        if global_path and global_path not in global_models:
            model = ArcFaceDescriptorV4(resolve(global_path), providers=provider)
            if global_provider == 'cuda' and model.runtime_providers[0] != 'CUDAExecutionProvider':
                raise RuntimeError('ArcFace silently fell back from CUDA')
            global_models[global_path] = model
    for method in methods:
        if method['scope'] == 'skin_only':
            actual_id = skin_models[method['skin_model_path']].feature_space_id
        elif method['scope'].startswith('whole_face'):
            actual_id = global_models[method['global_model_path']].feature_space_id
        elif method['scope'] == 'hybrid':
            identity_check = HybridFaceSkinDescriptorV4(
                resolve(method['global_model_path']), resolve(method['skin_model_path']),
                skin_weight=method['skin_weight'], preprocessor=processor, providers=['CPUExecutionProvider'])
            actual_id = identity_check.feature_space_id
            del identity_check
        else:
            raise ValueError('Unknown frozen method scope')
        if actual_id != method['feature_space_id']:
            raise ValueError('Frozen feature-space identity mismatch: ' + method['method_id'])
    header = dict(freeze_sha256=frozen_sha, dataset=dataset,
                  protocol_sha256=freeze['protocols'][dataset]['sha256'], global_provider=global_provider,
                  preprocessing_provider=preprocessing_provider,
                  common_preprocessing_providers=processor.runtime_providers)
    features = {m['method_id']: [] for m in methods}
    dimensions = {}
    for method in methods:
        skin_dim = skin_models[method['skin_model_path']].descriptor_dim if method.get('skin_model_path') else 0
        dimensions[method['method_id']] = (skin_dim if method['scope'] == 'skin_only' else
                                          512 if method['scope'].startswith('whole_face') else 512 + skin_dim)

    def matrix(method_id):
        return np.asarray(features[method_id], dtype=np.float64).reshape((-1, dimensions[method_id]))

    quality = {m['method_id']: [] for m in methods}
    started = stamp()
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        if any(saved.get(key) != value for key, value in header.items()):
            raise ValueError('Checkpoint freeze/runtime mismatch')
        started = saved['extraction_started_utc']
        quality = saved['quality']
        if len({len(q) for q in quality.values()}) != 1:
            raise ValueError('Checkpoint methods have different input counts')
        for method_id in features:
            if [q['path'] for q in quality[method_id]] != [r['path'] for r in rows[:len(quality[method_id])]]:
                raise ValueError('Checkpoint is not a prefix of the frozen inventory')
            path = output / f'{method_id}.checkpoint.npz'
            checked(path, saved['checkpoint_sha256'][method_id])
            with np.load(path, allow_pickle=False) as arrays:
                validate_checkpoint_records(quality[method_id], rows, arrays['embeddings'], dimensions[method_id])
                features[method_id] = list(arrays['embeddings'])
            if len(features[method_id]) != sum(q['status'] == 'accepted' for q in quality[method_id]):
                raise ValueError('Checkpoint embedding count mismatch')
    elif any(output.iterdir()):
        raise ValueError('Nonempty output has no validated checkpoint')

    def save_checkpoint():
        hashes = {}
        for method_id, vectors in features.items():
            path = output / f'{method_id}.checkpoint.npz'
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, embeddings=matrix(method_id))
            temporary.replace(path)
            hashes[method_id] = sha256(path)
        write_json(checkpoint, {**header, 'extraction_started_utc': started, 'quality': quality,
                               'checkpoint_sha256': hashes})

    clock = time.perf_counter()
    for index in range(len(next(iter(quality.values()))), len(rows)):
        row = rows[index]
        image_path = checked(row['path'], row['sha256'])
        record = dict(path=row['path'], subject_id=row['subject_id'], source_sha256=row['sha256'])
        image = read_image(image_path)
        strict_errors, detector_errors, computed = [], [], {}
        try:
            face = processor.process(image)
            record['target_selection'] = face.metadata.get('target_selection', {})
            packed, preproc_id, effective_preproc_id = prepared_tuple_from_face_v42(face)
            if effective_preproc_id != processor.preprocessing_id + ":strict":
                raise ValueError("Effective v4.2 packing contract mismatch")
            skin = {path: model.extract_features(packed, preproc_id) for path, model in skin_models.items()}
            kps = source_landmarks_from_cache(face.kps5, face.transform)
            global_features = {path: model.extract_features(image, kps) for path, model in global_models.items()}
            for method in methods:
                scope = method['scope']
                if scope == 'skin_only':
                    vector = skin[method['skin_model_path']]
                elif scope in ('whole_face', 'whole_face_detector_only'):
                    vector = global_features[method['global_model_path']]
                elif scope == 'hybrid':
                    vector = fuse_embeddings(global_features[method['global_model_path']],
                                             skin[method['skin_model_path']], method['skin_weight'])
                else:
                    raise ValueError(f'Unknown method scope: {scope}')
                computed[method['method_id']] = vector
        except QualityRejection as error:
            strict_errors = error.codes
            record['target_selection'] = error.metadata.get('target_selection', {})
            if any(m['scope'] == 'whole_face_detector_only' for m in methods):
                try:
                    kps, detection_quality = source_landmarks_for_target_face_v42(processor, image)
                    record['target_selection'] = detection_quality.get('target_selection', {})
                    global_features = {path: model.extract_features(image, kps) for path, model in global_models.items()}
                    for method in methods:
                        if method['scope'] == 'whole_face_detector_only':
                            computed[method['method_id']] = global_features[method['global_model_path']]
                except QualityRejection as detector_error:
                    detector_errors = detector_error.codes
        # The detector-only ablation always uses original detector landmarks;
        # it does not depend on a landmark106 round trip, even on accepted skin.
        if not strict_errors and any(m['scope'] == 'whole_face_detector_only' for m in methods):
            try:
                kps, detection_quality = source_landmarks_for_target_face_v42(processor, image)
                record['target_selection'] = detection_quality.get('target_selection', {})
                global_features = {path: model.extract_features(image, kps) for path, model in global_models.items()}
                for method in methods:
                    if method['scope'] == 'whole_face_detector_only':
                        computed[method['method_id']] = global_features[method['global_model_path']]
            except QualityRejection as detector_error:
                detector_errors = detector_error.codes
                for method in methods:
                    if method['scope'] == 'whole_face_detector_only':
                        computed.pop(method['method_id'], None)
        for method in methods:
            method_id = method['method_id']
            item = dict(record)
            if method_id in computed:
                # Match float32 persisted live-gallery values, float64 scores.
                vector = np.asarray(computed[method_id], dtype=np.float32).astype(np.float64)
                if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-6):
                    raise ValueError('Invalid unit embedding')
                item.update(status='accepted', embedding_index=len(features[method_id]), rejection_codes=[])
                features[method_id].append(vector)
            else:
                errors = detector_errors if method['scope'] == 'whole_face_detector_only' else strict_errors
                if not errors:
                    raise ValueError('Method has neither an embedding nor a quality rejection')
                item.update(status='rejected', embedding_index=None, rejection_codes=errors)
            quality[method_id].append(item)
        if (index + 1) % 100 == 0:
            save_checkpoint()
            counts = {key: len(value) for key, value in features.items()}
            print(f'{dataset} {index+1}/{len(rows)}; accepted={counts}; elapsed={time.perf_counter()-clock:.1f}s', flush=True)
    save_checkpoint()
    for method in methods:
        method_id = method['method_id']
        directory = output / method_id
        directory.mkdir(exist_ok=True)
        embeddings_path = directory / 'embeddings.npz'
        with embeddings_path.open('wb') as stream:
            np.savez_compressed(stream, embeddings=matrix(method_id))
        quality_path = directory / 'quality.json'
        write_json(quality_path, quality[method_id])
        write_json(directory / 'bundle.json', {
            **header, 'method_id': method_id, 'scope': method['scope'], 'protocol_id': protocol['protocol_id'],
            'feature_space_id': method['feature_space_id'],
            'preprocessing_id': method['preprocessing_id'], 'extraction_started_utc': started,
            'extraction_completed_utc': stamp(), 'freeze_path': str(freeze_path),
            'embeddings_path': str(embeddings_path), 'embeddings_sha256': sha256(embeddings_path),
            'quality_path': str(quality_path), 'quality_sha256': sha256(quality_path),
            'required_images': len(rows), 'accepted_images': len(features[method_id]),
            'global_runtime_providers': {path: model.runtime_providers for path, model in global_models.items()}})
    print(json.dumps({'status': 'completed', 'dataset': dataset, 'images': len(rows),
                      'accepted': {key: len(value) for key, value in features.items()}, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--dataset', choices=['lfw', 'orl', 'fei_holdout', 'cfp_ff', 'cfp_fp'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--global-provider', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--preprocessing-provider', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    extract(args.freeze, args.dataset, args.output, args.global_provider, args.preprocessing_provider)
