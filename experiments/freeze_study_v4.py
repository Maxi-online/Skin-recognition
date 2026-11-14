"""Freeze already selected FEI development methods before external inference."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing_v3 import PreprocessingConfigV3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    directory = ROOT / 'research/development/v4_hybrid_multicond_gpu'
    development_path = directory / 'freeze.json'
    margin_path = directory / 'search_operating_points_margin.json'
    development = json.loads(development_path.read_text(encoding='utf-8'))
    margin = json.loads(margin_path.read_text(encoding='utf-8'))
    if margin['source_freeze_sha256'] != sha(development_path):
        raise ValueError('Search operating point belongs to a different development run')
    configuration = development['model_configuration']
    global_path = str(Path(configuration['global_model_path']).resolve())
    skin_path = str(Path(configuration['skin_model_path']).resolve().relative_to(ROOT)).replace('\\', '/')
    if sha(global_path) != configuration['global_model_sha256'] or sha(ROOT / skin_path) != configuration['skin_model_sha256']:
        raise ValueError('Selected model checksum mismatch')
    candidates = {m['method_id']: m for m in development['methods']}
    operating_points = {m['method_id']: m for m in margin['methods']}
    selection = [('skin_texture', 'skin_only'), ('whole_face', 'global_only'),
                 ('hybrid', development['best_positive_hybrid_method_id']),
                 ('whole_face_detector_only', 'whole_face_detector_only')]
    methods = []
    for method_id, original_id in selection:
        method = dict(candidates[original_id])
        point = operating_points[original_id]
        if method['feature_space_id'] != point['feature_space_id'] or method['score_threshold'] != point['primary_score_threshold']:
            raise ValueError('Calibration and search-point contracts differ')
        method.update(method_id=method_id, development_method_id=original_id,
                      search_score_threshold=point['search_score_threshold'], search_threshold_rule=point['rule'],
                      previous_zero_false_match_threshold=point['previous_zero_false_match_threshold'],
                      search_calibration_metrics=point['metrics'])
        if method['scope'] == 'whole_face_plus_skin':
            method['scope'] = 'hybrid'
        method['preprocessing_id'] = (configuration['preprocessing_id'].removesuffix(':strict') + ':detector_only'
                                      if method['scope'] == 'whole_face_detector_only' else configuration['preprocessing_id'])
        if method['scope'] in ('skin_only', 'hybrid'):
            method['skin_model_path'] = skin_path
        if method['scope'] in ('whole_face', 'whole_face_detector_only', 'hybrid'):
            method['global_model_path'] = global_path
        methods.append(method)
    sources = [
        'experiments/freeze_study_v4.py', 'experiments/extract_external_v4.py',
        'experiments/external_evaluate_v4.py', 'experiments/external_evaluate_v3.py',
        'experiments/scientific_metrics.py', 'experiments/development_v3.py',
        'experiments/development_v4.py', 'experiments/development_v4_metrics.py',
        'experiments/hybrid_development_v4.py', 'experiments/calibrate_search_v3.py',
        'src/preprocessing_v3.py', 'src/descriptors/hybrid_v4.py',
        'src/descriptors/local_patches.py', 'src/descriptors/_image.py',
        'src/utils/data_loader.py', 'src/services/recognition_v4.py',
        'src/services/texture_v3.py', 'src/database/database.py']
    model_dir = Path(global_path).parent
    models = {global_path: sha(global_path), skin_path: sha(ROOT / skin_path)}
    for name in ('det_10g.onnx', '2d106det.onnx'):
        path = str(model_dir / name)
        models[path] = sha(path)
    protocols = {}
    for name, path in [('lfw', 'research/data/lfw/external_protocol_v4_1.json'),
                       ('orl', 'research/data/orl/external_protocol_v4_1.json'),
                       ('fei_holdout', 'research/data/fei/v4_subject_holdout_protocol.json')]:
        protocols[name] = dict(path=path, sha256=sha(ROOT / path))
    freeze = dict(schema_version='study-v4.1', timestamp=datetime.now(timezone.utc).isoformat(),
                  state='frozen_before_first_v4_external_feature_extraction', methods=methods,
                  selected_on_development='whole_face', positive_hybrid_weight=methods[2]['skin_weight'],
                  runtime=dict(global_provider='cuda', preprocessing_provider='cuda',
                               embedding_storage='float32', score_accumulation='scipy_cdist_float64',
                               gpu='NVIDIA GeForce RTX 3060 12GB', python_environment='.venv-gpu'),
                  preprocessing_config=asdict(PreprocessingConfigV3()), preprocessing_model_dir=str(model_dir),
                  source_hashes={path: sha(ROOT / path) for path in sources}, model_artifacts=models,
                  protocols=protocols, development_freeze=dict(path=str(development_path), sha256=sha(development_path)),
                  development_search_points=dict(path=str(margin_path), sha256=sha(margin_path)),
                  scope_notes=[
                      'LFW and ORL were acquired and structurally audited, with no prior v4 inference.',
                      'FEI heldout identities were excluded from all v4 fitting and selection; their frontal v2 benchmark was previously observed.',
                      'Georgia Tech is a known diagnostic benchmark and does not serve as the untouched v4 holdout.',
                      'Global recognition uses externally pretrained WebFace600K R50 weights; pretraining/evaluation identity overlap is unknown.',
                      'The first three methods share strict skin-quality gating; detector-only global is a separately declared coverage ablation.',
                      'Primary thresholds target FEI development pair FAR<=1%; separate search points maximize the FEI class gap when separable.',
                      'Only conventional secondary LFW ten-fold threshold calibration is allowed; models and fusion weights remain fixed.'])
    destination = ROOT / 'research/frozen/study_v4_1.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError('The frozen study manifest must not be overwritten')
    destination.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(path=str(destination), sha256=sha(destination), methods=[m['method_id'] for m in methods])))


if __name__ == '__main__':
    main()
