"""Freeze the native-scale input repair without refitting v4.1 recognition."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def main():
    from src.preprocessing_v42 import FacePreprocessorV42
    from src.preprocessing_v3 import PreprocessingConfigV3
    base_path = ROOT / 'research/frozen/study_v4_1.json'
    base = json.loads(base_path.read_text(encoding='utf-8'))
    for path, digest in base['source_hashes'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Frozen v4.1 source changed: ' + path)
    processor = FacePreprocessorV42(model_dir=base['preprocessing_model_dir'],
                                    config=PreprocessingConfigV3(**base['preprocessing_config']),
                                    providers=['CPUExecutionProvider'])
    record = copy.deepcopy(base)
    record.update(schema_version='study-v4.2', timestamp=datetime.now(timezone.utc).isoformat(),
                  state='native_scale_input_repair_frozen_before_first_CFP_inference',
                  base_freeze_path='research/frozen/study_v4_1.json', base_freeze_sha256=sha(base_path),
                  input_repair=dict(rule='For source frames smaller than the detector canvas, center on zero padding without magnification; map detections to original coordinates. If several faces are detected, select only a unique face bounding box containing the original image center; otherwise reject ambiguity.',
                                    development_evidence='Native-scale failure reproduced on three artificially reduced FEI training crops and a known ORL diagnostic image; CPU/GPU agreed. Known LFW multi-face failures motivated the explicit centered-target rule before CFP inference.',
                                    model_refitting=False, fusion_retuning=False, threshold_retuning=False,
                                    full_FEI_frame_behavior='640x480 inputs pass through without geometric change',
                                    effective_preprocessing_id=processor.preprocessing_id,
                                    legacy_descriptor_preprocessing_id=processor.legacy_preprocessing_id))
    for method in record['methods']:
        method['base_preprocessing_id'] = method['preprocessing_id']
        method['legacy_preprocessing_id'] = method['preprocessing_id']
        suffix = ':detector_only' if method['scope'] == 'whole_face_detector_only' else ':strict'
        method['preprocessing_id'] = processor.preprocessing_id + suffix
    new_sources = ['src/preprocessing_v42.py', 'src/services/recognition_v42.py',
                   'experiments/extract_external_v42.py', 'experiments/external_evaluate_v42.py',
                   'experiments/freeze_study_v42.py']
    record['source_hashes'].update({path: sha(ROOT / path) for path in new_sources})
    record['protocols'] = {}
    for name, information in base['protocols'].items():
        original = ROOT / information['path']
        protocol = json.loads(original.read_text(encoding='utf-8'))
        prior_scope = protocol.get('evaluation_scope')
        protocol.update(repair_protocol_created_utc=record['timestamp'],
                        parent_protocol_path=information['path'],
                        parent_protocol_sha256=information['sha256'],
                        prior_evaluation_scope=prior_scope,
                        evaluation_scope='known_benchmark_geometry_repair_audit_v4_2',
                        status_at_freeze='The v4.1 baseline was observed. This v4.2 rerun measures the declared input-geometry repair, with unchanged recognition models and thresholds.')
        path = original.parent / 'geometry_repair_protocol_v4_2.json'
        write_new(path, protocol)
        record['protocols'][name] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))
    for dataset, filename in [('cfp_ff', 'external_protocol_ff_v4_2.json'), ('cfp_fp', 'external_protocol_fp_v4_2.json')]:
        path = ROOT / 'research/data/cfp' / filename
        record['protocols'][dataset] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))
    record['scope_notes'] = [
        'CFP frontal-frontal is the new untouched primary external benchmark; CFP frontal-profile is a prespecified pose stress test.',
        'LFW and ORL v4.2 results are post-repair evaluations on known input distributions; all v4.1 baseline artifacts remain available.',
        'FEI heldout evaluation is retained with prior v2 frontal exposure and prior v4.1 measurement explicitly recorded.',
        'All v4.1 recognition weights, fusion weights, pair thresholds and search thresholds are preserved exactly.',
        'Externally pretrained WebFace600K R50 weights are used; pretraining identity/image overlap with public benchmarks is not audited.',
        'Secondary standard ten-fold threshold-CV metrics are separate from the transferred FEI decision thresholds.']
    destination = ROOT / 'research/frozen/study_v4_2.json'
    write_new(destination, record)
    print(json.dumps(dict(path=str(destination), sha256=sha(destination), preprocessing_id=processor.preprocessing_id)))


if __name__ == '__main__':
    main()
