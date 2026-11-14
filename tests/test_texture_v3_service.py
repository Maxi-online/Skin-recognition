"""V3 service contracts with synthetic inputs; no external-test dataset access."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

import src.services.texture_v3 as module
from src.preprocessing_v3 import PreprocessingConfigV3, QualityRejection

BASE_ID="synthetic-preprocessor-contract"
PREPROCESSING_ID=BASE_ID+":strict"
CONFIGURATION={"features":{"scope":"skin_regions","descriptor":"rootsift","photometric":"raw","grid":4},
               "transform":{"kind":"identity"}}


class StubDescriptor:
    feature_space_id="synthetic-frozen-descriptor"
    preprocessing_id=PREPROCESSING_ID
    descriptor_dim=2
    configuration=CONFIGURATION

    def extract_features(self, prepared, preprocessing_id):
        assert preprocessing_id==self.preprocessing_id
        angle=float(prepared[0,0,0])/100.0
        return np.array([np.cos(angle),np.sin(angle)],np.float32)


class StubPreprocessor:
    preprocessing_id=BASE_ID
    model_sha256={"detector":"synthetic-detector","landmarks":"synthetic-landmarks"}
    def __init__(self,**kwargs):
        self.images=[]
        self._lock=threading.RLock()
        self.detector=SimpleNamespace(detect=self._detect)

    def process(self,image):
        if image.shape[:2]!=(80,100):
            raise QualityRejection(["full_frame_shape_expected"])
        self.images.append(image.copy())
        return image

    def _detect(self,image,**kwargs):
        return np.array([[20,10,70,65,.9]],np.float32),np.ones((1,5,2),np.float32)*30


@pytest.fixture
def frozen(tmp_path,monkeypatch):
    model=tmp_path/"model.npz"
    model.write_bytes(b"synthetic-model-loader-is-patched-for-unit-tests")
    record={
        "schema_version":"skin-recognition-v3-freeze-1",
        "model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
        "feature_space_id":StubDescriptor.feature_space_id,
        "preprocessing_id":PREPROCESSING_ID,
        "preprocessor_base_id":BASE_ID,
        "preprocessing_mode":"landmark_strict",
        "preprocessing_config":asdict(PreprocessingConfigV3()),
        "preprocessing_model_sha256":StubPreprocessor.model_sha256,
        "configuration":CONFIGURATION,
        "score_definition":"negative_euclidean_distance",
        "score_threshold":-.8,
        "decision_rule":"accept if score >= score_threshold",
        "calibration_metrics":{"threshold":-.8},
        "primary_method":"synthetic-service-contract-test",
        "candidate_id":"synthetic-candidate",
        "source_sha256":{name:hashlib.sha256((module.ROOT/name).read_bytes()).hexdigest()
                         for name in ("experiments/development_v3.py","src/preprocessing_v3.py")},
    }
    calibration=tmp_path/"freeze.json"
    calibration.write_text(json.dumps(record),encoding="utf-8")
    monkeypatch.setattr(module.ClassicalTextureDescriptorV3,"load",lambda path:StubDescriptor())
    monkeypatch.setattr(module,"FacePreprocessorV3",StubPreprocessor)
    monkeypatch.setattr(module,"prepared_tuple_from_face",lambda image,mode:(image,PREPROCESSING_ID))
    return model,calibration,record


def make_image(value=0):
    return np.full((80,100,3),value,np.uint8)


def test_full_frame_input_squared_threshold_and_similarity(frozen):
    model,calibration,_=frozen
    service=module.SkinRecognitionSystemV3(model,calibration)
    service.register_person(make_image(),"a","Alice")
    name,similarity,candidates,distances=service.recognize(make_image(80),return_distances=True)
    assert name=="Alice"
    assert distances[0]==pytest.approx(2-2*np.cos(.8),abs=1e-7)
    assert service.recognition_threshold==pytest.approx(.64)
    assert similarity==pytest.approx(1-distances[0]/4)
    assert service.recognize(make_image(120))[0] is None
    assert all(image.shape==(80,100,3) for image in service.preprocessor.images)
    assert service.input_mode=="full_frame"
    stats=service.get_database_stats()
    assert stats["calibration_scope"]=="pairwise_verification_development"
    assert stats["identification_error_guarantee"] is False


def test_direct_overlay_uses_original_frame_coordinates(frozen):
    model,calibration,_=frozen
    service=module.SkinRecognitionSystemV3(model,calibration)
    information=service.detect_face_info(make_image())
    assert information["bbox"]==(20,10,50,55)
    assert "kps" in information and "lmk106" not in information
    service.preprocessor.detector.detect=lambda *args,**kwargs:(np.zeros((2,5)),None)
    with pytest.raises(QualityRejection,match="multiple_faces"):
        service.detect_face_info(make_image())


def test_gallery_save_reload_and_identity_ranking(frozen,tmp_path):
    model,calibration,_=frozen
    path=tmp_path/"gallery"
    service=module.SkinRecognitionSystemV3(model,calibration,database_path=path)
    service.register_person(make_image(),"a","Alice")
    service.register_person(make_image(10),"a","Alice")
    service.register_person(make_image(140),"b","Bob")
    loaded=module.SkinRecognitionSystemV3(model,calibration,database_path=path)
    name,_,candidates=loaded.recognize(make_image())
    assert name=="Alice"
    assert [x["person_id"] for x in candidates]==["a","b"]
    assert loaded.remove_person("a")==2
    assert loaded.get_database_stats()["total_embeddings"]==1


def test_cross_identity_exact_tie_is_rejected(frozen):
    model,calibration,_=frozen
    service=module.SkinRecognitionSystemV3(model,calibration)
    service.register_person(make_image(),"a","Alice")
    service.register_person(make_image(),"b","Bob")
    assert service.recognize(make_image())[0] is None


def test_enrollment_and_deletion_io_failures_restore_live_gallery(frozen,tmp_path,monkeypatch):
    model,calibration,_=frozen
    service=module.SkinRecognitionSystemV3(model,calibration,database_path=tmp_path/"gallery")
    service.register_person(make_image(),"a","Alice")
    before=service.get_database_stats()
    next_id=service.database.next_id
    def failure(path):raise OSError("simulated disk full")
    monkeypatch.setattr(service.database,"save",failure)
    with pytest.raises(OSError):
        service.register_person(make_image(20),"b","Bob")
    assert service.get_database_stats()==before
    assert service.database.next_id==next_id
    with pytest.raises(OSError):
        service.remove_person("a")
    assert service.get_database_stats()==before


def test_profile_checksum_mismatch_fails_closed(frozen):
    model,calibration,record=frozen
    record["model_sha256"]="0"*64
    calibration.write_text(json.dumps(record),encoding="utf-8")
    with pytest.raises(ValueError,match="checksum"):
        module.SkinRecognitionSystemV3(model,calibration)


def test_feature_space_mutation_is_detected(frozen):
    model,calibration,_=frozen
    service=module.SkinRecognitionSystemV3(model,calibration)
    service.descriptor.feature_space_id="changed-after-enrollment"
    with pytest.raises(ValueError,match="Descriptor changed"):
        service.recognize(make_image())


def test_source_code_contract_mismatch_fails_closed(frozen):
    model,calibration,record=frozen
    record["source_sha256"]["src/preprocessing_v3.py"]="0"*64
    calibration.write_text(json.dumps(record),encoding="utf-8")
    with pytest.raises(ValueError,match="source changed"):
        module.SkinRecognitionSystemV3(model,calibration)


def test_positive_reject_all_threshold_is_not_squared_to_accept(frozen):
    model,calibration,record=frozen
    record["score_threshold"]=np.nextafter(0.,1.)
    record["calibration_metrics"]["threshold"]=record["score_threshold"]
    calibration.write_text(json.dumps(record),encoding="utf-8")
    service=module.SkinRecognitionSystemV3(model,calibration)
    service.register_person(make_image(),"a","Alice")
    assert service.reject_all
    assert service.recognize(make_image())[0] is None


def test_missing_environment_does_not_create_default_gallery(monkeypatch):
    monkeypatch.delenv("SKIN_V3_MODEL_PATH",raising=False)
    monkeypatch.delenv("SKIN_V3_CALIBRATION_PATH",raising=False)
    with pytest.raises(ValueError,match="Set SKIN_V3_MODEL_PATH"):
        module.SkinRecognitionSystemV3.from_env()


def write_search_point(frozen, tmp_path, threshold):
    model,calibration,record=frozen
    point={"schema_version":"skin-recognition-v3-search-operating-point-1",
           "source_freeze_sha256":hashlib.sha256(calibration.read_bytes()).hexdigest(),
           "feature_space_id":record["feature_space_id"],
           "preprocessing_id":record["preprocessing_id"],
           "model_sha256":record["model_sha256"],
           "score_definition":"negative_euclidean_distance",
           "decision_rule":"accept if score >= score_threshold",
           "score_threshold":threshold}
    path=tmp_path/"search_point.json"
    path.write_text(json.dumps(point),encoding="utf-8")
    return path,point


def test_optional_search_threshold_preserves_nextafter_rejection(frozen,tmp_path):
    model,calibration,_=frozen
    descriptor=StubDescriptor()
    enrollment=descriptor.extract_features(make_image(),PREPROCESSING_ID)
    query=descriptor.extract_features(make_image(80),PREPROCESSING_ID)
    score=-module.cdist(query.astype(float)[None,:],enrollment.astype(float)[None,:])[0,0]
    path,_=write_search_point(frozen,tmp_path,float(np.nextafter(score,np.inf)))
    service=module.SkinRecognitionSystemV3(model,calibration,search_operating_point_path=path)
    service.register_person(make_image(),"a","Alice")
    assert service.recognize(make_image(80))[0] is None
    assert service.recognize(make_image())[0]=="Alice"
    stats=service.get_database_stats()
    assert stats["threshold_status"]=="development_zero_false_match_search_point"
    assert stats["primary_score_threshold"]==-.8
    assert stats["identification_error_guarantee"] is False


def test_search_point_from_other_freeze_is_rejected(frozen,tmp_path):
    model,calibration,_=frozen
    path,point=write_search_point(frozen,tmp_path,-.4)
    point["source_freeze_sha256"]="0"*64
    path.write_text(json.dumps(point),encoding="utf-8")
    with pytest.raises(ValueError,match="different primary freeze"):
        module.SkinRecognitionSystemV3(model,calibration,search_operating_point_path=path)

