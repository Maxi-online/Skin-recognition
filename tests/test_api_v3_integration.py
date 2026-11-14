"""API/UI v3 integration. Real biometric fixtures are FEI training images only."""
import importlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.services.texture_v3 import SkinRecognitionSystemV3
from src.ui.dash_app import create_dash_app

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/"research/development/v3_classical_landmark_strict"


def test_real_v3_full_frame_http_enrollment_recognition_quality_delete_reload(tmp_path,monkeypatch):
    neutral=ROOT/"data/fei/originals/1-11.jpg"
    smile=ROOT/"data/fei/originals/1-12.jpg"
    freeze_path=FROZEN/"freeze.json"
    if not all(path.is_file() for path in (neutral,smile,freeze_path)):
        pytest.skip("Local FEI development fixtures/frozen model are not installed")
    freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
    model_path=Path(freeze["model_path"])
    model_dir=Path.home()/".insightface/models/buffalo_l"
    if not model_path.is_file() or not (model_dir/"2d106det.onnx").is_file():
        pytest.skip("Local frozen descriptor/preprocessing models are not installed")
    gallery=tmp_path/"v3_api_gallery"
    service=SkinRecognitionSystemV3(model_path,freeze_path,database_path=gallery,
                                    search_operating_point_path=FROZEN/"search_operating_point.json")
    api=importlib.import_module("src.api.app")
    monkeypatch.setattr(api,"RECOGNITION_SYSTEM",service)

    def forbidden_legacy_crop(image):
        raise AssertionError("V3 API must not call the legacy face cropper")
    monkeypatch.setattr(api,"detect_face",forbidden_legacy_crop)
    original_process=service.preprocessor.process
    shapes=[]
    def record_full_frame(image):
        shapes.append(image.shape)
        return original_process(image)
    monkeypatch.setattr(service.preprocessor,"process",record_full_frame)

    with TestClient(api.app) as client:
        health=client.get("/health")
        assert health.status_code==200,health.text
        assert health.json()["database"]["input_mode"]=="full_frame"
        assert health.json()["database"]["version"]=="3.1"
        for path in ("/","/docs","/openapi.json","/_dash-layout","/_dash-dependencies"):
            response=client.get(path)
            assert response.status_code==200,(path,response.text[:500])
        registered=client.post("/api/v1/register",
                               files={"image":("fei_train_001_neutral.jpg",neutral.read_bytes(),"image/jpeg")},
                               data={"person_id":"fei_train_001","person_name":"FEI development 001"})
        assert registered.status_code==200,registered.text
        recognized=client.post("/api/v1/recognize",
                               files={"image":("fei_train_001_smile.jpg",smile.read_bytes(),"image/jpeg")})
        assert recognized.status_code==200,recognized.text
        result=recognized.json()
        assert result["matched"] is True and result["person_name"]=="FEI development 001"
        assert result["threshold_status"]=="development_zero_false_match_search_point"
        assert result["similarity"]==pytest.approx(result["confidence"])
        assert "not_probability" in result["score_type"]
        assert result["candidates"][0]["squared_l2"]<service.recognition_threshold
        with Image.open(neutral) as image:
            expected_shape=(image.height,image.width,3)
        assert shapes[:2]==[expected_shape,expected_shape]
        stats=client.get("/api/v1/stats").json()
        assert stats["total_persons"]==1
        assert stats["identification_error_guarantee"] is False
        assert stats["calibration_scope"]=="pairwise_verification_development"
        assert stats["score_threshold"]==pytest.approx(-1.1478443118004258)
        assert gallery.with_suffix(".npz").is_file()

        bad=client.post("/api/v1/recognize",files={"image":("broken.jpg",b"not an image","image/jpeg")})
        assert bad.status_code==400
        blank=io.BytesIO()
        Image.new("RGB",(160,120),(127,127,127)).save(blank,format="PNG")
        no_face=client.post("/api/v1/recognize",files={"image":("no_face.png",blank.getvalue(),"image/png")})
        assert no_face.status_code==400,no_face.text
        assert "no_face" in no_face.json()["detail"]

        deleted=client.delete("/api/v1/person/fei_train_001")
        assert deleted.status_code==200,deleted.text
        assert deleted.json()["deleted_embeddings"]==1
        assert client.get("/api/v1/stats").json()["total_embeddings"]==0

        # Re-initialize from durable storage, rather than trusting only live RAM.
        reloaded=SkinRecognitionSystemV3(model_path,freeze_path,database_path=gallery,
                                         search_operating_point_path=FROZEN/"search_operating_point.json")
        monkeypatch.setattr(api,"RECOGNITION_SYSTEM",reloaded)
        assert client.get("/api/v1/stats").json()["total_persons"]==0
        assert client.get("/health").status_code==200


def test_ui_passes_complete_frame_to_v3_registration_and_recognition():
    full_image=np.full((120,160,3),90,np.uint8)
    class FullFrameService:
        input_mode="full_frame"
        recognition_threshold=1.0
        def __init__(self):
            self.registered=[]
            self.queries=[]
        def register_person(self,image,**kwargs):
            np.testing.assert_array_equal(image,full_image)
            self.registered.append(image.shape)
            return len(self.registered)-1
        def recognize(self,image,**kwargs):
            np.testing.assert_array_equal(image,full_image)
            self.queries.append(image.shape)
            return "Fixture",.95,[{"person_name":"Fixture","squared_l2":.2}],[.2]
    service=FullFrameService()
    app=create_dash_app(init_system=lambda:service,
                        decode_image_bytes=lambda data:full_image,
                        detect_face_fn=lambda image:(20,20,80,80),
                        detect_face_info_fn=lambda image:{"bbox":(20,20,80,80)},
                        database_path="unused-ui-fixture")
    callbacks={}
    for entry in app.callback_map.values():
        callback=entry.get("callback")
        if callback and hasattr(callback,"__wrapped__"):
            raw=callback.__wrapped__
            callbacks[raw.__name__]=raw
    store,_,_=callbacks["start_enroll"](1,"Fixture")
    for index in range(3):
        result=callbacks["handle_frame"](
            {"action":"enroll","frame":"data:image/png;base64,AA==","ts":index},store)
        store=result[3]
    assert result[5]==100
    result=callbacks["handle_frame"](
        {"action":"recognize","frame":"data:image/png;base64,AA==","ts":10},store)
    assert "Распознан: Fixture" in result[0]
    assert "сходство: 0.950; не вероятность" in result[0]
    assert "сходство: 0.950" in result[1]
    assert service.registered==[(120,160,3)]*3
    assert service.queries==[(120,160,3)]


def test_ui_callback_outputs_have_unique_renderer_keys():
    """Dash rejects duplicate output keys before any Python callback can run."""
    app=create_dash_app(init_system=lambda:None,
                        decode_image_bytes=lambda data:None,
                        detect_face_fn=lambda image:None,
                        detect_face_info_fn=lambda image:None,
                        database_path="unused-ui-fixture")
    seen=set()
    duplicates=[]
    dependencies={}
    for callback in app._callback_list:
        output=callback["output"]
        keys=output[2:-2].split("...") if output.startswith("..") else [output]
        for key in keys:
            # Dash adds an input hash to explicitly permitted duplicate outputs.
            if key in seen:
                duplicates.append(key)
            seen.add(key)
            target=key.split("@",1)[0]
            for source in callback["inputs"]:
                source_key=f'{source["id"]}.{source["property"]}'
                dependencies.setdefault(source_key,set()).add(target)
    assert not duplicates, f"Browser-rejected duplicate callback output keys: {duplicates}"
    visited=set()
    active=set()
    def visit(key):
        assert key not in active, f"Callback dependency feedback loop at {key}"
        if key in visited:
            return
        active.add(key)
        for target in dependencies.get(key,()):
            visit(target)
        active.remove(key)
        visited.add(key)
    for key in dependencies:
        visit(key)

