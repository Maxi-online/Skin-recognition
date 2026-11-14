import importlib
import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.services import SkinRecognitionSystem


@pytest.fixture
def api(tmp_path, monkeypatch):
    module = importlib.import_module("src.api.app")
    system = SkinRecognitionSystem(database_path=str(tmp_path / "test_gallery"))
    monkeypatch.setattr(module, "RECOGNITION_SYSTEM", system)
    monkeypatch.setattr(module, "detect_face", lambda image: (0, 0, image.shape[1], image.shape[0]))
    return module, TestClient(module.app)


def image_bytes():
    image = np.random.default_rng(17).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def test_api_real_descriptor_roundtrip(api):
    module, client = api
    content = image_bytes()
    upload = {"image": ("fixture.png", content, "image/png")}
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    result = client.post("/api/v1/register", files=upload, data={"person_id": "fixture", "person_name": "Fixture"})
    assert result.status_code == 200, result.text
    result = client.post("/api/v1/recognize", files=upload)
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["matched"] is True and body["person_name"] == "Fixture"
    assert body["candidates"][0]["squared_l2"] == pytest.approx(0, abs=1e-6)
    assert "not_probability" in body["score_type"]
    assert client.delete("/api/v1/person/fixture").json()["deleted_embeddings"] == 1
    assert client.post("/api/v1/recognize", files=upload).json()["matched"] is False


def test_api_bad_inputs_are_client_errors(api):
    module, client = api
    assert client.post("/api/v1/recognize", files={"image": ("bad", b"bad")}).status_code == 400
    upload = {"image": ("fixture.png", image_bytes(), "image/png")}
    assert client.post("/api/v1/recognize", files=upload, data={"k": 0}).status_code == 400
    assert client.post("/api/v1/register", files=upload,
                       data={"person_id": "a", "person_name": "a", "metadata": "[]"}).status_code == 400


def test_image_decoder_palette_and_rgba(api):
    module, _ = api
    for mode in ("P", "RGBA", "L"):
        stream = io.BytesIO()
        Image.new(mode, (40, 40)).save(stream, "PNG")
        assert module.decode_image(stream.getvalue()).shape == (40, 40, 3)
