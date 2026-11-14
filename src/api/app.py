"""Local research API. One worker owns the file-backed gallery."""
from pathlib import Path
import io
import json
import logging
import os
import threading
import warnings
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.middleware.wsgi import WSGIMiddleware
from ..services import SkinRecognitionSystem, detect_face, detect_face_info
from ..ui.dash_app import create_dash_app

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "config" / "research_profile.json"
PROFILE = json.loads(PROFILE_PATH.read_text(encoding="utf-8")) if PROFILE_PATH.exists() else {}
BACKEND = os.getenv("SKIN_BACKEND", PROFILE.get("backend", "v2"))
app = FastAPI(title="Skin Recognition Research API", version="4.2.0",
              description="Reproducible skin-texture, whole-face and hybrid recognition with frozen model profiles.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8001", "http://127.0.0.1:8001"],
                   allow_credentials=False, allow_methods=["GET", "POST", "DELETE"],
                   allow_headers=["Content-Type"])
static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

RECOGNITION_SYSTEM = None
_INIT_LOCK = threading.Lock()
logger = logging.getLogger(__name__)
_active_method = os.getenv("SKIN_V4_METHOD") or PROFILE.get("method_id", "hybrid")
_default_gallery = (PROFILE.get("database_path") if PROFILE.get("backend", "v2") == BACKEND else None) or (
    f"runtime/skin_embeddings_{BACKEND}_{_active_method}" if BACKEND in ("v4", "v42") else f"runtime/skin_embeddings_{BACKEND}")
DATABASE_PATH = (os.getenv("SKIN_V4_DATABASE_PATH") if BACKEND in ("v4", "v42") else
                 os.getenv("SKIN_V3_DATABASE_PATH") if BACKEND == "v3" else None) or os.getenv(
    "DATABASE_PATH", str(ROOT / _default_gallery))
DESCRIPTOR_DIM = int(os.getenv("DESCRIPTOR_DIM", "512"))
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.7"))
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def init_recognition_system():
    global RECOGNITION_SYSTEM
    with _INIT_LOCK:
        if RECOGNITION_SYSTEM is None:
            if BACKEND in ("v4", "v42"):
                if BACKEND == "v42":
                    from ..services.recognition_v42 import RecognitionSystemV42 as RecognitionBackend
                else:
                    from ..services.recognition_v4 import RecognitionSystemV4 as RecognitionBackend
                freeze = os.getenv("SKIN_V4_FREEZE_PATH") or PROFILE.get("freeze_path")
                method = os.getenv("SKIN_V4_METHOD") or PROFILE.get("method_id")
                if not freeze or not method:
                    raise ValueError("Set the v4 freeze_path and method_id in the research profile")
                freeze = Path(freeze)
                RECOGNITION_SYSTEM = RecognitionBackend(
                    freeze if freeze.is_absolute() else ROOT / freeze, method, database_path=DATABASE_PATH)
            elif BACKEND == "v3":
                from ..services.texture_v3 import SkinRecognitionSystemV3
                def configured_path(env_name, profile_key, required=True):
                    value = os.getenv(env_name) or PROFILE.get(profile_key)
                    if not value:
                        if required:
                            raise ValueError(f"Set {env_name} or {profile_key} in research_profile.json")
                        return None
                    path = Path(value)
                    return str(path if path.is_absolute() else ROOT / path)
                RECOGNITION_SYSTEM = SkinRecognitionSystemV3(
                    model_path=configured_path("SKIN_V3_MODEL_PATH", "model_path"),
                    calibration_path=configured_path("SKIN_V3_CALIBRATION_PATH", "calibration_path"),
                    search_operating_point_path=configured_path("SKIN_V3_SEARCH_POINT_PATH", "search_operating_point_path", False),
                    database_path=DATABASE_PATH,
                    model_dir=os.getenv("SKIN_V3_PREPROCESSOR_MODEL_DIR"))
            elif BACKEND == "v2":
                RECOGNITION_SYSTEM = SkinRecognitionSystem(
                    database_path=DATABASE_PATH, descriptor_dim=DESCRIPTOR_DIM,
                    recognition_threshold=RECOGNITION_THRESHOLD,
                    descriptor_path=os.getenv("DESCRIPTOR_PATH"))
            else:
                raise ValueError("Unknown SKIN_BACKEND")
    return RECOGNITION_SYSTEM


def decode_image(image_bytes):
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be nonempty and no larger than 10 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as source:
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 16 megapixel limit")
                rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("Invalid or unsupported image") from error


def _system():
    try:
        return init_recognition_system()
    except Exception as error:
        logger.exception("Recognition initialization failed")
        raise HTTPException(503, "Recognition system is unavailable; check the local server log") from error


async def _face(upload):
    try:
        image = decode_image(await upload.read(MAX_IMAGE_BYTES + 1))
        if getattr(_system(), "input_mode", "face_crop") == "full_frame":
            return image
        bbox = detect_face(image)
        if bbox is None:
            raise ValueError("Face not detected")
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


class RecognizeResponse(BaseModel):
    success: bool
    matched: bool
    person_name: str | None
    confidence: float
    similarity: float
    score_type: str = "normalized_squared_l2_similarity_not_probability"
    threshold_status: str = "uncalibrated"
    candidates: list[dict]


@app.get("/legacy")
def legacy_root():
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health():
    try:
        system = init_recognition_system()
        return dict(status="ok", version=app.version, service="skin-recognition-api",
                    database=system.get_database_stats())
    except Exception:
        logger.exception("Health initialization failed")
        return JSONResponse(status_code=503, content=dict(status="unavailable", version=app.version))


@app.post("/api/v1/register")
async def register_person(image: UploadFile = File(...), person_id: str = Form(...),
                          person_name: str = Form(...), metadata: str | None = Form(None)):
    extra = None
    if metadata:
        try:
            extra = json.loads(metadata)
            if not isinstance(extra, dict):
                raise ValueError("Expected an object")
        except (ValueError, TypeError) as error:
            raise HTTPException(400, "metadata must contain a JSON object") from error
    face = await _face(image)
    try:
        embedding_id = _system().register_person(face, person_id, person_name, additional_metadata=extra)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return dict(success=True, embedding_id=embedding_id, person_id=person_id, person_name=person_name)


@app.post("/api/v1/recognize", response_model=RecognizeResponse)
async def recognize_person(image: UploadFile = File(...), k: int = Form(5)):
    if not 1 <= k <= 100:
        raise HTTPException(400, "k must be between 1 and 100")
    face = await _face(image)
    try:
        name, similarity, candidates = _system().recognize(face, k=k)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return RecognizeResponse(success=True, matched=name is not None, person_name=name,
                             threshold_status=_system().get_database_stats()["threshold_status"],
                             confidence=similarity, similarity=similarity, candidates=[
                                 {key: c.get(key) for key in ("person_id", "person_name", "squared_l2")}
                                 for c in candidates])


@app.get("/api/v1/stats")
def get_stats():
    return _system().get_database_stats()


@app.delete("/api/v1/person/{person_id}")
def delete_person(person_id: str):
    count = _system().remove_person(person_id)
    return dict(success=True, person_id=person_id, deleted_embeddings=count)


def _ui_face_info(image):
    system = init_recognition_system()
    detector = getattr(system, "detect_face_info", detect_face_info)
    return detector(image)


def _ui_face(image):
    info = _ui_face_info(image)
    return info.get("bbox") if info else None


dash_app = create_dash_app(fastapi_base_path="/", init_system=init_recognition_system,
                          decode_image_bytes=decode_image, detect_face_fn=_ui_face,
                          detect_face_info_fn=_ui_face_info, database_path=DATABASE_PATH)
app.mount("/", WSGIMiddleware(dash_app.server))
