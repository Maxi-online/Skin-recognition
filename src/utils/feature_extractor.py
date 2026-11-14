import numpy as np
from typing import Optional

try:
    import torch  # type: ignore
    from transformers import AutoImageProcessor, AutoModel  # type: ignore
    from PIL import Image  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    AutoImageProcessor = None  # type: ignore
    AutoModel = None  # type: ignore
    Image = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False

from ..descriptors import HybridSkinDescriptor

# Глобальный дескриптор для совместимости
_descriptor = HybridSkinDescriptor()

# Глобальный ViT (лениво инициализируется)
_vit_processor = None
_vit_model = None
_vit_model_name = None

# Для совместимости со старым кодом
def extract_gabor_features(image):
    """
    Извлекает признаки из изображения с использованием гибридного дескриптора.
    Используется для совместимости со старым кодом.
    """
    return _descriptor.extract_features(image)

def extract_vit_features(image):
    """
    Экспериментальный экстрактор признаков на базе Vision Transformer (ViT) без обучения.

    Важно: модель скачивается при первом запуске (нужен интернет).
    Возвращает L2-нормализованный вектор признаков (обычно размерность 768 для vit-base).
    """
    return extract_vit_features_advanced(image)


def extract_vit_features_advanced(
    image: np.ndarray,
    model_name: str = "google/vit-base-patch16-224-in21k",
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Извлечение признаков ViT (mean pooling по токенам).

    Args:
        image: np.ndarray (BGR или RGB, HxWxC)
        model_name: huggingface model id
        device: "cpu" / "cuda" (если None — авто)

    Returns:
        np.ndarray shape [D], float32, L2-normalized
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("Для ViT нужны зависимости: torch, torchvision, transformers, Pillow (см. requirements.txt)")

    global _vit_processor, _vit_model, _vit_model_name

    if _vit_model is None or _vit_processor is None or _vit_model_name != model_name:
        _vit_model_name = model_name
        _vit_processor = AutoImageProcessor.from_pretrained(model_name)
        _vit_model = AutoModel.from_pretrained(model_name)
        _vit_model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _vit_model.to(device)

    # Приводим к PIL RGB
    img = image
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    # heuristic: если похоже на BGR (OpenCV) — перевернем каналы
    # (не идеально, но безопасно для большинства пайплайнов OpenCV)
    img = img[:, :, ::-1].copy()

    pil = Image.fromarray(img.astype(np.uint8), mode="RGB")
    inputs = _vit_processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _vit_model(**inputs)
        # last_hidden_state: [1, tokens, dim]
        x = outputs.last_hidden_state.mean(dim=1)  # [1, dim]
        vec = x[0].detach().cpu().numpy().astype(np.float32)

    # L2 нормализация
    norm = float(np.linalg.norm(vec) + 1e-10)
    vec = vec / norm
    return vec

