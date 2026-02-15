"""OCR package facade preserving previous public imports."""

from .ai_client import (
    AiOcrClient,
    AiOcrTextRefiner,
    _is_multiline_candidate_for_linebreak_assist,
)
from .base import OcrProvider
from .deepseek_parser import _extract_deepseek_tagged_items
from .local_providers import (
    BaiduOcrClient,
    OcrManager,
    PaddleOcrClient,
    TesseractOcrClient,
    _dedupe_overlapping_ocr_items,
    create_ocr_manager,
    ocr_image_to_elements,
    probe_local_paddleocr,
    probe_local_tesseract,
)
from .utils import _coerce_bbox_xyxy
from .vendors import (
    AiOcrVendorAdapter,
    AiOcrVendorProfile,
    DeepSeekAiOcrAdapter,
    NovitaAiOcrAdapter,
    OpenAiAiOcrAdapter,
    PpioAiOcrAdapter,
    SiliconFlowAiOcrAdapter,
)

__all__ = [
    "AiOcrClient",
    "AiOcrTextRefiner",
    "AiOcrVendorAdapter",
    "AiOcrVendorProfile",
    "BaiduOcrClient",
    "DeepSeekAiOcrAdapter",
    "NovitaAiOcrAdapter",
    "OcrManager",
    "OcrProvider",
    "OpenAiAiOcrAdapter",
    "PaddleOcrClient",
    "PpioAiOcrAdapter",
    "SiliconFlowAiOcrAdapter",
    "TesseractOcrClient",
    "_coerce_bbox_xyxy",
    "_dedupe_overlapping_ocr_items",
    "_extract_deepseek_tagged_items",
    "_is_multiline_candidate_for_linebreak_assist",
    "create_ocr_manager",
    "ocr_image_to_elements",
    "probe_local_paddleocr",
    "probe_local_tesseract",
]
