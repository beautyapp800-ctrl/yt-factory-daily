"""Optional OCR safety net for generated images.

Demonstrably weak on its own: tested directly against a known-bad image (one with
visible "LIMITED EDITION" packaging text), Tesseract found nothing, at any page
segmentation mode, even upscaled 2x - an image model mostly draws pseudo-text, shapes
that look like letters to a human eye without forming real glyphs, which an OCR
engine trained on actual fonts does not recognise. The real defense is not sending a
text-bearing object to the generator in the first place (core.prompts.IMAGE_BANNED_WORDS,
checked before generation). This module exists as a cheap last-resort check for the
cases that are real, legible text, not as the primary line of defense.

Needs Tesseract-OCR installed as a system binary (not just the pytesseract package,
which is only a wrapper) and pillow. Both optional: entirely skipped, not an error,
wherever either is missing.
"""
from pathlib import Path

from core.logger import get_logger

log = get_logger("ocr")

# Common Windows install location for UB-Mannheim's Tesseract-OCR build, tried before
# falling back to PATH.
_WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_resolved_cmd = None   # "" once checked and unavailable, else the resolved command


def available():
    """Whether both Tesseract itself and the pytesseract/Pillow packages are usable."""
    global _resolved_cmd
    if _resolved_cmd is not None:
        return _resolved_cmd != ""
    try:
        import pytesseract      # noqa: F401
        from PIL import Image   # noqa: F401
    except ImportError:
        _resolved_cmd = ""
        return False

    import shutil
    if Path(_WINDOWS_DEFAULT).exists():
        _resolved_cmd = _WINDOWS_DEFAULT
    elif shutil.which("tesseract"):
        _resolved_cmd = "tesseract"
    else:
        _resolved_cmd = ""
    return _resolved_cmd != ""


def detect_text(image_path, min_confidence=60):
    """Characters Tesseract is at least min_confidence percent sure it read correctly.

    Returns (char_count, mean_confidence). (0, 0) when OCR is unavailable or nothing
    met the confidence bar - low confidence is exactly what an image model's fake
    lettering produces, so this intentionally does not count it.
    """
    if not available():
        return 0, 0
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = _resolved_cmd
    try:
        data = pytesseract.image_to_data(Image.open(image_path),
                                         output_type=pytesseract.Output.DICT)
    except Exception as e:
        log.warning("OCR failed on %s: %s", image_path, e)
        return 0, 0

    chars, confidences = 0, []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        text = text.strip()
        try:
            conf = int(float(conf))
        except (TypeError, ValueError):
            conf = -1
        if text and conf >= min_confidence:
            chars += len(text)
            confidences.append(conf)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0
    return chars, mean_conf
