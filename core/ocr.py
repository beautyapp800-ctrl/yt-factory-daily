"""Optional OCR safety net for generated images.

How it is asked matters more than whether Tesseract can read the picture. The default
page segmentation mode looks for a page of body text and finds nothing in a painting
with one word in it; PSM 11, "sparse text", looks for text anywhere in an image and is
the right question here. Measured on video 7's 72 images: the default mode read nothing
at all on the frame with NNNICE written across a sheet of paper, while PSM 11 read it at
92% confidence. An earlier note in this file concluded that Tesseract simply cannot see
an image model's lettering; that was the wrong conclusion from the wrong mode.

It is still a net with holes. Across those 72 images PSM 11 flagged 4, including the
worst case, and missed soft or stylised lettering on others. The real defence remains not
asking the generator for a text-bearing object in the first place
(core.prompts.IMAGE_BANNED_WORDS, checked before generation); this is the last line, not
the first.

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


def detect_text(image_path, min_confidence=50, psm=11, min_word_len=4, upscale=1,
                contrast=1.0):
    """Characters Tesseract is at least min_confidence percent sure it read correctly.

    Returns (char_count, mean_confidence). (0, 0) when OCR is unavailable or nothing met
    the bar. Two filters, both tuned on video 7's images: PSM 11 so the whole picture is
    searched for text rather than read as a page, and a minimum word length, because at
    sparse-text settings Tesseract returns a stream of one and two character fragments
    from ordinary edges and highlights. Real rendered lettering comes back as whole words
    at high confidence - NNNICE at 92, ARTIN at 91 - so the length filter costs nothing
    and removes almost all of the noise.

    `upscale` and `contrast` make the search harder, for callers that would rather have
    a false alarm than a miss. Measured over video 7's 72 images: the defaults flag 4
    and miss a serif INVOICE across a sheet of paper; at min_word_len 3, upscale 2 and
    contrast 1.6 that one is caught, 15 of 72 are flagged, and the extra flags are
    mostly three-letter noise. Worth it when the cost of a flag is skipping one
    candidate of seventy (pipeline/thumbnail.py), not when it is a regeneration
    (pipeline/images.py).
    """
    if not available():
        return 0, 0
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = _resolved_cmd
    try:
        image = Image.open(image_path).convert("L")
        if contrast != 1.0:
            from PIL import ImageEnhance
            image = ImageEnhance.Contrast(image).enhance(contrast)
        if upscale != 1:
            image = image.resize((image.width * upscale, image.height * upscale),
                                 Image.LANCZOS)
        data = pytesseract.image_to_data(image, config=f"--psm {psm}",
                                         output_type=pytesseract.Output.DICT)
    except Exception as e:
        log.warning("OCR failed on %s: %s", image_path, e)
        return 0, 0

    import re
    chars, confidences = 0, []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        text = re.sub(r"[^A-Za-z0-9$]", "", text)
        try:
            conf = int(float(conf))
        except (TypeError, ValueError):
            conf = -1
        if len(text) >= min_word_len and conf >= min_confidence:
            chars += len(text)
            confidences.append(conf)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0
    return chars, mean_conf


def read_text(image, psm=6):
    """Whatever text Tesseract reads from a PIL image, as one string ('' if OCR is
    unavailable or fails). Unlike detect_text this keeps low-confidence reads, because
    the caller wants to know what a legible-or-not piece of real lettering says."""
    if not available():
        return ""
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = _resolved_cmd
    try:
        return pytesseract.image_to_string(image, config=f"--psm {psm}")
    except Exception as e:
        log.warning("OCR failed: %s", e)
        return ""
