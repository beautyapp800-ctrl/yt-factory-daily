"""Stage 6: the thumbnail - 1280x720 JPEG, quality 90, a darkened picture with big words on it.

Written for one thing, which is being legible as a 210x118 postage stamp in a phone feed.
Every choice follows from that.

The picture. The most striking of the video's generated images, picked by measurement,
not taste: the highest contrast and the darkest edges, because a dark rim is what lets
white lettering and a vignette sit on it without a fight. Both are measured on the 16:9
crop that will actually be shown (the images are square, so the crop drops the top and
bottom), scaled to 0-1 across the candidates and added with equal weight.

The treatment. The picture is darkened by 35% and given a vignette so the text is the
brightest thing in the frame.

The words. 3-5 of them, upper case, lifted from the title (words_from_title): the
promise at its end, not the "10 Stoic Lessons" boilerplate in front of it. White Anton
(assets/fonts, SIL OFL) with a 4px black outline, left aligned in the bottom third. The
size is the largest at which the block fits the width and stays about a third of the
frame tall, trying every way to break the words into lines.

The check. The finished thumbnail is shrunk to 210x118 and saved next to it as
thumb_small.jpg - the size of a thumbnail in a phone feed - and the stage measures what
survives the shrink: how tall a capital letter is there, and how well white lettering
stands out from the picture under it. Both have thresholds, and missing either logs a
warning and writes an events row. It does not fail the video over it - a thumbnail is not
worth a day's upload - but the warning is there to be acted on, and the small file is
saved precisely so a person can look at it.

OCR is run on the small file too and logged, but it does not decide anything. First
version of this check gated on it, and it was wrong where it mattered: on a thumbnail
whose lettering is plainly readable at 210x118, Tesseract recovered one word of three. It
is trained on ordinary text, not on heavy condensed capitals with an outline.
"""
import itertools
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from core import db, ocr
from core.config import output_dir
from core.logger import get_logger

log = get_logger("thumbnail")

WIDTH, HEIGHT = 1280, 720
SMALL = (210, 118)
JPEG_QUALITY = 90
DARKEN = 0.65                    # 35% darker
VIGNETTE_STRENGTH = 0.55         # how much the extreme corners are darkened on top of that
STROKE = 4
MARGIN_X = 56
MARGIN_BOTTOM = 32
MAX_TEXT_HEIGHT = 205            # block height incl. outline: ~28% of the frame, inside the
                                 # bottom third (it starts at y=483, the third at y=480)
LINE_GAP = 0.08                  # between lines, as a fraction of the font size
MIN_WORDS, MAX_WORDS = 3, 5
FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Anton-Regular.ttf"
MIN_SMALL_CAP_PX = 10            # a capital letter must still be this tall at 210x118
MIN_CONTRAST = 4.5               # white text against the picture under it (WCAG AA ratio)

# Words that open a title without carrying its promise.
_BOILERPLATE = {"stoic", "stoicism", "lessons", "lesson", "ways", "things", "rules", "truths",
                "habits", "reasons", "secrets", "principles", "to", "that", "will", "the", "a",
                "an", "of", "for", "and", "or", "in", "on", "with", "how", "why", "is", "are",
                "this", "these", "from", "by", "at", "it", "be"}


# --- the words ---------------------------------------------------------------

def words_from_title(title):
    """3-5 upper-case words from the title: the window that ends the title (that is where
    the promise is) and neither starts nor ends on a connecting word. 'NEVER Explain
    Yourself Again' stays whole; '10 Stoic Lessons To Unchain Your Wealth' becomes
    'UNCHAIN YOUR WEALTH'. A title with a comma, as in '... as a Tool, Not a Status
    Symbol', is a setup and a payoff: the payoff is taken, not a window straddling the
    comma that reads as mush."""
    def tokens(text):
        return [w for w in re.findall(r"[A-Za-z0-9'’-]+", text or "") if not w.isdigit()]

    def best_window(ws):
        best, best_score = None, None
        for size in range(MIN_WORDS, MAX_WORDS + 1):
            for start in range(0, len(ws) - size + 1):
                window = ws[start:start + size]
                if window[0].lower() in _BOILERPLATE or window[-1].lower() in _BOILERPLATE:
                    continue
                # The tail carries the promise; longer reads better than shorter.
                score = (start + size == len(ws)) * 10 + size
                if best is None or score > best_score:
                    best, best_score = window, score
        return best

    words = tokens(title)
    if MIN_WORDS <= len(words) <= MAX_WORDS and words[0].lower() not in _BOILERPLATE:
        return [w.upper() for w in words]

    best = None
    for clause in reversed(re.split(r"\s*[,;:]\s*|\s+[-–—]\s+", title or "")):
        best = best_window(tokens(clause))
        if best:
            break
    best = best or best_window(words)
    if best is None:
        # No clean 3-5 word window (the promise is shorter than that: 'Calm Mind'). Take
        # the tail with its leading connecting words stripped, even if that is only two
        # words - better than a fragment that starts mid-phrase.
        tail = words[-MAX_WORDS:]
        while len(tail) > 1 and tail[0].lower() in _BOILERPLATE:
            tail = tail[1:]
        best = tail if len(tail) >= 2 else words[-MIN_WORDS:]
    return [w.upper() for w in best]


# --- the picture -------------------------------------------------------------

def crop_16_9(img):
    """The centred 16:9 part of an image, as it will appear in the thumbnail."""
    w, h = img.size
    target_h = min(h, round(w * 9 / 16))
    target_w = min(w, round(h * 16 / 9))
    left, top = (w - target_w) // 2, (h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def image_stats(path):
    """(contrast, edge_brightness) of the picture's 16:9 crop, both on a 0-255 scale:
    the standard deviation of luminance, and the mean luminance of the outer 12% rim."""
    with Image.open(path) as img:
        lum = np.asarray(crop_16_9(img.convert("L")).resize((320, 180)), dtype=np.float64)
    contrast = float(lum.std())
    bh, bw = int(lum.shape[0] * 0.12), int(lum.shape[1] * 0.12)
    rim = np.ones(lum.shape, dtype=bool)
    rim[bh:-bh, bw:-bw] = False
    return contrast, float(lum[rim].mean())


def pick_background(candidates):
    """The most striking path from [(path, provider)]: highest contrast, darkest rim.

    Pollinations images carry a watermark and duplicates are the same picture twice, so
    both are set aside - unless that leaves nothing, in which case anything will do.
    Returns (path, details).
    """
    seen, usable = set(), []
    for path, provider in candidates:
        if path in seen or not Path(path).exists():
            continue
        seen.add(path)
        usable.append((path, provider))
    clean = [c for c in usable if c[1] != "pollinations"] or usable
    if not clean:
        raise RuntimeError("no images to make a thumbnail from")

    stats = [image_stats(p) for p, _ in clean]
    contrast = np.array([s[0] for s in stats])
    rim = np.array([s[1] for s in stats])

    def unit(v):
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 0 else np.zeros_like(v)

    score = unit(contrast) + (1 - unit(rim))
    best = int(score.argmax())
    return clean[best][0], {"contrast": float(contrast[best]), "rim": float(rim[best]),
                            "score": float(score[best]), "candidates": len(clean)}


def vignette_mask():
    """A (HEIGHT, WIDTH) multiplier: 1 in the middle, falling to 1-VIGNETTE_STRENGTH at the
    corners, smoothly."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    r = np.sqrt(((x - WIDTH / 2) / (WIDTH / 2)) ** 2 + ((y - HEIGHT / 2) / (HEIGHT / 2)) ** 2)
    r = np.clip(r / np.sqrt(2), 0, 1)
    smooth = r * r * (3 - 2 * r)
    return 1 - VIGNETTE_STRENGTH * smooth


def background(path):
    """The picture, cropped to 16:9, sized to the thumbnail, darkened 35% and vignetted."""
    with Image.open(path) as img:
        img = crop_16_9(img.convert("RGB")).resize((WIDTH, HEIGHT), Image.LANCZOS)
    img = ImageEnhance.Brightness(img).enhance(DARKEN)
    arr = np.asarray(img, dtype=np.float64) * vignette_mask()[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# --- the text ----------------------------------------------------------------

def _line_breaks(words):
    """Every way to split the words, in order, into 1-3 lines."""
    n = len(words)
    for lines in (1, 2, 3):
        if lines > n:
            break
        for cuts in itertools.combinations(range(1, n), lines - 1):
            bounds = (0, *cuts, n)
            yield [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:])]


def _measure(lines, size, font_path):
    font = ImageFont.truetype(str(font_path), size)
    gap = round(size * LINE_GAP)
    boxes = [font.getbbox(line, stroke_width=STROKE) for line in lines]
    width = max(b[2] - b[0] for b in boxes)
    height = sum(b[3] - b[1] for b in boxes) + gap * (len(lines) - 1)
    return font, boxes, gap, width, height


def best_layout(words, font_path=FONT_PATH):
    """(lines, size): the way of breaking the words into lines whose block comes closest to
    filling MAX_TEXT_HEIGHT - about a third of the frame, which is how big the brief wants
    the text - and, among layouts that fill it equally well, the one with the largest font.
    Each layout is sized to the biggest font at which it still fits the width and the height.

    Filling the height first matters: three short words fit on one line at a larger font
    than on two, but that line is only a sixth of the frame tall."""
    best = None
    for lines in _line_breaks(words):
        for size in range(260, 40, -2):
            _, _, _, width, height = _measure(lines, size, font_path)
            if width <= WIDTH - 2 * MARGIN_X and height <= MAX_TEXT_HEIGHT:
                # Equal height and size: the more compact (so more evenly broken) layout.
                key = (height // 8, size, -width)
                if best is None or key > best[0]:
                    best = (key, lines, size)
                break
    if best is None:
        raise RuntimeError(f"cannot fit {words} on a thumbnail")
    return best[1], best[2]


def draw_text(img, words, font_path=FONT_PATH):
    """White text, black 4px outline, left aligned, its block sitting at the bottom."""
    lines, size = best_layout(words, font_path)
    font, boxes, gap, width, height = _measure(lines, size, font_path)
    draw = ImageDraw.Draw(img)
    mask = Image.new("L", (WIDTH, HEIGHT), 0)       # where the text and its outline are
    mask_draw = ImageDraw.Draw(mask)
    y = HEIGHT - MARGIN_BOTTOM - height
    for line, box in zip(lines, boxes):
        # box[0]/box[1] are the ink's offset from the drawing origin; cancel them so the
        # ink, not the font's empty ascender space, lands on the margins.
        at = (MARGIN_X - box[0], y - box[1])
        draw.text(at, line, font=font, fill="white", stroke_width=STROKE, stroke_fill="black")
        mask_draw.text(at, line, font=font, fill=255, stroke_width=STROKE, stroke_fill=255)
        y += (box[3] - box[1]) + gap
    cap = font.getbbox("H")
    return {"lines": lines, "font_size": size, "block_height": height, "block_width": width,
            "top": HEIGHT - MARGIN_BOTTOM - height, "cap_height": cap[3] - cap[1],
            "mask": mask}


# --- the check ---------------------------------------------------------------

def small_version(thumb_path, small_path):
    with Image.open(thumb_path) as img:
        img.resize(SMALL, Image.LANCZOS).save(small_path, "JPEG", quality=JPEG_QUALITY)
    return small_path


def _relative_luminance(gray):
    """WCAG relative luminance (0-1) of 0-255 sRGB grey values."""
    c = np.asarray(gray, dtype=np.float64) / 255
    return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def readability(small_path, backdrop, mask, cap_height, words):
    """What survives at 210x118, measured.

    cap_px      height of a capital letter in the small image
    contrast    white lettering against the picture under it, as the WCAG contrast ratio
                (L1+0.05)/(L2+0.05), taking the brighter 90th percentile of the picture
                under the text so a bright patch behind a letter counts against it
    ocr_share   share of the words Tesseract recovers - logged, not used to decide
    ok          cap_px and contrast both clear their thresholds
    """
    scale = SMALL[1] / HEIGHT
    cap_px = cap_height * scale

    under = np.asarray(backdrop.convert("L").resize(SMALL, Image.LANCZOS))
    where = np.asarray(mask.resize(SMALL, Image.LANCZOS)) > 40
    lum = _relative_luminance(under[where]) if where.any() else np.array([0.0])
    contrast = 1.05 / (float(np.percentile(lum, 90)) + 0.05)

    share, read = None, ""
    if ocr.available():
        with Image.open(small_path) as img:
            read = ocr.read_text(img.convert("L").resize((SMALL[0] * 4, SMALL[1] * 4),
                                                         Image.LANCZOS)).strip().replace("\n", " ")
        found = {w.lower() for w in re.findall(r"[A-Za-z']+", read)}
        wanted = [w.lower().strip("'-") for w in words]
        share = sum(1 for w in wanted if w in found) / len(wanted)
    return {"cap_px": cap_px, "contrast": contrast, "ocr_share": share, "ocr_read": read,
            "ok": cap_px >= MIN_SMALL_CAP_PX and contrast >= MIN_CONTRAST}


def make_thumbnail(candidates, title, out_dir):
    """Build thumbnail.jpg and thumb_small.jpg in out_dir. Returns a details dict."""
    out_dir = Path(out_dir)
    words = words_from_title(title)
    source, pick = pick_background(candidates)
    img = background(source)
    backdrop = img.copy()
    layout = draw_text(img, words)

    thumb = out_dir / "thumbnail.jpg"
    img.save(thumb, "JPEG", quality=JPEG_QUALITY)
    small = small_version(thumb, out_dir / "thumb_small.jpg")
    check = readability(small, backdrop, layout.pop("mask"), layout["cap_height"], words)
    return {"path": thumb, "small": small, "words": words, "source": source, **pick,
            **layout, "readability": check}


def run(video_id, cfg):
    video = db.get_video(video_id) or {}
    title = video.get("seo_title") or video.get("title")
    if not title:
        raise RuntimeError("the video has no title to take the thumbnail words from")

    rows = db.get_video_images(video_id)
    candidates = [(r["path"], r["provider"]) for r in rows if r["path"]]
    result = make_thumbnail(candidates, title, output_dir(video_id))

    log.info("background %s (contrast %.1f, rim brightness %.1f, best of %d)",
             Path(result["source"]).name, result["contrast"], result["rim"],
             result["candidates"])
    log.info("text %s: %d lines, font %dpx, block %dx%d, top at y=%d",
             " ".join(result["words"]), len(result["lines"]), result["font_size"],
             result["block_width"], result["block_height"], result["top"])
    check = result["readability"]
    summary = (f"at 210x118 a capital is {check['cap_px']:.0f}px tall (need {MIN_SMALL_CAP_PX}) "
               f"and the text has {check['contrast']:.1f}:1 contrast against the picture "
               f"(need {MIN_CONTRAST}:1)")
    if check["ocr_share"] is not None:
        summary += f"; OCR read {check['ocr_share']:.0%} of the words (informational)"
    if check["ok"]:
        log.info("thumb_small.jpg readable: %s", summary)
    else:
        msg = f"thumb_small.jpg may not be legible in a feed: {summary}"
        log.warning(msg)
        db.log_event(video_id, "thumbnail", "warning", msg)

    db.update_video(video_id, thumbnail_path=str(result["path"]))
    db.log_event(video_id, "thumbnail", "info",
                 f"{Path(result['source']).name} behind {' '.join(result['words'])!r}")
    return True
