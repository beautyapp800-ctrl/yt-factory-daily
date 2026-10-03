"""Stage 6: the thumbnail - 1280x720 JPEG, quality 90, built to survive a phone feed.

A thumbnail is seen at 210x118 before it is ever seen at full size, so every decision here
is made for the small version and checked on it.

The layout. A dark photograph with two lines of type in its left half: the first large and
white, the second smaller and warm, together at least a third of the frame tall. The right
half is left to the picture, so the type never fights the subject. Under the type is a dark
scrim that fades out upward and to the right, which is what keeps white lettering readable
over a picture that may be light in places. The picture itself is pushed in contrast,
darkened and heavily vignetted.

The words. Two phrases taken from the title. A title with a comma is a setup and a payoff
("... Money as a Tool, Not a Status Symbol"), and those two halves are the two lines; a
title without one is split where the two lines come out closest in width. The opening
"10 Stoic Lessons" is dropped either way - it is the part every title in the niche shares.

Choosing the layout. The code builds every sensible variant - each first-line size, each of
several size ratios between the two lines - measures each one, and keeps the best. The
measurements are the brief's own requirements: the block must be at least a third of the
frame tall, must fit the left half, and must pass the readability check. Variants are tried
largest first, so the one that is kept is the largest that reads.

The check. The finished thumbnail is shrunk to 210x118 and saved as thumb_small.jpg, and
what survives the shrink is measured: the height of a capital in the smaller line, and the
WCAG contrast of each line against what is actually behind it there. If every variant fails,
the stage keeps the largest and says so loudly rather than shipping it quietly.

OCR is run on the small file and logged, but decides nothing: on a thumbnail a person reads
without effort, Tesseract recovered one word in three. It is trained on body text, not on
heavy condensed capitals with an outline.
"""
import json
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

# --- the picture
CONTRAST = 1.30                  # pushed before darkening, so the darks stay separated
DARKEN = 0.62
VIGNETTE_STRENGTH = 0.70         # how far the extreme corners fall below the middle

# --- the type
TEXT_COLUMN = 0.52               # the type lives in this fraction of the width, from the left
MARGIN_X = 54
MARGIN_BOTTOM = 56
MIN_BLOCK_FRACTION = 1 / 3       # the brief: at least a third of the frame's height
MAX_BLOCK_FRACTION = 0.46        # above this it stops being a thumbnail and becomes a poster
STROKE = 5
LINE_GAP = 0.04                  # between the two lines, as a fraction of the first line's size
SECOND_LINE_RATIOS = (0.52, 0.60, 0.68)
WHITE = (255, 255, 255)
WARM = (240, 176, 84)            # amber, warm against the cold blues these pictures tend to
FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Anton-Regular.ttf"

# --- the scrim
SCRIM_DARKNESS = 0.82            # how black it is at its strongest
SCRIM_PAD_X, SCRIM_PAD_TOP = 90, 70     # how far it reaches past the type before fading out

# --- the check
MIN_SMALL_CAP_PX = 9             # a capital in the SMALLER line, at 210x118
MIN_CONTRAST = 4.5               # WCAG AA for large text
# Deliberately trigger-happy: a false alarm just moves to the next of seventy candidates,
# while a miss puts fake lettering on the thumbnail. See core.ocr.detect_text.
BACKGROUND_OCR = {"min_confidence": 50, "min_word_len": 3, "upscale": 2, "contrast": 1.6}
BACKGROUND_OCR_CHARS = 2         # characters of real-looking text that disqualify it

MAX_WORDS_PER_LINE = 4
MIN_WORDS, MAX_WORDS = 2, 6
_BOILERPLATE = {"stoic", "stoicism", "lessons", "lesson", "ways", "things", "rules", "truths",
                "habits", "reasons", "secrets", "principles", "to", "that", "will", "the", "a",
                "an", "of", "for", "and", "or", "in", "on", "with", "how", "why", "is", "are",
                "this", "these", "from", "by", "at", "it", "be", "use", "make"}
_CLAUSE_SPLIT = re.compile(r"\s*[,;:]\s*|\s+[-–—]\s+")


# --- the words ---------------------------------------------------------------

def _words(text):
    return [w for w in re.findall(r"[A-Za-z0-9'’-]+", text or "") if not w.isdigit()]


def _trim_boilerplate(words):
    """Drop the leading words every title in the niche shares, and any trailing connector."""
    out = list(words)
    while len(out) > 2 and out[0].lower() in _BOILERPLATE:
        out = out[1:]
    while len(out) > 2 and out[-1].lower() in _BOILERPLATE:
        out = out[:-1]
    return out or list(words)


def title_line_options(title):
    """Candidate (first line, second line) pairs, the most informative first.

    There is a real conflict in the brief, and it is geometric: a block a third of the frame
    tall makes the first line about 185px, and Anton at 185px fits roughly eight characters
    into the left half of a 1280px frame. A phrase like MONEY AS A TOOL cannot be that line.
    So the words are not chosen here and then forced to fit - every reasonable pair is
    offered, and compose() takes the first that the geometry and the readability check both
    accept. The order is what reasonable means: keep as many of the title words as possible,
    and prefer the split a reader would make.
    """
    clauses = [c for c in _CLAUSE_SPLIT.split(title or "") if _words(c)]
    pools = []
    if len(clauses) >= 2:
        # A comma is a setup and a payoff. The payoff alone is usually the better thumbnail,
        # so it is offered as its own pool as well as paired with the setup.
        setup = _trim_boilerplate(_words(clauses[-2]))[-MAX_WORDS_PER_LINE:]
        payoff = _trim_boilerplate(_words(clauses[-1]))[:MAX_WORDS_PER_LINE]
        if setup and payoff:
            pools.append((setup, payoff))
        pools.append(_trim_boilerplate(_words(clauses[-1])))
    pools.append(_trim_boilerplate(_words(title))[-MAX_WORDS:])

    options, seen = [], set()

    def offer(first, second):
        if not first or not second:
            return
        if max(len(first), len(second)) > MAX_WORDS_PER_LINE:
            return
        key = (tuple(first), tuple(second))
        if key in seen:
            return
        seen.add(key)
        options.append(([w.upper() for w in first], [w.upper() for w in second]))

    for pool in pools:
        if isinstance(pool, tuple):
            offer(*pool)
            continue
        # Every suffix of the pool, longest first, split every way; a suffix rather than any
        # sub-sequence, because a phrase that starts mid-thought reads as a mistake.
        for start in range(0, max(1, len(pool) - 1)):
            words = pool[start:]
            splits = sorted(range(1, len(words)),
                            key=lambda cut: abs(len(" ".join(words[:cut]))
                                                - len(" ".join(words[cut:]))))
            for cut in splits:
                offer(words[:cut], words[cut:])
    return options or [(["STOIC"], ["LESSONS"])]


def title_lines(title):
    """The first candidate pair, for callers that only want the words."""
    return title_line_options(title)[0]


# --- the picture -------------------------------------------------------------

def crop_16_9(img):
    """The centred 16:9 part of an image, as it will appear in the thumbnail."""
    w, h = img.size
    target_h = min(h, round(w * 9 / 16))
    target_w = min(w, round(h * 16 / 9))
    left, top = (w - target_w) // 2, (h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def image_stats(path):
    """What makes a picture work behind this layout, measured on the 16:9 crop that will
    actually be shown. All four are on comparable scales and all four are wanted low or
    high as noted:

    contrast  standard deviation of luminance - high is good
    rim       mean luminance of the outer 12% - low is good, it is where the vignette bites
    left      mean luminance of the left 45%, under the type - low is good
    detail    mean absolute Laplacian, i.e. how much texture there is - high is good

    `detail` is the one that earns its place. Contrast alone rewards a big flat bright shape
    on a dark ground, and the first picture it chose for video 7 was a blank sheet of paper
    held up to the camera: maximum contrast, nothing to look at. Texture tells those apart.
    """
    with Image.open(path) as img:
        lum = np.asarray(crop_16_9(img.convert("L")).resize((320, 180)), dtype=np.float64)
    bh, bw = int(lum.shape[0] * 0.12), int(lum.shape[1] * 0.12)
    rim = np.ones(lum.shape, dtype=bool)
    rim[bh:-bh, bw:-bw] = False
    laplacian = (4 * lum[1:-1, 1:-1] - lum[:-2, 1:-1] - lum[2:, 1:-1]
                 - lum[1:-1, :-2] - lum[1:-1, 2:])
    return (float(lum.std()), float(lum[rim].mean()),
            float(lum[:, :int(lum.shape[1] * 0.45)].mean()), float(np.abs(laplacian).mean()))


def pick_background(candidates):
    """The most striking path from [(path, provider)]: highest contrast, darkest rim.

    Pollinations images carry a watermark and duplicates are the same picture twice, so
    both are set aside - unless that leaves nothing, in which case anything will do.

    Then the candidates are walked from best to worst and each is read by OCR, because the
    picture generator sometimes draws lettering and the very images that carry it - a sheet
    of paper, a sign, a screen - are exactly the high-contrast, dark-edged ones this scoring
    likes. Caught live: the first run of this layout put NNNICE across the top of video 7's
    thumbnail, and after that was fixed the next-best candidate had a serif INVOICE on it.
    Both came down to how Tesseract was asked - see core/ocr.py - and the settings here
    are the strict ones, because skipping a candidate costs nothing. On video 7 that
    leaves 57 of 70 usable. It is still a net with holes.
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

    stats = np.array([image_stats(p) for p, _ in clean])
    contrast, rim, left, detail = stats.T

    def unit(v):
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 0 else np.zeros_like(v)

    score = unit(contrast) + (1 - unit(rim)) + (1 - unit(left)) + unit(detail)
    order = list(np.argsort(-score))
    rejected = []
    for rank, i in enumerate(order):
        chars, _ = (ocr.detect_text(clean[i][0], **BACKGROUND_OCR)
                    if ocr.available() else (0, 0))
        if chars <= BACKGROUND_OCR_CHARS:
            return clean[i][0], {"contrast": float(contrast[i]), "rim": float(rim[i]),
                                 "left": float(left[i]), "detail": float(detail[i]),
                                 "score": float(score[i]), "candidates": len(clean),
                                 "rank": rank + 1, "rejected_for_text": rejected}
        rejected.append(f"{Path(clean[i][0]).name} ({chars} characters)")
        log.info("background %s skipped: OCR reads %d characters on it",
                 Path(clean[i][0]).name, chars)

    best = int(score.argmax())
    log.warning("every candidate has readable text on it; using the best-scoring one anyway")
    return clean[best][0], {"contrast": float(contrast[best]), "rim": float(rim[best]),
                            "left": float(left[best]), "detail": float(detail[best]),
                            "score": float(score[best]), "candidates": len(clean),
                            "rank": 1, "rejected_for_text": rejected}


def vignette_mask():
    """A (HEIGHT, WIDTH) multiplier: 1 in the middle, falling towards the corners."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    r = np.sqrt(((x - WIDTH / 2) / (WIDTH / 2)) ** 2 + ((y - HEIGHT / 2) / (HEIGHT / 2)) ** 2)
    r = np.clip(r / np.sqrt(2), 0, 1)
    smooth = r * r * (3 - 2 * r)
    return 1 - VIGNETTE_STRENGTH * smooth


def background(path):
    """The picture: cropped to 16:9, contrast pushed, darkened, vignetted."""
    with Image.open(path) as img:
        img = crop_16_9(img.convert("RGB")).resize((WIDTH, HEIGHT), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(CONTRAST)
    img = ImageEnhance.Brightness(img).enhance(DARKEN)
    arr = np.asarray(img, dtype=np.float64) * vignette_mask()[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_scrim(img, box):
    """Darken the area behind the type, fading out upward and to the right.

    `box` is (left, top, right, bottom) of the type. The scrim is at full strength across
    the block and the frame's bottom-left corner and falls away over SCRIM_PAD past its
    edges, so there is no visible rectangle - only a part of the picture that got darker.
    """
    _, top, right, _ = box
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    right_falloff = np.clip((right + SCRIM_PAD_X - x) / SCRIM_PAD_X, 0, 1)
    top_falloff = np.clip((y - (top - SCRIM_PAD_TOP)) / SCRIM_PAD_TOP, 0, 1)
    mask = np.minimum(right_falloff, top_falloff)
    mask = mask * mask * (3 - 2 * mask)                 # smoothstep, so there is no banding
    arr = np.asarray(img, dtype=np.float64) * (1 - SCRIM_DARKNESS * mask)[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# --- laying out the type -----------------------------------------------------

def _measure(line, size, font_path=FONT_PATH):
    font = ImageFont.truetype(str(font_path), size)
    box = font.getbbox(line, stroke_width=STROKE)
    return font, box, box[2] - box[0], box[3] - box[1]


def layouts(first, second, font_path=FONT_PATH):
    """Every candidate layout, largest type first.

    Only those that fit the text column and land inside the height the brief asks for are
    offered, so anything this returns already satisfies the shape requirements.
    """
    column = WIDTH * TEXT_COLUMN - MARGIN_X
    first_text, second_text = " ".join(first), " ".join(second)
    out = []
    for size in range(220, 60, -4):
        for ratio in SECOND_LINE_RATIOS:
            small = max(24, round(size * ratio))
            _, box1, w1, h1 = _measure(first_text, size, font_path)
            _, box2, w2, h2 = _measure(second_text, small, font_path)
            gap = round(size * LINE_GAP)
            height, width = h1 + gap + h2, max(w1, w2)
            if width > column:
                continue
            if not (HEIGHT * MIN_BLOCK_FRACTION <= height <= HEIGHT * MAX_BLOCK_FRACTION):
                continue
            out.append({"size": size, "small_size": small, "gap": gap, "height": height,
                        "width": width, "boxes": (box1, box2),
                        "first": first_text, "second": second_text})
    return out


def draw_lines(img, layout, font_path=FONT_PATH):
    """Draw both lines. Returns the block's (left, top, right, bottom) and a mask per line."""
    box1, box2 = layout["boxes"]
    top = HEIGHT - MARGIN_BOTTOM - layout["height"]
    draw = ImageDraw.Draw(img)
    masks, y = {}, top
    for text, size, colour, box, key in (
            (layout["first"], layout["size"], WHITE, box1, "first"),
            (layout["second"], layout["small_size"], WARM, box2, "second")):
        font = ImageFont.truetype(str(font_path), size)
        # box[0]/box[1] are the ink's offset from the drawing origin; cancelling them puts
        # the ink itself on the margin, not the font's empty ascender space.
        at = (MARGIN_X - box[0], y - box[1])
        draw.text(at, text, font=font, fill=colour, stroke_width=STROKE, stroke_fill="black")
        mask = Image.new("L", (WIDTH, HEIGHT), 0)
        ImageDraw.Draw(mask).text(at, text, font=font, fill=255,
                                  stroke_width=STROKE, stroke_fill=255)
        masks[key] = mask
        y += (box[3] - box[1]) + layout["gap"]
    return (MARGIN_X, top, MARGIN_X + layout["width"], top + layout["height"]), masks


# --- the check ---------------------------------------------------------------

def _relative_luminance(gray):
    c = np.asarray(gray, dtype=np.float64) / 255
    return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _contrast_ratio(colour, backdrop_small, mask_small):
    """WCAG contrast of `colour` against what is behind it, at thumbnail-feed size.

    The 90th percentile of the backdrop is used, not its mean, so one bright patch showing
    through a letter counts against the layout instead of being averaged away.
    """
    where = mask_small > 40
    if not where.any():
        return 0.0
    behind = float(np.percentile(_relative_luminance(backdrop_small[where]), 90))
    grey = 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]
    text = float(_relative_luminance(np.array([grey]))[0])
    return (max(text, behind) + 0.05) / (min(text, behind) + 0.05)


def readability(backdrop, masks, layout, font_path=FONT_PATH):
    """What survives at 210x118: capital height of the smaller line, contrast of each line."""
    scale = SMALL[1] / HEIGHT
    cap = ImageFont.truetype(str(font_path), layout["small_size"]).getbbox("H")
    cap_px = (cap[3] - cap[1]) * scale

    small_backdrop = np.asarray(backdrop.convert("L").resize(SMALL, Image.LANCZOS))
    ratios = {key: _contrast_ratio(colour, small_backdrop,
                                   np.asarray(masks[key].resize(SMALL, Image.LANCZOS)))
              for key, colour in (("first", WHITE), ("second", WARM))}
    return {"cap_px": cap_px, "contrast_first": ratios["first"],
            "contrast_second": ratios["second"],
            "ok": cap_px >= MIN_SMALL_CAP_PX and min(ratios.values()) >= MIN_CONTRAST}


def _ocr_reading(small_path, first, second):
    if not ocr.available():
        return None, ""
    with Image.open(small_path) as img:
        read = ocr.read_text(img.convert("L").resize((SMALL[0] * 4, SMALL[1] * 4), Image.LANCZOS))
    read = read.strip().replace("\n", " ")
    found = {w.lower() for w in re.findall(r"[A-Za-z']+", read)}
    wanted = [w.lower().strip("'-") for w in first + second]
    return sum(1 for w in wanted if w in found) / len(wanted), read


# --- putting it together -----------------------------------------------------

def compose(source, options, font_path=FONT_PATH):
    """Try every (words, layout) pair and return (image, layout, readability, tried).

    `options` is title_line_options(): candidate word pairs, most informative first. For each
    pair the layouts come largest-first, so the first combination that passes the readability
    check keeps as much of the title as the geometry allows and sets it as large as it can.
    """
    base = background(source)
    best, tried = None, 0
    for first, second in options:
        for layout in layouts(first, second, font_path):
            tried += 1
            box, masks = draw_lines(base.copy(), layout, font_path)
            backdrop = apply_scrim(base.copy(), box)
            final = backdrop.copy()
            draw_lines(final, layout, font_path)
            check = readability(backdrop, masks, layout, font_path)
            if check["ok"]:
                return final, layout, check, tried
            if best is None or check["cap_px"] > best[2]["cap_px"]:
                best = (final, layout, check, tried)
    if best is None:
        raise RuntimeError("no two-line layout fits in the left half of the frame for "
                           + repr(options[0]))
    return best[0], best[1], best[2], tried


def make_thumbnail(candidates, title, out_dir, font_path=FONT_PATH):
    """Build thumbnail.jpg and thumb_small.jpg in out_dir. Returns a details dict."""
    out_dir = Path(out_dir)
    options = title_line_options(title)
    source, pick = pick_background(candidates)
    img, layout, check, tried = compose(source, options, font_path)
    first, second = layout["first"].split(), layout["second"].split()

    thumb = out_dir / "thumbnail.jpg"
    img.save(thumb, "JPEG", quality=JPEG_QUALITY)
    small = out_dir / "thumb_small.jpg"
    with Image.open(thumb) as full:
        full.resize(SMALL, Image.LANCZOS).save(small, "JPEG", quality=JPEG_QUALITY)
    share, read = _ocr_reading(small, first, second)

    return {"path": thumb, "small": small, "first": first, "second": second, "source": source,
            "font_size": layout["size"], "small_font_size": layout["small_size"],
            "block_height": layout["height"], "block_width": layout["width"],
            "block_fraction": round(layout["height"] / HEIGHT, 3),
            "layouts_tried": tried, "word_options": len(options),
            "readability": {**check, "ocr_share": share, "ocr_read": read}, **pick}


def run(video_id, cfg):
    video = db.get_video(video_id) or {}
    title = video.get("seo_title") or video.get("title")
    if not title:
        raise RuntimeError("the video has no title to take the thumbnail words from")

    rows = db.get_video_images(video_id)
    candidates = [(r["path"], r["provider"]) for r in rows if r["path"]]
    result = make_thumbnail(candidates, title, output_dir(video_id))

    log.info("background %s (contrast %.1f, rim %.1f, left %.1f, detail %.1f, ranked %d of "
             "%d%s)", Path(result["source"]).name, result["contrast"], result["rim"],
             result["left"], result["detail"], result["rank"], result["candidates"],
             f", {len(result['rejected_for_text'])} skipped for lettering"
             if result["rejected_for_text"] else "")
    log.info("type: %r at %dpx over %r at %dpx, block %dx%d = %.0f%% of the frame height "
             "(%d of %d word pairs were on the table, %d combinations tried)",
             " ".join(result["first"]), result["font_size"], " ".join(result["second"]),
             result["small_font_size"], result["block_width"], result["block_height"],
             result["block_fraction"] * 100, 1, result["word_options"], result["layouts_tried"])

    check = result["readability"]
    summary = (f"at 210x118 a capital in the smaller line is {check['cap_px']:.0f}px "
               f"(need {MIN_SMALL_CAP_PX}), contrast {check['contrast_first']:.1f}:1 white and "
               f"{check['contrast_second']:.1f}:1 amber (need {MIN_CONTRAST}:1)")
    if check["ocr_share"] is not None:
        summary += f"; OCR read {check['ocr_share']:.0%} of the words (informational)"
    if check["ok"]:
        log.info("thumb_small.jpg readable: %s", summary)
    else:
        msg = f"no layout passed the feed-size check, keeping the largest: {summary}"
        log.warning(msg)
        db.log_event(video_id, "thumbnail", "warning", msg)

    (output_dir(video_id) / "thumbnail.json").write_text(
        json.dumps({k: v for k, v in result.items() if k not in ("path", "small")},
                   indent=2, default=str), encoding="utf-8")
    db.update_video(video_id, thumbnail_path=str(result["path"]))
    db.log_event(video_id, "thumbnail", "info",
                 f"{Path(result['source']).name} behind "
                 f"{' '.join(result['first'])} / {' '.join(result['second'])}")
    return True
