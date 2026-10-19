"""The channel's thumbnail: a marble bust on the left, two or three words on the right.

The look, fixed by the owner and followed exactly:

- left, about 45% of the width: a close-up marble bust of a Stoic philosopher - classical
  Greco-Roman sculpture, dramatic side light, real marble texture, the gaze a little aside;
- behind it a dark grey ground with a little smoke or mist, never pure black;
- the bottom-right corner, 25% of the width by 20% of the height, is left empty: YouTube
  draws the duration stamp there and it covered the yellow line once. The type is moved up
  until it clears the corner, and the check below measures that on the drawn pixels;
- right: two or three words, large, bold, white - what the video is about, e.g. SELF
  DISCIPLINE - and under them in yellow "- MARCUS AURELIUS", "- SENECA" or "- EPICTETUS".
  The words are drawn by code with a light shadow, never by the image model.
- A philosopher is named only when the script really leans on him, counted in the narration
  itself; otherwise the line says "- STOIC WISDOM". The bust is the philosopher named.

How the bust is got. Cloudflare's flux-1-schnell, the same model the narration frames use,
at more steps than they get, three or four candidates at a time. Gemini looks at the
candidates side by side and picks the one with real-looking marble, an undeformed face, no
lettering of its own and the right composition. If Gemini is not available the pick falls to
measurements and OCR.

How the result is checked. Every composed thumbnail is measured: the bust's mass sits in
the left part of the frame and stays out of the type, every line of type is inside the frame
and still readable at 336x189 (cap height and WCAG contrast against what is behind it), and
the ground is not black. A thumbnail that fails is drawn again with fresh candidates, up to
three times; after that the best one made is used anyway, because a weaker thumbnail costs a
little and a missing video costs the day.

Neurons. The narration frames come first: this runs after them and spends only what
core.neurons says the UTC day has left over, keeping its reserve. Less room means fewer
candidates; no room at all means no bust today and the caller falls back to the older
thumbnail. Pollinations is not used: it stamps its logo on the picture and does not draw
marble - both were caught on the first live run.

Measured live on 18 October: flux-1-schnell costs about 19 + 38 Neurons a step, so 326 an
image at 8 steps against 96 at the frames' 2. After a day's frames there is room for about
three candidates, which is why the count is set by the budget rather than fixed.
"""
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from core import db, images as image_api, neurons, ocr
from core.llm import LLMError, complete_json, vision_json
from core.logger import get_logger

log = get_logger("bust")

WIDTH, HEIGHT = 1280, 720
CHECK_SIZE = (336, 189)
JPEG_QUALITY = 90
FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Anton-Regular.ttf"

WHITE = (255, 255, 255)
YELLOW = (255, 204, 0)
DASH = "–"

DEFAULTS = {
    "candidates": 4,
    "attempts": 3,
    "steps": 8,
    "fallback_steps": 4,             # if the model refuses 8
    "reserve_neurons": neurons.DEFAULT_RESERVE,
    "frames_reserve_neurons": 9250,  # held back for the day's narration frames (seen: 8256-9216)
    "vision_model": "gemini-flash-latest",
}

# --- who ---------------------------------------------------------------------

PHILOSOPHERS = {
    "marcus": {
        "label": "MARCUS AURELIUS",
        "pattern": r"\bmarcus(?:\s+aurelius)?\b|\baurelius\b",
        "look": "the Roman emperor Marcus Aurelius, a middle-aged man with thick curly hair "
                "and a full curly beard",
    },
    "seneca": {
        "label": "SENECA",
        "pattern": r"\bseneca\b|\blucilius\b",
        "look": "the Roman philosopher Seneca, an older man with short thinning hair, a short "
                "rough beard, deep-set eyes and a lined, gaunt face",
    },
    "epictetus": {
        "label": "EPICTETUS",
        "pattern": r"\bepictetus\b|\benchiridion\b",
        "look": "the Greek Stoic philosopher Epictetus, an elderly man with a long full beard, "
                "short hair and a calm, stern expression",
    },
}
GENERIC = {"label": "STOIC WISDOM",
           "look": "an ancient Greek Stoic philosopher, an older man with a long full beard and "
                   "short curly hair"}

# Named only when the narration leans on him: at least this many mentions, and at least
# DOMINANCE times as many as the next one. Three passing quotes is a video about the topic,
# not about the man.
MIN_MENTIONS = 4
DOMINANCE = 2.0


def mentions(text):
    return {key: len(re.findall(p["pattern"], text or "", re.I))
            for key, p in PHILOSOPHERS.items()}


def philosopher_for(text):
    """(key or None, counts). None means the line reads STOIC WISDOM."""
    counts = mentions(text)
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    (top, n), (_, runner_up) = ranked[0], ranked[1]
    if n >= MIN_MENTIONS and n >= DOMINANCE * runner_up:
        return top, counts
    return None, counts


def attribution(key):
    return f"{DASH} {(PHILOSOPHERS[key] if key else GENERIC)['label']}"


def video_text(video_id):
    """The narration of a video, from the database, which is what CI still has of it."""
    with db._connect() as conn:
        rows = conn.execute("SELECT text FROM scenes WHERE video_id = ? ORDER BY idx",
                            (video_id,)).fetchall()
    return " ".join(r["text"] or "" for r in rows)


# --- the words ---------------------------------------------------------------

MIN_WORDS, MAX_WORDS = 2, 3
MAX_CHARS = 20
MAX_WORD_CHARS = 11
_FUNCTION = {"a", "an", "the", "to", "of", "for", "and", "or", "in", "on", "with", "how",
             "why", "when", "while", "that", "this", "these", "from", "by", "at", "it", "is",
             "are", "be", "your", "you", "before", "after", "first", "every", "no", "one"}
_EMPTY = {"stoic", "stoicism", "lessons", "lesson", "ways", "things", "rules", "tips",
          "philosophy", "wisdom"}
_PREFIX = re.compile(r"^\s*\d+\s+(stoic\s+)?(lessons|things|ways|rules|habits|truths)"
                     r"(\s+(to|that|for|from))?\s+", re.I)


def headline_problems(text):
    words = (text or "").split()
    problems = []
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        problems.append(f"{len(words)} words, need {MIN_WORDS}-{MAX_WORDS}")
    if any(not re.fullmatch(r"[A-Z][A-Z'\-]*", w) for w in words):
        problems.append("only capital letters, no digits or punctuation")
    if len(text or "") > MAX_CHARS:
        problems.append(f"{len(text)} characters, at most {MAX_CHARS}")
    if any(len(w) > MAX_WORD_CHARS for w in words):
        problems.append(f"a word longer than {MAX_WORD_CHARS} letters")
    if words and words[0].lower() in _FUNCTION:
        problems.append(f"opens on '{words[0]}'")
    if words and all(w.lower() in _EMPTY | _FUNCTION for w in words):
        problems.append("says nothing the niche does not already say")
    return problems


def headline_from_title(title):
    rest = _PREFIX.sub("", title or "")
    words = [w.upper() for w in re.findall(r"[A-Za-z']+", rest)
             if w.lower() not in _FUNCTION | _EMPTY and len(w) <= MAX_WORD_CHARS]
    for n in (3, 2):
        if len(words) >= n and len(" ".join(words[:n])) <= MAX_CHARS:
            return " ".join(words[:n])
    return "INNER PEACE"


HEADLINE_PROMPT = """Write the headline for a YouTube thumbnail of a Stoic philosophy video.

Video title: {title}
Topic: {topic}
{promise}
The headline is two or three words in capitals that name the core of the video, the way a
book spine would: SELF DISCIPLINE, INNER PEACE, LET IT GO, FEAR NOTHING, CONTROL YOUR MIND,
MASTER YOUR TIME. Plain words a viewer gets in half a second. Not a fragment of the title,
not a question, no digits, no punctuation, at most {max_chars} characters in total, and
do not use the words STOIC, STOICISM, LESSONS or WISDOM.
{retry}
Reply with JSON: {{"headline": "..."}}"""


def headline_for(title, topic, promise=""):
    """(headline, source). Source is "llm" or "title" when the model could not be used."""
    complaint = ""
    for _ in range(2):
        try:
            answer = complete_json(HEADLINE_PROMPT.format(
                title=title, topic=topic or title, max_chars=MAX_CHARS,
                promise=f"What the viewer gets: {promise}\n" if promise else "",
                retry=f"\nYour last answer was refused: {complaint}.\n" if complaint else ""),
                max_tokens=200, temperature=0.6)
        except Exception as e:                                      # noqa: BLE001
            log.warning("the headline could not be written: %s", e)
            break
        text = " ".join(str((answer or {}).get("headline", "")).upper().split())
        problems = headline_problems(text)
        if not problems:
            return text, "llm"
        complaint = f'"{text}": ' + "; ".join(problems)
    return headline_from_title(title), "title"


# --- the bust ----------------------------------------------------------------

def bust_prompt(key):
    look = (PHILOSOPHERS[key] if key else GENERIC)["look"]
    return ("photorealistic close-up photograph of an ancient white marble bust of "
            f"{look}, classical Greco-Roman sculpture, head and shoulders filling most of the "
            "frame, three-quarter view, head turned slightly to the right, eyes gazing off to "
            "the side, dramatic side lighting from the left, deep sculpted shadows, highly "
            "detailed realistic marble texture with subtle grey veins and weathering, the bust "
            "centred slightly left, dark charcoal grey background with soft drifting smoke and "
            "mist, museum photography, 85mm lens, sharp focus, no text, no letters, no "
            "inscription, no plaque, no watermark")


class _Drawer:
    """Draws bust candidates from Cloudflare within a Neuron budget."""

    def __init__(self, cfg, settings, budget, out_dir):
        self.cfg, self.s, self.budget, self.out = cfg, settings, budget, Path(out_dir)
        self.steps = settings["steps"]
        self.per_image = None
        self.spent = 0.0
        self.cf_calls = 0
        self.cf_dead = not image_api.cloudflare_available()
        self.counter = 0

    def per_image_estimate(self):
        return self.per_image or neurons_per_image(self.steps)

    def affordable(self):
        return int((self.budget - self.spent) // self.per_image_estimate())

    def _cf_cfg(self):
        cfg = json.loads(json.dumps(self.cfg))
        cfg.setdefault("images", {}).setdefault("cloudflare", {})["steps"] = self.steps
        return cfg

    def _cloudflare(self, prompt, path):
        for attempt in range(2):
            try:
                _, used = image_api.synthesize_cloudflare(prompt, path, 0, self._cf_cfg())
            except image_api.ImageError as e:
                text = str(e)
                if "steps" in text.lower() and "400" in text and self.steps != \
                        self.s["fallback_steps"]:
                    log.warning("the model refused %d steps, using %d: %s", self.steps,
                                self.s["fallback_steps"], text[:160])
                    self.steps = self.s["fallback_steps"]
                    continue
                if image_api._worth_asking_again(e) and attempt == 0:
                    continue
                if re.search(r"4006|daily|quota|limit|auth", text, re.I):
                    self.cf_dead = True
                log.warning("cloudflare bust failed: %s", text[:200])
                return False
            self.cf_calls += 1
            cost = float(used) if used is not None else self.per_image_estimate()
            self.spent += cost
            self.per_image = cost if self.per_image is None else max(self.per_image, cost)
            return True
        return False

    def draw(self, prompt, wanted):
        """[(path, provider)], as many as the budget allows up to `wanted`."""
        made = []
        cf_count = 0 if self.cf_dead else min(wanted, self.affordable())
        if cf_count < wanted:
            log.info("neurons: %.0f free for the thumbnail, about %.0f an image - %d of %d "
                     "candidates from Cloudflare", self.budget - self.spent,
                     self.per_image_estimate(), cf_count, wanted)
        for _ in range(cf_count):
            if self.cf_dead or self.affordable() < 1:
                break
            path = self.out / f"bust_{self.counter:02d}.jpg"
            self.counter += 1
            if self._cloudflare(prompt, path):
                made.append((str(path), "cloudflare"))
        return made


# --- choosing a candidate ------------------------------------------------------

VISION_PROMPT = """You are choosing the background picture for a YouTube thumbnail.
The {n} images follow in order; the first is index 0.

The target: a photorealistic classical Greco-Roman WHITE MARBLE BUST of {who}, a close-up of
head and shoulders, dramatic side light, realistic marble texture, the gaze slightly to the
side, on a dark grey smoky background (not pure black), the bust in the centre or left of
centre and not cut off at the top of the head.

For every image report:
- "marble": 0-10, how convincingly it is real carved marble (not plastic, not a painting, not
  a living person);
- "face_ok": true only if the face is undeformed - two matching eyes, an intact nose and
  mouth, no melted, doubled or missing features;
- "text": true if ANY letters, inscription, signature, logo or watermark is visible anywhere;
- "composition": 0-10, how well it matches the target composition above;
- "likeness": 0-10, how well it fits {who}.

Reply with JSON: {{"images": [{{"index": 0, "marble": 0, "face_ok": true, "text": false,
"composition": 0, "likeness": 0}}], "best": 0}}"""


def _jpeg_bytes(path, size=448):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return buf.getvalue()


def _detail(path):
    with Image.open(path) as img:
        lum = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float64)
    lap = (4 * lum[1:-1, 1:-1] - lum[:-2, 1:-1] - lum[2:, 1:-1] - lum[1:-1, :-2]
           - lum[1:-1, 2:])
    return float(np.abs(lap).mean())


def neurons_per_image(steps):
    """flux-1-schnell's cost for one 1024x1024 image, as measured: 96 at 2 steps, 326 at 8."""
    return 19.2 + 38.4 * steps


def rank_candidates(paths, key, models):
    """[(path, score, acceptable, notes)] best first. Gemini judges; measurements if not."""
    who = (PHILOSOPHERS[key]["label"].title() if key else "an ancient Stoic philosopher")
    verdicts = {}
    judged_by = "gemini"
    pictures = [_jpeg_bytes(p) for p in paths]
    for model in [m for i, m in enumerate(models) if m and m not in models[:i]]:
        try:
            answer = vision_json(VISION_PROMPT.format(n=len(paths), who=who), pictures,
                                 model=model)
            for item in (answer or {}).get("images", []):
                i = int(item.get("index", -1))
                if 0 <= i < len(paths):
                    verdicts[i] = item
            break
        except (LLMError, ValueError, TypeError) as e:
            log.warning("Gemini %s could not judge the busts (%s)", model, str(e)[:160])
    if not verdicts:
        log.warning("no Gemini model judged the busts; ranking by measurement")
        judged_by = "measurement"

    ranked = []
    for i, path in enumerate(paths):
        text_chars = 0
        if ocr.available():
            text_chars, _ = ocr.detect_text(path, min_confidence=50, min_word_len=3, upscale=2,
                                            contrast=1.6)
        v = verdicts.get(i)
        if v is not None:
            score = (float(v.get("marble", 0)) + float(v.get("composition", 0))
                     + 0.5 * float(v.get("likeness", 0)))
            ok = (bool(v.get("face_ok")) and not bool(v.get("text"))
                  and float(v.get("marble", 0)) >= 6 and float(v.get("composition", 0)) >= 5
                  and text_chars <= 2)
            notes = (f"marble {v.get('marble')}, composition {v.get('composition')}, likeness "
                     f"{v.get('likeness')}, face {'ok' if v.get('face_ok') else 'DEFORMED'}"
                     f"{', TEXT' if v.get('text') else ''}")
        else:
            score = _detail(path) / 4
            ok = text_chars <= 2
            notes = f"not judged by Gemini, detail {score:.1f}"
        if text_chars > 2:
            notes += f", OCR {text_chars} chars"
        ranked.append((path, score, ok, notes))
    ranked.sort(key=lambda r: (not r[2], -r[1]))
    return ranked, judged_by


# --- composing ---------------------------------------------------------------

BUST_CENTRE_X = 0.235          # of the width
FADE_FROM, FADE_TO = 0.40, 0.53
TEXT_LEFT = 0.515
MARGIN = 56
MIN_BG_LUM, MAX_BG_LUM = 34, 78


def _lum(arr):
    return arr[..., :3] @ np.array([0.2126, 0.7152, 0.0722])


def ground_colour(bust):
    """A dark grey taken from the candidate's own background, kept out of black."""
    arr = np.asarray(bust, dtype=np.float64)
    edge = np.concatenate([arr[:, -int(arr.shape[1] * 0.08):].reshape(-1, 3),
                           arr[:int(arr.shape[0] * 0.06)].reshape(-1, 3)])
    colour = np.median(edge, axis=0)
    grey = colour.mean()
    colour = 0.35 * colour + 0.65 * grey               # mostly neutral
    lum = float(_lum(colour))
    target = float(np.clip(lum, MIN_BG_LUM, MAX_BG_LUM))
    return colour * (target / max(lum, 1.0))


def smoke(seed, size=(HEIGHT, WIDTH)):
    """Soft wisps in 0..1: a few octaves of blurred noise, thresholded."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(size)
    for grid, amp in (((5, 9), 1.0), ((9, 16), 0.55), ((18, 32), 0.3), ((36, 64), 0.15)):
        small = Image.fromarray((rng.random(grid) * 255).astype(np.uint8))
        acc += amp * np.asarray(small.resize((size[1], size[0]), Image.BICUBIC),
                                dtype=np.float64) / 255
    acc = (acc - acc.min()) / max(acc.max() - acc.min(), 1e-6)
    return np.clip((acc - 0.45) / 0.55, 0, 1) ** 1.6


def ground(colour, seed):
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    glow = np.exp(-(((x - WIDTH * 0.72) / (WIDTH * 0.45)) ** 2
                    + ((y - HEIGHT * 0.45) / (HEIGHT * 0.7)) ** 2))
    shade = 0.90 + 0.22 * glow                          # a little lighter behind the type
    wisps = smoke(seed)
    arr = colour[None, None, :] * shade[:, :, None] + 34 * wisps[:, :, None]
    return np.clip(arr, 0, 255)


def place_bust(bust, colour, seed):
    """The 16:9 backdrop: the ground, with the square bust laid in on the left."""
    canvas = ground(colour, seed)
    square = np.asarray(bust.convert("RGB").resize((HEIGHT, HEIGHT), Image.LANCZOS),
                        dtype=np.float64)
    floor = 0.55 * colour                               # lift the blacks: no pure black
    square = floor + square * (1 - floor / 255)
    left = round(WIDTH * BUST_CENTRE_X - HEIGHT / 2)
    x = np.arange(WIDTH, dtype=np.float64)
    alpha = np.clip((WIDTH * FADE_TO - x) / (WIDTH * (FADE_TO - FADE_FROM)), 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    lo, hi = max(0, left), min(WIDTH, left + HEIGHT)
    region = square[:, lo - left:hi - left]
    a = alpha[lo:hi][None, :, None]
    canvas[:, lo:hi] = region * a + canvas[:, lo:hi] * (1 - a)
    wisps = smoke(seed + 1)[:, :, None]
    canvas = canvas + 14 * wisps * (1 - 0.6 * alpha[None, :, None])   # one air over both
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def subject_mask(bust, colour):
    """Where the bust is in the placed backdrop: pixels well away from the ground."""
    square = np.asarray(bust.convert("L").resize((HEIGHT, HEIGHT)), dtype=np.float64)
    bg = float(np.median(np.concatenate([square[:, -40:].ravel(), square[:30].ravel()])))
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    left = round(WIDTH * BUST_CENTRE_X - HEIGHT / 2)
    lo, hi = max(0, left), min(WIDTH, left + HEIGHT)
    mask[:, lo:hi] = np.abs(square[:, lo - left:hi - left] - bg) > 38
    return mask


def _font(size):
    return ImageFont.truetype(str(FONT_PATH), size)


def line_options(words):
    n = len(words)
    options = [[" ".join(words)]]
    for cut in range(1, n):
        options.append([" ".join(words[:cut]), " ".join(words[cut:])])
    if n == 3:
        options.append(list(words))
    return options


# Where YouTube's duration stamp sits: nothing of ours may touch it.
CORNER_W, CORNER_H = 0.25, 0.20
CORNER = (round(WIDTH * (1 - CORNER_W)), round(HEIGHT * (1 - CORNER_H)), WIDTH, HEIGHT)
CORNER_PAD = 10                  # clear air above and left of it, for the shadow


def _intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _inks(layout, top):
    """[(name, (x0, y0, x1, y1))] of each line of type with its block's top edge at `top`."""
    out, y = [], top
    for box in layout["boxes"]:
        out.append(("head", (layout["left"], y, layout["left"] + box[2] - box[0],
                             y + box[3] - box[1])))
        y += (box[3] - box[1]) + layout["gap"]
    y += round(layout["size"] * 0.22) - layout["gap"]
    abox = layout["abox"]
    out.append(("attrib", (layout["left"], y, layout["left"] + abox[2] - abox[0],
                           y + abox[3] - abox[1])))
    return out


def place(layout):
    """The top edge for this block: as near the middle as it can be while every line stays
    out of the corner. None if no position does."""
    keep_out = (CORNER[0] - CORNER_PAD, CORNER[1] - CORNER_PAD, WIDTH, HEIGHT)
    preferred = round((HEIGHT - layout["total"]) / 2 - HEIGHT * 0.02)
    for top in range(preferred, MARGIN - 1, -2):
        if not any(_intersects(ink, keep_out) for _, ink in _inks(layout, top)):
            return top
    return None


def layout_text(headline, attrib):
    """The largest arrangement of the headline that fits the right-hand column and clears
    the bottom-right corner."""
    words = headline.split()
    left, right = round(WIDTH * TEXT_LEFT), WIDTH - MARGIN
    column = right - left
    best = None
    for lines in line_options(words):
        for size in range(230, 60, -4):
            font = _font(size)
            boxes = [font.getbbox(line) for line in lines]
            widths = [b[2] - b[0] for b in boxes]
            if max(widths) > column:
                continue
            gap = round(size * 0.10)
            head_h = sum(b[3] - b[1] for b in boxes) + gap * (len(lines) - 1)
            small = max(40, round(size * 0.34))
            while small > 30 and _font(small).getbbox(attrib)[2] > column:
                small -= 2
            abox = _font(small).getbbox(attrib)
            total = head_h + round(size * 0.22) + (abox[3] - abox[1])
            if total > HEIGHT - 2 * MARGIN - 40:
                continue
            candidate = {"lines": lines, "size": size, "small": small, "gap": gap,
                         "head_h": head_h, "total": total, "boxes": boxes, "abox": abox,
                         "left": left}
            candidate["top"] = place(candidate)
            if candidate["top"] is None:
                continue                      # too big to keep out of the corner: try smaller
            # Bigger type wins; at the same size, fewer lines.
            if best is None or (size, -len(lines)) > (best["size"], -len(best["lines"])):
                best = candidate
            break
    if best is None:
        raise RuntimeError(f"the headline {headline!r} does not fit the right-hand column")
    return best


def draw_text(img, layout, attrib):
    """Draw the type with a soft shadow. Returns (image, {name: mask}, [ink boxes])."""
    top = layout["top"]
    items, y = [], top
    for line, box in zip(layout["lines"], layout["boxes"]):
        items.append(("head", line, layout["size"], WHITE, (layout["left"] - box[0], y - box[1])))
        y += (box[3] - box[1]) + layout["gap"]
    y += round(layout["size"] * 0.22) - layout["gap"]
    abox = layout["abox"]
    items.append(("attrib", attrib, layout["small"], YELLOW,
                  (layout["left"] - abox[0], y - abox[1])))

    masks = {"head": Image.new("L", img.size, 0), "attrib": Image.new("L", img.size, 0)}
    inks = []
    for name, text, size, _, at in items:
        ImageDraw.Draw(masks[name]).text(at, text, font=_font(size), fill=255)
        bb = _font(size).getbbox(text)
        inks.append((at[0] + bb[0], at[1] + bb[1], at[0] + bb[2], at[1] + bb[3]))
    shadow_src = Image.fromarray(np.maximum(np.asarray(masks["head"]),
                                            np.asarray(masks["attrib"])))
    shadow = Image.new("L", img.size, 0)
    shadow.paste(shadow_src, (5, 6))
    shadow = shadow.filter(ImageFilter.GaussianBlur(7))
    arr = np.asarray(img, dtype=np.float64)
    arr = arr * (1 - 0.60 * np.asarray(shadow, dtype=np.float64)[:, :, None] / 255)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for _, text, size, colour, at in items:
        draw.text(at, text, font=_font(size), fill=colour)
    return out, masks, inks


def _rel_lum(v):
    c = np.asarray(v, dtype=np.float64) / 255
    return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _contrast(colour, backdrop_small, mask_small):
    where = mask_small > 60
    if not where.any():
        return 0.0
    behind = float(np.percentile(_rel_lum(backdrop_small[where]), 90))
    text = float(_rel_lum(np.array([_lum(np.array(colour, dtype=np.float64))]))[0])
    return (max(text, behind) + 0.05) / (min(text, behind) + 0.05)


MIN_HEAD_CAP_PX = 14            # at 336x189
MIN_ATTRIB_CAP_PX = 7
MIN_CONTRAST = 4.5
MAX_BLACK_SHARE = 0.02          # of pixels darker than BLACK_LUM
BLACK_LUM = 16
MIN_GROUND_LUM = 28
BUST_MAX_CENTROID = 0.42        # of the width
BUST_MIN_LEFT_SHARE = 0.85      # of the bust's mass left of the type column
BUST_MIN_AREA = 0.12            # of the left 45%: a close-up, not a figurine


def check(backdrop, final, masks, inks, layout, subject):
    """Every requirement measured. Returns a dict with "ok" and the reasons it is not."""
    problems = []
    xs = np.nonzero(subject)[1]
    left_cols = int(WIDTH * 0.45)
    if xs.size == 0:
        problems.append("no bust found on the left")
        centroid, share, area = 1.0, 0.0, 0.0
    else:
        centroid = float(xs.mean() / WIDTH)
        share = float((xs < WIDTH * TEXT_LEFT - 20).mean())
        area = float(subject[:, :left_cols].mean())
        if centroid > BUST_MAX_CENTROID:
            problems.append(f"the bust's centre is at {centroid:.0%} of the width")
        if share < BUST_MIN_LEFT_SHARE:
            problems.append(f"only {share:.0%} of the bust is left of the type")
        if area < BUST_MIN_AREA:
            problems.append(f"the bust fills only {area:.0%} of the left side")

    for box in inks:
        if box[0] < 16 or box[1] < 16 or box[2] > WIDTH - 16 or box[3] > HEIGHT - 16:
            problems.append(f"type is cut off at {box}")

    corner = np.zeros((HEIGHT, WIDTH), dtype=bool)
    corner[CORNER[1]:, CORNER[0]:] = True
    drawn = (np.asarray(masks["head"]) > 20) | (np.asarray(masks["attrib"]) > 20)
    if (drawn & corner).any():
        problems.append("type reaches the bottom-right corner, where YouTube puts the duration")
    for _, ink in _inks(layout, layout["top"]):
        if _intersects(ink, (CORNER[0] - CORNER_PAD, CORNER[1] - CORNER_PAD, WIDTH, HEIGHT)):
            problems.append("type is within a few pixels of the bottom-right corner")
            break

    scale = CHECK_SIZE[1] / HEIGHT
    head_cap = (_font(layout["size"]).getbbox("H")[3] - _font(layout["size"]).getbbox("H")[1]) \
        * scale
    attrib_cap = (_font(layout["small"]).getbbox("H")[3]
                  - _font(layout["small"]).getbbox("H")[1]) * scale
    if head_cap < MIN_HEAD_CAP_PX:
        problems.append(f"headline capitals are {head_cap:.1f}px at 336x189")
    if attrib_cap < MIN_ATTRIB_CAP_PX:
        problems.append(f"name capitals are {attrib_cap:.1f}px at 336x189")
    small_back = np.asarray(backdrop.convert("RGB").resize(CHECK_SIZE, Image.LANCZOS),
                            dtype=np.float64)
    small_back = _lum(small_back)
    ratios = {name: _contrast(colour, small_back,
                              np.asarray(masks[name].resize(CHECK_SIZE, Image.LANCZOS)))
              for name, colour in (("head", WHITE), ("attrib", YELLOW))}
    for name, ratio in ratios.items():
        if ratio < MIN_CONTRAST:
            problems.append(f"{name} contrast {ratio:.1f}:1 at 336x189")

    lum = _lum(np.asarray(backdrop.convert("RGB"), dtype=np.float64))
    black = float((lum < BLACK_LUM).mean())
    ground_lum = float(np.median(lum[:, round(WIDTH * TEXT_LEFT):]))
    if black > MAX_BLACK_SHARE:
        problems.append(f"{black:.0%} of the frame is black")
    if ground_lum < MIN_GROUND_LUM:
        problems.append(f"the ground is too dark ({ground_lum:.0f})")
    return {"ok": not problems, "problems": problems, "bust_centroid": round(centroid, 3),
            "bust_left_share": round(share, 3), "bust_area": round(area, 3),
            "head_cap_px": round(head_cap, 1), "attrib_cap_px": round(attrib_cap, 1),
            "contrast_head": round(ratios["head"], 1),
            "contrast_attrib": round(ratios["attrib"], 1),
            "black_share": round(black, 4), "ground_lum": round(ground_lum, 1)}


def compose(bust_path, headline, attrib, seed):
    with Image.open(bust_path) as img:
        bust = img.convert("RGB")
    colour = ground_colour(bust)
    backdrop = place_bust(bust, colour, seed)
    layout = layout_text(headline, attrib)
    final, masks, inks = draw_text(backdrop, layout, attrib)
    result = check(backdrop, final, masks, inks, layout, subject_mask(bust, colour))
    return final, result


# --- the whole job ---------------------------------------------------------------

def settings(cfg):
    return {**DEFAULTS, **((cfg or {}).get("thumbnail") or {})}


def make(video_id, title, topic, text, out_dir, cfg, promise="", budget=None):
    """Build out_dir/thumbnail.jpg. Returns a details dict, "neurons" among them.

    Raises RuntimeError only when no bust at all could be drawn; the caller falls back.
    """
    s = settings(cfg)
    out_dir = Path(out_dir)
    work = out_dir / "bust"
    work.mkdir(parents=True, exist_ok=True)
    if budget is None:
        budget = neurons.available(s["reserve_neurons"])

    key, counts = philosopher_for(text)
    attrib = attribution(key)
    headline, headline_source = headline_for(title, topic, promise)
    prompt = bust_prompt(key)
    drawer = _Drawer(cfg, s, budget, work)
    log.info("thumbnail: %r %s (mentions %s), %.0f neurons free", headline, attrib, counts,
             budget)

    best, tries = None, []
    for attempt in range(1, s["attempts"] + 1):
        drawn = drawer.draw(prompt, s["candidates"])
        if not drawn:
            log.warning("attempt %d: no bust could be drawn", attempt)
            break
        ranked, judged_by = rank_candidates(
            [p for p, _ in drawn], key,
            [s["vision_model"], ((cfg or {}).get("llm") or {}).get("gemini_model")])
        provider = dict(drawn)
        for rank, (path, score, acceptable, notes) in enumerate(ranked):
            if not acceptable and rank > 0:
                continue                       # the top one is composed even if flawed
            final, result = compose(path, headline, attrib, int(video_id) * 7 + attempt)
            entry = {"attempt": attempt, "source": Path(path).name,
                     "provider": provider[path], "vision_ok": acceptable,
                     "vision": notes, "judged_by": judged_by, **result}
            tries.append(entry)
            quality = (result["ok"] and acceptable, acceptable, result["ok"], score)
            if best is None or quality > best[0]:
                best = (quality, final, entry)
            if result["ok"] and acceptable:
                break
        if best and best[0][0]:
            break
        log.warning("attempt %d of %d did not pass: %s", attempt, s["attempts"],
                    "; ".join(tries[-1]["problems"]) or tries[-1]["vision"])

    if best is None:
        raise RuntimeError("no bust could be drawn: no Neurons left today, or Cloudflare failed")

    _, final, chosen = best
    path = out_dir / "thumbnail.jpg"
    final.save(path, "JPEG", quality=JPEG_QUALITY)
    small = out_dir / "thumb_small.jpg"
    final.resize(CHECK_SIZE, Image.LANCZOS).save(small, "JPEG", quality=JPEG_QUALITY)
    details = {"style": "bust", "headline": headline, "headline_source": headline_source,
               "attribution": attrib, "philosopher": key, "mentions": counts,
               "passed": bool(best[0][0]), "chosen": chosen, "tries": tries,
               "neurons": round(drawer.spent), "cloudflare_calls": drawer.cf_calls,
               "neurons_per_image": drawer.per_image, "steps": drawer.steps,
               "budget": round(budget), "path": str(path), "small": str(small)}
    (out_dir / "thumbnail.json").write_text(json.dumps(details, indent=2, default=str),
                                            encoding="utf-8")
    return details


def summary(details):
    """One line for the events table. Carries "<n> neurons" so core.neurons counts it."""
    c = details["chosen"]
    return (f"bust: {details['headline']} / {details['attribution']} "
            f"({details['philosopher'] or 'no philosopher dominates'}), "
            f"{c['source']} from {c['provider']}, attempt {c['attempt']}, "
            f"{'all checks passed' if details['passed'] else 'BEST AVAILABLE: ' + '; '.join(c['problems'] or [c['vision']])}, "
            f"{details['neurons']} neurons over {details['cloudflare_calls']} Cloudflare calls "
            f"at {details['steps']} steps")
