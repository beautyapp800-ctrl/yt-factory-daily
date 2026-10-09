"""The picture behind the thumbnail, drawn for the thumbnail.

Every frame in the video is composed for a viewer who is already watching: wide, quiet, evenly
lit, the subject in the middle band. That is right for 1280x720 held for twenty seconds, and
wrong for 210x118 passed in half a second. Shrunk to a feed, those frames are a grey smear
with nothing to stop the eye.

A thumbnail needs the opposite picture: one source of light in a dark frame, and everything
else given up to get it. So the thumbnail no longer borrows the best of seventy narration
frames - it asks for its own, with a prompt written for a single bright thing against dark,
and falls back on the narration frames only if that cannot be had.

The cost is three images, 288 Neurons of a daily 10,000 - under four per cent of a day to fix
the one image that decides whether anybody clicks.

The picture rules of the project still apply in full and are checked the same way: no writing
and nothing that carries writing, no hands, no faces, no portrait closer than the waist. A
figure is allowed here only as the thing the light is behind - a silhouette, a back, a shape
in a doorway - which is exactly what this composition wants anyway.
"""
from core import images as image_api
from core.config import output_dir
from core.logger import get_logger
from core.prompts import find_banned_anatomy_words, find_banned_image_words

log = get_logger("cover")

CANDIDATES = 3

# The composition, stated as the generator understands it. Written once, here, rather than
# asked of the language model: the shape of a thumbnail is a fixed requirement, not something
# that should vary with the topic, and a model asked for it every day will drift off it.
COMPOSITION = ("one very large bright light source filling the right side of the frame, "
               "strong chiaroscuro, deep black shadows on the left, extreme contrast between "
               "the bright light and the dark room, subject rim-lit from behind in silhouette, "
               "the left third of the frame almost black and empty, dramatic low-key lighting, "
               "the light source is big and blown out white")

# Six ways of putting one light in a dark frame. The video's number picks one, so neighbouring
# videos on the channel do not repeat a composition any more than they repeat a colour.
SUBJECTS = [
    "a lone figure in silhouette standing in front of a huge bright window that fills the "
    "right half of the frame",
    "a wide doorway on the right thrown open onto blinding daylight, the room around it black",
    "a single large lamp glowing on the right of a black room, its light pooling wide across "
    "the floor",
    "a distant figure in silhouette at the end of a dark corridor, a tall bright opening "
    "behind them on the right",
    "a broad shaft of hard daylight falling from the right across a dark stone floor",
    "a tall arched window on the right burning with light, the stone hall around it in deep "
    "shadow",
]


# Fragments of the configured style that say the opposite of what a thumbnail needs. The
# style is written for frames a viewer is already watching - wide, soft, quiet, subject in the
# middle band - and asking for that and for a single hard light in the same breath gets the
# average of the two, which is the flat frame this exists to stop. The rest of the style is
# kept, bans included, so the cover still looks like the channel.
STYLE_CONTRADICTIONS = ("wide horizontal composition", "key subject within the central "
                        "horizontal band", "soft volumetric light", "quiet atmosphere")


def cover_style(style):
    """The configured style with the parts that fight a thumbnail composition taken out."""
    parts = [p.strip() for p in (style or "").split(",")]
    kept = [p for p in parts if p and p.lower() not in
            {c.lower() for c in STYLE_CONTRADICTIONS}]
    return ", ".join(kept)


def prompt_for(video_id, topic, style):
    """The image prompt for this video's cover. Style first, as the generator weights it."""
    subject = SUBJECTS[int(video_id) % len(SUBJECTS)]
    return f"{cover_style(style)}, {subject}, {COMPOSITION}"


def banned_words_in(subject_and_composition):
    """The project's own picture bans, applied to a prompt this module wrote itself.

    The rules exist because of what the generator does with certain objects, not because of
    who wrote the prompt. A prompt built in code gets checked exactly like one the model
    wrote - and this one has been through that check, so a hit here means somebody edited
    SUBJECTS or COMPOSITION without reading them.

    The configured style is not scanned, for the same reason pipeline/images.py does not scan
    it: the style ends in "no text, no lettering, no watermark, no modern branding", and a
    checker that reads those as the words they ban would flag every prompt ever written.
    """
    return (find_banned_anatomy_words(subject_and_composition)
            + find_banned_image_words(subject_and_composition))


def draw(video_id, topic, cfg, count=CANDIDATES):
    """[(path, provider)] of freshly drawn covers. Never raises: an empty list means the
    caller should fall back to the narration frames, which is a worse thumbnail but still
    a thumbnail."""
    style = (cfg.get("image_style") or "").strip()
    subject = SUBJECTS[int(video_id) % len(SUBJECTS)]
    prompt = prompt_for(video_id, topic, style)
    banned = banned_words_in(f"{subject}, {COMPOSITION}")
    if banned:
        log.error("the cover prompt breaks the project's own picture rules (%s); "
                  "falling back to the narration frames", ", ".join(banned))
        return []

    out = output_dir(video_id) / "cover"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for i in range(count):
        path = out / f"cover_{i}.jpg"
        if path.exists() and path.stat().st_size > 1000:
            # Resumability, the same as every other paid-for thing: a rerun of this video
            # does not buy its covers twice.
            made.append((str(path), "cloudflare"))
            continue
        try:
            provider, neurons = image_api.synthesize(prompt, path, 9000 + int(video_id) * 10 + i,
                                                     cfg)
        except image_api.ImageError as e:
            log.warning("cover %d of %d failed: %s", i + 1, count, e)
            continue
        made.append((str(path), provider))
        log.info("cover %d of %d drawn by %s%s", i + 1, count, provider,
                 f" ({neurons:.0f} neurons)" if neurons else "")
    if made:
        log.info("%d cover candidate(s) drawn for the thumbnail: %s", len(made), prompt[:120])
    return made
