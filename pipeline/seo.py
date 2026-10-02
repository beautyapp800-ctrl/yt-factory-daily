"""Stage 7: the YouTube title, description and tags.

One LLM request, built from the video's own title, promise and lesson plan, returns the
title, the opening hook of the description, a short summary and the tags. Everything
that can be computed is computed instead of asked for: the chapter timestamps come from
timings.json, the call to subscribe is fixed text, and every constraint on the model's
answer is checked in code - a title that is too long, carries a colon or an emoji, or
promises a different number of lessons than the video has goes back for a rewrite.

The description is laid out as

    <hook: a complete thought in the first 150 characters, which is all search shows>

    <3-4 sentence summary>

    Lessons in this video:
    0:00 Introduction
    1:05 <lesson 1 title>
    ...
    m:ss Closing thoughts

    <call to subscribe>

Chapter lines are what YouTube turns into chapters: they have to start at 0:00, there has
to be at least three, and each must be at least ten seconds long, which a lesson always is.

Results are written to videos.seo_title, videos.description and videos.tags (a JSON list).
videos.title keeps the script's working title.
"""
import json
import re

from core import db, text as txt
from core.config import output_dir
from core.llm import complete_json
from core.logger import get_logger

log = get_logger("seo")

MAX_TITLE = 70
HOOK_MAX = 150                  # what search results show of a description
MIN_TAGS, MAX_TAGS = 12, 15
MAX_TAG_LEN = 30
MAX_TAGS_CHARS = 450            # YouTube allows 500 in total; commas and quotes count
MAX_DESCRIPTION = 5000
ATTEMPTS = 3
# The channel makes no medical claims (core/prompts.py STYLE_RULES). A title, hook or tag is
# a promise to the viewer, so these may not appear in them; the summary only describes.
MEDICAL_WORDS = ("anxiety", "anxious", "depression", "depressed", "cure", "heal", "healing",
                 "therapy", "disorder", "ptsd", "trauma", "mental health", "burnout")
BROAD_TAGS = ["stoicism", "stoic philosophy"]
CALL_TO_ACTION = ("If this helped, subscribe, and the next video on practical Stoicism "
                  "will find you.")

SYSTEM = """You write YouTube metadata for a channel about Stoicism and practical philosophy.
The audience is adults looking for something solid to stand on, not hustle content.
Be concrete and honest: never promise a cure for anxiety or any medical outcome, never claim
more than the video delivers. Reply with valid JSON only."""

# Characters models like to emit that look like a hyphen or are invisible: a non-breaking
# hyphen (U+2011) renders as a box in some places, and a soft hyphen or zero-width space is
# invisible junk in a title.
_HYPHENS = dict.fromkeys(map(ord, "\u2010\u2011\u2012"), "-")
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u2060\ufeff"), None)
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")
_NUMBER = re.compile(r"\b(\d+)\b")


def format_timestamp(seconds):
    """YouTube's chapter format: m:ss, or h:mm:ss from an hour up."""
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chapter_lines(video_id, lesson_titles):
    """['0:00 Introduction', '1:05 <lesson>', ..., 'm:ss Closing thoughts'], timed from
    timings.json, or [] if the lesson boundaries cannot be reconstructed.

    Times are the narration's own, i.e. measured from the start of voice.mp3. If an intro
    is ever put in front of the narration, every timestamp here has to shift by its length.
    """
    from pipeline.tts import _lesson_boundary_indices

    out = output_dir(video_id)
    timings_path = out / "timings.json"
    if not timings_path.exists():
        log.warning("no timings.json; the description will have no chapter list")
        return []
    timings = json.loads(timings_path.read_text(encoding="utf-8"))
    scenes = db.get_scenes(video_id)
    total = sum(len(txt.split_sentences(s["text"])) or 1 for s in scenes)
    starts = sorted(_lesson_boundary_indices(video_id, total))
    if len(starts) < len(lesson_titles) or total != len(timings):
        log.warning("cannot place the %d lessons in %d timed sentences (%d boundaries "
                    "found); the description will have no chapter list",
                    len(lesson_titles), len(timings), len(starts))
        return []

    lines = ["0:00 Introduction"]
    for title, start in zip(lesson_titles, starts):
        lines.append(f"{format_timestamp(timings[start]['start_s'])} {title}")
    if len(starts) > len(lesson_titles):
        lines.append(f"{format_timestamp(timings[starts[len(lesson_titles)]]['start_s'])} "
                     "Closing thoughts")
    return lines


# --- the request -------------------------------------------------------------

def _prompt(title, promise, lessons, complaints):
    listed = "\n".join(f"  {i}. {t}" for i, t in enumerate(lessons, 1))
    fix = ""
    if complaints:
        fix = ("\n\nYour previous answer was rejected for these reasons - fix every one:\n"
               + "\n".join(f"  - {c}" for c in complaints))
    return f"""A video is finished. Write its YouTube metadata.

Working title: {title}
What it promises the viewer: {promise}
It has exactly {len(lessons)} lessons:
{listed}

Return JSON with these keys:

"title": the YouTube title. At most {MAX_TITLE} characters. The format that works in this niche
  is a number plus a promise, in the manner of "10 Stoic Lessons That Will Make You Mentally
  Unbreakable" or "NEVER Explain Yourself Again". If you use a number it must be {len(lessons)},
  the real lesson count. No colon, no emoji, no hashtags. Specific to THIS video's subject, not
  a title that could sit on any Stoicism video.

"hook": the first lines of the description. At most {HOOK_MAX} characters in total, and it
  must stand alone: search results show only this much, so it must be a complete thought that
  makes someone want to watch, not the start of a longer sentence. No emoji.

"summary": 3 or 4 plain sentences saying what the video covers and who it is for. No emoji.

"tags": between {MIN_TAGS} and {MAX_TAGS} tags, each under {MAX_TAG_LEN} characters, lower case.
  A mix, and the mix matters: at least five BROAD tags that the whole niche is searched by
  ("stoicism", "stoic philosophy", "stoic lessons", "marcus aurelius", "stoic mindset" - pick
  those that fit), and at least five NARROW ones for this video's own subject, each a phrase
  a viewer would really type into search ("stoicism and money", "stop comparing yourself"),
  not a fragment lifted from a lesson title. No hashtags, no duplicates.{fix}"""


def _tidy(value):
    """Normalise one string from the model: look-alike hyphens to '-', invisible characters
    out, runs of whitespace folded (a description keeps its own newlines, so only blanks
    inside a line are folded)."""
    value = str(value or "").translate(_HYPHENS).translate(_INVISIBLE)
    return re.sub(r"[ \t]+", " ", value).strip()


def _problems(data, lesson_count):
    """What is wrong with the model's answer, as sentences it can act on. Empty = fine."""
    problems = []
    title = (data.get("title") or "").strip()
    if not title:
        problems.append("title is missing")
    else:
        if len(title) > MAX_TITLE:
            problems.append(f"title is {len(title)} characters, the limit is {MAX_TITLE}")
        if ":" in title:
            problems.append("title contains a colon")
        if _EMOJI.search(title):
            problems.append("title contains an emoji")
        if "#" in title:
            problems.append("title contains a hashtag")
        wrong = [n for n in _NUMBER.findall(title) if int(n) != lesson_count]
        if wrong:
            problems.append(f"title says {wrong[0]} but the video has {lesson_count} lessons")

    hook = (data.get("hook") or "").strip()
    if not hook:
        problems.append("hook is missing")
    else:
        if len(hook) > HOOK_MAX:
            problems.append(f"hook is {len(hook)} characters, the limit is {HOOK_MAX}")
        if hook[-1] not in ".!?":
            problems.append("hook does not end on a full stop, so it is not a complete thought")
        if _EMOJI.search(hook):
            problems.append("hook contains an emoji")

    summary = (data.get("summary") or "").strip()
    sentences = len(txt.split_sentences(summary))
    if not 3 <= sentences <= 4:
        problems.append(f"summary has {sentences} sentences, it must have 3 or 4")
    if _EMOJI.search(summary):
        problems.append("summary contains an emoji")

    tags = _clean_tags(data.get("tags"))
    if len(tags) < MIN_TAGS:
        problems.append(f"only {len(tags)} usable tags, need at least {MIN_TAGS}")

    promised = " ".join([title, hook] + tags).lower()
    for word in MEDICAL_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", promised):
            problems.append(f"'{word}' appears in the title, hook or tags: this channel makes "
                            "no medical or psychiatric promises, so leave it out")
            break
    return problems


def _clean_tags(raw):
    """Lower-cased, de-duplicated, hash-free tags no longer than MAX_TAG_LEN, in order."""
    seen, out = set(), []
    for tag in raw if isinstance(raw, list) else []:
        tag = re.sub(r"\s+", " ", str(tag).replace("#", "")).strip().lower()
        if tag and len(tag) <= MAX_TAG_LEN and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _final_tags(raw):
    """The broad tags guaranteed in, then the model's own, capped by count and by the
    total length YouTube accepts."""
    tags = _clean_tags(BROAD_TAGS + _clean_tags(raw))
    out, used = [], 0
    for tag in tags:
        cost = len(tag) + 1
        if len(out) >= MAX_TAGS or used + cost > MAX_TAGS_CHARS:
            break
        out.append(tag)
        used += cost
    return out


def build_description(hook, summary, chapters):
    parts = [hook.strip(), summary.strip()]
    if chapters:
        parts.append("Lessons in this video:\n" + "\n".join(chapters))
    parts.append(CALL_TO_ACTION)
    return "\n\n".join(parts)


def run(video_id, cfg):
    out = output_dir(video_id)
    outline_path = out / "outline.json"
    if not outline_path.exists():
        raise RuntimeError(f"{outline_path} is missing; did the script stage run?")
    outline = json.loads(outline_path.read_text(encoding="utf-8"))
    lessons = [l["title"] for l in outline["lessons"]]
    title = outline.get("title") or (db.get_video(video_id) or {}).get("title") or ""
    promise = outline.get("promise") or (outline.get("concept") or {}).get("promise") or ""

    data, complaints = {}, []
    for attempt in range(1, ATTEMPTS + 1):
        data = complete_json(_prompt(title, promise, lessons, complaints), system=SYSTEM,
                             max_tokens=1200, temperature=0.8)
        if not isinstance(data, dict):
            complaints = ["the answer was not a JSON object"]
            continue
        data = {**data, **{k: _tidy(data.get(k)) for k in ("title", "hook", "summary")},
                "tags": [_tidy(t) for t in data["tags"]] if isinstance(data.get("tags"), list) else []}
        complaints = _problems(data, len(lessons))
        if not complaints:
            break
        log.warning("seo attempt %d/%d rejected: %s", attempt, ATTEMPTS, "; ".join(complaints))
    if complaints:
        raise RuntimeError(f"seo metadata still invalid after {ATTEMPTS} attempts: "
                           f"{'; '.join(complaints)}")

    chapters = chapter_lines(video_id, lessons)
    description = build_description(data["hook"], data["summary"], chapters)
    if len(description) > MAX_DESCRIPTION:
        raise RuntimeError(f"description is {len(description)} characters, YouTube's limit "
                           f"is {MAX_DESCRIPTION}")
    tags = _final_tags(data["tags"])
    seo_title = data["title"].strip()
    if not _NUMBER.search(seo_title):
        log.warning("title has no number: %s", seo_title)

    db.update_video(video_id, seo_title=seo_title, description=description,
                    tags=json.dumps(tags, ensure_ascii=False))
    log.info("title (%d chars): %s", len(seo_title), seo_title)
    log.info("%d tags, %d chapter lines, description %d chars", len(tags), len(chapters),
             len(description))
    db.log_event(video_id, "seo", "info",
                 f"title {len(seo_title)} chars, {len(tags)} tags, {len(chapters)} chapters")
    return True
