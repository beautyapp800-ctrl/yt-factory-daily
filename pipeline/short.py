"""Stage 9: one Short a day, cut from today's video and pointing back to it.

Vertical 1080x1920, 40 to 55 seconds, one lesson of the full video:

  the lesson     chosen by the text model from the video's own chapters - the one a stranger
                 understands with no context - and its scenes are what the Short is written
                 from, so it says nothing the full video does not;
  the words      a strong first sentence of at most twelve words, the lesson's one idea with
                 one concrete example from it, and a last sentence that sends the viewer to
                 the full video. Checked in code: length, the first sentence, the last one,
                 the words this channel never promises;
  the voice      en-US-AndrewMultilingualNeural through edge-tts, with word timings, so the
                 captions are the words actually being said, as they are said;
  the picture    the video's own bust for the opening, then the pictures of that lesson's
                 scenes, each with a slow eased push-in drawn by one affine resample a frame
                 (exact, nothing rounded - the zoompan judder measured on yt-kids does not
                 happen here) and a crossfade between them;
  the captions   large, centred, two or three words at a time, the spoken word in yellow.

It costs no Cloudflare neurons: the pictures already exist and the text model is Gemini or
Groq. It costs one videos.insert from the uploads bucket. It is published at the same moment
as the full video, so the link it ends on is never to a private video.
"""
import asyncio
import json
import math
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from core import db
from core.config import ASSETS_DIR, output_dir
from core.llm import complete_json
from core.logger import get_logger

log = get_logger("short")

W, H, FPS = 1080, 1920, 30
VOICE = "en-US-AndrewMultilingualNeural"
RATE = "-5%"
MIN_WORDS, MAX_WORDS = 100, 140
MIN_SECONDS, MAX_SECONDS = 40.0, 55.0
HOOK_MAX_WORDS = 12
LEAD, TAIL = 0.3, 0.9                       # silence before the first word and after the last
FONT = ASSETS_DIR / "fonts" / "Anton-Regular.ttf"
CAPTION_SIZE = 128
CAPTION_Y = 0.60                            # the captions' centre, as a share of the height
SPOKEN = (255, 214, 0)
ATTEMPTS = 3
CHAPTER = re.compile(r"^(\d+):(\d\d)\s+(.+)$")


# --- which lesson -------------------------------------------------------------------------

def lessons_of(video_id):
    """[(title, [scene rows])] for the video's lessons, from its chapters and its scenes.

    The chapters in the description say where each lesson starts; the scenes say how long
    each scene ran. A scene belongs to the lesson whose chapter has started by the time the
    scene does - with a little slack, because the chapter times include the pauses between
    scenes and the scene lengths do not.
    """
    video = db.get_video(video_id) or {}
    chapters = []
    for line in (video.get("description") or "").splitlines():
        m = CHAPTER.match(line.strip())
        if m:
            chapters.append((int(m.group(1)) * 60 + int(m.group(2)), m.group(3).strip()))
    with db._connect() as conn:
        scenes = [dict(r) for r in conn.execute(
            "SELECT idx, text, image_path, duration_s FROM scenes WHERE video_id = ? "
            "ORDER BY idx", (video_id,))]
    starts, t = [], 0.0
    for s in scenes:
        starts.append(t)
        t += float(s.get("duration_s") or 0)
    scale = (chapters[-1][0] / starts[-1]) if chapters and starts and starts[-1] else 1.0
    out = []
    for n, (start, title) in enumerate(chapters):
        if title.lower() in ("introduction", "closing thoughts"):
            continue
        end = chapters[n + 1][0] if n + 1 < len(chapters) else float("inf")
        rows = [s for s, at in zip(scenes, starts) if start - 3 <= at * scale < end - 3]
        if rows:
            out.append((title, rows))
    return out


def choose_lesson(lessons, video_title):
    """The index of the lesson that stands on its own best, asked of the text model."""
    listed = "\n".join(f"{i}. {t}" for i, (t, _r) in enumerate(lessons))
    data = complete_json(f"""A YouTube video called "{video_title}" has these lessons:

{listed}

One of them will be cut into a 45-second vertical Short for people who have never seen the
channel. Which lesson can a stranger understand and use with no context at all?

Return JSON: {{"index": <number>, "why": "<one sentence>"}}""",
                         system="You choose clips for short videos. JSON only.",
                         max_tokens=300, temperature=0.2)
    try:
        index = int(data.get("index"))
    except (TypeError, ValueError, AttributeError):
        index = 0
    return index if 0 <= index < len(lessons) else 0, (data or {}).get("why", "")


# --- what is said -------------------------------------------------------------------------

# Look-alike dashes and quotes to plain ones. The model wrote "five‑year" with a non-breaking
# hyphen (U+2011); Anton has no glyph for it and the caption showed a box.
_PLAIN = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-",
                        "—": " - ", "−": "-", "‘": "'", "’": "'",
                        "“": '"', "”": '"', " ": " "})


def plain(text):
    return re.sub(r"\s+", " ", str(text).translate(_PLAIN)).strip()


def _words(text):
    return re.findall(r"[A-Za-z0-9']+", text or "")


def script_problems(lines, video_title, medical_words=()):
    """Why a Short's script is unusable, as sentences the model can act on. Empty = fine."""
    problems = []
    if not lines:
        return ["there are no lines"]
    text = " ".join(lines)
    n = len(_words(text))
    if not MIN_WORDS <= n <= MAX_WORDS:
        problems.append(f"the script is {n} words; it must be {MIN_WORDS} to {MAX_WORDS}")
    first = lines[0]
    if len(_words(first)) > HOOK_MAX_WORDS:
        problems.append(f"the first sentence is {len(_words(first))} words; at most "
                        f"{HOOK_MAX_WORDS}")
    if re.match(r"\s*(in this video|today|hey|hi|welcome|did you know)\b", first, re.I):
        problems.append("the first sentence opens like an introduction; open on the idea")
    last = lines[-1].lower()
    if "full video" not in last or "channel" not in last:
        problems.append('the last sentence must name "the full video" and say it is on this '
                        'channel')
    if re.search(r"[#@]|[\U0001F300-\U0001FAFF]", text):
        problems.append("no hashtags, handles or emoji in the spoken words")
    low = text.lower()
    for word in medical_words:
        if re.search(rf"\b{re.escape(word)}\b", low):
            problems.append(f"'{word}': this channel makes no medical promises")
            break
    return problems


def write_script(lesson_title, scenes, video_title, complaints=(), seconds=None):
    source = "\n\n".join(s["text"] for s in scenes)
    fix = ""
    if complaints:
        fix = ("\n\nYour previous answer was rejected - fix every one of these:\n"
               + "\n".join(f"  - {c}" for c in complaints))
    if seconds:
        fix += (f"\n\nSpoken, the last version ran {seconds:.0f} seconds; it must run "
                f"{MIN_SECONDS:.0f} to {MAX_SECONDS:.0f}.")
    # A fixed shape, filled in by the model and assembled in code. Asked for free-form lines,
    # the model stuffed details from three different scenes of the lesson into forty seconds -
    # a steering wheel, an uncle, foam squares - which a stranger with no context cannot follow.
    return complete_json(f"""Write the voice-over of a vertical YouTube Short, cut from a longer
video called "{video_title}". It covers one lesson of that video, "{lesson_title}", and must
not say anything the lesson below does not.

The lesson, as the full video tells it:
{source}

Fill in these parts, spoken in this order, {MIN_WORDS + 5} to {MAX_WORDS - 10} words in all:
"hook": one sentence of at most {HOOK_MAX_WORDS} words that stops the scroll - a sharp claim or
  a question about the viewer's own life. Never "In this video", "Today" or a greeting.
"idea": one or two plain sentences saying the lesson's single idea, so a stranger gets it.
"moment": two to four sentences telling ONE moment from the lesson above, said to "you" - one
  place, one situation. Do not mix details from different scenes.
"takeaway": one sentence: what the viewer does differently from today.
"closing": one sentence sending the viewer to the full video: what it is about in a few plain
  words, that it is on this channel, using the words "the full video" - for example "The full
  video on letting go of rigid plans is on this channel."

Short sentences, plain spoken English. No hashtags, no emoji, no lesson numbers.

Return JSON: {{"hook": "...", "idea": "...", "moment": "...", "takeaway": "...",
"closing": "..."}}{fix}""",
                         system="You write voice-overs for short vertical videos. JSON only.",
                         max_tokens=900, temperature=0.7)


PARTS = ("hook", "idea", "moment", "takeaway", "closing")


def assemble(data):
    """The parts in order, as sentences; [] when a part is missing."""
    if not isinstance(data, dict) or any(not plain(data.get(k, "")) for k in PARTS):
        return []
    out = []
    for k in PARTS:
        out += [x for x in re.split(r"(?<=[.!?])\s+", plain(data[k])) if x]
    return out


# --- the voice ----------------------------------------------------------------------------

def speak(text, mp3_path, voice=VOICE, rate=RATE, attempts=5):
    """[(start_s, end_s, word)] and an mp3, from edge-tts with word boundaries on.

    Retried with a growing wait: Microsoft's service answers "No audio was received" when it
    has had too many requests lately, whatever the text - measured while building the first
    Short, three sentences in a row failed and the next five, no different, went through.
    A daily run makes one or two requests; the waits are for the day it is unlucky.
    """
    import time
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return _speak_once(text, mp3_path, voice, rate)
        except Exception as e:                                   # noqa: BLE001
            last = e
            log.warning("edge-tts attempt %d/%d failed: %s", attempt, attempts, str(e)[:100])
            time.sleep((5, 15, 30, 60, 0)[attempt - 1])
    raise RuntimeError(f"edge-tts gave no audio after {attempts} attempts: {last}")


def _speak_once(text, mp3_path, voice, rate):
    import edge_tts

    async def run():
        words = []
        comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
        with open(mp3_path, "wb") as f:
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start = chunk["offset"] / 1e7
                    words.append((start, start + chunk["duration"] / 1e7, chunk["text"]))
        return words
    return asyncio.run(run())


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True, timeout=60)
    return float(out.stdout.strip())


def caption_chunks(words, size=3, script=""):
    """[(start, end, [words])] - up to three words at a time, never across a sentence.

    edge-tts reports each word without its punctuation, so where a sentence ends is taken
    from the script: when edge's words and the script's line up one to one (120 and 120 on
    the first Short) the script's own commas and full stops say where to break. Otherwise a
    pause in the voice stands in - measured, the pauses alone overlap (0.20 s after a full
    stop at the shortest, 0.34 s inside a phrase at the longest), so it is the fallback only.
    """
    tokens = script.split()
    if len(tokens) == len(words):
        ends = [bool(re.search(r"[.,!?;:]$", t)) for t in tokens]
    else:
        ends = [(n + 1 < len(words) and words[n + 1][0] - w[1] > 0.35)
                for n, w in enumerate(words)]
    # Phrases first, then each phrase cut into even pieces: ten words go 3-3-2-2, not
    # 3-3-3-1, which left a lone "crisis" on screen at the end of a sentence.
    phrases, cur = [], []
    for w, end in zip(words, ends):
        cur.append(w)
        if end:
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)
    chunks = []
    for phrase in phrases:
        pieces = math.ceil(len(phrase) / size)
        base, extra = divmod(len(phrase), pieces)
        at = 0
        for k in range(pieces):
            n = base + (1 if k < extra else 0)
            chunks.append(phrase[at:at + n])
            at += n
    out = []
    for n, c in enumerate(chunks):
        end = chunks[n + 1][0][0] if n + 1 < len(chunks) else c[-1][1] + 0.4
        out.append((c[0][0], end, c))
    return out


# --- the picture --------------------------------------------------------------------------

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 4 * t ** 3 if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


def vertical(img):
    """A picture made 1080x1920. A wide one is cut to its middle; a square one (the bust) sits
    in the middle on a blurred, darkened copy of itself, because cutting a 720 px square to
    9:16 would enlarge it to mush."""
    img = img.convert("RGB")
    if img.width >= img.height * 1.2:
        crop_w = img.height * W / H
        left = (img.width - crop_w) / 2
        return img.transform((W, H), Image.EXTENT, (left, 0, left + crop_w, img.height),
                             Image.BICUBIC)
    back = img.resize((H, H), Image.BICUBIC).crop(((H - W) // 2, 0, (H - W) // 2 + W, H))
    back = Image.blend(back.filter(ImageFilter.GaussianBlur(40)), Image.new("RGB", (W, H)), 0.45)
    side = W
    fg = img.resize((side, round(img.height * side / img.width)), Image.LANCZOS)
    back.paste(fg, (0, (H - fg.height) // 2 - 160))
    return back


def _pushed(img, scale):
    s = 1.0 / scale
    cx, cy = W / 2, H / 2
    return img.transform((W, H), Image.AFFINE, (s, 0, cx - s * cx, 0, s, cy - s * cy),
                         Image.BICUBIC)


def _font(size):
    return ImageFont.truetype(str(FONT), size)


def draw_caption(frame, chunk_words, now):
    """The chunk, centred, the word being said in yellow."""
    text = [w[2].upper() for w in chunk_words]
    size = CAPTION_SIZE
    d = ImageDraw.Draw(frame)
    while size > 60:
        font = _font(size)
        widths = [d.textlength(t, font=font) for t in text]
        space = d.textlength(" ", font=font)
        if sum(widths) + space * (len(text) - 1) <= W - 120:
            break
        size -= 6
    font = _font(size)
    widths = [d.textlength(t, font=font) for t in text]
    space = d.textlength(" ", font=font)
    x = (W - (sum(widths) + space * (len(text) - 1))) / 2
    y = H * CAPTION_Y - size / 2
    # One word lit at a time: the latest to have started. Edge's word spans overlap, and
    # lighting every word whose span covered the moment lit two at once.
    started = [n for n, (start, _e, _w) in enumerate(chunk_words) if start <= now]
    lit = started[-1] if started else -1
    for n, ((start, end, _w), t, wd) in enumerate(zip(chunk_words, text, widths)):
        colour = SPOKEN if n == lit else (255, 255, 255)
        d.text((x, y), t, font=font, fill=colour, stroke_width=11, stroke_fill=(0, 0, 0))
        x += wd + space


def frames(pictures, chunks, seconds, workers=6):
    """Raw RGB frames: pictures with eased push-ins and crossfades, captions on top."""
    n_frames = int(round(seconds * FPS))
    base = [vertical(p) for p in pictures]
    shade = Image.new("RGB", (W, H), (0, 0, 0))
    span = seconds / len(base)
    fade = 0.45

    def picture_at(k, now):
        local = (now - k * span) / (span + fade)
        img = _pushed(base[k], 1.0 + 0.10 * ease(local))
        return Image.blend(img, shade, 0.22)

    def frame_at(i):
        now = i / FPS
        k = min(int(now / span), len(base) - 1)
        img = picture_at(k, now)
        into = now - (k + 1) * span + fade
        if k + 1 < len(base) and into > 0:
            img = Image.blend(img, picture_at(k + 1, now), ease(into / fade))
        for start, end, words in chunks:
            if start <= now < end:
                draw_caption(img, words, now)
                break
        return img.tobytes()

    with ThreadPoolExecutor(workers) as pool:
        for begin in range(0, n_frames, workers * 2):
            for data in pool.map(frame_at, range(begin, min(n_frames, begin + workers * 2))):
                yield data


def encode(frame_iter, audio, out_path, timeout=1800):
    args = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(audio),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            str(out_path)]
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=errors)
        try:
            for f in frame_iter:
                proc.stdin.write(f)
        finally:
            proc.stdin.close()
        if proc.wait(timeout=timeout) != 0:
            errors.seek(0)
            raise RuntimeError("ffmpeg could not encode the Short: "
                               + errors.read()[-400:].decode("utf-8", "replace"))
    return out_path


# --- sound --------------------------------------------------------------------------------

def soundtrack(voice_mp3, seconds, work, cfg, video_id):
    """Voice at -14 LUFS from LEAD on, with the channel's own music under it, `seconds` long."""
    from core import music
    work = Path(work)
    voice = work / "short_voice.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(voice_mp3), "-af",
                    f"adelay={int(LEAD * 1000)}:all=1,loudnorm=I=-14:TP=-1.5:LRA=11,"
                    f"apad=whole_dur={seconds:.3f}", "-t", f"{seconds:.3f}",
                    "-ar", "48000", "-ac", "2", str(voice)], check=True, capture_output=True)
    out = work / "short_audio.m4a"
    tracks = music.available_tracks()
    if not tracks:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(voice), "-c:a", "aac",
                        str(out)], check=True, capture_output=True)
        return out
    bed, _details = music.build_bed(music.pick_track(video_id, tracks), seconds,
                                    work / "short_bed.wav", cfg, music.loudness(voice))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(voice), "-i", str(bed),
                    "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0:duration=first[a]",
                    "-map", "[a]", "-c:a", "aac", "-b:a", "192k", str(out)],
                   check=True, capture_output=True)
    return out


# --- the whole Short ----------------------------------------------------------------------

def build(video_id, cfg, pictures_for=None, work=None):
    """Make short.mp4 for this video. Returns a report dict."""
    from pipeline.seo import MEDICAL_WORDS
    video = db.get_video(video_id) or {}
    title = video.get("seo_title") or video.get("title") or ""
    work = Path(work or output_dir(video_id))
    work.mkdir(parents=True, exist_ok=True)

    lessons = lessons_of(video_id)
    if not lessons:
        raise RuntimeError("the video has no chapters to cut a Short from")
    index, why = choose_lesson(lessons, title)
    lesson_title, scenes = lessons[index]
    log.info("Short from lesson %d, '%s': %s", index + 1, lesson_title, why)

    lines, complaints, seconds, words = [], [], None, []
    mp3 = work / "short_voice.mp3"
    for attempt in range(1, ATTEMPTS + 1):
        data = write_script(lesson_title, scenes, title, complaints, seconds)
        lines = assemble(data)
        complaints = (script_problems(lines, title, MEDICAL_WORDS) if lines else
                      [f"every one of {', '.join(PARTS)} must be filled in"])
        if complaints:
            log.warning("Short script attempt %d rejected: %s", attempt, "; ".join(complaints))
            seconds = None
            continue
        words = speak(" ".join(lines), mp3)
        seconds = LEAD + duration(mp3) + TAIL
        if MIN_SECONDS <= seconds <= MAX_SECONDS:
            break
        complaints = [f"spoken it runs {seconds:.0f} s"]
        log.warning("Short attempt %d runs %.1f s", attempt, seconds)
    if complaints:
        raise RuntimeError(f"no usable Short script after {ATTEMPTS} attempts: "
                           f"{'; '.join(complaints)}")

    seconds = round(seconds * FPS) / FPS
    chunks = [(s + LEAD, e + LEAD, [(a + LEAD, b + LEAD, t) for a, b, t in ws])
              for s, e, ws in caption_chunks(words, script=" ".join(lines))]
    pictures = (pictures_for or scene_pictures)(video_id, scenes)
    audio = soundtrack(mp3, seconds, work, cfg, video_id)
    out = encode(frames(pictures, chunks, seconds), audio, work / "short.mp4")
    report = {"lesson": lesson_title, "why": why, "seconds": seconds,
              "words": len(_words(" ".join(lines))), "lines": lines,
              "pictures": len(pictures), "path": str(out)}
    (work / "short.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Short: %.1f s, %d words, %d pictures, lesson '%s'", seconds, report["words"],
             len(pictures), lesson_title)
    return report


def scene_pictures(video_id, scenes, limit=4):
    """The bust first, for the opening line, then the lesson's own scene pictures."""
    pictures = []
    busts = sorted((ASSETS_DIR / "busts").glob(f"v{int(video_id):04d}_*.jpg"))
    if busts:
        pictures.append(Image.open(busts[-1]))
    for s in scenes:
        path = s.get("image_path")
        if path and Path(path).exists():
            pictures.append(Image.open(path))
        if len(pictures) >= limit + 1:
            break
    if not pictures:
        raise RuntimeError("no picture for the Short")
    return pictures


def frames_from_video(video_id, scenes, limit=4):
    """The lesson's pictures taken from the finished video itself, at the middle of each
    scene - for a run that resumes on a runner where the scene files are gone."""
    video = db.get_video(video_id) or {}
    path = video.get("video_path")
    if not path or not Path(path).exists():
        return []
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT idx, duration_s FROM scenes WHERE video_id = ? ORDER BY idx", (video_id,))]
    start, t = {}, 0.0
    for r in rows:
        start[r["idx"]] = t
        t += float(r["duration_s"] or 0)
    scale = duration(path) / t if t else 1.0
    out = []
    work = output_dir(video_id) / "short_frames"
    work.mkdir(parents=True, exist_ok=True)
    for s in scenes[:limit]:
        at = (start.get(s["idx"], 0.0) + float(s.get("duration_s") or 0) / 2) * scale
        dst = work / f"scene_{s['idx']}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}", "-i", str(path),
                        "-frames:v", "1", str(dst)], check=True, capture_output=True,
                       timeout=120)
        out.append(Image.open(dst))
    return out


def pictures(video_id, scenes, limit=4):
    """The bust, then the lesson's scene pictures - from their files, or else from the video."""
    pics = scene_pictures(video_id, scenes, limit)
    if len(pics) < 2:
        pics = pics[:1] + frames_from_video(video_id, scenes, limit)
    return pics


# --- upload -------------------------------------------------------------------------------

def short_body(video, report, cfg):
    """The videos.insert body: the Short goes public at the same moment as its full video."""
    from pipeline import upload
    from pipeline.search_topics import display
    full_url = f"https://youtu.be/{video['youtube_id']}"
    hook = report["lines"][0].rstrip(".")
    title = hook if len(hook) <= 90 else hook[:87].rsplit(" ", 1)[0] + "..."
    title = f"{title} #shorts"
    phrase = video.get("search_phrase") or ""
    description = "\n\n".join(x for x in (
        f"The full video: {video.get('seo_title') or ''}\n{full_url}",
        " ".join(report["lines"][1:3]),
        "#stoicism #stoic #shorts") if x)
    tags = ["stoicism", "stoic", "shorts", "stoic philosophy"] + ([phrase] if phrase else [])
    status = {"privacyStatus": cfg.get("privacy_status", "private"),
              "selfDeclaredMadeForKids": upload.MADE_FOR_KIDS,
              "containsSyntheticMedia": upload.CONTAINS_SYNTHETIC_MEDIA}
    if video.get("published_at"):
        status["privacyStatus"] = "private"
        status["publishAt"] = video["published_at"]
    yt = cfg.get("youtube") or {}
    return {"snippet": {"title": title, "description": description, "tags": tags,
                        "categoryId": str(yt.get("category_id", upload.CATEGORY_ID)),
                        "defaultLanguage": yt.get("default_language", upload.DEFAULT_LANGUAGE)},
            "status": status}


def run(video_id, cfg, service=None):
    """Build the Short and upload it. Optional: a day without a Short still has its video."""
    if not cfg.get("shorts", True):
        return True
    video = db.get_video(video_id) or {}
    if video.get("short_youtube_id"):
        log.info("video %s already has its Short, %s", video_id, video["short_youtube_id"])
        return True
    if not video.get("youtube_id"):
        raise RuntimeError("the full video is not on YouTube, so a Short would point nowhere")
    out = output_dir(video_id)
    marker = out / "short_uploaded.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        db.update_video(video_id, short_youtube_id=saved["youtube_id"])
        return True

    report_path = out / "short.json"
    if (out / "short.mp4").exists() and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = build(video_id, cfg, pictures_for=pictures, work=out)

    from pipeline import upload
    body = short_body(video, report, cfg)
    upload.assert_mandatory_flags(body)
    if service is None:
        from core import youtube
        service = youtube.build_service(youtube.load_credentials())
    chunk_mb = int((cfg.get("youtube") or {}).get("chunk_mb", upload.DEFAULT_CHUNK_MB))
    response, _stats = upload.upload_video(service, body, out / "short.mp4",
                                           out / "short_session.json", chunk_mb)
    youtube_id = response.get("id")
    if not youtube_id:
        raise RuntimeError(f"YouTube accepted the Short but returned no id: {response}")
    marker.write_text(json.dumps({"youtube_id": youtube_id}), encoding="utf-8")
    db.update_video(video_id, short_youtube_id=youtube_id)
    log.info("Short uploaded: https://youtube.com/shorts/%s, public at %s, quota: one "
             "videos.insert from the uploads bucket", youtube_id,
             body["status"].get("publishAt") or "once processed")
    db.log_event(video_id, "short", "info",
                 f"https://youtube.com/shorts/{youtube_id} ({report['seconds']:.0f} s, "
                 f"lesson '{report['lesson']}')")
    return True
