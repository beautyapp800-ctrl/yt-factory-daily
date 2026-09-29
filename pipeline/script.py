"""Stage 2: full narration script, built in four stages instead of one request.

A: outline (title, promise, lesson titles)
B: each lesson as its own request
C: hook and outro
D: programmatic scene split, then image prompts in batches
"""
import json

from core import db, text as txt
from core.config import output_dir
from core.llm import complete, complete_json
from core.logger import get_logger
from core.prompts import IMAGE_SYSTEM, NARRATION_SYSTEM, OUTLINE_SYSTEM

log = get_logger("script")

HOOK_WORDS = 200
OUTRO_WORDS = 200
WORD_TOLERANCE = 0.15          # total script must land within +/-15% of budget
MIN_SCENES, MAX_SCENES = 40, 60
SCENE_TARGET, SCENE_MIN, SCENE_MAX = 92, 75, 110
IMAGE_BATCH = 10
MAX_FIX_ROUNDS = 3
LESSONS_PER_FIX_ROUND = 5


# --- stage A ---------------------------------------------------------------

def make_outline(concept, cfg, lesson_count):
    prompt = f"""Plan a {cfg['target_duration_min']} minute video about {concept['topic']}.

The promise to the viewer: {concept['promise']}
What keeps it fresh: {concept['different']}

Return JSON with exactly these keys:
"title": the video title, in the format this channel uses. Examples of the format:
  "10 Stoic Lessons That Will Make You Mentally Unbreakable"
  "10 Things You Should Quietly Remove From Your Life"
  Start with {lesson_count}. Under 70 characters. No colon, no subtitle.
"promise": one sentence, what the viewer walks away able to do.
"lessons": an array of exactly {lesson_count} objects, each with
   "title": 3 to 8 words naming the lesson, and
   "focus": one sentence on the specific situation this lesson covers.

The lessons must be concrete and clearly different from each other. No two may
cover the same move in different words. Order them so the video builds.

JSON only."""
    outline = complete_json(prompt, system=OUTLINE_SYSTEM, max_tokens=2000, temperature=0.85)
    lessons = outline.get("lessons") or []
    if len(lessons) != lesson_count:
        raise ValueError(f"outline has {len(lessons)} lessons, expected {lesson_count}")
    for i, lesson in enumerate(lessons, 1):
        if not str(lesson.get("title", "")).strip():
            raise ValueError(f"lesson {i} has no title")
        lesson["title"] = txt.clean(str(lesson["title"]))
        lesson["focus"] = txt.clean(str(lesson.get("focus", "")))
    outline["title"] = txt.clean(str(outline.get("title", ""))).strip(" .")
    if not outline["title"]:
        raise ValueError("outline has no title")
    return outline


# --- stage B ---------------------------------------------------------------

def write_lesson(index, outline, written, target_words, cfg):
    """One lesson. Gets the whole plan plus what is already written, to avoid repeats."""
    lesson = outline["lessons"][index]
    plan = "\n".join(f"{i}. {l['title']}: {l.get('focus', '')}"
                     for i, l in enumerate(outline["lessons"], 1))
    already = "\n\n".join(f"Lesson {i + 1} ({outline['lessons'][i]['title']}):\n{body}"
                          for i, body in sorted(written.items())) or "Nothing yet."

    prompt = f"""Video title: {outline['title']}
Promise: {outline['promise']}

The full plan of {len(outline['lessons'])} lessons:
{plan}

Already written, do not repeat these points, examples or phrasings:
{already}

Now write lesson {index + 1}, "{lesson['title']}". Focus: {lesson.get('focus', '')}

About {target_words} words. Build it in three movements, with no labels or breaks between them:
First, one concrete everyday situation the viewer recognises. A specific moment, a real
room, a real hour of the day. Not a category of problem.
Then the Stoic principle that cuts through it, in plain language.
Then one thing to do tomorrow morning, specific enough to actually do.

Do not name the lesson number. Do not write the lesson title as a heading.
Start straight into the situation. Narration only."""

    # x6, not x2: reasoning models spend part of the ceiling before the prose starts.
    raw = complete(prompt, system=NARRATION_SYSTEM,
                   max_tokens=int(target_words * 6), temperature=0.85)
    return txt.clean(raw)


# --- stage C ---------------------------------------------------------------

def write_hook(outline, cfg):
    prompt = f"""Video title: {outline['title']}
Promise: {outline['promise']}
The lessons ahead: {', '.join(l['title'] for l in outline['lessons'])}

Write the opening of this video, about {HOOK_WORDS} words.

The first two sentences have to hold someone who is one thumb-scroll from leaving.
Open on a specific moment they know, at a specific time of day. No greeting, no
welcome, no channel name, no "in this video". Do not list what is coming.
Name the tension, then promise that there is a way to stand inside it.
Narration only."""
    return txt.clean(complete(prompt, system=NARRATION_SYSTEM,
                              max_tokens=HOOK_WORDS * 6, temperature=0.9))


def write_outro(outline, cfg):
    titles = ", ".join(l["title"] for l in outline["lessons"])
    prompt = f"""Video title: {outline['title']}
The lessons just covered: {titles}

Write the ending of this video, about {OUTRO_WORDS} words.

Pull the lessons into one idea, without listing them again. Leave the viewer with
one thing to carry into tomorrow. Then a short, quiet invitation to subscribe if
the video was worth their time, no more than two sentences, no enthusiasm, no
asking for likes or comments or the bell.
Narration only."""
    return txt.clean(complete(prompt, system=NARRATION_SYSTEM,
                              max_tokens=OUTRO_WORDS * 6, temperature=0.85))


# --- stage D ---------------------------------------------------------------

def build_scenes(full_text):
    """Split into scenes locally, nudging the target until the count lands in range."""
    scenes = txt.split_scenes(full_text, SCENE_TARGET, SCENE_MIN, SCENE_MAX)
    if MIN_SCENES <= len(scenes) <= MAX_SCENES:
        return scenes

    total = txt.word_count(full_text)
    wanted = (MIN_SCENES + MAX_SCENES) // 2
    retarget = max(40, min(200, round(total / wanted)))
    log.warning("scene split gave %d scenes (want %d-%d), retrying with target %d words",
                len(scenes), MIN_SCENES, MAX_SCENES, retarget)
    scenes = txt.split_scenes(full_text, retarget,
                              max(30, int(retarget * 0.8)), int(retarget * 1.25))
    return scenes


def add_image_prompts(scenes, cfg, outline):
    """Ask for one image prompt per scene, in batches. Returns a list as long as scenes."""
    style = cfg["image_style"]
    prompts = [None] * len(scenes)

    for start in range(0, len(scenes), IMAGE_BATCH):
        batch = scenes[start:start + IMAGE_BATCH]
        for attempt in (1, 2):
            try:
                got = _ask_image_batch(batch, start, outline)
            except (ValueError, KeyError) as e:
                log.warning("image batch at scene %d failed (attempt %d): %s", start + 1, attempt, e)
                continue
            for offset in range(len(batch)):
                value = got.get(start + offset)
                if value:
                    prompts[start + offset] = f"{value.rstrip(' .,')}, {style}"
            if all(prompts[start:start + len(batch)]):
                break

    # Anything still missing gets a plain prompt built from the video's own subject,
    # so a scene is never left without one.
    for i, value in enumerate(prompts):
        if not value:
            log.warning("scene %d got no image prompt, using the fallback", i + 1)
            prompts[i] = (f"empty stone courtyard at dawn, a single worn wooden bench, "
                          f"long shadows across the flagstones, {style}")
    return prompts


def _ask_image_batch(batch, start, outline):
    listed = "\n\n".join(f"SCENE {start + i + 1}:\n{scene}" for i, scene in enumerate(batch))
    prompt = f"""These are consecutive scenes from the narration of a video titled
"{outline['title']}". Write one image prompt for each scene, illustrating what that
scene talks about.

{listed}

Return JSON: {{"prompts": [{{"scene": <number>, "prompt": "<15 to 30 words>"}}, ...]}}
One entry for every scene listed, using the same scene numbers. JSON only."""
    data = complete_json(prompt, system=IMAGE_SYSTEM, max_tokens=1800, temperature=0.8)
    entries = data.get("prompts") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("no prompts array in response")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            scene_no = int(entry.get("scene", entry.get("idx", 0)))
        except (TypeError, ValueError):
            continue
        value = txt.clean(str(entry.get("prompt", "")))
        if value and start < scene_no <= start + len(batch):
            result[scene_no - 1] = value
    if not result:
        raise ValueError("no usable entries in response")
    return result


# --- validation ------------------------------------------------------------

def check_scenes(scenes, prompts):
    """Problems that must be fixed before anything is written to the database."""
    problems = []
    if not MIN_SCENES <= len(scenes) <= MAX_SCENES:
        problems.append(f"scene count {len(scenes)} outside {MIN_SCENES}-{MAX_SCENES}")
    for i, scene in enumerate(scenes, 1):
        if not scene.strip():
            problems.append(f"scene {i} is empty")
        if txt.has_markdown(scene):
            problems.append(f"scene {i} contains markdown characters")
    for i, prompt in enumerate(prompts, 1):
        if not (prompt or "").strip():
            problems.append(f"scene {i} has no image prompt")
    return problems


def fix_word_count(parts, outline, budget, cfg):
    """Regenerate only the lessons that push the total out of range, not the whole script.

    Each round aims the lessons at a uniform target that would put the total on the
    budget itself, then rewrites the few lessons furthest from that target.
    """
    lesson_count = len(outline["lessons"])
    low, high = budget * (1 - WORD_TOLERANCE), budget * (1 + WORD_TOLERANCE)

    for round_no in range(1, MAX_FIX_ROUNDS + 1):
        total = sum(txt.word_count(p) for p in parts.values())
        if low <= total <= high:
            return True
        fixed = txt.word_count(parts[("hook", 0)]) + txt.word_count(parts[("outro", 0)])
        per_lesson = max(200, round((budget - fixed) / lesson_count))
        log.warning("round %d: %d words, need %d-%d, aiming lessons at %d words each",
                    round_no, total, round(low), round(high), per_lesson)

        worst = sorted(range(lesson_count),
                       key=lambda i: abs(txt.word_count(parts[("lesson", i)]) - per_lesson),
                       reverse=True)[:LESSONS_PER_FIX_ROUND]
        written = {j: parts[("lesson", j)] for j in range(lesson_count)}
        for index in worst:
            was = txt.word_count(parts[("lesson", index)])
            log.info("regenerating lesson %d: %d words -> ~%d", index + 1, was, per_lesson)
            context = {k: v for k, v in written.items() if k != index}
            parts[("lesson", index)] = write_lesson(index, outline, context, per_lesson, cfg)

    return low <= sum(txt.word_count(p) for p in parts.values()) <= high


# --- assembly --------------------------------------------------------------

def assemble(parts, lesson_count):
    """Stitch the parts into one narration in reading order."""
    ordered = [parts[("hook", 0)]]
    ordered += [parts[("lesson", i)] for i in range(lesson_count)]
    ordered.append(parts[("outro", 0)])
    return "\n\n".join(p.strip() for p in ordered if p.strip())


def run(video_id, cfg):
    concept_path = output_dir(video_id) / "concept.json"
    if concept_path.exists():
        concept = json.loads(concept_path.read_text(encoding="utf-8"))
    else:
        # Topic stage was skipped or its file is gone; fall back to what the DB knows.
        video = db.get_video(video_id) or {}
        topic = video.get("topic") or cfg["channel_topic"]
        concept = {"topic": topic, "promise": "", "different": ""}
        log.warning("no concept.json, working from the topic in the database: %s", topic)

    lesson_count = int(cfg.get("lessons_per_video", 10))
    budget = int(cfg["target_duration_min"]) * int(cfg["words_per_minute"])
    per_lesson = max(200, round((budget - HOOK_WORDS - OUTRO_WORDS) / lesson_count))
    log.info("word budget %d (~%d per lesson across %d lessons)", budget, per_lesson, lesson_count)

    log.info("stage A: outline")
    outline = make_outline(concept, cfg, lesson_count)
    log.info("title: %s", outline["title"])

    parts = {}
    log.info("stage B: %d lessons", lesson_count)
    for i in range(lesson_count):
        written = {j: parts[("lesson", j)] for j in range(i)}
        parts[("lesson", i)] = write_lesson(i, outline, written, per_lesson, cfg)
        log.info("lesson %d/%d: %s (%d words)", i + 1, lesson_count,
                 outline["lessons"][i]["title"], txt.word_count(parts[("lesson", i)]))

    log.info("stage C: hook and outro")
    parts[("hook", 0)] = write_hook(outline, cfg)
    parts[("outro", 0)] = write_outro(outline, cfg)
    log.info("hook %d words, outro %d words",
             txt.word_count(parts[("hook", 0)]), txt.word_count(parts[("outro", 0)]))

    if not fix_word_count(parts, outline, budget, cfg):
        total = sum(txt.word_count(p) for p in parts.values())
        raise RuntimeError(f"script is {total} words, outside 15% of the {budget} word budget")

    full_text = assemble(parts, lesson_count)
    total_words = txt.word_count(full_text)

    log.info("stage D: scenes and image prompts")
    scenes = build_scenes(full_text)
    log.info("%d scenes, %.0f words each on average", len(scenes), total_words / len(scenes))
    prompts = add_image_prompts(scenes, cfg, outline)

    problems = check_scenes(scenes, prompts)
    if problems:
        for problem in problems:
            log.error("validation: %s", problem)
            db.log_event(video_id, "script", "error", f"validation: {problem}")
        raise RuntimeError(f"script validation failed: {problems[0]}")

    for opener in txt.banned_openers(full_text):
        # Style slip, not a reason to throw away a finished script.
        log.warning("style: sentence starts with a banned phrase: %s", opener)

    seconds_per_word = 60.0 / int(cfg["words_per_minute"])
    db.clear_scenes(video_id)
    for i, (scene, prompt) in enumerate(zip(scenes, prompts), 1):
        db.add_scene(video_id, i, text=scene, image_prompt=prompt,
                     duration_s=round(txt.word_count(scene) * seconds_per_word, 2))

    estimated_s = round(total_words * seconds_per_word, 1)
    db.update_video(video_id, title=outline["title"])

    out = output_dir(video_id)
    (out / "script.txt").write_text(
        f"{outline['title']}\n\n{full_text}\n", encoding="utf-8")
    (out / "outline.json").write_text(
        json.dumps({**outline, "concept": concept, "word_budget": budget,
                    "total_words": total_words, "scene_count": len(scenes),
                    "estimated_duration_s": estimated_s}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    log.info("done: %d words, %d scenes, ~%.1f min", total_words, len(scenes), estimated_s / 60)
    db.log_event(video_id, "script", "info",
                 f"{total_words} words, {len(scenes)} scenes, ~{estimated_s / 60:.1f} min")
    return True
