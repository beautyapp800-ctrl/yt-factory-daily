"""Stage 2: full narration script, built in four stages instead of one request.

A: outline (title, promise, lesson titles), with the refusal/subtraction form enforced
B: hook, written before the lessons so every lesson can be told not to echo it
C: each lesson as its own request, in its own setting
D: programmatic scene split, then image prompts in batches

Openings are tracked across every part: two parts starting with the same four words
is treated as a defect and the later one is rewritten.
"""
import json
import random
import re

from core import db, text as txt
from core.config import output_dir
from core.llm import complete, complete_json
from core.logger import get_logger
from core.prompts import (CONCRETE_DETAIL_RULE, DOMAINS, EXAMPLE_TITLES, IMAGE_SYSTEM,
                          LESSON_FORM_RULES, MAX_LESSONS_PER_DOMAIN,
                          NARRATION_SYSTEM, OUTLINE_SYSTEM, SCENE_SETTINGS,
                          TITLE_BANNED_WORDS, domain_of, forbidden_tics_rule)

log = get_logger("script")

HOOK_WORDS = 200
# 140, not 200: the model gave 126, 142 and 159 across three runs whatever it was
# asked for, and a short close is not a fault, so the budget now matches reality.
OUTRO_WORDS = 140
WORD_TOLERANCE = 0.15          # total script must land within +/-15% of budget
# Ordering 340 for a wanted 395 (ratio 0.86) landed video 4 at 3710 words, twelve words
# above the floor that fails the whole script: with the tighter prompts the model stopped
# overshooting and delivered 262-383 words per lesson. The two failure modes cost very
# different amounts, because trim_to_budget() fixes an over-long script for free while a
# short one has to be bought back with rewrites, so the order now aims at the full target
# and lets trimming take back any excess.
ORDER_RATIO = 1.0
TRIM_RATIO = 1.20              # a lesson over target by this much is cut, not redone
MIN_SCENES, MAX_SCENES = 40, 60
SCENE_TARGET, SCENE_MIN, SCENE_MAX = 92, 75, 110
IMAGE_BATCH = 10
MAX_FIX_ROUNDS = 3
LESSONS_PER_FIX_ROUND = 5
OUTLINE_ATTEMPTS = 3
MAX_HOOK_TITLE_HITS = 2        # a hook naming more titles than this is previewing
# Fraction of the runtime reserved for pause silence (sentence/scene/lesson gaps),
# not available for speaking. words_per_minute is a measured rate with no pauses in
# it, so the raw budget has to shrink by this much to still fit target_duration_min.
PAUSE_RESERVE_RATIO = 0.07
OPENING_WORDS = 4              # parts sharing this many opening words clash
OPENING_REWRITES = 2           # attempts to shift a clashing opening before giving up
MAX_NAME_USES = 3              # a name used more than this is swapped for another
TICS_FROM_VIDEOS = 10          # how far back to look for names and colours to avoid


def ordered_words(actual_target):
    """Turn the word count we want into the smaller number we ask the model for."""
    return max(180, round(actual_target * ORDER_RATIO))


# --- stage A ---------------------------------------------------------------

def _normalise_title(title):
    return re.sub(r"[^a-z ]", "", title.lower()).strip()


def title_problems(title, strict=True):
    """Why a lesson title is unusable. An empty list means it passes."""
    problems = []
    low = title.lower()
    if strict and _normalise_title(title) in {_normalise_title(t) for t in EXAMPLE_TITLES}:
        problems.append(f'"{title}" is copied straight from the examples in the instructions')
    for banned in TITLE_BANNED_WORDS:
        if re.search(rf"\b{re.escape(banned)}\b", low):
            problems.append(f'"{title}" uses the banned word "{banned}"')
    if strict:
        words = title.split()
        if not 4 <= len(words) <= 8:
            problems.append(f'"{title}" is {len(words)} words, needs 4 to 8')
        if ":" in title:
            problems.append(f'"{title}" contains a colon')
    return problems


def _outline_prompt(concept, cfg, lesson_count, complaints=(), tics=""):
    seed = concept.get("seed") or ""
    hints = concept.get("angle_hints") or []
    anchor = ""
    if seed:
        listed = "; ".join(hints)
        anchor = f"""

THE STOIC IDEA THIS VIDEO IS ABOUT: {seed}
Angles that belong to it: {listed}

Every one of the {lesson_count} lessons has to work directly on {seed} as the Stoics
meant it. A lesson that would fit equally well in a video about any other idea does not
belong here. "Become calmer", "be more present" and "build better habits" are not about
{seed} and are not acceptable. If you cannot tie a lesson back to {seed} in one sentence,
replace it.

Spread them across a life: work, family, money, the body, friendship, the past. Do not
let the whole video become one subject."""

    prompt = f"""Plan a {cfg['target_duration_min']} minute video about {concept['topic']}.

The promise to the viewer: {concept['promise']}
What keeps it fresh: {concept['different']}{anchor}

{LESSON_FORM_RULES}{tics}

Return JSON with exactly these keys:
"title": the video title. Examples of the format this channel uses:
  10 Stoic Lessons That Will Make You Mentally Unbreakable
  10 Things You Should Quietly Remove From Your Life
  Start with {lesson_count}. Under 70 characters. No colon, no subtitle.
"promise": one sentence, what the viewer walks away able to do.
"lessons": an array of exactly {lesson_count} objects, each with
   "title": the lesson title, following the rules above, and
   "focus": one sentence naming the specific thing being refused or dropped.

The lessons must be concrete and clearly different from each other. No two may
refuse the same thing in different words. Order them so the video builds.

JSON only."""
    if complaints:
        listed = "\n".join(f"- {c}" for c in complaints)
        prompt += ("\n\nYour previous attempt was rejected. Fix every one of these and "
                   f"do not introduce new violations:\n{listed}")
    return prompt


def domain_complaints(lessons):
    """Reject a plan that crowds most of its lessons into one area of life."""
    tally = {}
    for lesson in lessons:
        domain = domain_of(f"{lesson['title']} {lesson.get('focus', '')}")
        if domain:
            tally[domain] = tally.get(domain, 0) + 1
    problems = []
    for domain, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        if count > MAX_LESSONS_PER_DOMAIN:
            others = ", ".join(d for d in DOMAINS if d != domain)
            problems.append(f"too many lessons about {domain} ({count} of {len(lessons)}); "
                            f"give lessons about {others} instead")
    return problems


def make_outline(concept, cfg, lesson_count, tics=""):
    complaints = []
    for attempt in range(1, OUTLINE_ATTEMPTS + 1):
        outline = complete_json(_outline_prompt(concept, cfg, lesson_count, complaints, tics),
                                system=OUTLINE_SYSTEM, max_tokens=2500, temperature=0.85)
        lessons = outline.get("lessons") or []
        if len(lessons) != lesson_count:
            raise ValueError(f"outline has {len(lessons)} lessons, expected {lesson_count}")
        for i, lesson in enumerate(lessons, 1):
            if not str(lesson.get("title", "")).strip():
                raise ValueError(f"lesson {i} has no title")
            lesson["title"] = txt.clean(str(lesson["title"])).strip(" .")
            lesson["focus"] = txt.clean(str(lesson.get("focus", "")))
        outline["title"] = txt.clean(str(outline.get("title", ""))).strip(" .")
        if not outline["title"]:
            raise ValueError("outline has no title")

        complaints = []
        for lesson in lessons:
            complaints += title_problems(lesson["title"])
        # The video title carries a leading number, so only the word ban applies.
        complaints += title_problems(outline["title"], strict=False)
        complaints += domain_complaints(lessons)
        if not complaints:
            return outline
        log.warning("outline attempt %d/%d rejected: %s", attempt, OUTLINE_ATTEMPTS,
                    "; ".join(complaints[:4]))

    raise RuntimeError("outline still breaks the title rules after "
                       f"{OUTLINE_ATTEMPTS} attempts: {complaints[0]}")


# --- shared prompt fragments ----------------------------------------------

def _avoid_openings(openings):
    """Name the exact word sequences already used, not the whole sentences.

    Handing over full sentences told the model what to avoid writing about; what it
    kept repeating was the construction. The forbidden first words are the thing to
    state, so they are listed on their own as well.
    """
    if not openings:
        return ""
    prefixes = sorted({txt.opening_words(o, 3) for o in openings if o.strip()})
    listed = "\n".join(f"  - {o}" for o in openings)
    banned = "\n".join(f'  - do not begin with "{p}"' for p in prefixes if p)
    return f"""

These word sequences are already used by parts of this video. Your first sentence must
not begin with any of them:
{banned}

For context, those parts open like this:
{listed}
Do not reuse their place, their time of day, or the shape of their first sentence. If
they open on a person standing or sitting somewhere, open on something else entirely."""


# --- stage B: hook --------------------------------------------------------

def write_hook(concept, cfg, setting, openings=(), tics=""):
    """Deliberately given no lesson list: it previewed all ten when it had one."""
    prompt = f"""The subject of this video: {concept['topic']}
The promise to the viewer: {concept['promise']}

Write the opening of this video, about {ordered_words(HOOK_WORDS)} words.

Open inside one situation, in one place, at one hour: {setting}. Two sentences in, the
viewer has to recognise themselves in it. Then turn hard, and say that there is another
way to stand inside this.

Never preview, list, number or summarise what the video will cover. You are not being
told what the lessons are, and you must not guess at them. No greeting, no welcome, no
channel name, no phrase like in this video or by the end.{tics}{_avoid_openings(openings)}

Narration only."""
    return txt.clean(complete(prompt, system=NARRATION_SYSTEM,
                              max_tokens=HOOK_WORDS * 6, temperature=0.9))


# --- stage C: lessons and outro -------------------------------------------

def write_lesson(index, outline, written, target_words, cfg, setting, openings=(), tics=""):
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

Now write lesson {index + 1}, {lesson['title']}. Focus: {lesson.get('focus', '')}

About {ordered_words(target_words)} words. Build it in three movements, with no labels
or breaks between them:
First, one concrete situation the viewer recognises, set here: {setting}. That is where
this happens. Not a kitchen and not a morning unless the setting says so.
Then the Stoic principle that cuts through it, in plain language.
Then one thing to stop doing tomorrow, specific enough to actually do.

{CONCRETE_DETAIL_RULE}

This lesson is a refusal, not a new habit. You are taking something away from the
viewer. Never hand them a tool, a list, a log or a routine.

Do not name the lesson number. Do not write the lesson title as a heading.
Start straight into the situation.{tics}{_avoid_openings(openings)}

Narration only."""

    # x6, not x2: reasoning models spend part of the ceiling before the prose starts.
    raw = complete(prompt, system=NARRATION_SYSTEM,
                   max_tokens=int(target_words * 6), temperature=0.85)
    body = txt.clean(raw)

    # Over target by more than TRIM_RATIO: cut it instead of paying for a rewrite.
    limit = int(target_words * TRIM_RATIO)
    if txt.word_count(body) > limit:
        was = txt.word_count(body)
        body = txt.trim_to_words(body, limit)
        log.info("lesson %d trimmed from %d to %d words", index + 1, was, txt.word_count(body))
    return body


def write_outro(outline, cfg, openings=(), tics=""):
    titles = ", ".join(l["title"] for l in outline["lessons"])
    prompt = f"""Video title: {outline['title']}
The lessons just covered: {titles}

Write the ending of this video, about {ordered_words(OUTRO_WORDS)} words.

Pull the lessons into one idea, without listing them again. Leave the viewer with one
thing to carry into tomorrow. Then a short, quiet invitation to subscribe if the video
was worth their time, no more than two sentences, no enthusiasm, no asking for likes or
comments or the bell.{tics}{_avoid_openings(openings)}

Narration only."""
    return txt.clean(complete(prompt, system=NARRATION_SYSTEM,
                              max_tokens=OUTRO_WORDS * 6, temperature=0.85))


# --- opening diversity ----------------------------------------------------

def first_sentence(text):
    sentences = txt.split_sentences(text)
    return sentences[0] if sentences else ""


def _label(key):
    kind, index = key
    return kind if kind != "lesson" else f"lesson {index + 1}"


# --- model tics -----------------------------------------------------------

def swap_overused_names(parts, avoid):
    """Replace any name used more than MAX_NAME_USES times with an unused one.

    Done in code rather than by asking again: the model returns to its favourite names
    whatever the prompt says, and a find-and-replace costs nothing.
    """
    full = " ".join(parts.values())
    counts = txt.proper_names(full)
    # Collision check over every capitalised word, not just the confirmed names: a name
    # that only ever opens a sentence is invisible to proper_names() and would otherwise
    # be handed out as a replacement for the name being removed.
    taken = {w.lower() for w in txt.capitalised_words(full)} | {a.lower() for a in avoid}
    spare = [n for n in txt.NAME_POOL if n.lower() not in taken]

    for name, uses in sorted(counts.items(), key=lambda kv: -kv[1]):
        if uses <= MAX_NAME_USES:
            continue
        if not spare:
            log.warning("%s used %d times but no replacement name is left", name, uses)
            break
        replacement = spare.pop(0)
        log.info("%s appears %d times, replacing it with %s", name, uses, replacement)
        for key, body in parts.items():
            parts[key] = txt.replace_name(body, name, replacement)


def collect_tics(text, min_uses=2):
    """The (kind, value) pairs worth remembering so later videos avoid them.

    Only words used at least min_uses times count. A tic is by definition something
    repeated, and the single-use capitals are mostly false positives: ordinary words
    that happen to be capitalised after a colon or inside quoted speech, like Prepare
    or Supplies, which it would be actively harmful to forbid.
    """
    pairs = [("name", n) for n, uses in txt.proper_names(text).items() if uses >= min_uses]
    pairs += [("colour", c) for c, uses in txt.colours_used(text).items() if uses >= min_uses]
    return pairs


def fix_duplicate_openings(parts, order, regenerate):
    """Rewrite the later of any two parts that open with the same four words."""
    seen = {}
    for key in order:
        opener = txt.opening_words(parts[key], OPENING_WORDS)
        if not opener:
            continue
        if opener in seen:
            log.warning("%s opens like %s (%s), rewriting the later one",
                        _label(key), _label(seen[opener]), opener)
            for attempt in range(1, OPENING_REWRITES + 1):
                openings = [first_sentence(parts[k]) for k in order
                            if k != key and parts.get(k)]
                parts[key] = regenerate(key, openings)
                opener = txt.opening_words(parts[key], OPENING_WORDS)
                if opener not in seen:
                    log.info("%s now opens differently (%s)", _label(key), opener)
                    break
                log.warning("%s still opens the same way after rewrite %d/%d",
                            _label(key), attempt, OPENING_REWRITES)
            if opener in seen:
                continue
        seen[opener] = key


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
    return txt.split_scenes(full_text, retarget,
                            max(30, int(retarget * 0.8)), int(retarget * 1.25))


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
                log.warning("image batch at scene %d failed (attempt %d): %s",
                            start + 1, attempt, e)
                continue
            for offset in range(len(batch)):
                value = got.get(start + offset)
                if value:
                    prompts[start + offset] = f"{value.rstrip(' .,')}, {style}"
            if all(prompts[start:start + len(batch)]):
                break

    # Anything still missing gets a plain prompt, so a scene is never left without one.
    for i, value in enumerate(prompts):
        if not value:
            log.warning("scene %d got no image prompt, using the fallback", i + 1)
            prompts[i] = ("empty stone courtyard at dawn, a single worn wooden bench, "
                          f"long shadows across the flagstones, {style}")
    return prompts


def _ask_image_batch(batch, start, outline):
    listed = "\n\n".join(f"SCENE {start + i + 1}:\n{scene}" for i, scene in enumerate(batch))
    prompt = f"""These are consecutive scenes from the narration of a video titled
{outline['title']}. Write one image prompt for each scene, illustrating what that scene
talks about.

{listed}

For every prompt, pick one action or one object that the scene's own words mention, and
make it the thing the picture shows. A viewer who saw only the image should be able to
point at which scene it belongs to. A general mood piece - a landscape, an empty room,
light through a window with nothing happening - could sit under any scene in any video
and is a failure here, however pretty.

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


def _total_words(parts):
    return sum(txt.word_count(p) for p in parts.values())


def trim_to_budget(parts, lesson_count, max_total):
    """Cut the longest lessons until the whole script fits inside max_total.

    Caps every lesson at the highest length that makes the total fit, so the long
    lessons give up words and the short ones are left alone. Trimming is free and
    exact, which a rewrite is not: asking the model for fewer words just produces
    another overshoot, so an over-long script is never regenerated.
    """
    if _total_words(parts) <= max_total:
        return
    lengths = [txt.word_count(parts[("lesson", i)]) for i in range(lesson_count)]
    fixed = _total_words(parts) - sum(lengths)

    low, high = 60, max(lengths)
    while low < high:
        cap = (low + high + 1) // 2
        if fixed + sum(min(n, cap) for n in lengths) <= max_total:
            low = cap
        else:
            high = cap - 1
    cap = low

    for i in range(lesson_count):
        was = txt.word_count(parts[("lesson", i)])
        if was > cap:
            parts[("lesson", i)] = txt.trim_to_words(parts[("lesson", i)], cap)
            log.info("trimmed lesson %d from %d to %d words", i + 1, was,
                     txt.word_count(parts[("lesson", i)]))


def fix_word_count(parts, outline, budget, cfg, regenerate):
    """Bring the script inside the budget: trim when it is too long, rewrite when short."""
    lesson_count = len(outline["lessons"])
    low, high = budget * (1 - WORD_TOLERANCE), budget * (1 + WORD_TOLERANCE)

    if _total_words(parts) > high:
        log.warning("%d words is over the %d ceiling, trimming to the %d budget",
                    _total_words(parts), round(high), budget)
        trim_to_budget(parts, lesson_count, budget)
        log.info("after trimming: %d words", _total_words(parts))

    # Only a rewrite can add words, so a short script is the one case worth paying for.
    for round_no in range(1, MAX_FIX_ROUNDS + 1):
        total = _total_words(parts)
        if total >= low:
            break
        fixed = txt.word_count(parts[("hook", 0)]) + txt.word_count(parts[("outro", 0)])
        per_lesson = max(200, round((budget - fixed) / lesson_count))
        log.warning("round %d: %d words, need at least %d, aiming lessons at %d words each",
                    round_no, total, round(low), per_lesson)

        shortest = sorted(range(lesson_count),
                          key=lambda i: txt.word_count(parts[("lesson", i)])
                          )[:LESSONS_PER_FIX_ROUND]
        for index in shortest:
            was = txt.word_count(parts[("lesson", index)])
            log.info("regenerating lesson %d: %d words -> ~%d", index + 1, was, per_lesson)
            parts[("lesson", index)] = regenerate(("lesson", index), (), per_lesson)
        if _total_words(parts) > high:
            trim_to_budget(parts, lesson_count, budget)

    return low <= _total_words(parts) <= high


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
    # words_per_minute is a measured speaking rate (see scripts/calibrate_tempo.py),
    # not counting the silence our own pause system adds between sentences, scenes
    # and lessons. PAUSE_RESERVE_RATIO of the runtime goes to that silence, so the
    # word budget is shrunk by the same fraction to still land in target_duration_min.
    raw_budget = int(cfg["target_duration_min"]) * int(cfg["words_per_minute"])
    budget = round(raw_budget * (1 - PAUSE_RESERVE_RATIO))
    per_lesson = max(200, round((budget - HOOK_WORDS - OUTRO_WORDS) / lesson_count))
    log.info("word budget %d (%d wpm minus %.0f%% for pauses; ~%d per lesson wanted, "
             "ordering %d)", budget, cfg["words_per_minute"], PAUSE_RESERVE_RATIO * 100,
             per_lesson, ordered_words(per_lesson))

    seen = db.recent_tics(TICS_FROM_VIDEOS)
    tics = forbidden_tics_rule(seen.get("name", []), seen.get("colour", []))
    if seen:
        log.info("avoiding %d names and %d colour words from recent videos",
                 len(seen.get("name", [])), len(seen.get("colour", [])))

    log.info("stage A: outline")
    outline = make_outline(concept, cfg, lesson_count, tics)
    log.info("title: %s", outline["title"])

    # One distinct setting per part, so ten lessons do not share a kitchen.
    pool = random.sample(SCENE_SETTINGS, min(len(SCENE_SETTINGS), lesson_count + 1))
    settings = {"hook": pool[0], **{i: pool[i + 1] for i in range(lesson_count)}}
    spare = [s for s in SCENE_SETTINGS if s not in pool]

    parts = {}

    def regenerate(key, openings=(), target=None):
        """Rewrite one part, in a fresh setting when one is left, with an optional target."""
        kind, index = key
        if spare:
            settings[index if kind == "lesson" else "hook"] = spare.pop(0)
        if kind == "hook":
            return write_hook(concept, cfg, settings["hook"], openings, tics)
        if kind == "outro":
            return write_outro(outline, cfg, openings, tics)
        written = {j: parts[("lesson", j)] for j in range(lesson_count)
                   if j != index and ("lesson", j) in parts}
        return write_lesson(index, outline, written, target or per_lesson, cfg,
                            settings[index], openings, tics)

    log.info("stage B: hook (written first, with no sight of the lessons)")
    parts[("hook", 0)] = write_hook(concept, cfg, settings["hook"], tics=tics)
    log.info("hook %d words, opens: %s", txt.word_count(parts[("hook", 0)]),
             txt.opening_words(parts[("hook", 0)], 6))

    log.info("stage C: %d lessons, then the outro", lesson_count)
    for i in range(lesson_count):
        written = {j: parts[("lesson", j)] for j in range(i)}
        openings = [first_sentence(parts[("hook", 0)])]
        openings += [first_sentence(parts[("lesson", j)]) for j in range(i)]
        parts[("lesson", i)] = write_lesson(i, outline, written, per_lesson, cfg,
                                            settings[i], openings, tics)
        log.info("lesson %d/%d: %s (%d words) in %s", i + 1, lesson_count,
                 outline["lessons"][i]["title"], txt.word_count(parts[("lesson", i)]),
                 settings[i])

    all_openings = [first_sentence(parts[("hook", 0)])]
    all_openings += [first_sentence(parts[("lesson", i)]) for i in range(lesson_count)]
    parts[("outro", 0)] = write_outro(outline, cfg, all_openings, tics)
    log.info("outro %d words", txt.word_count(parts[("outro", 0)]))

    order = [("hook", 0)] + [("lesson", i) for i in range(lesson_count)] + [("outro", 0)]
    fix_duplicate_openings(parts, order, lambda key, openings: regenerate(key, openings))

    # A hook naming the lessons is previewing them, which it was told not to do.
    titles = [l["title"] for l in outline["lessons"]]
    for attempt in (1, 2):
        hits = txt.titles_present_in(parts[("hook", 0)], titles)
        if len(hits) <= MAX_HOOK_TITLE_HITS:
            break
        log.warning("hook previews %d lesson titles (%s), rewriting it",
                    len(hits), "; ".join(hits[:3]))
        db.log_event(video_id, "script", "warning",
                     f"hook previewed {len(hits)} lesson titles, rewritten")
        parts[("hook", 0)] = regenerate(
            ("hook", 0), [first_sentence(parts[("lesson", i)]) for i in range(lesson_count)])

    swap_overused_names(parts, seen.get("name", []))

    if not fix_word_count(parts, outline, budget, cfg, regenerate):
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
    db.add_tics(video_id, collect_tics(full_text))

    out = output_dir(video_id)
    (out / "script.txt").write_text(f"{outline['title']}\n\n{full_text}\n", encoding="utf-8")
    (out / "outline.json").write_text(
        json.dumps({**outline, "concept": concept, "word_budget": budget,
                    "total_words": total_words, "scene_count": len(scenes),
                    "estimated_duration_s": estimated_s,
                    "settings": {str(k): v for k, v in settings.items()}},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("done: %d words, %d scenes, ~%.1f min", total_words, len(scenes), estimated_s / 60)
    db.log_event(video_id, "script", "info",
                 f"{total_words} words, {len(scenes)} scenes, ~{estimated_s / 60:.1f} min")
    return True
