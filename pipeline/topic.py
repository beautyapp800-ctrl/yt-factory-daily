"""Stage 1: pick a seed theme and turn it into a concrete video concept."""
import json
import random
from difflib import SequenceMatcher

from core import db
from core.concede import concede
from core.config import ASSETS_DIR, output_dir
from core.llm import complete_json
from core.logger import get_logger
from core.prompts import OUTLINE_SYSTEM

log = get_logger("topic")

SEEDS_PATH = ASSETS_DIR / "topic_seeds.json"
MAX_ATTEMPTS = 5
SIMILARITY_THRESHOLD = 0.75
RECENT_TOPICS_CHECKED = 200


def load_seeds(path=SEEDS_PATH):
    seeds = json.loads(path.read_text(encoding="utf-8"))
    if not seeds:
        raise RuntimeError(f"No seeds in {path}")
    return seeds


def pick_seeds(seeds, count):
    """Least-used seeds first, shuffled within each usage count so the pick stays random."""
    counts = db.seed_counts()
    buckets = {}
    for seed in seeds:
        buckets.setdefault(counts.get(seed["theme"], 0), []).append(seed)
    ordered = []
    for used in sorted(buckets):
        bucket = buckets[used][:]
        random.shuffle(bucket)
        ordered.extend(bucket)
    return ordered[:count]


def similarity(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def is_duplicate(topic, recent):
    """True when the topic already exists or reads too much like a recent one."""
    if db.topic_exists(topic):
        log.warning("topic already in the database: %s", topic)
        return True
    for old in recent:
        score = similarity(topic, old)
        if score >= SIMILARITY_THRESHOLD:
            log.warning("topic too similar (%.2f) to an earlier one: %s", score, old)
            return True
    return False


def _ask_concept(seed, cfg):
    hints = "; ".join(seed.get("angle_hints", []))
    prompt = f"""Design one video for a YouTube channel about {cfg['channel_topic']}.

The seed theme is "{seed['theme']}". Possible angles: {hints}.

Pick one narrow angle, not the whole theme. The video runs about {cfg['target_duration_min']} minutes.

Return JSON with exactly these keys:
"topic": the working topic, 4 to 10 words, naming the specific angle. Not a title, no numbers, no colon.
"promise": one sentence saying what the viewer can do differently after watching.
"different": one sentence on how this avoids the obvious take that every other channel makes on this theme.

JSON only."""
    concept = complete_json(prompt, system=OUTLINE_SYSTEM, max_tokens=700, temperature=0.95)
    missing = [k for k in ("topic", "promise", "different") if not str(concept.get(k, "")).strip()]
    if missing:
        raise ValueError(f"concept missing keys: {', '.join(missing)}")
    concept["topic"] = " ".join(str(concept["topic"]).split()).strip(" .\"'")
    concept["seed"] = seed["theme"]
    # The script stage anchors all ten lessons to these, so they have to travel with it.
    concept["angle_hints"] = seed.get("angle_hints", [])
    return concept


def _ask_search_concept(phrase, cfg):
    """The video that answers one real search, built around the phrase rather than a seed."""
    prompt = f"""Someone typed "{phrase}" into YouTube's search box.

Design the one video on a channel about {cfg['channel_topic']} that gives that person exactly
what they were looking for. It runs about {cfg['target_duration_min']} minutes as
{cfg.get('lessons_per_video', 10)} lessons drawn from Stoic practice.

Return JSON with exactly these keys:
"topic": the working topic, 4 to 10 words, containing the main words of the search. No colon.
"promise": one sentence saying what the viewer can do differently after watching.
"different": one sentence on how this avoids the obvious take every other channel makes.
"angle_hints": five short angles, each a concrete everyday situation this search is about.

JSON only."""
    concept = complete_json(prompt, system=OUTLINE_SYSTEM, max_tokens=900, temperature=0.8)
    missing = [k for k in ("topic", "promise", "different") if not str(concept.get(k, "")).strip()]
    if missing:
        raise ValueError(f"concept missing keys: {', '.join(missing)}")
    concept["topic"] = " ".join(str(concept["topic"]).split()).strip(" .\"'")
    hints = concept.get("angle_hints")
    concept["angle_hints"] = [str(h) for h in hints][:5] if isinstance(hints, list) else []
    concept["seed"] = phrase
    concept["search_phrase"] = phrase
    return concept


def run_from_search(video_id, cfg):
    """Choose today's subject from what people type into YouTube. True, or False when the
    source could not give one, in which case the caller falls back to the seeds."""
    from pipeline import search_topics
    recent = db.recent_topics(RECENT_TOPICS_CHECKED)
    used = db.used_search_phrases()
    recent_phrases = [v for v in db.recent_search_phrases(search_topics.BASE_REST)]
    try:
        options = search_topics.choose(used=used, recent=recent_phrases, count=MAX_ATTEMPTS)
    except Exception as e:                                       # noqa: BLE001
        log.warning("no subject from search: %s", str(e)[:160])
        return False
    for attempt, (phrase, row) in enumerate(options, 1):
        log.info("attempt %d/%d with the search phrase '%s' (score %.1f, in %d lists)",
                 attempt, len(options), phrase, row["score"], row["lists"])
        try:
            concept = _ask_search_concept(phrase, cfg)
        except (ValueError, KeyError) as e:
            log.warning("attempt %d: bad concept (%s)", attempt, e)
            continue
        if is_duplicate(concept["topic"], recent):
            continue
        db.update_video(video_id, topic=concept["topic"], seed=phrase, search_phrase=phrase)
        (output_dir(video_id) / "concept.json").write_text(
            json.dumps(concept, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("topic from search '%s': %s", phrase, concept["topic"])
        db.log_event(video_id, "topic", "info",
                     f"search phrase '{phrase}' -> topic: {concept['topic']}")
        return True
    return False


def run(video_id, cfg):
    # What people search for comes first. The seeds stay as the fallback for a day when the
    # autocomplete cannot be reached, so a network fault does not cost a video - but it is
    # recorded, because a video built from a seed is a video nobody searched for.
    if cfg.get("topics_from_search", True):
        if run_from_search(video_id, cfg):
            return True
        concede(video_id, "topic", "no subject could be taken from YouTube search",
                using="the seed themes, as before")
    seeds = load_seeds()
    recent = db.recent_topics(RECENT_TOPICS_CHECKED)
    candidates = pick_seeds(seeds, MAX_ATTEMPTS)

    for attempt, seed in enumerate(candidates, 1):
        log.info("attempt %d/%d with seed '%s'", attempt, len(candidates), seed["theme"])
        try:
            concept = _ask_concept(seed, cfg)
        except (ValueError, KeyError) as e:
            log.warning("attempt %d: bad concept (%s)", attempt, e)
            continue
        if is_duplicate(concept["topic"], recent):
            continue

        db.update_video(video_id, topic=concept["topic"], seed=seed["theme"])
        (output_dir(video_id) / "concept.json").write_text(
            json.dumps(concept, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("topic: %s", concept["topic"])
        log.info("promise: %s", concept["promise"])
        db.log_event(video_id, "topic", "info", f"topic chosen from seed '{seed['theme']}': {concept['topic']}")
        return True

    # The model refusing to phrase a concept is not a reason to skip a day:
    # the seed itself carries a theme and angle hints, which is enough to
    # write from. The topic will be blunter than usual, and recorded as such.
    seed = candidates[0]
    hint = (seed.get("angle_hints") or [""])[0]
    concept = {"topic": " ".join(f"{seed['theme']} {hint}".split())[:70].strip(),
               "promise": "", "different": "", "seed": seed["theme"],
               "angle_hints": seed.get("angle_hints", [])}
    db.update_video(video_id, topic=concept["topic"], seed=seed["theme"])
    (output_dir(video_id) / "concept.json").write_text(
        json.dumps(concept, indent=2, ensure_ascii=False), encoding="utf-8")
    concede(video_id, "topic",
            f"no usable concept after {len(candidates)} attempts",
            using=f"the seed theme directly: {concept['topic']}")
    return True
