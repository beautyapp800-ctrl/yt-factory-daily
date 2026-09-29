"""Stage 1: pick a seed theme and turn it into a concrete video concept."""
import json
import random
from difflib import SequenceMatcher

from core import db
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
    return concept


def run(video_id, cfg):
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

    raise RuntimeError(f"no usable topic after {len(candidates)} attempts")
