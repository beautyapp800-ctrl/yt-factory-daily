"""What people actually type into YouTube, as the source of each video's subject and title.

The channel is not penalised - its videos come up when searched by their exact titles, and
every feature in Studio is green - but nobody searches for those titles. "10 Phantoms You Must
Stop Craving" was written by a model for a model's idea of a hook; no one has ever typed it.
So the subject now starts from YouTube's own search autocomplete, which lists what people do
type, most-typed first, and the title starts with that phrase word for word.

How the phrases are gathered: each base query ("stoicism", "how to stop overthinking"...) is
asked as it is and followed by each letter of the alphabet ("stoicism a", "stoicism b"...),
which is how autocomplete gives up its long tail. A phrase scores for every list it appears
in, more for appearing near the top, and most for appearing under the bare base query, where
the list is pure popularity. Long-tail phrases of two to seven words are preferred to the bare
head term: a small channel can rank for "stoicism for beginners", not for "stoicism".

What is thrown out, and why, is written next to each list below. The source was checked from a
GitHub runner (scripts/runner_probe.py): it answers there exactly as it does from a laptop.
"""
import json
import re
import string
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from core.logger import get_logger

log = get_logger("search")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "search_cache.json"
URL = "https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&hl=en&gl=us&q="

BASES = ["stoicism", "marcus aurelius", "seneca", "epictetus", "stoic",
         "how to stop overthinking", "how to control emotions"]

# A phrase must touch the channel's subject: a Stoic name or idea, or one of the everyday
# problems the channel's lessons are about.
ON_TOPIC = {"stoic", "stoicism", "stoics", "marcus", "aurelius", "seneca", "epictetus",
            "meditations", "discourses", "enchiridion", "overthinking", "overthink", "emotions",
            "emotion", "anger", "angry", "anxiety", "worry", "worrying", "discipline",
            "mindset", "calm", "control", "fear", "grief", "ego", "letting", "stress",
            "resilience", "mental", "peace", "focus", "habits", "philosophy", "virtue",
            "ruminating", "jealousy", "regret", "loneliness", "confidence"}

# Thrown out: other meanings of the same words, and formats this channel does not make. Seen in
# the first harvest: "seneca village", "seneca crane death", "stoichiometry chemistry", "how to
# control emotions in trading", "marcus aurelius gladiator".
OFF_TOPIC = {"stoichiometry", "chemistry", "village", "crane", "falls", "college", "lake",
             "project", "county", "nation", "gladiator", "movie", "film", "actor", "russell",
             "crowe", "documentary", "audiobook", "audio", "book", "books", "pdf", "full",
             "reading", "read", "sleep", "asmr", "music", "song", "rap", "edit", "phonk",
             "tiktok", "shorts", "short", "meme", "memes", "game", "trading", "trader", "poker",
             "basketball", "football", "soccer", "golf", "tennis", "gym", "christian",
             "christianity", "islam", "muslim", "bible", "jesus", "god", "catholic", "buddhism",
             "women", "woman", "girl", "girls", "hindi", "urdu", "tamil", "telugu", "espanol",
             "spanish", "español", "deutsch", "kids", "child", "children", "dog", "cat",
             "boyfriend", "girlfriend", "husband",
             "ex", "crush", "texting", "dating", "ai", "chatgpt", "minecraft", "fortnite",
             "roblox", "anime", "lyrics", "ringtone", "wallpaper", "tattoo", "statue",
             "death", "died", "wife", "son", "daughter", "net", "worth", "age", "height",
             "quotes", "quote", "vs", "versus", "school", "high", "beats", "period",
             "man", "men", "male", "biography", "born", "wikipedia", "podcast", "course"}

MIN_WORDS, MAX_WORDS = 2, 7


def _suggest(query, timeout=20):
    req = urllib.request.Request(URL + urllib.parse.quote(query),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    return [str(s).strip().lower() for s in (data[1] if len(data) > 1 else [])]


def harvest(bases=BASES, pause=0.25, use_cache=True):
    """{phrase: {"score", "lists", "best", "bases", "head"}} from today's autocomplete.

    Cached for the day: a retry of the daily run should not ask Google two hundred questions
    again, and the answers do not change by the hour.
    """
    today = date.today().isoformat()
    if use_cache and CACHE.exists():
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        if cached.get("date") == today and cached.get("bases") == bases:
            return cached["phrases"]
    found, asked, failed = {}, 0, 0
    for base in bases:
        for suffix in [""] + [f" {c}" for c in string.ascii_lowercase]:
            query = base + suffix
            try:
                listed = _suggest(query)
            except Exception as e:                               # noqa: BLE001
                failed += 1
                log.warning("autocomplete for %r failed: %s", query, str(e)[:80])
                continue
            asked += 1
            for rank, phrase in enumerate(listed):
                row = found.setdefault(phrase, {"score": 0.0, "lists": 0, "best": 99,
                                                "bases": [], "head": False})
                row["score"] += (10 - rank) / 10
                row["lists"] += 1
                row["best"] = min(row["best"], rank)
                if base not in row["bases"]:
                    row["bases"].append(base)
                if suffix == "":
                    row["head"] = True
                    row["score"] += 2.0
            time.sleep(pause)
    if asked == 0:
        raise RuntimeError(f"YouTube autocomplete answered none of {failed} queries")
    log.info("autocomplete: %d queries answered, %d failed, %d distinct phrases",
             asked, failed, len(found))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"date": today, "bases": bases, "phrases": found}, indent=1),
                     encoding="utf-8")
    return found


def words(phrase):
    return re.findall(r"[a-z0-9']+", phrase.lower())


def why_not(phrase):
    """'' if the phrase can be a video on this channel, else the reason it cannot."""
    w = words(phrase)
    if not MIN_WORDS <= len(w) <= MAX_WORDS:
        return f"{len(w)} words"
    bad = set(w) & OFF_TOPIC
    if bad:
        return "off topic: " + ", ".join(sorted(bad))
    if not set(w) & ON_TOPIC:
        return "not about the channel's subject"
    if any(len(x) == 1 and x not in ("a", "i") for x in w):
        return "a stray letter from the expansion"
    return ""


INTENT_CACHE = ROOT / "data" / "search_intent.json"
# What a searcher can want, offered as a closed list so the answer can be counted. Only the
# first two are what this channel makes. The question is open - "what is this person looking
# for?" - never "is this a good topic?", which a model answers yes to.
INTENTS = ["stoic-lessons", "self-help-advice", "biography", "book-or-audiobook",
           "music-or-beats", "school-or-place", "person-or-celebrity", "fiction-or-anime",
           "health-or-medical", "other"]
WANTED = {"stoic-lessons", "self-help-advice"}


def intents(phrases):
    """{phrase: intent} for each phrase, asked of the text model once and kept for good."""
    from core.llm import complete_json
    cache = json.loads(INTENT_CACHE.read_text(encoding="utf-8")) if INTENT_CACHE.exists() else {}
    todo = [p for p in phrases if p not in cache]
    # Fifteen at a time: at thirty the model once returned 29 answers, which cannot be
    # lined up with the questions, so the whole batch was lost.
    for start in range(0, len(todo), 15):
        batch = todo[start:start + 15]
        listed = "\n".join(f"{i}. {p}" for i, p in enumerate(batch, 1))
        prompt = f"""Each line below is something a person typed into YouTube's search box.

{listed}

For each line give three things:
"label": what that person is most likely looking for, one of: {", ".join(INTENTS)}
  "stoic-lessons" = practical ideas from Stoic philosophy to apply to their life;
  "self-help-advice" = practical advice for a personal problem such as worry or anger.
  A book title alone, a life story, a song, a school, a team, a film or an anime character
  are not lessons, whatever words they share.
"names_a_modern_person": true if the line contains the name of any person who is not an
  ancient philosopher (a speaker, a celebrity, a teacher, a fictional character), else false.
"english": true if every word of the line is English, else false.

Return JSON: {{"items": [{{"label": ..., "names_a_modern_person": ..., "english": ...}}, ...]}}
with exactly {len(batch)} items in the same order as the lines."""
        items = None
        for _attempt in range(2):
            try:
                data = complete_json(prompt, system="You classify search queries. JSON only.",
                                     max_tokens=2500, temperature=0.0)
                items = data.get("items") if isinstance(data, dict) else None
            except Exception as e:                               # noqa: BLE001
                log.warning("could not classify %d phrases: %s", len(batch), str(e)[:100])
            if isinstance(items, list) and len(items) == len(batch):
                break
        if not isinstance(items, list) or len(items) != len(batch):
            log.warning("classifier returned %s items for %d phrases; batch skipped",
                        len(items) if isinstance(items, list) else "no", len(batch))
            continue
        for phrase, item in zip(batch, items):
            item = item if isinstance(item, dict) else {}
            label = str(item.get("label", "")).strip().lower()
            label = label if label in INTENTS else "other"
            # A modern name or a foreign word turns any label into "other": a video cannot
            # answer "how to stop overthinking mel robbins" without being about Mel Robbins.
            if item.get("names_a_modern_person") is True or item.get("english") is False:
                label = "other"
            cache[phrase] = label
        INTENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        INTENT_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    return {p: cache.get(p) for p in phrases}


# Forms of one word that a searcher treats as the same: "stoic philosophy" and "stoicism
# philosophy" are one video.
SAME = {"stoicism": "stoic", "stoics": "stoic", "emotional": "emotions", "emotion": "emotions",
        "your": "", "the": "", "a": "", "to": "", "how": "", "and": "", "of": "", "in": ""}


def _core(phrase):
    return {SAME.get(w, w) for w in words(phrase)} - {""}


def same_video(a, b, at=0.75):
    """True when two phrases would make the same video: one only extends the other, or their
    meaningful words overlap by `at` or more."""
    if a.startswith(b) or b.startswith(a):
        return True
    ca, cb = _core(a), _core(b)
    return bool(ca and cb) and len(ca & cb) / len(ca | cb) >= at


def ranked(found, used=()):
    """Usable phrases, best first, as [(phrase, row)], none the same video as one used."""
    used = {u.lower() for u in used}
    out = []
    for phrase, row in found.items():
        if why_not(phrase) or phrase in used:
            continue
        if any(same_video(phrase, u) for u in used):
            continue
        out.append((phrase, row))
    out.sort(key=lambda pr: (-pr[1]["score"], pr[1]["best"], pr[0]))
    return out


def display(phrase):
    """The phrase as it goes at the head of a title: the same words, in title case."""
    small = {"a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with", "at", "by"}
    parts = phrase.split()
    return " ".join(p if (i and p in small) else (p[:1].upper() + p[1:])
                    for i, p in enumerate(parts))


def starts_with_phrase(title, phrase):
    """True when the title opens with the search phrase word for word (case aside)."""
    t, p = words(title), words(phrase)
    return t[:len(p)] == p


def base_of(phrase):
    """The base query a phrase grew from: the longest base it starts with, else its first word."""
    hits = [b for b in BASES if phrase.startswith(b)]
    return max(hits, key=len) if hits else (words(phrase)[:1] or [""])[0]


BASE_REST = 2   # a base query is not the root of two videos within this many days


def choose(used=(), found=None, look_at=40, count=1, recent=()):
    """[(phrase, row)] for today's video(s): the best-scored phrases not used before whose
    searcher wants what this channel makes."""
    found = found if found is not None else harvest()
    options = ranked(found, used)[:look_at]
    labels = intents([p for p, _r in options])
    good = [(p, r) for p, r in options if labels.get(p) in WANTED]
    log.info("of the top %d phrases, %d are searches for lessons or advice; the rest: %s",
             len(options), len(good),
             ", ".join(f"{p} ({labels.get(p)})" for p, _r in options
                       if labels.get(p) not in WANTED)[:600])
    if not good:
        raise RuntimeError("autocomplete gave no usable phrase that has not been used")
    # Spread across the bases: autocomplete's long tail for "how to stop overthinking" is deep,
    # and taken in score order four of ten videos came out as variations of it.
    picked, history = [], [base_of(p) for p in recent][:BASE_REST]
    for phrase, row in good:
        base = base_of(phrase)
        if base in history[:BASE_REST]:
            continue
        if any(same_video(phrase, p) for p, _r in picked):
            continue
        picked.append((phrase, row))
        history.insert(0, base)
        if len(picked) == count:
            break
    return picked
