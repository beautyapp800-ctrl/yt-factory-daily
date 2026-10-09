"""The words on the thumbnail, written as their own job.

They used to be cut out of the YouTube title, and the cut landed wherever the geometry
allowed: "WHILE PROGRESS DRAGS", "WHILE STAYING TRUE". Both are grammatical fragments hanging
off a sentence the viewer cannot see, and in a feed they mean nothing at all - the eye reads
three words, finds no thought in them, and moves on.

So the phrase is now asked for on its own terms. The model is told what the video is about and
asked for a thumbnail line, not for a piece of a title, and the answer has to be a finished
thought that either promises a consequence or names a tension.

What the code enforces, rather than hopes for:

- two to four words, because that is what fits at a size a phone feed can read;
- it does not open on a function word. A phrase that starts with while, when, to, that, how
  or for is a fragment by construction - it is the hinge of a longer sentence, and the sentence
  is missing. The same is true of articles, conjunctions and the other prepositions, so the
  whole class is refused;
- it is not a slice of the title in disguise, which is the failure mode this replaces;
- it says something: a line made only of the words every title in the niche already uses
  ("STOIC LESSONS", "THE STOIC WAY") is refused.

Every rejection is handed back to the model in the next attempt, named. If three attempts
still produce nothing usable the caller is told so and falls back to the old title-cutting -
visibly, in the database and in the run email, because a quiet fallback would hide exactly the
defect this module exists to fix.
"""
import re

from core.llm import complete_json
from core.logger import get_logger

log = get_logger("phrase")

ATTEMPTS = 3
MIN_WORDS, MAX_WORDS = 2, 4

# The six the brief names, and the rest of the same class. Every one of them is a hinge: it
# joins a clause to something else, so a phrase that begins with one is announcing a sentence
# it never finishes.
BANNED_OPENERS = {
    "while", "when", "to", "that", "how", "for",            # named in the brief
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "if", "as", "than", "then",
    "because", "since", "unless", "until", "after", "before", "once", "whether", "though",
    "although", "whereas", "of", "in", "on", "at", "by", "from", "with", "without", "into",
    "onto", "over", "under", "about", "against", "between", "through", "during", "upon",
    "is", "are", "was", "were", "be", "being", "been", "do", "does", "did", "can", "will",
    "what", "which", "who", "whom", "whose", "why", "where",
}

# Words that say nothing because every video in the niche already says them. A phrase made
# only of these is not a thought, it is the channel's own wallpaper.
EMPTY_WORDS = {
    "stoic", "stoics", "stoicism", "lesson", "lessons", "rule", "rules", "way", "ways",
    "thing", "things", "truth", "truths", "secret", "secrets", "habit", "habits", "step",
    "steps", "reason", "reasons", "principle", "principles", "wisdom", "philosophy",
    "life", "mind", "daily", "every", "day", "your", "you", "yours", "self",
}

SYSTEM = """You write the words that go on a YouTube thumbnail.

A thumbnail line is not a title and not a piece of one. It is read in about half a second, at
the size of a postage stamp, by someone scrolling past. It has to land on its own.

You write in capitals-ready plain English. No punctuation, no emoji, no quotation marks."""

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def _words(text):
    return _WORD.findall(text or "")


def _prompt(topic, promise, lessons, title, complaints=()):
    listed = "\n".join(f"- {l}" for l in lessons[:10])
    prompt = f"""This video is about: {topic}

What the viewer is promised: {promise}

The lessons in it:
{listed}

Write ONE thumbnail phrase for it.

Rules, all of them hard:
- TWO to FOUR words. Not five, not one.
- A finished thought. Someone who sees only these words, with no title underneath and no
  picture around them, must understand something. "WHILE PROGRESS DRAGS" and "WHILE STAYING
  TRUE" are failures: they are the middle of a sentence nobody can see.
- It must do one of two things: promise a consequence ("SILENCE WINS ARGUMENTS", "ANGER COSTS MORE"),
  or name a tension ("PATIENCE OR PRIDE", "KINDNESS HAS LIMITS").
- It must NOT begin with while, when, to, that, how, for, or any other joining word, article
  or preposition. Begin on a noun, a verb or an adjective that carries weight.
- Do not quote the title. The title is "{title}" - the phrase has to be its own line, not a
  slice of that one.
- No words that every video in this niche already uses on its own: stoic, lessons, rules,
  ways, things, wisdom, life, mind, daily. They can appear inside a real thought, but a
  phrase built only out of them says nothing.

Return JSON: {{"phrase": "YOUR PHRASE"}}"""
    if complaints:
        listed = "\n".join(f"- {c}" for c in complaints)
        prompt += ("\n\nYour previous answer was rejected. Fix every one of these and do not "
                   f"introduce a new fault:\n{listed}")
    return prompt


def problems(phrase, title="", fits=None):
    """Why a phrase is unusable. An empty list means it passes.

    `fits` is an optional callable the caller supplies that answers whether the phrase can
    actually be set on the thumbnail at a readable size. The type is heavy condensed capitals
    a third of the frame tall, which leaves room for about eight characters on the large line,
    so a perfectly good thought made of long words - "SILENCE SHATTERS ARGUMENTS" - cannot be
    drawn. Rather than writing a character limit here and hoping it matches the layout, the
    layout itself is asked. Measured, not guessed, and it stays true if the font or the frame
    ever changes.
    """
    words = _words(phrase)
    out = []
    if not words:
        return ["the phrase has no words in it"]
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        out.append(f'"{phrase}" is {len(words)} words, needs {MIN_WORDS} to {MAX_WORDS}')
    if words[0].lower() in BANNED_OPENERS:
        out.append(f'"{phrase}" begins with "{words[0]}", which is a joining word: the phrase '
                   f"reads as the middle of a sentence. Begin on a noun, verb or adjective")
    if all(w.lower() in EMPTY_WORDS for w in words):
        out.append(f'"{phrase}" is made only of words every video in this niche uses; it says '
                   f"nothing specific about this one")
    if re.search(r"[.,;:!?\"'“”]", (phrase or "").replace("'", "").replace("’", "")):
        out.append(f'"{phrase}" contains punctuation')
    if title:
        # A run of three of the title's words in the same order is the old defect wearing a
        # different hat: the model has cut the title up again rather than written a line.
        low = [w.lower() for w in _words(title)]
        said = [w.lower() for w in words]
        for i in range(len(said) - 2):
            run = said[i:i + 3]
            if any(low[j:j + 3] == run for j in range(max(0, len(low) - 2))):
                out.append(f'"{phrase}" is lifted out of the title word for word')
                break
    # "PAT IENCE BREEDS SUCCESS" came back from the model and passed every rule above: four
    # words, a real thought, no banned opener. It reached a finished tile before anyone saw
    # that PATIENCE had been broken in half. Two neighbouring words that join into a word the
    # video itself uses are not two words.
    context = f"{title} {' '.join(_words(phrase))}".lower()
    for i in range(len(words) - 1):
        joined = (words[i] + words[i + 1]).lower()
        if len(joined) > 4 and joined in context.replace(" ", " ") and joined in title.lower():
            out.append(f'"{phrase}" has "{joined}" broken across two words')
            break
    if fits and not out and not fits(phrase):
        longest = max(words, key=len)
        out.append(f'"{phrase}" will not fit the thumbnail at a readable size - "{longest}" '
                   f"is too long. Use shorter words; nothing on the big line should be more "
                   f"than about eight letters")
    return out


def write_phrase(topic, promise, lessons, title, attempts=ATTEMPTS, fits=None):
    """(phrase, complaints). complaints is empty when the phrase passed every rule.

    On failure the last candidate is returned with its complaints, so the caller can decide
    between using it and falling back - and can say which it did.
    """
    complaints, phrase = [], ""
    for attempt in range(1, attempts + 1):
        data = complete_json(_prompt(topic, promise, lessons, title, complaints),
                             system=SYSTEM, max_tokens=200, temperature=0.9)
        phrase = (data or {}).get("phrase") if isinstance(data, dict) else ""
        phrase = re.sub(r"\s+", " ", str(phrase or "")).strip().strip('"“”')
        complaints = problems(phrase, title, fits)
        if not complaints:
            log.info("thumbnail phrase: %r (attempt %d)", phrase, attempt)
            return phrase, []
        log.warning("phrase attempt %d/%d rejected: %s", attempt, attempts,
                    "; ".join(complaints))
    return phrase, complaints


def line_options(phrase):
    """Every way of breaking the phrase across the thumbnail's two lines, best first.

    Two to four words, so the choices are few; they are ordered by how even the two lines
    are in character count, because that is what the layout can set largest.
    """
    words = [w.upper() for w in _words(phrase)]
    if len(words) < 2:
        return []
    cuts = sorted(range(1, len(words)),
                  key=lambda cut: abs(len(" ".join(words[:cut])) - len(" ".join(words[cut:]))))
    return [(words[:cut], words[cut:]) for cut in cuts]
