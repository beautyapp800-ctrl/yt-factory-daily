"""Narration text handling: cleanup, word counting, sentence and scene splitting."""
import re

# Characters that mean the model slipped into markdown instead of plain narration.
MARKDOWN_CHARS = "*#`_[]|<>"

_ABBREVIATIONS = ("Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sr", "Jr", "vs", "etc", "e.g", "i.e")
_BANNED_OPENERS = ("in conclusion", "remember that", "it is important to")

# Typography the model emits that a speech synthesizer either mispronounces or
# swallows. The non-breaking hyphen is the worst offender: it is not a hyphen to
# most voices, so "half-finished" comes out as one run-together word.
_TYPOGRAPHY = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‐": "-", "‑": "-", "−": "-",
    "…": ".", " ": " ", " ": " ", " ": " ", "﻿": "",
}


def clean(text):
    """Strip markdown, parentheticals and stray labels so a TTS engine reads it cleanly."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    text = text.translate(str.maketrans(_TYPOGRAPHY))
    # Fenced blocks and inline code.
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    # Emphasis first, so a label wrapped in ** is visible to the label pass below.
    text = re.sub(r"\*+([^*\n]*?)\*+", r"\1", text)    # **bold** / *italic*
    text = re.sub(r"_+([^_\n]*?)_+", r"\1", text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*(?:#{1,6}\s*|[-*•]\s+|\d+[.)]\s+)", "", line)
        # A bare "Lesson 3:" or "Hook:" style label the model sometimes prepends.
        line = re.sub(r"^\s*(?:lesson|part|section|hook|outro|intro)\s*\d*\s*[:.]\s*", "",
                      line, flags=re.I)
        lines.append(line.strip())
    text = " ".join(l for l in lines if l)
    # Remarks in parentheses or brackets, including the brackets themselves.
    text = re.sub(r"[(\[][^)\]]*[)\]]", " ", text)
    text = text.translate({ord(c): None for c in MARKDOWN_CHARS})
    text = re.sub(r"\s*-{2,}\s*", ", ", text)          # -- reads badly aloud
    text = text.replace("—", ", ").replace("–", ", ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Not between digits: a time like 6:15 and a figure like 1,000 must stay intact.
    text = re.sub(r"(?<!\d)([,;:])(?=\S)", r" ", text)
    return text.strip()


def word_count(text):
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def has_markdown(text):
    return any(c in (text or "") for c in MARKDOWN_CHARS)


def banned_openers(text):
    """Sentences that start with a phrase the style rules forbid."""
    found = []
    for sentence in split_sentences(text):
        low = sentence.lstrip().lower()
        for phrase in _BANNED_OPENERS:
            if low.startswith(phrase):
                found.append(sentence[:60])
    return found


def split_sentences(text):
    """Split on sentence boundaries, keeping common abbreviations intact."""
    if not text:
        return []
    guarded = text
    for abbr in _ABBREVIATIONS:
        guarded = re.sub(rf"\b{re.escape(abbr)}\.", f"{abbr}<DOT>", guarded)
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+", guarded)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def split_scenes(text, target=92, minimum=75, maximum=110):
    """Group sentences into scenes of roughly `target` words.

    Sentence boundaries are never broken. A sentence longer than `maximum` becomes
    its own scene rather than being cut mid-thought.
    """
    scenes, current, current_words = [], [], 0
    for sentence in split_sentences(text):
        words = word_count(sentence)
        # Starting the next scene would overshoot, and this one is already long enough.
        if current and current_words + words > maximum and current_words >= minimum:
            scenes.append(" ".join(current))
            current, current_words = [sentence], words
            continue
        current.append(sentence)
        current_words += words
        if current_words >= target:
            scenes.append(" ".join(current))
            current, current_words = [], 0
    if current:
        tail = " ".join(current)
        # Fold a stub tail into the previous scene when it would be too short alone.
        if scenes and word_count(tail) < minimum // 2:
            scenes[-1] = f"{scenes[-1]} {tail}"
        else:
            scenes.append(tail)
    return scenes


# Words too common to signal that a lesson title is being previewed. Every title
# opens with one of stop/quit/drop/let go, so those cannot count as evidence.
_TITLE_STOPWORDS = {
    "the", "a", "an", "your", "you", "yours", "to", "of", "that", "this", "and",
    "for", "in", "on", "at", "it", "is", "are", "be", "with", "from", "who",
    "what", "already", "every", "own", "stop", "quit", "let", "go", "drop",
    "give", "up", "keep", "have", "has", "do", "not",
}


def opening_words(text, count=4):
    """The first `count` words, lowercased, for spotting two parts that open alike."""
    words = re.findall(r"[A-Za-z']+", text or "")
    return " ".join(w.lower() for w in words[:count])


def trim_to_words(text, max_words):
    """Drop whole sentences off the end until the text fits inside max_words.

    Sentence granularity, not paragraph: clean() has already folded the narration
    into one continuous block for the speech synthesizer, so paragraph boundaries
    no longer exist by the time a lesson is measured. Never cuts mid-sentence.
    """
    if word_count(text) <= max_words:
        return text
    kept, total = [], 0
    for sentence in split_sentences(text):
        words = word_count(sentence)
        if kept and total + words > max_words:
            break
        kept.append(sentence)
        total += words
    return " ".join(kept) if kept else text


def titles_present_in(text, titles, threshold=0.6):
    """Which of `titles` the text appears to preview, by content-word overlap."""
    low = (text or "").lower()
    hits = []
    for title in titles:
        words = [w for w in re.findall(r"[a-z']+", title.lower())
                 if w not in _TITLE_STOPWORDS and len(w) > 2]
        if len(words) < 2:
            continue
        found = sum(1 for w in words if w in low)
        if found / len(words) >= threshold:
            hits.append(title)
    return hits


# Capitalised words that are not names, so the tic extractor does not ban ordinary
# nouns it happens to meet mid-sentence.
_NOT_NAMES = {
    "i", "i'm", "i'll", "stoic", "stoics", "stoicism", "god", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
    "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "mr", "mrs", "ms", "dr", "st", "lego", "crossfit", "tv",
    "ok", "okay", "english", "roman", "rome", "greek", "marcus", "aurelius", "seneca",
    "epictetus", "hospital", "street", "avenue", "road", "station", "monday's",
    # Openers of quoted speech: split_sentences does not break inside quotation marks,
    # so these land mid-sentence and would otherwise read as names.
    "did", "hey", "hello", "why", "what", "when", "where", "how", "who", "yes", "no",
    "oh", "well", "sorry", "thanks", "look", "listen", "please", "you", "your", "the",
    "can", "are", "there", "this", "that", "just", "maybe", "let", "don't", "it's",
}

# Rare colour words the model reaches for over and over. Plain colours are fine.
COLOUR_WORDS = [
    "teal", "amber", "crimson", "azure", "ochre", "mauve", "taupe", "sepia", "indigo",
    "magenta", "turquoise", "beige", "olive", "charcoal", "slate", "ivory", "lavender",
    "coral", "emerald", "scarlet", "vermilion", "chartreuse", "periwinkle", "russet",
    "umber", "sienna", "cerulean", "puce", "fuchsia",
]

# Replacements for a name the model has leaned on too hard.
NAME_POOL = [
    "Nadia", "Tomas", "Bea", "Rafi", "Ines", "Dov", "Petra", "Kwame", "Liesel",
    "Omar", "Sabine", "Hugo", "Mira", "Yusuf", "Freja", "Tariq", "Noor", "Emil",
]


def proper_names(text):
    """Proper names and how many times each appears.

    Two passes. A word opening a sentence is capitalised by grammar and proves nothing,
    so candidates are only collected from mid-sentence positions. Every occurrence of a
    confirmed candidate is then counted, including the sentence-initial ones, or a name
    that likes to start sentences would be undercounted. Words in _NOT_NAMES, acronyms,
    and words that also occur lowercase in the same text are dropped as ordinary words.
    """
    if not text:
        return {}
    lowercase_elsewhere = {w.lower() for w in re.findall(r"\b[a-z][a-z']+\b", text)}
    candidates = set()
    for sentence in split_sentences(text):
        for word in re.findall(r"\b[A-Za-z][A-Za-z']*\b", sentence)[1:]:
            word = word.split("'")[0]                   # Marion's -> Marion
            if len(word) < 3 or not word[0].isupper() or word.isupper():
                continue
            if word.lower() in _NOT_NAMES or word.lower() in lowercase_elsewhere:
                continue
            candidates.add(word)
    return {name: len(re.findall(rf"\b{re.escape(name)}\b", text)) for name in candidates}


def capitalised_words(text):
    """Every capitalised token, wherever it sits in a sentence.

    Unlike proper_names(), this makes no judgement about what is a name: it exists so a
    substitution can avoid picking a replacement that already appears in the text. A
    name that only ever opens sentences is invisible to proper_names() but shows up here.
    """
    return {w.split("'")[0] for w in re.findall(r"\b[A-Z][A-Za-z']*\b", text or "")
            if len(w) > 2 and w.lower() not in _NOT_NAMES and not w.isupper()}


def colours_used(text):
    """Which of the rare colour words this text uses, and how often."""
    low = (text or "").lower()
    return {c: len(re.findall(rf"\b{c}\b", low)) for c in COLOUR_WORDS
            if re.search(rf"\b{c}\b", low)}


def replace_name(text, old, new):
    """Swap one proper name for another, keeping any possessive form intact."""
    return re.sub(rf"\b{re.escape(old)}\b", new, text)
