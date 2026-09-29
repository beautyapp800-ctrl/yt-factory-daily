"""Narration text handling: cleanup, word counting, sentence and scene splitting."""
import re

# Characters that mean the model slipped into markdown instead of plain narration.
MARKDOWN_CHARS = "*#`_[]|<>"

_ABBREVIATIONS = ("Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sr", "Jr", "vs", "etc", "e.g", "i.e")
_BANNED_OPENERS = ("in conclusion", "remember that", "it is important to")


def clean(text):
    """Strip markdown, parentheticals and stray labels so a TTS engine reads it cleanly."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
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
    text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
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
