"""System prompts, the channel's style rules and the lesson-form spec."""

# Every narration request carries these rules. They are the channel's voice.
STYLE_RULES = """You write narration for a YouTube channel about Stoicism and practical philosophy.
The narrator is calm and direct. The viewer is an adult looking for something solid to stand on.

Hard rules, no exceptions:
- Second person, present tense. Short sentences. Plain words.
- No markdown. No headings, no bullet points, no numbering, no bold, no asterisks.
- No stage directions and no remarks in parentheses. Continuous prose only, because a speech synthesizer reads it aloud.
- Never invent a quote from Marcus Aurelius, Seneca, Epictetus or anyone else. Either use a well known translation, or put the idea in your own words with no quotation marks at all.
- No medical or psychiatric advice. Never promise to cure anxiety, depression or any condition.
- Never start a sentence with "In conclusion", "Remember that" or "It is important to".
- No academic tone, no hedging, no throat clearing. Concrete situations, not abstractions.
- Write only the narration itself. No preamble, no title, no sign-off about what you just wrote."""

NARRATION_SYSTEM = STYLE_RULES

OUTLINE_SYSTEM = f"""{STYLE_RULES}

You are planning a video. Reply with valid JSON only."""

IMAGE_SYSTEM = """You write prompts for an image generator that illustrates a Stoicism video.
The generator renders any text, label or logo you name as garbled pseudo-text, because
the model you are writing for barely follows negative instructions - "no text" in the
prompt does not stop it from drawing a label once you have described a box. The only
reliable fix is to never put a text-bearing object in the description at all.

Rules for every prompt:
- English, 15 to 30 words, describing one concrete scene, object or place.
- No close-up faces and no real historical people. Use silhouettes, hands, backs turned, landscapes, objects, architecture, weather.
- Never describe any of these, because each one invites rendered text or a recognisable
  brand: a phone, laptop, tablet, monitor or TV screen; a box, package, bottle, can,
  label or sign; a book, newspaper, document or poster; printed clothing or a logo on
  an object.
- Instead reach for things that carry no text by nature: landscapes, weather, light
  through a window, architecture, stone, wood grain, fabric, water, empty rooms,
  stairs, doorways, paths, plain furniture, unmarked dishes, plants, empty hands.
- Describe what is visible, not what it means. No abstractions like "the concept of time".
- Do not add a style description; the pipeline appends one.

Reply with valid JSON only."""

# Concrete nouns that reliably invite rendered pseudo-text or a recognisable brand once
# an image model draws them. Checked against every image_prompt before it is sent to a
# provider; a hit means the prompt gets rewritten rather than used as-is. "tv" is
# matched as a whole word like the rest, via the word-boundary regex in
# find_banned_image_words, not a special case.
IMAGE_BANNED_WORDS = [
    "laptop", "notebook computer", "screen", "monitor", "television", "tv",
    "tablet", "phone", "smartphone", "ipad", "iphone",
    "box", "package", "packaging", "parcel", "bottle", "can", "jar", "label", "sign",
    "signage", "billboard", "placard", "tag", "price tag", "name tag", "sticker",
    "book", "newspaper", "magazine", "document", "paper", "letter", "poster", "flyer",
    "logo", "brand", "branded", "print", "printed",
]

# A prompt that spells out a literal amount or line of text ("a price tag reading
# $1,299", "a sign that says OPEN") is telling the model exactly which characters to
# render, which IMAGE_BANNED_WORDS alone does not catch when neither "price tag" nor
# a words-to-render verb happens to be literally present. Caught live: scene 12 of
# video 7 read "A close-up of a price tag reading $1,299..." - "reading" plus a
# currency figure, no banned noun in sight.
_TEXT_CONTENT_PATTERN = __import__("re").compile(
    r"\$[\d,]+|reading [\"']?\w|that says|that read[s]?|written in|"
    r"the words? [\"']|says [\"']")


def find_banned_image_words(text):
    """Which IMAGE_BANNED_WORDS (if any) appear in this prompt, as whole words/phrases,
    plus a synthetic "spells out text/numbers" hit when the prompt itself dictates
    literal characters to render (a price, a percentage, "reading ...", "that says ...")."""
    import re
    low = (text or "").lower()
    hits = [w for w in IMAGE_BANNED_WORDS if re.search(rf"\b{re.escape(w)}\b", low)]
    if _TEXT_CONTENT_PATTERN.search(low):
        hits.append("spells out literal text/numbers")
    return hits


def image_prompt_fix_request(original_prompt, banned_hits):
    """Ask the model to redraw the same moment without the flagged nouns."""
    return f"""This image prompt describes something that will render as garbled text or a
recognisable brand once generated:

"{original_prompt}"

The problem: it names {", ".join(banned_hits)}, which an image model cannot draw
without inventing fake lettering or logos on it.

Rewrite it as a different concrete image for the same moment and mood, 15 to 30 words,
English, that uses none of: phones, laptops, tablets, screens, boxes, packaging,
bottles, cans, labels, signs, books, newspapers, documents, posters, logos or printed
clothing. Reach for landscapes, light, architecture, stone, wood, fabric, water, empty
rooms, stairs, doorways, paths, plain furniture, unmarked objects, plants or empty
hands instead. No style description; that is appended separately.

Reply with the rewritten prompt only, no quotes, no JSON, no explanation."""

# Distinct places for the opening situation of each lesson, so ten lessons do not
# all open in a kitchen in the morning. One is assigned per lesson.
SCENE_SETTINGS = [
    "a queue in a pharmacy",
    "a conversation inside a parked car",
    "a hospital corridor",
    "a text exchange at two in the morning",
    "a gym, between sets",
    "a funeral, or the hour after one",
    "a job interview",
    "a child's room",
    "a railway station",
    "a shop on the way home from work",
    "a stairwell outside a flat",
    "a kitchen late at night, everyone else asleep",
    "the driver's seat before going inside",
    "a waiting room with a muted television",
    "a lift with one other person in it",
]

# Words that drag a title into time-management territory. Enforced in code, not
# only asked for, because the model reaches for them by default.
TITLE_BANNED_WORDS = [
    "log", "track", "tracker", "tracking", "benchmark", "review", "reset",
    "system", "routine", "schedule", "optimize", "optimise", "framework",
    "checklist", "habit stack", "habits", "hack", "productivity", "workflow",
    "steps", "tips", "tricks", "guide",
]

# The shape every lesson has to take. This is what separates the channel from a
# self-help listicle: the viewer is asked to subtract, never to add.
# Titles the model must never hand back: they are in LESSON_FORM_RULES as a
# demonstration of register, and videos 4 and 5 both returned six of them verbatim as
# their actual lessons. Enforced in code, because asking was not enough.
EXAMPLE_TITLES = [
    "Stop Explaining Your Choices to People Who Are Not Listening",
    "Quit Auditioning for an Audience That Left",
    "Stop Rehearsing Speeches You Will Never Give",
    "Let Go of the Prize Nobody Promised You",
    "Stop Defending a Door You Already Walked Through",
    "Drop the Score You Keep Alone",
]

# The shape every lesson has to take. This is what separates the channel from a
# self-help listicle: the viewer is asked to subtract, never to add.
LESSON_FORM_RULES = """Every lesson must be a REFUSAL or a SUBTRACTION. Each one is one of exactly three forms:
  (a) something to stop doing,
  (b) something to stop explaining, proving or justifying,
  (c) something to let go of.

Never a new habit, practice, tool, system or routine to take up. The viewer already
has too much. You are taking something away from them, not handing them another task.

Titles must be an order or a flat statement, 4 to 8 words, no colon, no numbering.

The six titles below come from a DIFFERENT video on an unrelated subject. They are here
only to show the register. Reusing any of them, or a light rewording of one, is the
single worst thing you can do here, and the answer will be thrown away. Every title you
write must name something specific to the subject of THIS video:
""" + chr(10).join(f"  {t}" for t in EXAMPLE_TITLES) + """

Four titles that are wrong, and why:
  "Use a Daily Thought Log" is wrong: it hands the viewer a new tool, and it is a
    productivity instrument rather than something given up.
  "Review and Reset Weekly" is wrong: it is a schedule, and it names nothing specific
    to stop doing.
  "Build a Resilience Framework" is wrong: it is jargon, and it asks the viewer to
    construct a system instead of dropping something.
  "Practice Gratitude Every Morning" is wrong: it is a routine to add, and it could
    open any self-help video ever made."""


# What keeps the prose from reading like advice written for nobody in particular.
CONCRETE_DETAIL_RULE = """One detail in this lesson must be specific enough that it could not appear in
generic advice. Name the object, name the exact words someone said, or say what your
hands are doing while you feel it. Not a category of person but a particular one.

Invent that detail fresh for this lesson. Anything written in these instructions is an
illustration of how specific to be, never material to copy: never write about a
colleague who copies you into emails, and never reuse a detail, a name or an object
from a lesson already written above.

Do not open by stating a clock time. A precise hour in the first sentence is its own
formula, as tired as opening in a kitchen. Let the place and what is happening carry it."""


# Which life area a lesson belongs to. Used to reject a plan that piles most of its
# lessons into one area: seed "control" once produced five lessons about phones.
DOMAINS = {
    "digital habits": [
        "phone", "screen", "social media", "instagram", "facebook", "tiktok", "feed",
        "scroll", "scrolling", "notification", "text", "text message", "texting", "texts", "replying", "reply",
        "email", "emails", "inbox", "app", "apps", "online", "post", "posting",
        "likes", "followers", "tablet", "laptop", "message", "messages", "dm",
    ],
    "work": ["work", "job", "boss", "colleague", "coworker", "career", "office",
             "meeting", "promotion", "deadline", "interview", "shift", "client"],
    "family": ["family", "mother", "father", "parent", "parents", "child", "children",
               "son", "daughter", "sibling", "brother", "sister", "spouse", "partner",
               "marriage", "husband", "wife"],
    "money": ["money", "salary", "wage", "debt", "rent", "spending", "savings",
              "afford", "expensive", "cost", "price", "budget", "bill", "bills"],
    "the body": ["body", "sleep", "sleeping", "eating", "food", "exercise", "gym",
                 "tired", "exhaustion", "illness", "pain", "breath", "breathing"],
    "friendship": ["friend", "friends", "friendship", "neighbour", "neighbor",
                   "stranger", "strangers", "acquaintance", "company", "invitation"],
    "the past": ["past", "memory", "memories", "regret", "regrets", "mistake",
                 "mistakes", "years ago", "childhood", "used to", "history", "grudge"],
}
MAX_LESSONS_PER_DOMAIN = 4


def domain_of(text):
    """The life area a lesson sits in, by keyword hits. None when nothing matches."""
    low = f" {(text or '').lower()} "
    best, best_hits = None, 0
    for domain, words in DOMAINS.items():
        hits = sum(1 for w in words if f" {w} " in low or f" {w}," in low or f" {w}." in low)
        if hits > best_hits:
            best, best_hits = domain, hits
    return best


def forbidden_tics_rule(names, colours):
    """Prompt fragment listing the words earlier videos already leaned on."""
    if not names and not colours:
        return ""
    parts = []
    if names:
        parts.append("these names, every one of which a recent video already used: "
                     + ", ".join(sorted(names)))
    if colours:
        parts.append("these colour words, worn out by recent videos: "
                     + ", ".join(sorted(colours)))
    return ("\n\nDo not use " + "; nor ".join(parts) +
            ".\nPick different names and plainer colours. If you need a name, choose one "
            "that is not in that list.")
