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

Your first job is SPECIFICITY. The image must show the particular situation the scene
describes, not its general mood. A scene about a man checking his phone in bed at two in
the morning is not illustrated by "a quiet landscape at dawn" - that could sit under any
scene in any video. Name one action or one object that the scene's own text mentions, and
make it the thing the viewer sees.

Your second job is to keep rendered text out of the frame, and the way to do that is NOT
to avoid the object. Earlier versions banned every text-bearing noun outright and the
results drifted into interchangeable scenery - safe, generic, illustrating nothing. Show
the object WITHOUT its writing instead:
  not "a box with a label"        but "hands holding a plain unmarked parcel"
  not "a laptop on a desk"        but "screen light on a face in a dark room, the screen itself out of frame"
  not "a shop sign above a door"  but "a lit doorway at dusk, the lettering lost in glare"
  not "a newspaper on a table"    but "folded newsprint, face down, a cold mug beside it"
The generator will draw garbled pseudo-text on any surface you tell it carries writing,
so keep writing off-frame, turned away, out of focus or lost in light - but keep the
object, because the object is what ties the picture to the scene.

Your third job is to keep human anatomy out of reach of the generator. It draws hands with
six fingers and three fingers, and faces that are almost right and therefore worse than
wrong. Nothing downstream can repair that, so the only fix is not to ask for it:

NEVER write hands, fingers, a palm, a grip, a gesture, or anyone holding, touching,
reaching, gripping, pressing, wiping or clasping anything. NEVER write a face, a portrait,
an expression, eyes, a close-up of a person, or any view closer than the whole figure.

A person may appear only where the generator cannot get them wrong:
  - a silhouette against a window, a lamp, a doorway
  - seen from behind, walking away, shoulders and coat
  - far enough away that a hand is a few pixels: across a platform, down a corridor
  - implied and absent: the chair they left, the coat on the hook, the light they turned on

This channel is about Stoicism, and Stoicism does not need a close-up. It needs the room
after the argument, the platform in the rain, the window at four in the morning. Most
prompts should be a place, an object or the light in a room, with no person in them at all.

When the scene's own sentence IS about the hands - signing, paying, reaching for a phone -
build the picture from everything around them instead, and let the act be understood:
  not "a hand reaching for a phone on the nightstand"
      but "a phone face down on a nightstand, screen light on the ceiling, the bed empty"
  not "fingers gripping a cold metal railing"
      but "a worn brass railing in an empty stairwell, winter light along it"
  not "hands counting banknotes at a kitchen table"
      but "a kitchen table at night, notes and coins left in a loose pile, one chair pushed back"
  not "a hand pressing the lift button"
      but "a lift door closing on an empty marble lobby, the floor indicator lit"

Rules for every prompt:
- English, 15 to 30 words, describing one concrete moment.
- Name one action or object taken from the scene's own text. That element must be visible.
- No hands, no fingers, no faces, no portraits, no real historical people. A person, if any,
  is a silhouette, a back, or a distant figure.
- Describe what is visible, not what it means. No abstractions like "the concept of time".
- Do not add a style description; the pipeline appends one.

Reply with valid JSON only."""


# Words that put a hand or a face in frame. The generator is bad at both in a way nothing
# downstream can fix - six fingers, three fingers, a face that is almost right - so these are
# refused in code rather than only asked against. Checked as whole words, so "handle",
# "handful", "beforehand" and "surface" are not hits.
#
# "arm", "shoulder" and "silhouette" are deliberately NOT here: a figure at a distance or in
# outline is what the channel wants, and the generator renders it well.
IMAGE_ANATOMY_WORDS = [
    # the hand itself
    "hand", "hands", "finger", "fingers", "fingertip", "fingertips", "thumb", "thumbs",
    "palm", "palms", "knuckle", "knuckles", "fist", "fists", "wrist", "wrists",
    "fingernail", "fingernails",
    # what a hand is doing
    "holding", "holds", "held", "gripping", "grips", "grip", "clutching", "clasping",
    "clasped", "touching", "touches", "reaching", "reaches", "grasping", "grabbing",
    "pressing", "presses", "typing", "scrolling", "wiping", "tracing", "pointing",
    "gesture", "gestures", "gesturing", "handshake", "signing", "writing with",
    "picking up", "putting down", "turning a page", "counting out",
    # the face
    "face", "faces", "facial", "portrait", "close-up of a man", "close-up of a woman",
    "closeup", "close-up face", "eyes", "eye contact", "gaze", "stare", "staring at the camera",
    "expression", "smile", "smiling", "frown", "frowning", "tears", "crying",
    "mouth", "lips", "cheek", "cheeks", "forehead", "chin", "nose",
    # framing that brings either one close
    "close-up", "closeup", "macro", "extreme close", "head and shoulders", "bust shot",
    "selfie",
]

# Words that ARE the writing rather than the object carrying it. A box can be drawn
# unmarked and a laptop can be drawn as light on a face, but a "label" or a "logo" has
# nothing left once you take the lettering away, so naming one guarantees pseudo-text.
#
# This list used to be far longer and banned the objects themselves - phone, laptop,
# box, bottle, book, paper. It worked: no more fake lettering. It also pushed the model
# into interchangeable scenery, landscapes and doorways that illustrated no particular
# scene, which cost more than the pseudo-text did. Objects are allowed again; only the
# writing is not. See IMAGE_SYSTEM for how to show the object without its text.
IMAGE_BANNED_WORDS = [
    "label", "labelled", "labeled", "sign", "signage", "signpost", "billboard",
    "placard", "poster", "banner", "tag", "price tag", "name tag", "sticker",
    "logo", "brand", "branded", "headline", "caption", "slogan", "nameplate",
    "inscription", "engraving", "lettering", "writing on", "text on",
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


def find_banned_anatomy_words(text):
    """Which IMAGE_ANATOMY_WORDS appear in this prompt, as whole words or phrases.

    Whole words matter here: "handle", "handful", "beforehand", "surface" and "household"
    all contain "hand" and none of them puts a hand in the frame.
    """
    import re
    low = (text or "").lower()
    return [w for w in IMAGE_ANATOMY_WORDS if re.search(rf"\b{re.escape(w)}\b", low)]


def anatomy_fix_request(original_prompt, hits):
    """Ask for the same moment, built from what is around the hand instead of the hand."""
    return f"""This image prompt puts human hands or a face in the frame:

"{original_prompt}"

The problem: {", ".join(hits)}. An image generator draws hands with six or three fingers and
faces that are almost right, and nothing later in the pipeline can repair either, so the
picture has to be built without them.

Rewrite it for the SAME moment and the SAME place, 15 to 30 words, English, showing what is
AROUND the action instead of the body performing it - the object left behind, the empty
chair, the room, the light, the view. A person may appear only as a silhouette, from behind,
or far enough away that a hand is a few pixels. Keep it specific to this scene: a generic
landscape that could sit under any scene is not an improvement.

Reply with the rewritten prompt only, no quotes, no JSON, no explanation."""


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
    """Ask the model to keep the same object and moment, minus the writing on it."""
    return f"""This image prompt names writing that an image generator will render as
garbled pseudo-text:

"{original_prompt}"

The problem: {", ".join(banned_hits)}. Those are the lettering itself, not the thing
carrying it.

Rewrite it for the SAME moment and the SAME object, 15 to 30 words, English, showing
that object without any writing on it - unmarked, turned away, face down, out of focus,
or with the lettering lost in glare. Do not replace the object with scenery: a quiet
landscape would illustrate nothing in particular, and keeping the object is the whole
point. No style description; that is appended separately.

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
