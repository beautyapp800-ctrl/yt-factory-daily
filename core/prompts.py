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

Your second job is to keep every kind of writing and every mark of ownership out of the
frame. Two different costs: a recognisable trademark in a video is a claim from whoever owns
the mark, and lettering of any kind looks cheap, because the generator renders it as garbled
pseudo-letters.

So NEVER name a thing whose purpose is to carry writing: a sign, a board, a label, a logo,
an emblem, a badge, a document, an invoice, a receipt, a ticket, a timetable, a newspaper, a
magazine, a book or its cover, packaging, a number plate, or a screen with an interface on it.

Describing it as blank or unmarked does not work. An earlier version of these rules allowed
the object and banned only the writing on it - "a plain unmarked parcel", "folded newsprint,
face down". Measured on a finished video: the generator put INVOICE in serif capitals across
the frame drawn from "a modest funeral invoice lying open on a table", and no checker could
read it afterwards to catch it. If the thing is meant to be a document, it will have writing
on it. Choose something else:
  not "a funeral invoice on a table"  but "a candle burned down on a table, chairs pushed back"
  not "a departure board overhead"    but "an empty platform under a long roof, rain blowing in"
  not "a phone showing a message"     but "a phone face down on a nightstand, its glow on the ceiling"
  not "an open book on a desk"        but "a reading lamp left on over an empty desk"

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

# Anything that puts writing or a mark of ownership in the frame. Two different costs: a
# recognisable trademark in a video is a claim from whoever owns the mark, and lettering of
# any kind looks cheap, because the generator renders it as garbled pseudo-text.
#
# This list bans the OBJECT, not only the writing on it, and that is a deliberate reversal.
# An earlier version banned only the lettering words ("label", "logo") and let the object
# through, on the grounds that a box can be drawn unmarked. Measured on video 7: it cannot.
# The frame the generator drew for "a modest funeral invoice lying open on a table" has
# INVOICE across it in serif capitals, and Tesseract cannot read it at any setting tried, so
# no checker downstream will catch it either. A sheet of paper that is meant to be a document
# will have writing on it. The only reliable way not to get the writing is not to ask for the
# document.
IMAGE_BANNED_WORDS = [
    # the writing itself
    "text", "lettering", "letters", "writing on", "written", "handwriting", "printed",
    "inscription", "engraving", "engraved", "caption", "subtitle", "headline", "slogan",
    "word", "words", "number", "numbers", "digit", "digits", "price",
    # things whose whole purpose is to carry writing
    "label", "labelled", "labeled", "sign", "signage", "signpost", "signboard", "billboard",
    "placard", "poster", "banner", "tag", "price tag", "name tag", "sticker", "nameplate",
    "plaque", "notice", "noticeboard", "menu", "receipt", "invoice", "bill", "statement",
    "cheque", "check stub", "payslip", "paycheck", "paystub", "ticket", "boarding pass",
    "timetable", "departure board", "arrivals board", "scoreboard", "number plate",
    "licence plate", "license plate", "banknote", "banknotes", "certificate", "diploma",
    "contract", "document", "letter", "envelope", "postcard", "calendar", "chart", "graph",
    "newspaper", "newsprint", "magazine", "book", "books", "bookshelf", "notebook",
    "journal", "ledger", "diary", "page", "pages", "cover", "spine",
    # marks of ownership
    "logo", "emblem", "crest", "insignia", "monogram", "brand", "branded", "branding",
    "trademark", "badge", "flag", "packaging", "wrapper", "carton", "bottle label",
    # screens with an interface on them
    "interface", "app", "application", "dashboard", "notification", "message on",
    "screen showing", "screen displaying", "displaying", "display showing", "monitor showing",
    "website", "browser", "keyboard", "spreadsheet", "chat", "inbox", "feed",
]

# Words that would otherwise be caught by the list above but are harmless, because they are
# not the thing that carries writing. Checked as whole phrases before the ban is applied.
IMAGE_BANNED_EXCEPTIONS = [
    "screen light", "screen glow", "lamp light", "light on the screen", "face down",
    "turned away", "blank screen", "dark screen", "screen dark", "unlit screen",
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
    literal characters to render (a price, a percentage, "reading ...", "that says ...").

    Phrases in IMAGE_BANNED_EXCEPTIONS are blanked out first, so "screen light on the
    ceiling" is allowed while "screen showing a message" is not.
    """
    import re
    low = (text or "").lower()
    for allowed in IMAGE_BANNED_EXCEPTIONS:
        low = low.replace(allowed, " ")
    hits = [w for w in IMAGE_BANNED_WORDS if re.search(rf"\b{re.escape(w)}\b", low)]
    if _TEXT_CONTENT_PATTERN.search(low):
        hits.append("spells out literal text/numbers")
    return hits


# --- the shape of a shot -----------------------------------------------------------------
#
# Left to itself the generator drifts into one kind of picture. Measured on video 7 after the
# hands were banned: corridors, lifts and counters, with almost no outdoors at all, and the
# mean similarity between frames rose from +0.013 to +0.033. So the kind of shot is decided
# here, in code, and the model is told which one to write.
SHOT_TYPES = ("landscape", "interior", "object", "light")

# Nine shots, three of them outdoors, and no two neighbours of the same kind - including
# across the wrap, so the pattern can repeat for as many scenes as a video has.
SHOT_CYCLE = ("landscape", "interior", "object",
              "landscape", "light", "interior",
              "landscape", "object", "light")

SHOT_TYPE_BRIEF = {
    "landscape": ("an outdoor view, open air, no room around it: a street in rain, a station "
                  "platform, a field at dusk, rooftops, a river, a road. If a person is in it "
                  "they are far away and small."),
    "interior": ("the inside of a room or building, empty or with one distant silhouette: a "
                 "stairwell, a kitchen at night, a waiting room, a hall, a shop after closing."),
    "object": ("one or two things resting somewhere, seen from a normal distance - not a "
               "macro shot: a coat over a chair, a cup going cold, a key on a sill."),
    "light": ("light itself as the subject: a shaft across a floor, a lamp in a dark room, "
              "headlights through a window, the shadow of a railing, dawn on a wall."),
}

# A prompt that calls itself a landscape has to show some sign of being outdoors. Without
# this the model labels a corridor "landscape" and the rotation means nothing.
_OUTDOOR_CUES = [
    "sky", "skyline", "horizon", "clouds", "cloud", "rain", "snow", "fog", "mist", "wind",
    "street", "road", "pavement", "alley", "courtyard", "square", "park", "field", "meadow",
    "hill", "hills", "mountain", "mountains", "river", "canal", "sea", "shore", "coast",
    "beach", "lake", "forest", "trees", "tree", "garden", "rooftops", "roof", "bridge",
    "platform", "station", "tracks", "railway", "city", "town", "village", "outdoors",
    "outside", "dusk", "dawn", "sunset", "sunrise", "moonlight", "streetlight", "traffic",
    "car park", "bus stop", "harbour", "dock", "farmland", "valley", "path", "lane",
]


def shot_type_for(index):
    """The kind of shot the scene at this position must be."""
    return SHOT_CYCLE[index % len(SHOT_CYCLE)]


def looks_outdoors(text):
    """Whether a prompt shows any sign of being out of doors."""
    import re
    low = (text or "").lower()
    return any(re.search(rf"\b{re.escape(cue)}\b", low) for cue in _OUTDOOR_CUES)


def shot_type_problem(prompt, wanted):
    """None if the prompt fits the kind of shot it was asked for, else a short reason."""
    if wanted not in SHOT_TYPES:
        return None
    if wanted == "landscape" and not looks_outdoors(prompt):
        return "it was asked for an outdoor view but nothing in it is outdoors"
    if wanted != "landscape" and prompt and looks_outdoors(prompt) and wanted == "interior":
        return None            # a window onto a street is still an interior
    return None


def text_free_request(original_prompt, hits):
    """Ask for the same moment with nothing in it that could carry writing at all.

    The last resort, after redrawing the same prompt has failed to produce a frame without
    lettering. At that point the subject itself is the problem - a document will have writing
    on it however it is described - so the subject has to go.
    """
    return f"""An image generator keeps drawing garbled text into this picture:

"{original_prompt}"

{("It names " + ", ".join(hits) + ". ") if hits else ""}Redrawing it has not helped, so the
subject itself has to change: anything that normally carries writing - paper, a document, a
screen with an interface, a sign, a board, a book, packaging - will be drawn with lettering
on it whatever the prompt says.

Rewrite it for the SAME place and the SAME moment, 15 to 30 words, English, using ONLY
things that never carry writing: furniture, cloth, stone, metal, glass, water, plants, food,
weather, light and shadow. No paper, no screens, no signs, no packaging, no books. Keep it
specific to this scene rather than retreating to a generic view.

Reply with the rewritten prompt only, no quotes, no JSON, no explanation."""


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
