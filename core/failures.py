"""What a failure means, and what the owner has to do about it - if anything.

The factory runs unattended, so every way it can stop has to end in one of two places: it
heals itself on a later run, or an email names the one thing a person must do. "The run went
red, go and read the log" is neither, and that is what this module removes: it turns the error
text the pipeline recorded into a kind, a yes/no on whether waiting fixes it, and a sentence
of instruction. notify.py puts that sentence in the email; heartbeat.py uses it when the queue
stops growing.

The patterns are matched against the stage name and the error message together, most specific
first. Anything unrecognised is reported as unknown - explicitly, rather than silently looking
like a handled case.
"""
import re

# (kind, heals_itself, regex, action)
#
# `heals_itself` is the honest question: will doing nothing at all fix this? A spent daily
# allowance heals - tomorrow it is full again. A revoked token does not heal, ever, and the
# longer nobody is told the longer the channel runs dry.
RULES = [
    ("cloudflare-quota", True,
     r"(cloudflare|neuron).{0,80}(quota|limit|exceed|10000|10,000)|"
     r"(quota|limit|exceed).{0,80}(cloudflare|neuron)|cloudflare http 429",
     "Nothing. Cloudflare's 10,000 Neurons a day are spent; the allowance resets and the "
     "next run redraws only the frames that are missing - the ones already paid for are "
     "kept in the run cache."),

    ("youtube-upload-quota", True,
     r"uploadlimitexceeded|quotaexceeded|dailylimitexceeded|exceeded.{0,40}quota",
     "Nothing. YouTube's daily upload allowance is used up. The finished video stays on "
     "disk at 'ready' and the next run uploads it before starting anything new."),

    ("youtube-auth", False,
     r"(token|credential|oauth|refresh).{0,60}(expired|revoked|invalid|not work|failed)|"
     r"invalid_grant|unauthorized_client|401.{0,40}youtube|youtube.{0,40}401",
     "Re-issue the YouTube token: run `python scripts/youtube_auth.py` on your own machine, "
     "then paste the new token.json into the YOUTUBE_TOKEN_JSON repository secret. Nothing "
     "uploads until that is done."),

    ("secret-missing", False,
     r"secrets not set|not set in \.env|api[_ ]?key.{0,30}(missing|not set)",
     "A repository secret is missing or empty. Settings > Secrets and variables > Actions, "
     "and set the one the message names."),

    ("tts-blocked", False,
     r"edge-tts.{0,80}(403|refus|blocked)|403.{0,40}edge-tts",
     "Microsoft is refusing the runner's address for free speech synthesis. This does not "
     "clear by itself: see the README section on running in CI for the fallback voice."),

    ("rate-limited", True,
     r"429|rate.?limit|too many requests",
     "Nothing for now - the service is throttling and the next run tries again. If the same "
     "message arrives three days running, the free tier is too small for one video a day and "
     "the model or the key has to change."),

    ("service-down", True,
     r"\b5\d\d\b|timed? ?out|timeout|connection (reset|refused|aborted)|"
     r"temporarily unavailable|network error|urlerror|bad gateway",
     "Nothing. A service the pipeline uses was unreachable; the next run resumes from the "
     "stage that failed and does not repeat what is already paid for."),

    ("script-quality", True,
     r"outline still breaks|narration is .* minutes|gave up after \d+ scripts|"
     r"too many lessons|lesson \d+ has no title",
     "Nothing. The written script did not pass its own quality gate. The next run writes a "
     "fresh one; the pictures and the voice of a rejected script are thrown away deliberately."),

    ("render", False,
     r"ffmpeg|ffprobe|no such file or directory.{0,40}\.(mp4|mp3|wav)|moov atom",
     "The render failed on the video tool itself rather than on anything external. This one "
     "needs a look at the run log: it is the only failure here that is likely a defect in "
     "the code rather than a service having a bad day."),

    ("no-images", True,
     r"no images were produced",
     "Nothing immediately - both picture services failed for every frame at once, which is "
     "almost always both being down. If it happens twice, check the Cloudflare token."),
]

UNKNOWN_ACTION = ("This failure is not one the factory knows how to classify. Read the run "
                  "log at the link below; if it turns out to be a recurring kind, it belongs "
                  "in core/failures.py so the next email can say what to do.")


def classify(stage, message):
    """(kind, heals_itself, action) for a recorded failure."""
    text = f"{stage or ''}: {message or ''}".lower()
    for kind, heals, pattern, action in RULES:
        if re.search(pattern, text):
            return kind, heals, action
    return "unknown", False, UNKNOWN_ACTION


def action_for(stage, message):
    """Just the sentence of instruction, for putting straight into an email."""
    return classify(stage, message)[2]
