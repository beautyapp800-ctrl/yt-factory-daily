"""Checks a CI run can succeed, before it spends anything.

    python scripts/ci_preflight.py

Runs first in the workflow and exits non-zero with a plain explanation if something that would
fail the run later is already wrong. Each check is here because skipping it costs money or a
day:

- every secret the pipeline needs is set. A missing key otherwise surfaces three stages in,
  after the script has already used up Groq tokens.
- ffmpeg, ffprobe and tesseract are on the PATH.
- edge-tts can actually speak from this machine. Microsoft's free endpoint is known to refuse
  some datacentre addresses with a 403 (see core/tts.py), and a GitHub runner is a datacentre
  address. This is the one genuine unknown about running in CI, and finding it out on the
  first sentence is better than finding it out after the images are drawn.
- the YouTube token loads and refreshes, which costs no quota.

It never prints a secret, only whether it is set.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm import get_key

REQUIRED = ["GROQ_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "YOUTUBE_TOKEN_JSON"]

# Where the findings are left for the step that writes the email. Empty file = all checks passed.
PROBLEMS_PATH = Path(__file__).resolve().parent.parent / "preflight-problems.txt"


def main():
    problems = []

    missing = [name for name in REQUIRED if not get_key(name)]
    print("secrets:", ", ".join(f"{n}={'set' if n not in missing else 'MISSING'}" for n in REQUIRED))
    if missing:
        problems.append(f"secrets not set in this workflow: {', '.join(missing)}")

    for tool in ("ffmpeg", "ffprobe", "tesseract"):
        found = shutil.which(tool)
        print(f"{tool}: {found or 'MISSING'}")
        if not found:
            problems.append(f"{tool} is not installed on this runner")

    # Name the provider that is actually configured. This message used to say
    # "edge-tts ... if this is a 403, Microsoft is refusing the runner" no matter
    # what, so a missing piper module read as a Microsoft block and sent the
    # reader hunting the wrong fault entirely.
    provider = "unknown"
    try:
        from core import tts
        from core.config import load_config
        cfg = load_config()
        provider = (cfg.get("tts", {}) or {}).get("provider", "unknown")
        with tempfile.TemporaryDirectory() as tmp:
            seconds = tts.synthesize("This is a short test of the narrator's voice.",
                                     Path(tmp) / "preflight.wav", cfg)
        print(f"tts[{provider}]: spoke {seconds:.1f}s of audio")
    except Exception as e:                                       # noqa: BLE001
        print(f"tts[{provider}]: FAILED: {e}")
        hint = ""
        if provider == "edge":
            hint = (" If this is a 403, Microsoft is refusing the runner's address; "
                    "see the README, section on running in CI, for the fallback.")
        elif provider == "piper":
            hint = (" piper needs both the package (requirements-piper.txt) and the "
                    "voice model in assets/voices/, which is gitignored and must be "
                    "downloaded by the workflow.")
        problems.append(f"the configured TTS provider '{provider}' cannot speak "
                        f"from this machine: {e}.{hint}")

    # The fallback is only a fallback if it answers. It sat unconfigured and
    # pointed at a model id that 404s, so the day Groq ran dry there was nothing
    # behind it - and that is a failure nobody sees until it is the only thing
    # left.
    try:
        from core import llm
        if not llm.get_key("GEMINI_API_KEY"):
            print("gemini fallback: NOT CONFIGURED (no GEMINI_API_KEY)")
            problems.append("no Gemini fallback: GROQ is the only provider, so a "
                            "Groq outage or exhausted daily quota stops the channel")
        else:
            reply = llm._call_gemini("Reply with OK.", None, 64, 0.0, False)
            print(f"gemini fallback: answers ({reply.strip()[:20]})")
    except Exception as e:                                       # noqa: BLE001
        print(f"gemini fallback: FAILED: {e}")
        problems.append(f"the Gemini fallback does not answer: {e}")

    try:
        from core import youtube
        creds = youtube.load_credentials()
        print(f"youtube token: loads and refreshes (valid={creds.valid})")
    except Exception as e:                                       # noqa: BLE001
        print(f"youtube token: FAILED: {e}")
        problems.append(f"the YouTube token does not work: {e}")

    # The email is written by a later step, after this process has exited, and a workflow log
    # is not something the owner should have to open. Leaving the findings on disk is what
    # lets notify.py say which check failed instead of listing everything it could have been.
    try:
        PROBLEMS_PATH.write_text("\n".join(problems), encoding="utf-8")
    except OSError as e:                                         # noqa: BLE001
        print(f"(could not write {PROBLEMS_PATH}: {e})")

    if problems:
        print("\nPREFLIGHT FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\npreflight ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
