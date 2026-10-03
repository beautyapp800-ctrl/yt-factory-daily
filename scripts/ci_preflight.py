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

    try:
        from core import tts
        from core.config import load_config
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            seconds = tts.synthesize("This is a short test of the narrator's voice.",
                                     Path(tmp) / "preflight.wav", cfg)
        print(f"edge-tts: spoke {seconds:.1f}s of audio")
    except Exception as e:                                       # noqa: BLE001
        print(f"edge-tts: FAILED: {e}")
        problems.append(f"edge-tts cannot speak from this machine: {e}. If this is a 403, "
                        "Microsoft is refusing the runner's address; see the README, section "
                        "on running in CI, for the fallback.")

    try:
        from core import youtube
        creds = youtube.load_credentials()
        print(f"youtube token: loads and refreshes (valid={creds.valid})")
    except Exception as e:                                       # noqa: BLE001
        print(f"youtube token: FAILED: {e}")
        problems.append(f"the YouTube token does not work: {e}")

    if problems:
        print("\nPREFLIGHT FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\npreflight ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
