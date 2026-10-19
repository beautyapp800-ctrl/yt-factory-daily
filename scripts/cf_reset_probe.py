"""When does Cloudflare's daily Neuron allowance actually reset?

    python scripts/cf_reset_probe.py [max_minutes]

Asks for one tiny image (1 step, about 60 Neurons) every minute until one is delivered, and
prints the UTC time of the last refusal and of the first success. A refused call costs
nothing, so this is safe to leave running while the allowance is spent. Used on 19 October
2026 after Cloudflare refused a cover at 06:44 and again at 14:41 UTC, which fits neither the
documented 00:00 UTC nor Pacific midnight (07:00 UTC); see core/neurons.py.
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import images as image_api
from core.config import OUTPUT_DIR

CFG = {"images": {"cloudflare": {"model": "@cf/black-forest-labs/flux-1-schnell", "steps": 1}}}


def main():
    limit = float(sys.argv[1]) if len(sys.argv) > 1 else 240
    out = OUTPUT_DIR / "_probe" / "reset.jpg"
    started, last_refused = time.monotonic(), None
    while time.monotonic() - started < limit * 60:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            _, used = image_api.synthesize_cloudflare("a grey stone", out, 0, CFG)
        except image_api.ImageError as e:
            refused = "daily free allocation" in str(e) or "429" in str(e)
            print(f"{now} UTC refused ({'allowance' if refused else str(e)[:120]})", flush=True)
            last_refused = now if refused else last_refused
            time.sleep(60)
            continue
        print(f"{now} UTC DELIVERED ({used} Neurons); last refusal {last_refused}", flush=True)
        return 0
    print("still refused when the time limit ran out; last refusal", last_refused)
    return 1


if __name__ == "__main__":
    sys.exit(main())
