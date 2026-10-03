"""One Cloudflare image call, and a line in logs/cloudflare_window.jsonl saying what came back.

    python scripts/cf_window_probe.py

On 2 October 2026 every call answered HTTP 429 with error 4006 ("you have used up your daily
free allocation of 10,000 neurons") at 06:46 UTC, and the same account generated 70 images
without trouble at 20:50 UTC. The dashboard showed 0 of 10k used. So when, exactly, the
allowance comes back is not what the documentation says (00:00 UTC), and a daily run needs to
know it. Calling this at the same time of day over several days, and reading the file, is how
that gets found out instead of guessed.

The call is the same one the pipeline makes (flux-1-schnell, 2 steps), so a success costs the
usual 96 Neurons and a failure costs nothing. Each line records the UTC time, the HTTP status,
Cloudflare's error code and message if any, and the Neurons the response reports.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import images
from core.config import load_config
from core.llm import get_key

LOG = Path(__file__).resolve().parent.parent / "logs" / "cloudflare_window.jsonl"
PROMPT = "hands holding a plain unmarked parcel, grey winter light, painterly illustration"


def probe():
    cfg = load_config()
    settings = images.cloudflare_settings(cfg)
    url = images.CLOUDFLARE_API_BASE.format(account_id=get_key("CLOUDFLARE_ACCOUNT_ID"),
                                            model=settings["model"])
    body = json.dumps({"prompt": PROMPT, "steps": settings["steps"]}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_key('CLOUDFLARE_API_TOKEN')}",
        "User-Agent": images.USER_AGENT})
    started = time.time()
    record = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "model": settings["model"], "steps": settings["steps"]}
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw, record["http"] = resp.read(), resp.status
    except urllib.error.HTTPError as e:
        raw, record["http"] = e.read(), e.code
    record["seconds"] = round(time.time() - started, 1)
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = {}
    errors = payload.get("errors") or []
    record["error_codes"] = [e.get("code") for e in errors]
    record["error_message"] = (errors[0].get("message") if errors else None)
    record["success"] = bool(payload.get("success")) and bool((payload.get("result") or {}).get("image"))
    record["neurons"] = ((payload.get("result") or {}).get("usage") or {}).get("neurons")
    return record


def main():
    record = probe()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=1, ensure_ascii=False))
    return 0 if record["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
