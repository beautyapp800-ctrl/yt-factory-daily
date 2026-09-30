"""Image generation: two providers.

pollinations.ai is primary. As of writing it no longer matches its old reputation as
"free, no key, just rate-limited": anonymous requests now intermittently answer with
HTTP 402 (payment required, an x402 crypto-payment scheme), seemingly tied to load or
which parameters are set (the `seed` parameter alone was enough to trigger it in
testing), not a documented, stable free quota. It still succeeds often enough to be
worth trying first and free when it does. A POLLINATIONS_TOKEN in .env, if the
account has one, is sent as a bearer token, which is a best-effort guess: the current
docs could not be fetched to confirm the exact header.

pexels is the fallback for stock photography, used only once pollinations has failed
three attempts for a given prompt. PEXELS_API_KEY from .env; if that key is absent,
pexels is simply unavailable, not an error.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from core.llm import get_key
from core.logger import get_logger
from core.retry import retry

log = get_logger("images")

USER_AGENT = "yt-factory/1.0"

POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/{prompt}"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_MODEL = "flux"
DEFAULT_DELAY_S = 16
POLLINATIONS_ATTEMPTS = 3

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
# Pexels' own documented free-tier caps: 200 requests/hour, 20000/month. Not enforced
# here (Pexels enforces it server-side and answers 429), just for anyone reading this.
PEXELS_REQUESTS_PER_HOUR = 200
PEXELS_REQUESTS_PER_MONTH = 20000


class ImageError(Exception):
    pass


# --- pollinations --------------------------------------------------------------

_last_pollinations_call = 0.0


def _pollinations_wait(delay_s):
    """Block until at least delay_s has passed since the last pollinations request."""
    global _last_pollinations_call
    now = time.monotonic()
    wait = delay_s - (now - _last_pollinations_call)
    if wait > 0:
        time.sleep(wait)
    _last_pollinations_call = time.monotonic()


def pollinations_settings(cfg):
    block = ((cfg.get("images") or {}).get("pollinations")) or {}
    return {
        "width": block.get("width", DEFAULT_WIDTH),
        "height": block.get("height", DEFAULT_HEIGHT),
        "model": block.get("model", DEFAULT_MODEL),
        "delay_s": block.get("delay_s", DEFAULT_DELAY_S),
    }


def synthesize_pollinations(prompt, out_path, seed, cfg):
    """One image from pollinations.ai. Waits out the configured delay first."""
    settings = pollinations_settings(cfg)
    _pollinations_wait(settings["delay_s"])

    params = {"width": settings["width"], "height": settings["height"],
             "model": settings["model"], "nologo": "true", "seed": seed}
    url = POLLINATIONS_BASE.format(prompt=urllib.parse.quote(prompt)) + \
        "?" + urllib.parse.urlencode(params)

    headers = {"User-Agent": USER_AGENT}
    token = get_key("POLLINATIONS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        body = e.read()[:200]
        if e.code == 402:
            raise ImageError(
                "pollinations answered 402 Payment Required. Anonymous access is no "
                "longer reliably free; get a token at enter.pollinations.ai and set "
                "POLLINATIONS_TOKEN in .env, or rely on the pexels fallback.") from e
        raise ImageError(f"pollinations HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise ImageError(f"pollinations network error: {e.reason}") from e

    if not content_type.startswith("image/") or len(data) < 1000:
        raise ImageError(f"pollinations returned something that is not an image "
                         f"({content_type}, {len(data)} bytes)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path


# --- pexels ----------------------------------------------------------------------

def pexels_available():
    return get_key("PEXELS_API_KEY") is not None


def synthesize_pexels(prompt, out_path, cfg):
    """One stock photo from Pexels matching prompt. Raises if no key or no match."""
    key = get_key("PEXELS_API_KEY")
    if not key:
        raise ImageError("PEXELS_API_KEY is not set; pexels is unavailable")

    settings = (cfg.get("images") or {}).get("pexels") or {}
    per_page = settings.get("per_page", 1)
    # Pexels searches literal keywords, not a scene-illustration prompt; the style
    # suffix core/prompts.py appends ("cinematic painterly illustration...") would
    # just pollute the search, so only the descriptive part before it is used.
    query = prompt.split(",")[0].strip() or prompt
    params = urllib.parse.urlencode(
        {"query": query, "orientation": "landscape", "per_page": per_page})
    req = urllib.request.Request(f"{PEXELS_SEARCH_URL}?{params}",
                                 headers={"Authorization": key, "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise ImageError("pexels rate limit hit (200/hour, 20000/month)") from e
        raise ImageError(f"pexels HTTP {e.code}: {e.read()[:200]}") from e
    except urllib.error.URLError as e:
        raise ImageError(f"pexels network error: {e.reason}") from e

    photos = data.get("photos") or []
    if not photos:
        raise ImageError(f"pexels found no photo for: {query[:80]!r}")

    image_url = photos[0]["src"]["large2x"]
    img_req = urllib.request.Request(image_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(img_req, timeout=60) as resp:
            data = resp.read()
    except urllib.error.URLError as e:
        raise ImageError(f"pexels image download failed: {e.reason}") from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path, photos[0].get("photographer")


# --- combined ----------------------------------------------------------------

def images_per_scene(duration_s, images_per_seconds):
    return max(1, round(duration_s / images_per_seconds))


def synthesize(prompt, out_path, seed, cfg):
    """Pollinations first (POLLINATIONS_ATTEMPTS tries), pexels once it has failed
    that many times and a key is available. Raises ImageError only once nothing
    worked, naming the last error from whichever provider was tried last."""
    provider = (cfg.get("images") or {}).get("provider", "pollinations")
    last_error = None

    if provider in ("pollinations", "auto"):
        for attempt in range(1, POLLINATIONS_ATTEMPTS + 1):
            try:
                synthesize_pollinations(prompt, out_path, seed, cfg)
                return "pollinations"
            except ImageError as e:
                last_error = e
                log.warning("pollinations attempt %d/%d failed: %s",
                           attempt, POLLINATIONS_ATTEMPTS, e)

    if pexels_available():
        try:
            path, photographer = synthesize_pexels(prompt, out_path, cfg)
            log.info("pexels photo by %s", photographer or "unknown")
            return "pexels"
        except ImageError as e:
            last_error = e
            log.warning("pexels failed: %s", e)
    elif provider == "pollinations":
        log.warning("pexels unavailable (no PEXELS_API_KEY); nothing left to try")

    raise ImageError(str(last_error) if last_error else "no image provider available")
