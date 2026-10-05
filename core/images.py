"""Image generation: Cloudflare Workers AI primary, pollinations.ai fallback.

Cloudflare Workers AI (flux-1-schnell) needs a free Cloudflare account, no credit
card, plus CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env. The free plan
gives every account 10000 Neurons/day, resetting daily - see
pipeline/images.py's live per-image measurement for whether that actually covers
a full video, since Cloudflare's own docs do not list a fixed cost for this model
and the blog-post figures floating around ("~50-100 neurons per image") turned out,
same as pollinations before it, to be worth verifying rather than trusting.

pollinations.ai is the fallback once Cloudflare is unavailable or fails. It no
longer matches its old reputation as "free, no key, just rate-limited": anonymous
requests intermittently answer HTTP 402 (payment required, an x402 crypto-payment
scheme), tied to load rather than a documented quota. A POLLINATIONS_TOKEN
measurably raises the success rate, sent as the query parameter ?token=..., not an
Authorization header (confirmed against the live endpoint both ways).

Deciding what to do when BOTH fail for one image (duplicate the previous frame
rather than leave a gap) is pipeline/images.py's job, not this module's: this file
only knows how to ask each provider for one image and say plainly when that failed.
"""
import base64
import json
import re
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

# --- cloudflare workers ai -------------------------------------------------

CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
CLOUDFLARE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
CLOUDFLARE_STEPS = 4   # this model's own max is 4; more steps is not accepted

# --- pollinations ------------------------------------------------------------

POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/{prompt}"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_MODEL = "flux"
DEFAULT_DELAY_S = 16
POLLINATIONS_ATTEMPTS = 3


class ImageError(Exception):
    pass


# --- cloudflare --------------------------------------------------------------

CLOUDFLARE_ATTEMPTS = 3
CLOUDFLARE_RETRY_DELAY_S = 4

# What a second attempt could plausibly fix: the service being briefly unreachable or
# overloaded. Everything else - a bad token, a spent daily allowance, a prompt the model
# refuses - answers identically however long it is asked for, so retrying only burns time.
_WORTH_RETRYING = re.compile(
    r"\b5\d\d\b|network error|timed? ?out|timeout|connection (reset|refused|aborted)|"
    r"non-json|decoded to only", re.I)
_NOT_WORTH_RETRYING = re.compile(r"auth rejected|not set in \.env|quota|limit", re.I)


def _worth_asking_again(error):
    text = str(error)
    return bool(_WORTH_RETRYING.search(text)) and not _NOT_WORTH_RETRYING.search(text)


def cloudflare_available():
    return bool(get_key("CLOUDFLARE_ACCOUNT_ID")) and bool(get_key("CLOUDFLARE_API_TOKEN"))


def cloudflare_settings(cfg):
    block = ((cfg.get("images") or {}).get("cloudflare")) or {}
    return {"model": block.get("model", CLOUDFLARE_MODEL),
           "steps": block.get("steps", CLOUDFLARE_STEPS)}


def synthesize_cloudflare(prompt, out_path, seed, cfg):
    """One image from Cloudflare Workers AI. Returns (path, neurons_used_or_None):
    the neuron figure is whatever the response body actually contains, under
    whichever key name it turns out to use - not assumed, read live once real
    credentials exist."""
    account_id = get_key("CLOUDFLARE_ACCOUNT_ID")
    token = get_key("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        raise ImageError("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set in .env")

    settings = cloudflare_settings(cfg)
    url = CLOUDFLARE_API_BASE.format(account_id=account_id, model=settings["model"])
    # The live API rejects "seed" outright ("Additional or unevaluated properties
    # '/seed' not allowed") despite the published docs listing it as accepted -
    # confirmed against the real endpoint, not assumed. `seed` is still accepted as
    # this function's own parameter and recorded in the db for traceability; it is
    # just not sent on, so each call draws its own random image.
    body = json.dumps({"prompt": prompt, "steps": settings["steps"]}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    })

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read()[:300]
        if e.code in (401, 403):
            raise ImageError(f"cloudflare auth rejected (HTTP {e.code}): check "
                             f"CLOUDFLARE_ACCOUNT_ID and the token's permissions "
                             f"({detail})") from e
        raise ImageError(f"cloudflare HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise ImageError(f"cloudflare network error: {e.reason}") from e

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ImageError(f"cloudflare returned non-JSON ({len(raw)} bytes)") from e

    if not payload.get("success", True) and payload.get("errors"):
        raise ImageError(f"cloudflare reported errors: {payload['errors']}")

    result = payload.get("result") or {}
    image_b64 = result.get("image")
    if not image_b64:
        raise ImageError(f"cloudflare response had no image field: "
                         f"{str(payload)[:200]}")
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]

    try:
        data = base64.b64decode(image_b64)
    except (ValueError, base64.binascii.Error) as e:
        raise ImageError(f"cloudflare image field was not valid base64: {e}") from e
    if len(data) < 1000:
        raise ImageError(f"cloudflare image decoded to only {len(data)} bytes")

    # Read whatever usage/cost figure the response actually carries, under
    # whichever key it turns out to use, rather than assuming one shape.
    neurons = None
    for container in (payload, result, payload.get("usage") or {}, result.get("usage") or {}):
        for key in ("neurons", "neuron_count", "cost", "usage"):
            value = container.get(key) if isinstance(container, dict) else None
            if isinstance(value, (int, float)):
                neurons = value
                break
        if neurons is not None:
            break

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path, neurons


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
    # A query parameter, not an Authorization header: confirmed by testing both against
    # the live endpoint. Authorization: Bearer <token> left flux 402ing exactly as if
    # anonymous; ?token=<token> is what actually gets through some of the time.
    token = get_key("POLLINATIONS_TOKEN")
    if token:
        params["token"] = token
    url = POLLINATIONS_BASE.format(prompt=urllib.parse.quote(prompt)) + \
        "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        body = e.read()[:200]
        if e.code == 402:
            raise ImageError(
                "pollinations answered 402 Payment Required (known to happen even "
                "with a valid token when their own service is under strain - see "
                "pollinations/pollinations#10028).") from e
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


# --- combined ------------------------------------------------------------------

def images_per_scene(duration_s, images_per_seconds):
    return max(1, round(duration_s / images_per_seconds))


def synthesize(prompt, out_path, seed, cfg):
    """Cloudflare first, pollinations second - that IS the "cloudflare" default, a
    cascade, not an exclusive choice. "cloudflare-only" skips the pollinations leg
    (useful when isolating a neuron measurement); "pollinations" skips straight past
    Cloudflare (useful if its credentials are being debugged). Raises ImageError
    only once whatever is allowed has been tried; the caller decides what to do next
    (pipeline/images.py duplicates the previous frame rather than leaving a gap).

    Returns (provider, neurons_or_None).
    """
    provider_pref = (cfg.get("images") or {}).get("provider", "cloudflare")
    try_cloudflare = provider_pref in ("cloudflare", "cloudflare-only")
    try_pollinations = provider_pref in ("cloudflare", "pollinations")
    last_error = None

    if try_cloudflare and cloudflare_available():
        for attempt in range(1, CLOUDFLARE_ATTEMPTS + 1):
            try:
                _, neurons = synthesize_cloudflare(prompt, out_path, seed, cfg)
                return "cloudflare", neurons
            except ImageError as e:
                last_error = e
                # A timeout or a 502 is the service having a bad second, and the frame it
                # cost would otherwise fall through to the weaker provider - or be a copy of
                # the frame before it - for no reason. A rejected token or a spent allowance
                # will say the same thing however many times it is asked, so those are not
                # retried: the waiting would be the only result.
                if attempt < CLOUDFLARE_ATTEMPTS and _worth_asking_again(e):
                    log.warning("cloudflare attempt %d/%d failed, retrying: %s",
                                attempt, CLOUDFLARE_ATTEMPTS, e)
                    time.sleep(CLOUDFLARE_RETRY_DELAY_S * attempt)
                    continue
                log.warning("cloudflare failed: %s", e)
                break
    elif try_cloudflare:
        log.warning("cloudflare unavailable (CLOUDFLARE_ACCOUNT_ID/"
                    "CLOUDFLARE_API_TOKEN not set)%s", "; trying pollinations"
                    if try_pollinations else "")

    if try_pollinations:
        for attempt in range(1, POLLINATIONS_ATTEMPTS + 1):
            try:
                synthesize_pollinations(prompt, out_path, seed, cfg)
                return "pollinations", None
            except ImageError as e:
                last_error = e
                log.warning("pollinations attempt %d/%d failed: %s",
                           attempt, POLLINATIONS_ATTEMPTS, e)

    raise ImageError(str(last_error) if last_error else "no image provider available")
