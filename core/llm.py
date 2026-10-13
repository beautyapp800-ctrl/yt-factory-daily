"""LLM access: Groq primary, Gemini fallback. Stdlib only."""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock

from core.logger import get_logger
from core.retry import retry

log = get_logger("llm")

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODEL = "gemini-flash-lite"

# Cloudflare sits in front of api.groq.com and answers "error code: 1010" to
# urllib's default User-Agent, so every request carries an explicit one.
USER_AGENT = "yt-factory/1.0"

MAX_RPM = 25
# Groq's free tier refuses on tokens per minute, not on requests, and this
# pipeline's prompts grow as a script accumulates. Measured on one video: 57
# calls, median 3,884 tokens, largest 6,912, 215,000 in total. The published
# free-tier ceiling for this model is 8,000 tokens a minute, so a single call
# can be most of a minute's budget and pacing by request count cannot stay
# inside it.
MAX_TPM = 8_000
_RATE_WINDOW = 60.0
# Groq answers a 429 with "try again in Xs" and the code honours it. Five waits
# was not enough patience to cross the busiest minute of a long script; the run
# that failed had already finished most of its work.
_MAX_429_WAITS = 12
# No single call may spend longer than this waiting out 429s before it gives up
# and lets the other provider try. Twelve exponential waits total eighteen
# minutes, which is a silent stall rather than a retry.
_MAX_WAIT_PER_CALL_S = 150.0


class LLMError(Exception):
    pass


_settings_cache = None


def _settings():
    """Model names and the rate cap, from config.json's optional "llm" block."""
    global _settings_cache
    if _settings_cache is None:
        try:
            from core.config import load_config
            block = load_config().get("llm") or {}
        except Exception as e:  # config problems are the config module's to report
            log.warning("could not read llm settings from config (%s), using defaults", e)
            block = {}
        _settings_cache = {
            "groq_model": block.get("groq_model") or GROQ_MODEL,
            "gemini_model": block.get("gemini_model") or GEMINI_MODEL,
            "max_rpm": int(block.get("max_rpm") or MAX_RPM),
            "max_tpm": int(block.get("max_tpm") or MAX_TPM),
            "primary": (block.get("primary") or "gemini").lower(),
        }
    return _settings_cache


# --- .env ------------------------------------------------------------------

_env_cache = None


def load_env(path=ENV_PATH):
    """Parse a simple KEY=value .env file. Later keys win; os.environ wins over the file."""
    global _env_cache
    if _env_cache is not None:
        return _env_cache
    values = {}
    path = Path(path)
    if path.exists():
        # utf-8-sig: PowerShell's > and Out-File write a BOM, which would otherwise
        # end up glued to the first key name.
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # "export FOO=bar" is a common shape to paste in; treat it as FOO=bar.
            key = key.strip().removeprefix("export ").strip().lstrip("﻿")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key:
                values[key] = value
    _env_cache = values
    return values


def get_key(name):
    """Environment variable first, then .env. Returns None when unset or blank."""
    value = os.environ.get(name) or load_env().get(name)
    return value.strip() or None if value else None


# --- rate limit ------------------------------------------------------------

_calls = []
_calls_lock = Lock()


def _rate_limit(estimated_tokens=0):
    """Block until both the request and the token budget have room.

    `_calls` holds (timestamp, tokens) for the trailing minute. Tokens are the
    binding constraint on Groq's free tier; requests are kept as a second guard
    because a burst of tiny calls can still trip the per-request limit.
    """
    settings = _settings()
    cap, token_cap = settings["max_rpm"], settings["max_tpm"]
    while True:
        with _calls_lock:
            now = time.monotonic()
            _calls[:] = [c for c in _calls if now - c[0] < _RATE_WINDOW]
            used = sum(c[1] for c in _calls)
            room_for_calls = len(_calls) < cap
            # A prompt larger than the entire per-minute budget could never
            # fit, and this loop would spin on it forever: reserve the whole
            # budget and let the API be the one to object.
            want = min(estimated_tokens, token_cap)
            room_for_tokens = used + want <= token_cap
            if room_for_calls and room_for_tokens:
                _calls.append((now, want))
                return
            oldest = _calls[0][0] if _calls else now
            wait = _RATE_WINDOW - (now - oldest) + 0.05
            reason = "tokens" if not room_for_tokens else "requests"
        log.info("rate limit (%s): %d calls / %d tokens in the last minute, "
                 "waiting %.1fs", reason, len(_calls), used, wait)
        time.sleep(max(0.1, wait))


def _record_actual(tokens):
    """Correct the last reservation once the real usage is known."""
    if not tokens:
        return
    with _calls_lock:
        if _calls:
            when, _ = _calls[-1]
            _calls[-1] = (when, tokens)


# --- http ------------------------------------------------------------------

def _post(url, payload, headers, timeout=180):
    """POST JSON. Returns (status, body_text). HTTP errors come back as values, not exceptions."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        raise LLMError(f"network error calling {url}: {e.reason}") from e


def _retry_after_seconds(body, attempt):
    """Seconds to wait after a 429. Honours the API's own hint when it gives one.

    The two providers phrase the hint differently: Groq writes it into the
    message ("try again in 7.5s"), Gemini returns a RetryInfo detail with
    retryDelay. Reading only Groq's phrasing meant every Gemini 429 fell
    through to the exponential ladder below, which is how one run spent
    eighteen minutes per call waiting out a quota that resets tomorrow.
    """
    match = re.search(r"try again in ([\d.]+)s", body)
    if match:
        return min(float(match.group(1)) + 1.0, 120.0)
    match = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', body)
    if match:
        return min(float(match.group(1)) + 1.0, 120.0)
    return min(5.0 * 2 ** attempt, 120.0)


def _is_daily_quota(body):
    """True when a 429 is a per-day allowance rather than a per-minute burst.

    Waiting is pointless here - the free tier allows twenty requests a day per
    model, and one video needs more than that - so the call should fail over to
    the other provider at once instead of sleeping.
    """
    return bool(re.search(r"PerDay|per day|requests per day", body, re.I))


# --- providers -------------------------------------------------------------

def _call_groq(prompt, system, max_tokens, temperature, json_mode):
    key = get_key("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set (put it in .env)")
    model = _settings()["groq_model"]
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        # These models bill their reasoning against max_tokens; keep it short so the
        # answer itself is not the part that gets cut off.
        payload["reasoning_effort"] = "low"

    # Reserve roughly what this call will cost before making it: prompt length
    # is known, the answer is bounded by max_tokens. Corrected below from the
    # usage the API reports, so the running total stays honest.
    # max_tokens is a ceiling, not a forecast: lessons ask for 2,370 and answer
    # in about 500. Reserving the ceiling throttled the pipeline against a limit
    # it was nowhere near. Reserve the prompt in full plus a realistic share of
    # the ceiling, then correct from the reported usage below.
    estimate = (len(prompt) // 4 + len(system or "") // 4
                + min(int(max_tokens), int(max_tokens) // 3 + 200))

    for attempt in range(_MAX_429_WAITS):
        _rate_limit(estimate)
        started = time.monotonic()
        status, body = _post(GROQ_URL, payload, {"Authorization": f"Bearer {key}"})
        elapsed = time.monotonic() - started
        if status == 429:
            wait = _retry_after_seconds(body, attempt)
            log.warning("groq 429 (attempt %d/%d), waiting %.1fs", attempt + 1, _MAX_429_WAITS, wait)
            time.sleep(wait)
            continue
        if status != 200:
            raise LLMError(f"groq HTTP {status}: {body[:400]}")
        data = json.loads(body)
        usage = data.get("usage") or {}
        # Replace the pre-call estimate with what the call actually cost.
        _record_actual(usage.get("total_tokens"))
        choice = data["choices"][0]
        log.info("groq %s | %s tokens (%s in, %s out) | %.1fs", model,
                 usage.get("total_tokens", "?"), usage.get("prompt_tokens", "?"),
                 usage.get("completion_tokens", "?"), elapsed)
        if choice.get("finish_reason") == "length":
            log.warning("groq hit the %d token ceiling; the answer is cut off", max_tokens)
        content = (choice.get("message") or {}).get("content") or ""
        if not content.strip():
            raise LLMError("groq returned empty content")
        return content
    raise LLMError(f"groq still rate-limited after {_MAX_429_WAITS} waits")


def _call_gemini(prompt, system, max_tokens, temperature, json_mode):
    key = get_key("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    model = _settings()["gemini_model"]
    url = GEMINI_URL.format(model=model)

    waited = 0.0
    for attempt in range(_MAX_429_WAITS):
        _rate_limit()
        started = time.monotonic()
        status, body = _post(url, payload, {"x-goog-api-key": key})
        elapsed = time.monotonic() - started
        if status == 429:
            if _is_daily_quota(body):
                raise LLMError(
                    f"gemini daily quota is used up, falling over: {body[:200]}")
            wait = _retry_after_seconds(body, attempt)
            waited += wait
            if waited > _MAX_WAIT_PER_CALL_S:
                raise LLMError(
                    f"gemini rate-limited {waited:.0f}s on one call, falling over")
            log.warning("gemini 429 (attempt %d/%d), waiting %.1fs", attempt + 1, _MAX_429_WAITS, wait)
            time.sleep(wait)
            continue
        if status != 200:
            raise LLMError(f"gemini HTTP {status}: {body[:400]}")
        data = json.loads(body)
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError(f"gemini returned no candidates: {body[:400]}")
        usage = data.get("usageMetadata") or {}
        log.info("gemini %s | %s tokens | %.1fs", model,
                 usage.get("totalTokenCount", "?"), elapsed)
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text.strip():
            raise LLMError("gemini returned empty text")
        return text
    raise LLMError(f"gemini still rate-limited after {_MAX_429_WAITS} waits")


# --- public API ------------------------------------------------------------

@retry(attempts=3, base_delay=4)
def complete(prompt, system=None, max_tokens=4096, temperature=0.8, json_mode=False):
    """The configured primary provider, falling back to the other one.

    Order is set by llm.primary in config.json. Groq led until a measured video
    turned out to cost 215,000 tokens against a 200,000-a-day free ceiling - one
    video a day did not fit, which is why the channel kept stopping. Gemini
    leads now and Groq is the spare.
    """
    order = _settings().get("primary", "gemini")
    providers = [("gemini", _call_gemini), ("groq", _call_groq)]
    if order == "groq":
        providers.reverse()

    first_error = None
    for name, call in providers:
        key = "GEMINI_API_KEY" if name == "gemini" else "GROQ_API_KEY"
        if not get_key(key):
            log.info("%s has no key configured, skipping", name)
            continue
        try:
            return call(prompt, system, max_tokens, temperature, json_mode)
        except LLMError as e:
            if first_error is None:
                first_error = e
            log.warning("%s failed (%s), trying the next provider", name, e)
    raise first_error or LLMError("no provider is configured: set GEMINI_API_KEY "
                                  "or GROQ_API_KEY")


def extract_json(text):
    """Pull a JSON value out of a model reply, including ```json fenced blocks."""
    if not text:
        raise ValueError("empty response")
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    candidate = fence.group(1).strip() if fence else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] span in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in response: {text[:200]}")


def complete_json(prompt, system=None, max_tokens=4096, temperature=0.7):
    """complete() that returns parsed JSON, with one stricter re-ask on a parse failure."""
    text = complete(prompt, system, max_tokens, temperature, json_mode=True)
    try:
        return extract_json(text)
    except ValueError as e:
        log.warning("JSON parse failed (%s), re-asking with a stricter instruction", e)
        strict = (f"{prompt}\n\n"
                  "Your previous answer was not valid JSON. Reply with valid JSON only. "
                  "No prose, no markdown, no code fences.")
        text = complete(strict, system, max_tokens, temperature=0.2, json_mode=True)
        return extract_json(text)
