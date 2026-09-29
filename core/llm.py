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
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODEL = "gemini-flash-lite"

MAX_RPM = 25
_RATE_WINDOW = 60.0
_MAX_429_WAITS = 5


class LLMError(Exception):
    pass


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
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
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


def _rate_limit():
    """Block until fewer than MAX_RPM calls sit inside the trailing 60s window."""
    while True:
        with _calls_lock:
            now = time.monotonic()
            _calls[:] = [t for t in _calls if now - t < _RATE_WINDOW]
            if len(_calls) < MAX_RPM:
                _calls.append(now)
                return
            wait = _RATE_WINDOW - (now - _calls[0]) + 0.05
        log.info("rate limit: %d calls in the last minute, waiting %.1fs", MAX_RPM, wait)
        time.sleep(wait)


# --- http ------------------------------------------------------------------

def _post(url, payload, headers, timeout=180):
    """POST JSON. Returns (status, body_text). HTTP errors come back as values, not exceptions."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        raise LLMError(f"network error calling {url}: {e.reason}") from e


def _retry_after_seconds(body, attempt):
    """Seconds to wait after a 429. Honours the API's own hint when it gives one."""
    match = re.search(r"try again in ([\d.]+)s", body)
    if match:
        return min(float(match.group(1)) + 1.0, 120.0)
    return min(5.0 * 2 ** attempt, 120.0)


# --- providers -------------------------------------------------------------

def _call_groq(prompt, system, max_tokens, temperature, json_mode):
    key = get_key("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set (put it in .env)")
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    payload = {"model": GROQ_MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(_MAX_429_WAITS):
        _rate_limit()
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
        log.info("groq %s | %s tokens (%s in, %s out) | %.1fs", GROQ_MODEL,
                 usage.get("total_tokens", "?"), usage.get("prompt_tokens", "?"),
                 usage.get("completion_tokens", "?"), elapsed)
        return data["choices"][0]["message"]["content"]
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
    url = GEMINI_URL.format(model=GEMINI_MODEL)

    for attempt in range(_MAX_429_WAITS):
        _rate_limit()
        started = time.monotonic()
        status, body = _post(url, payload, {"x-goog-api-key": key})
        elapsed = time.monotonic() - started
        if status == 429:
            wait = _retry_after_seconds(body, attempt)
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
        log.info("gemini %s | %s tokens | %.1fs", GEMINI_MODEL,
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
    """Groq, falling back to Gemini. Raises LLMError when no provider answers."""
    try:
        return _call_groq(prompt, system, max_tokens, temperature, json_mode)
    except LLMError as groq_error:
        if not get_key("GEMINI_API_KEY"):
            raise
        log.warning("groq failed (%s), falling back to gemini", groq_error)
        return _call_gemini(prompt, system, max_tokens, temperature, json_mode)


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
