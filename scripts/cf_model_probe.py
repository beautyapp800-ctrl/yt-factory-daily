"""Does a Cloudflare image model honour width/height, and what does it cost in Neurons?

    python scripts/cf_model_probe.py              # all three candidates
    python scripts/cf_model_probe.py light dream  # some of them

flux-1-schnell, the current model, takes only `prompt` and `steps` and always returns a
1024x1024 square, which the render stage then crops to 16:9, throwing away 44% of the
picture. These three accept a size, so one of them could give a native 16:9 image:

    sdxl   @cf/stabilityai/stable-diffusion-xl-base-1.0
    light  @cf/bytedance/stable-diffusion-xl-lightning
    dream  @cf/lykon/dreamshaper-8-lcm

Each is asked once for 1280x720 with the same prompt. For every call this prints the real
size of the image that came back (not the size asked for), the time, and any Neuron figure
the response carries, and saves the picture to output/_probe2/ to be looked at. Cost per
image decides whether a model is usable at all: the daily allowance is 10,000 Neurons and
a 30 minute video needs ~80 images.

All models draw on the same daily allowance, so while it is used up (error 4006) every
call fails in under a second without costing anything.
"""
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import OUTPUT_DIR
from core.llm import get_key

OUT = OUTPUT_DIR / "_probe2"
PROMPT = ("hands holding a plain unmarked parcel in a pharmacy queue, grey winter light "
          "from a window, painterly illustration, muted palette")
CASES = {
    "sdxl": ("@cf/stabilityai/stable-diffusion-xl-base-1.0",
             {"prompt": PROMPT, "width": 1280, "height": 720, "num_steps": 20}),
    "light": ("@cf/bytedance/stable-diffusion-xl-lightning",
              {"prompt": PROMPT, "width": 1280, "height": 720, "num_steps": 4}),
    "dream": ("@cf/lykon/dreamshaper-8-lcm",
              {"prompt": PROMPT, "width": 1280, "height": 720, "num_steps": 4}),
}


def image_size(data):
    """(width, height) of a PNG or JPEG, read from its header."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        if data[i + 1] in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return None


def call(model, body, tag):
    account, token = get_key("CLOUDFLARE_ACCOUNT_ID"), get_key("CLOUDFLARE_API_TOKEN")
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {token}",
        "User-Agent": "yt-factory/1.0"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw, headers = resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:300]!r}"}
    info = {"seconds": round(time.time() - started, 1), "bytes": len(raw)}
    info["usage_headers"] = {k: v for k, v in headers.items()
                             if any(w in k.lower() for w in ("neuron", "usage", "cost"))}
    data = raw
    if "json" in headers.get("Content-Type", ""):
        payload = json.loads(raw)
        result = payload.get("result") or {}
        info["payload_keys"] = list(payload)
        info["usage"] = payload.get("usage") or result.get("usage")
        data = base64.b64decode(result["image"]) if result.get("image") else None
    if data:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{model.split('/')[-1]}_{tag}.jpg"
        path.write_bytes(data)
        info["returned_size"] = image_size(data)
        info["file"] = str(path)
    return info


def main():
    chosen = sys.argv[1:] or list(CASES)
    for name in chosen:
        model, body = CASES[name]
        print(f"{name}: {model}, asked for {body['width']}x{body['height']}")
        print("   ", json.dumps(call(model, body, "720p"), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
