"""Two questions only a GitHub runner can answer, asked from the runner itself.

    python scripts/runner_probe.py

1. Does YouTube's search autocomplete answer from a datacenter address? Topics are going to
   be chosen from it, and a source that works from a laptop and returns nothing - or a consent
   page - from Azure would leave the daily run with no topic.
2. Does edge-tts speak with en-US-AndrewMultilingualNeural from here? The daily videos are
   read by piper, offline; the Shorts are to be read by Andrew, which is a Microsoft service
   that has been known to refuse datacenter addresses.

Exits non-zero if either fails, and says which.
"""
import asyncio
import json
import sys
import time
import urllib.parse
import urllib.request

SEEDS = ["stoicism", "marcus aurelius", "seneca", "epictetus", "stoic",
         "how to stop overthinking", "how to control emotions"]
URL = "https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&hl=en&gl=us&q="


def suggest(query):
    req = urllib.request.Request(URL + urllib.parse.quote(query),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        body = r.read().decode("utf-8", "replace")
    data = json.loads(body)
    return data[1] if isinstance(data, list) and len(data) > 1 else []


def main():
    ok = True
    print("== autocomplete")
    for q in SEEDS:
        try:
            got = suggest(q)
            print(f"  {q:28} {len(got):2d} suggestions: {', '.join(got[:4])}")
            if not got:
                ok = False
        except Exception as e:                                   # noqa: BLE001
            print(f"  {q:28} FAILED: {str(e)[:100]}")
            ok = False
        time.sleep(0.5)

    print("== edge-tts, Andrew")
    try:
        import edge_tts

        async def go():
            n = 0
            c = edge_tts.Communicate("Most of what troubles you has not happened yet.",
                                     "en-US-AndrewMultilingualNeural")
            async for chunk in c.stream():
                if chunk["type"] == "audio":
                    n += len(chunk["data"])
            return n
        n = asyncio.run(go())
        print(f"  spoke: {n} bytes of audio")
        ok = ok and n > 5000
    except Exception as e:                                       # noqa: BLE001
        print(f"  FAILED: {str(e)[:160]}")
        ok = False
    print("RESULT", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
