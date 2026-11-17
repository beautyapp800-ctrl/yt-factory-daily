"""The shared guards, checked in every channel's smoke run (owner 17.11). No network, no secrets.

    python scripts/smoke_guards.py

  - every workflow file parses (17.11: one colon in a run line stopped a channel's builds for 40 min);
  - core/ytguard.py: the marker finds an uploaded episode, a short channel list is a Doubt (no upload),
    a token of another channel is a Doubt, the days held on YouTube are taken, git add one path at a time.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
bad = []


def check(name, ok, detail=""):
    print(("[ok] " if ok else "[FAIL] ") + name + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        bad.append(name)


def main():
    import yaml
    errors = []
    for wf in sorted((ROOT / ".github" / "workflows").glob("*.y*ml")):
        try:
            yaml.safe_load(wf.read_text(encoding="utf-8"))
        except Exception as e:                                       # noqa: BLE001
            errors.append(f"{wf.name}: {str(e)[:90]}")
    check("workflows: every file parses as YAML", not errors, "; ".join(errors))

    from core import ytguard as g

    class Req:
        def __init__(self, v):
            self.v = v

        def execute(self):
            if isinstance(self.v, Exception):
                raise self.v
            return self.v

    class FakeYT:
        def __init__(self, ids, total=None, cid="UC1"):
            self.ids, self.total, self.cid = ids, total if total is not None else len(ids), cid

        def channels(self):
            yt = self

            class C:
                def list(self, **k):
                    return Req({"items": [{"id": yt.cid, "snippet": {"title": "t"},
                                           "contentDetails": {"relatedPlaylists": {"uploads": "UU"}}}]})
            return C()

        def playlistItems(self):
            yt = self

            class P:
                def list(self, pageToken=None, **k):
                    start = int(pageToken or 0)
                    page = yt.ids[start:start + 2]
                    nxt = str(start + 2) if start + 2 < len(yt.ids) else None
                    return Req({"items": [{"contentDetails": {"videoId": i}} for i in page],
                                "pageInfo": {"totalResults": yt.total}, **({"nextPageToken": nxt} if nxt else {})})
            return P()

        def videos(self):
            class V:
                def list(self, id, **k):
                    return Req({"items": [{"id": i, "snippet": {"title": f"T {i}", "description":
                                                                  ("x\n" + g.ref("ch", "s-" + i)),
                                                                  "publishedAt": "2026-11-17T15:00:00Z"},
                                           "status": {"privacyStatus": "private" if i == "c" else "public",
                                                      "publishAt": "2026-11-21T15:00:00Z" if i == "c" else None}}
                                          for i in id.split(",")]})
            return V()

    vids = g.all_uploads(FakeYT(["a", "b", "c"]), known_ids=["a"])
    check("guard: the whole list is read, page by page", [v["id"] for v in vids] == ["a", "b", "c"])
    check("guard: the marker finds the uploaded episode", (g.find_existing(vids, g.ref("ch", "s-b")) or {}).get("id") == "b")
    check("guard: a new episode is not found", g.find_existing(vids, g.ref("ch", "new"), "Other") is None)
    try:
        g.all_uploads(FakeYT(["a", "b"], total=5))
        short = False
    except g.Doubt:
        short = True
    check("guard: a list shorter than YouTube says is a Doubt (no upload)", short)
    try:
        g.all_uploads(FakeYT(["a"]), known_ids=["zz"])
        lost = False
    except g.Doubt:
        lost = True
    check("guard: a video the state knows but the list lacks is a Doubt", lost)
    try:
        g.check_channel(FakeYT([], cid="UCother"), "UC1")
        wrong = False
    except g.Doubt:
        wrong = True
    check("guard: a token of another channel is a Doubt", wrong)
    days = g.taken_days(vids)
    check("guard: days held on YouTube are taken (scheduled and published)",
          {str(d) for d in days} == {"2026-11-17", "2026-11-21"}, str(days))
    src = (ROOT / "core" / "ytguard.py").read_text(encoding="utf-8")
    check("guard: git add one path at a time", 'git("add", "--", str(p))' in src)
    print("GUARDS", "PASSED" if not bad else f"FAILED: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
