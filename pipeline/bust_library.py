"""Every bust this channel has drawn, kept in git, for the day Cloudflare will not draw one.

A video's bust is drawn first thing in the daily run and asked for again every half hour until
its publish time (scripts/bust_pending.py). When the daily run cannot get one, the video's phrase
is set at once on a bust from here, as a temporary cover, so that it has a cover in the channel's
format from the moment it is uploaded; its own bust replaces that when it is drawn. Never
YouTube's own pick and never a frame from the video. Owner's choice, 22 October.

The store only grows. Each bust is a JPEG with a small JSON beside it saying where it came from;
nothing is ever rewritten, so the daily run and the half-hourly job can both add to it without
their pushes colliding. Which bust was reused for which video is recorded by the job that reuses
them - as a "bust temporary" event in the database, and by bust-pending in
data/bust_pending.json - and the one picked is the one longest out of use.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

LIB_DIR = Path(__file__).resolve().parent.parent / "assets" / "busts"
SIZE = 720                       # compose() lays the bust in as a 720x720 square anyway
QUALITY = 92


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _when(value):
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def add(path, video_id, philosopher=None, judged_by="", notes="", drawn_at=None,
        lib_dir=None):
    """Keep a copy of a bust. Returns its file name; a picture already kept is not kept twice."""
    lib = Path(lib_dir or LIB_DIR)
    lib.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as img:
        img = img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
        digest = hashlib.sha256(img.tobytes()).hexdigest()[:10]
    name = f"v{int(video_id):04d}_{digest}.jpg"
    if (lib / name).exists():
        return name
    img.save(lib / name, "JPEG", quality=QUALITY)
    (lib / name).with_suffix(".json").write_text(json.dumps({
        "video_id": int(video_id), "philosopher": philosopher, "judged_by": judged_by,
        "notes": notes, "drawn_at": drawn_at or _now()}, indent=2) + "\n", encoding="utf-8")
    return name


def entries(lib_dir=None):
    """[(file name, metadata)] of everything kept, oldest drawn first."""
    lib = Path(lib_dir or LIB_DIR)
    out = []
    for jpg in sorted(lib.glob("*.jpg")):
        meta_path = jpg.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        out.append((jpg.name, meta))
    return sorted(out, key=lambda e: _when(e[1].get("drawn_at")))


TEMPORARY = "bust temporary"


def uses_from_events():
    """[{"file", "at"}] of every kept bust that was put on a video as its temporary cover,
    read from the events table - the daily run can write that, but not data/bust_pending.json."""
    import re
    from core import db
    out = []
    with db._connect() as conn:
        rows = conn.execute("SELECT message, created_at FROM events WHERE stage = 'thumbnail' "
                            "AND message LIKE ?", (TEMPORARY + "%",)).fetchall()
    for row in rows:
        found = re.search(r"kept bust (\S+\.jpg)", row["message"] or "")
        if found:
            out.append({"file": found.group(1), "at": row["created_at"]})
    return out


def pick(uses, lib_dir=None):
    """The bust longest out of use: its last appearance - on the video it was drawn for, or
    reused since (`uses`: [{"file", "at"}]) - is the oldest. None if nothing is kept."""
    last = {}
    for name, meta in entries(lib_dir):
        last[name] = _when(meta.get("drawn_at"))
    for use in uses or []:
        if use.get("file") in last:
            last[use["file"]] = max(last[use["file"]], _when(use.get("at")))
    if not last:
        return None
    name = min(last, key=lambda n: (last[n], n))
    return Path(lib_dir or LIB_DIR) / name
