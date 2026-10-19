"""How much of Cloudflare's daily 10,000 Neurons is still free to spend.

Cloudflare does not tell this token what is left, so it is worked out from what this
project has recorded spending: the images stage and the thumbnail stage each log a line
with "<n> neurons" in it, and the thumbnail refresh job keeps its own ledger in
data/thumb_refresh.json (it runs outside the video pipeline and does not write the
database).

The window is the day of America/Los_Angeles, i.e. it turns over at 07:00 UTC in summer and
08:00 UTC after the clocks go back, not at 00:00 UTC as the documentation says. Inferred from
the record, and it is the only reading that fits all of it: Cloudflare refused the bust on
19 October at 06:44 UTC after 9,216 Neurons of frames, which only adds up to 10,000 with the
979 spent at 18:08 UTC the evening before; the frames of 16, 17 and 18 October, drawn between
05:15 and 06:25 UTC each, each had a full allowance; and the probes in the README are blocked
at 06:46 UTC and open at 20:50 and 07:29 UTC. So the daily frames, which run at 05-06 UTC, are
the LAST spend of a window, and everything drawn after 07:00 UTC belongs to the next one.

The narration frames come first. The thumbnail stage runs after them, so whatever it sees
here is what they left; the refresh job runs only after a successful daily run and keeps
the same reserve.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from core import db

RESET_ZONE = ZoneInfo("America/Los_Angeles")
DAILY_FREE_NEURONS = 10000
DEFAULT_RESERVE = 300
LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "thumb_refresh.json"

_SPENT = re.compile(r"(\d+(?:\.\d+)?) neurons\b")


def _parse(when):
    try:
        moment = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def load_ledger(path=LEDGER_PATH):
    path = Path(path)
    if not path.exists():
        return {"done": {}, "spend": [], "set_log": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, empty in (("done", {}), ("spend", []), ("set_log", [])):
        data.setdefault(key, empty)
    return data


def window_start(now=None):
    """The moment the current allowance window began, as an aware UTC datetime."""
    now = (now or datetime.now(timezone.utc)).astimezone(RESET_ZONE)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def window_events(stage_like, now=None):
    """[(created_at, message)] of this window's events whose message has "<n> neurons"."""
    since = window_start(now)
    try:
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT message, created_at FROM events WHERE stage IN ('images', 'thumbnail') "
                "AND message LIKE '%neurons%'").fetchall()
    except Exception:                                               # noqa: BLE001
        return []
    return [(_parse(r["created_at"]), r["message"] or "") for r in rows
            if _parse(r["created_at"]) and _parse(r["created_at"]) >= since
            and stage_like in (r["message"] or "")]


def frames_done(now=None):
    """True once this window's narration frames have been drawn: their summary line is the
    only one that says "redrawn for lettering"."""
    return bool(window_events("redrawn for lettering", now))


def spent_today(now=None, ledger_path=LEDGER_PATH):
    """Neurons recorded in the current allowance window (see the module docstring)."""
    since = window_start(now)
    total = 0.0
    try:
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT message, created_at FROM events "
                "WHERE stage IN ('images', 'thumbnail') AND message LIKE '%neurons%'").fetchall()
    except Exception:                                               # noqa: BLE001
        rows = []
    for row in rows:
        moment = _parse(row["created_at"])
        if moment is None or moment < since:
            continue
        total += sum(float(n) for n in _SPENT.findall(row["message"] or ""))
    for entry in load_ledger(ledger_path).get("spend", []):
        moment = _parse(entry.get("at"))
        if moment is not None and moment >= since:
            total += float(entry.get("neurons") or 0)
    return total


def available(reserve=DEFAULT_RESERVE, now=None, ledger_path=LEDGER_PATH, frames_reserve=0):
    """Neurons free to spend now. `frames_reserve` is held back for the window's narration
    frames when they have not been drawn yet - they come first, then the new video's cover,
    then everything else."""
    held = 0 if frames_done(now) else frames_reserve
    return max(0.0, DAILY_FREE_NEURONS - reserve - held - spent_today(now, ledger_path))
