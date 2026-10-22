"""Stage: draw the thumbnail's bust, before the frames.

The bust is one image, about 330 Neurons, and nothing can replace it: the old picture-and-two-
lines thumbnail cut from a narration frame is no longer allowed on the channel. The frames are
fifty times that and can wait a day. So the cheap, irreplaceable picture is drawn first, while
the day's allowance is still there, and the type is set on it later by the thumbnail stage once
the final title is known (pipeline/bust_thumbnail.py, draw and finish).

On 22 October the order was the other way round, the frames took 6,912 Neurons, and video 28
went out with a frame for a thumbnail. Never again: if no bust can be drawn here, the stage
defers and the video waits for the next run. The queue on YouTube is days deep.
"""
from core import db
from core.concede import Deferred
from core.config import output_dir
from core.logger import get_logger

log = get_logger("bust")


def run(video_id, cfg):
    style = ((cfg or {}).get("thumbnail") or {}).get("style")
    if style != "bust":
        return True                       # another thumbnail style draws nothing up front
    from pipeline import bust_thumbnail
    try:
        record = bust_thumbnail.draw(video_id, bust_thumbnail.video_text(video_id),
                                     output_dir(video_id), cfg)
    except Deferred:
        raise
    except Exception as e:                                           # noqa: BLE001
        raise Deferred(f"the bust could not be drawn ({e}); the video waits for the next "
                       f"run") from e
    top = record["ranked"][0]
    db.log_event(video_id, "bust", "info",
                 f"bust drawn for {record['philosopher'] or 'a generic Stoic'}, "
                 f"{len(record['ranked'])} candidate(s) judged by {record['judged_by']}, "
                 f"top: {top['notes']}; {record['neurons']} neurons over "
                 f"{record['cloudflare_calls']} Cloudflare calls")
    return True
