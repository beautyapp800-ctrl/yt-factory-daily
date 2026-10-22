"""Stage: draw the thumbnail's bust, before the frames.

The bust is one image, about 330 Neurons, and nothing may stand in for it on a published video:
not the old picture-and-two-lines thumbnail, not YouTube's own pick. The frames are twenty times
that. So the bust is asked for first, while the day's allowance is still there, and the type is
set on it later by the thumbnail stage once the final title is known (pipeline/bust_thumbnail.py,
draw and finish). On 22 October the order was the other way round and video 28 went out with a
frame for a thumbnail.

If Cloudflare refuses, the run does not stop: the video is still made and uploaded, private and
scheduled, without a thumbnail of its own, and scripts/bust_pending.py (a workflow every half
hour) keeps asking for the bust until the publish time and sets it with thumbnails.set.
"""
from core import db
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
    except Exception as e:                                           # noqa: BLE001
        msg = (f"bust pending: none could be drawn now ({e}); the video goes ahead and "
               f"scripts/bust_pending.py sets the bust before the publish time")
        log.warning(msg)
        db.log_event(video_id, "bust", "warning", msg)
        return True
    top = record["ranked"][0]
    db.log_event(video_id, "bust", "info",
                 f"bust drawn for {record['philosopher'] or 'a generic Stoic'}, "
                 f"{len(record['ranked'])} candidate(s) judged by {record['judged_by']}, "
                 f"top: {top['notes']}; {record['neurons']} neurons over "
                 f"{record['cloudflare_calls']} Cloudflare calls")
    return True
