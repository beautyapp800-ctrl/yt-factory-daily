"""Shipping something imperfect, on the record, rather than nothing at all.

The channel publishes at 21:00 every day. That is the one hard requirement,
and for a while the pipeline was built the other way round: every stage had a
quality gate, and any gate that failed threw the whole day away. Four separate
gates did exactly that in one week - a word budget, a pace figure measured on
a retired voice, an under-delivering prompt, a repair loop that could make a
script shorter - and each fix only exposed the next one.

So the rule is inverted. A gate that is not met tries to fix itself, then
concedes and records what it conceded. A run ends without a video only when
there is physically nothing to assemble: no narration, or no frames.

Every concession is written to the events table, which is what the run email
is built from, so "shipped anyway" is never silent. A relaxation nobody can
see is indistinguishable from the rule not existing.
"""
from core import db
from core.logger import get_logger

log = get_logger("concede")


def concede(video_id, stage, problem, using=None):
    """Record a target that was missed and let the run carry on."""
    message = f"below target, shipped anyway: {problem}"
    if using:
        message += f"; used {using}"
    log.warning("%s conceded: %s", stage, message)
    try:
        db.log_event(video_id, stage, "warning", message)
    except Exception as e:                                       # noqa: BLE001
        log.error("could not record the concession: %s", e)
    return message


def impossible(stage, problem):
    """The one kind of failure that is still allowed to stop a run."""
    return RuntimeError(f"{stage}: nothing can be assembled - {problem}")
