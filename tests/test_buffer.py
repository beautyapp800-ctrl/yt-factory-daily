"""Offline tests for the buffer loop. Run: python tests/test_buffer.py

The loop runs unattended for hours on a free runner, so the things that matter
are that it stops: on a full buffer, on a video it cannot assemble, and on a
video that finished without reaching YouTube.
"""
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core.logger  # noqa: F401,E402
from core import db  # noqa: E402
import run as runner  # noqa: E402

# Captured before any test swaps it out: the earlier tests stub
# runner.queue_depth_days, and this is the real one under test.
REAL_DEPTH = runner.queue_depth_days
REAL_PROCESS = runner.process_video

CFG = {"timezone": "Europe/Kiev", "publish_time": "21:00", "buffer_days": 3,
       "run_time_budget_min": 240, "publish_days_ahead": 3}
failures = []


def check(ok, msg):
    print(f"  {'ok' if ok else 'FAIL'}: {msg}")
    if not ok:
        failures.append(msg)


def _fresh(tmp, name):
    db.DB_PATH = tmp / f"{name}.db"
    db.init_db()


def test_stops_when_the_buffer_is_full(tmp):
    print("test_stops_when_the_buffer_is_full")
    _fresh(tmp, "full")
    depth = {"days": 3}
    runner.queue_depth_days = lambda cfg, now=None: depth["days"]
    calls = []
    runner.process_video = lambda vid, cfg: calls.append(vid) or True
    made, because = runner.build_until_buffered(CFG)
    check(made == 0, f"nothing was built ({made})")
    check(not calls, "and no video was even started")
    check("3 days deep" in because, f"the reason names the depth ({because})")


def test_builds_until_the_buffer_fills(tmp):
    print("test_builds_until_the_buffer_fills")
    _fresh(tmp, "fill")
    depth = {"days": 0}
    runner.queue_depth_days = lambda cfg, now=None: depth["days"]

    def produce(vid, cfg):
        depth["days"] += 1          # each upload takes the next day's slot
        return True
    runner.process_video = produce
    made, because = runner.build_until_buffered(CFG)
    check(made == 3, f"three videos filled a three day buffer ({made})")
    check("3 days deep" in because, f"and it stopped because it was full ({because})")


def test_stops_on_a_video_that_cannot_be_assembled(tmp):
    print("test_stops_on_a_video_that_cannot_be_assembled")
    _fresh(tmp, "broken")
    runner.queue_depth_days = lambda cfg, now=None: 0
    runner.process_video = lambda vid, cfg: False
    made, because = runner.build_until_buffered(CFG)
    check(made == 0, f"nothing was built ({made})")
    check("could not be assembled" in because, f"and it says so ({because})")


def test_stops_when_a_finished_video_never_reaches_youtube(tmp):
    print("test_stops_when_a_finished_video_never_reaches_youtube")
    _fresh(tmp, "stuck")
    runner.queue_depth_days = lambda cfg, now=None: 0      # never deepens
    seen = []
    runner.process_video = lambda vid, cfg: seen.append(vid) or True
    made, because = runner.build_until_buffered(CFG)
    check(made == 1, f"it built one and then stopped ({made})")
    check("did not reach YouTube" in because, f"naming the real problem ({because})")


def test_depth_counts_whole_days_ahead():
    print("test_depth_counts_whole_days_ahead")
    now = datetime(2026, 10, 13, 20, 0, tzinfo=timezone.utc)
    for days in (0, 1, 5):
        db.last_scheduled_publish = (
            lambda d=days: now + timedelta(days=d) if d else now)
        got = REAL_DEPTH(CFG, now=now)
        check(got == days, f"{days} day(s) ahead reads as {got}")
    db.last_scheduled_publish = lambda: None
    check(REAL_DEPTH(CFG, now=now) == 0, "an empty queue is zero days")


def test_a_deferred_video_is_carried_on_and_not_counted_as_a_failure(tmp):
    print("test_a_deferred_video_is_carried_on_and_not_counted_as_a_failure")
    _fresh(tmp, "deferred")
    runner.queue_depth_days = lambda cfg, now=None: 2
    vid = db.create_video("waiting for the allowance")
    db.update_video(vid, status="images")
    real = runner.process_video
    seen = []
    runner.process_video = lambda v, cfg: seen.append(v) or None
    try:
        made, because = runner.build_until_buffered({**CFG, "buffer_days": 5})
    finally:
        runner.process_video = real
    check(seen == [vid], f"the waiting video was the one carried on with ({seen})")
    check(made == 0 and "waiting for the image allowance" in because,
          f"the run stops and says why ({because})")
    check((db.get_video(vid).get("attempts") or 0) == 0,
          "and the wait did not use up one of its attempts")
    again = db.oldest_unfinished(runner.MAX_ATTEMPTS)
    check(again is not None and again["id"] == vid, "so the next run picks it up again")


def test_process_video_returns_none_when_a_stage_defers(tmp):
    print("test_process_video_returns_none_when_a_stage_defers")
    _fresh(tmp, "defer_stage")
    from core.concede import Deferred
    vid = db.create_video("t")
    stages = runner.STAGES

    class Stage:
        def __init__(self, deferred=False):
            self.deferred = deferred

        def run(self, video_id, cfg):
            if self.deferred:
                raise Deferred("Cloudflare's free allowance is spent")
            return True
    runner.STAGES = [("topic", Stage(), "pending"), ("images", Stage(True), "images"),
                     ("render", Stage(), "rendering")]
    try:
        got = REAL_PROCESS(vid, CFG)
    finally:
        runner.STAGES = stages
    video = db.get_video(vid)
    check(got is None, f"process_video reports a deferral, not a failure ({got})")
    check(video["status"] == "images" and not video.get("error"),
          f"the video waits at the images stage, not failed ({video['status']})")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        real_process, real_depth, real_last = (
            runner.process_video, runner.queue_depth_days, db.last_scheduled_publish)
        try:
            test_stops_when_the_buffer_is_full(tmp)
            test_builds_until_the_buffer_fills(tmp)
            test_stops_on_a_video_that_cannot_be_assembled(tmp)
            test_stops_when_a_finished_video_never_reaches_youtube(tmp)
            test_depth_counts_whole_days_ahead()
            test_a_deferred_video_is_carried_on_and_not_counted_as_a_failure(tmp)
            test_process_video_returns_none_when_a_stage_defers(tmp)
        finally:
            runner.process_video, runner.queue_depth_days = real_process, real_depth
            db.last_scheduled_publish = real_last
    if failures:
        print(f"\n{len(failures)} BUFFER TEST(S) FAILED")
        sys.exit(1)
    print("\nALL BUFFER TESTS PASSED")
