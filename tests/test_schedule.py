"""Offline tests for the twice-a-week schedule. Run: python tests/test_schedule.py

Since 14 November 2026: Sunday and Wednesday at 21:00 Kyiv, a Short at the same moment (it
takes the video's own publishAt), and a video is built only for a slot, never more than one
ahead. Videos already scheduled on YouTube are left exactly as they are.
"""
import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import db
import ci_should_run
import check_queue_depth
import run as runner
from pipeline import upload

utc = timezone.utc
CFG = {"timezone": "Europe/Kyiv", "publish_time": "21:00", "publish_weekdays": ["Sun", "Wed"],
       "queue_videos": 1}
failures = []


def check(ok, msg):
    print(f"  {'ok' if ok else 'FAIL'}: {msg}")
    if not ok:
        failures.append(msg)


def fresh(tmp, name):
    db.DB_PATH = tmp / f"{name}.db"
    db.init_db()


def add_scheduled(when_utc, title="v"):
    vid = db.create_video(title)
    db.update_video(vid, youtube_id=f"yt{vid}", published_at=when_utc, status="published")
    return vid


def local(slot):
    return datetime.fromisoformat(slot.replace("Z", "+00:00")).astimezone(
        upload.zone("Europe/Kyiv"))


def test_slots_are_sunday_and_wednesday_at_21_kyiv():
    print("test_slots_are_sunday_and_wednesday_at_21_kyiv")
    saturday = datetime(2026, 11, 14, 21, 40, tzinfo=utc)
    check(upload.publish_at(CFG, saturday, taken=[]) == "2026-11-15T19:00:00Z",
          "with nothing taken, Saturday evening's next slot is Sunday 15 Nov 21:00 Kyiv (19:00Z)")
    check(upload.publish_at(CFG, saturday, taken=[datetime(2026, 11, 15, 19, tzinfo=utc)])
          == "2026-11-18T19:00:00Z", "Sunday taken: the next is Wednesday 18 Nov")
    check(upload.publish_at(CFG, saturday, taken=[datetime(2026, 11, 15, 19, tzinfo=utc),
                                                  datetime(2026, 11, 18, 19, tzinfo=utc)])
          == "2026-11-22T19:00:00Z", "Wednesday taken too: Sunday 22 Nov")
    day, seen, taken = saturday, set(), []
    for _ in range(14):
        slot = upload.publish_at(CFG, day, taken=taken)
        taken.append(datetime.fromisoformat(slot.replace("Z", "+00:00")))
        seen.add(local(slot).strftime("%a %H:%M"))
    check(seen == {"Sun 21:00", "Wed 21:00"}, f"fourteen slots land only on Sun/Wed 21:00 ({seen})")
    summer = upload.publish_at(CFG, datetime(2026, 6, 1, 8, tzinfo=utc), taken=[])
    check(local(summer).strftime("%a %H:%M") == "Wed 21:00" and summer.endswith("T18:00:00Z"),
          f"in summer 21:00 Kyiv is 18:00Z ({summer})")
    old = upload.publish_at({**CFG, "timezone": "Europe/Kiev"}, saturday, taken=[])
    check(old == "2026-11-15T19:00:00Z", "the old zone name still resolves")
    every_day = upload.publish_at({**CFG, "publish_weekdays": None}, saturday, taken=[])
    check(every_day == "2026-11-15T19:00:00Z" and upload.publish_at(
        {**CFG, "publish_weekdays": None}, saturday, taken=[datetime(2026, 11, 15, 19, tzinfo=utc)]
    ) == "2026-11-16T19:00:00Z", "without publish_weekdays it is daily, as before")


def test_the_real_config_and_the_videos_already_scheduled(tmp):
    print("test_the_real_config_and_the_videos_already_scheduled")
    from core.config import load_config
    cfg = load_config()
    check(cfg["publish_weekdays"] == ["Sun", "Wed"] and cfg["publish_time"] == "21:00"
          and cfg["queue_videos"] == 1 and cfg["videos_per_week"] == 2,
          "config.json says Sun/Wed 21:00, one video ahead, two a week")
    fresh(tmp, "real")
    # What the channel really has on 14 November: video 51 for Sunday 15 November.
    add_scheduled("2026-11-15T19:00:00Z", "already on YouTube")
    now = datetime(2026, 11, 14, 21, 40, tzinfo=utc)
    check(len(db.scheduled_after(now)) == 1, "one video is scheduled")
    check(upload.publish_at(cfg, now, taken=db.scheduled_publish_times())
          == "2026-11-18T19:00:00Z", "the next video's slot is Wednesday 18 Nov 21:00 Kyiv")
    ahead, target, unit = runner.queue_depth(cfg, now)
    check((ahead, target) == (1, 1), f"the queue is full at one video ({ahead}/{target} {unit})")
    check(db.scheduled_publish_times() == [datetime(2026, 11, 15, 19, tzinfo=utc)],
          "and the scheduled video's time was not touched")


def test_it_builds_only_when_nothing_is_scheduled_ahead(tmp):
    print("test_it_builds_only_when_nothing_is_scheduled_ahead")
    fresh(tmp, "build")
    add_scheduled("2026-11-15T19:00:00Z")
    before = datetime(2026, 11, 15, 8, 0, tzinfo=utc)     # Sunday morning: still scheduled
    run_it, why = ci_should_run.decide(before, queue_videos=1)
    check(not run_it and "already scheduled" in why,
          f"Sunday morning, the video is not out yet: nothing is built ({why})")
    after = datetime(2026, 11, 16, 1, 30, tzinfo=utc)     # Monday, after it went public
    run_it, why = ci_should_run.decide(after, queue_videos=1)
    check(run_it, f"Monday, the queue is empty: the next video is built ({why})")

    made = []
    real = runner.process_video
    state = {"scheduled": 0}
    real_count = runner.queue_count
    runner.queue_count = lambda now=None: state["scheduled"]

    def produce(vid, cfg):
        state["scheduled"] += 1
        made.append(vid)
        return True
    runner.process_video = produce
    try:
        n, because = runner.build_until_buffered({**CFG, "run_time_budget_min": 240})
    finally:
        runner.process_video, runner.queue_count = real, real_count
    check(n == 1 and len(made) == 1, f"one run builds exactly one video ({n})")
    check("1 videos deep" in because, f"and stops because the queue holds one ({because})")


def test_the_evening_check_looks_at_the_next_slot(tmp):
    print("test_the_evening_check_looks_at_the_next_slot")
    fresh(tmp, "alert")
    cfg = {**CFG}

    def asked(now, scheduled):
        for w in scheduled:
            add_scheduled(w)
        slot = check_queue_depth.next_slot(now, cfg)
        rows = check_queue_depth.upcoming(now)
        covered = any(abs((w - slot).total_seconds()) < 3600 for w, _t, _v in rows)
        hours = (slot - now).total_seconds() / 3600
        return slot, covered, hours

    sunday_evening = datetime(2026, 11, 15, 19, 11, tzinfo=utc)       # just after it went out
    slot, covered, hours = asked(sunday_evening, [])
    check(slot.strftime("%a") == "Wed" and not covered and hours > 60,
          f"Sunday evening, empty queue: the next slot is Wednesday, {hours:.0f} h away - "
          f"above 60 h, so no alert")
    monday_evening = datetime(2026, 11, 16, 19, 11, tzinfo=utc)
    slot, covered, hours = asked(monday_evening, [])
    check(not covered and hours < 60, f"Monday evening with nothing built: {hours:.0f} h left "
                                      f"and no video - this is the alert")
    add_scheduled("2026-11-18T19:00:00Z")
    slot, covered, hours = asked(monday_evening, [])
    check(covered, "once the Wednesday video is scheduled the check is satisfied")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_slots_are_sunday_and_wednesday_at_21_kyiv()
        test_the_real_config_and_the_videos_already_scheduled(tmp)
        test_it_builds_only_when_nothing_is_scheduled_ahead(tmp)
        test_the_evening_check_looks_at_the_next_slot(tmp)
    if failures:
        print(f"\n{len(failures)} SCHEDULE TEST(S) FAILED")
        sys.exit(1)
    print("\nALL SCHEDULE TESTS PASSED")
