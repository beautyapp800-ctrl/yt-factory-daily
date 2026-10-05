"""Offline tests for the automation: resuming a video, the run notification, the weekly report.

Run: python tests/test_ci.py
No network, no ffmpeg: stages are replaced by recorders, the database is a temp file.
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

from core import config, db
import run
import ci_state
import ci_should_run
import heartbeat
import notify
import weekly_report as weekly

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def _fresh(tmp, name):
    db.DB_PATH = tmp / f"{name}.db"
    config.OUTPUT_DIR = tmp / f"out_{name}"
    db.init_db()


def _video(created_days_ago=0, **fields):
    vid = db.create_video("topic")
    created = (datetime.now(timezone.utc) - timedelta(days=created_days_ago)).isoformat(timespec="seconds")
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.execute("UPDATE videos SET created_at = ? WHERE id = ?", (created, vid))
        conn.commit()
    if fields:
        db.update_video(vid, **fields)
    return vid


# --- which video a run works on ----------------------------------------------------------

def test_next_id_is_the_sequence_not_the_highest_survivor(tmp):
    print("test_next_id_is_the_sequence_not_the_highest_survivor")
    _fresh(tmp, "seq")
    check(ci_state.next_video_id() == 1, "an empty database starts at 1")
    for _ in range(3):
        _video()
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.execute("DELETE FROM videos WHERE id = 3")
        conn.commit()
    check(ci_state.next_video_id() == 4,
          "AUTOINCREMENT does not reuse a deleted id, so the cache key matches the video the "
          "run will really create")


def test_an_unfinished_video_is_resumed_and_the_rest_are_not(tmp):
    print("test_an_unfinished_video_is_resumed_and_the_rest_are_not")
    _fresh(tmp, "pick")
    done = _video(status="published", youtube_id="abc")
    exhausted = _video(status="failed", attempts=db.MAX_ATTEMPTS)
    stale = _video(created_days_ago=9, status="failed", attempts=1)
    live = _video(status="failed", attempts=1)
    newer = _video(status="scripting", attempts=0)
    picked = db.oldest_unfinished(db.MAX_ATTEMPTS)
    check(picked["id"] == live,
          f"the oldest video that has attempts left and is recent is chosen (video {picked['id']})")
    check(picked["id"] not in (done, exhausted, stale),
          "not one already on YouTube, not one out of attempts, not one over a week old")
    db.update_video(live, attempts=db.MAX_ATTEMPTS)
    check(db.oldest_unfinished(db.MAX_ATTEMPTS)["id"] == newer, "and then the next one")
    db.update_video(newer, attempts=db.MAX_ATTEMPTS)
    check(db.oldest_unfinished(db.MAX_ATTEMPTS) is None, "with none left, a new video is started")


# --- resuming skips what is done ---------------------------------------------------------

class Stage:
    def __init__(self, name, log, fail_first=False):
        self.name, self.log, self.fail = name, log, fail_first

    def run(self, video_id, cfg):
        self.log.append(self.name)
        if self.fail:
            self.fail = False
            raise RuntimeError(f"{self.name} fell over")
        return True


class Voicer:
    """A tts stage that sets the narration's length from a script of durations, and a script
    stage that counts how many times it was asked to write."""

    def __init__(self, lengths_min, log):
        self.lengths, self.log, self.calls = list(lengths_min), log, 0

    def run(self, video_id, cfg):
        minutes = self.lengths[min(self.calls, len(self.lengths) - 1)]
        self.calls += 1
        self.log.append(f"tts({minutes})")
        db.update_video(video_id, duration_s=minutes * 60)
        return True


def test_a_failed_run_resumes_at_the_stage_that_failed(tmp):
    print("test_a_failed_run_resumes_at_the_stage_that_failed")
    _fresh(tmp, "resume")
    vid = _video()
    log = []
    # the tts stage has to leave a publishable length behind, or the length gate rejects it
    stages = [(n, Voicer([30.0], log) if n == "tts" else Stage(n, log, fail_first=(n == "render")), s)
              for n, s in (("topic", "pending"), ("script", "scripting"), ("tts", "tts"),
                           ("images", "images"), ("render", "rendering"), ("upload", "uploading"))]
    original = run.STAGES
    run.STAGES = stages
    try:
        check(run.process_video(vid, {}) is False, "the first run fails at the render")
        first = list(log)
        log.clear()
        check(run.process_video(vid, {}) is True, "the second run completes")
    finally:
        run.STAGES = original
    check(first == ["topic", "script", "tts(30.0)", "images", "render"], f"first run: {first}")
    check(log == ["render", "upload"],
          f"second run starts at the render, so the pictures are not drawn again ({log})")
    check(db.get_video(vid)["status"] == "ready", "and the video ends ready")
    check({"topic", "script", "tts", "images"} <= db.finished_stages(vid),
          "finished stages come from the events log")

    # A video that reached YouTube must not be relabelled "ready" by the end of the run.
    # Caught on the first real run: the upload stage set published and this overwrote it.
    db.update_video(vid, youtube_id="abc123")
    original = run.STAGES
    run.STAGES = [("topic", Stage("topic", []), "pending")]
    try:
        run.process_video(vid, {})
    finally:
        run.STAGES = original
    check(db.get_video(vid)["status"] == "published",
          "a video with a youtube_id ends published, not ready")


# --- the length gate ---------------------------------------------------------------------

def _gate_run(tmp, name, lengths, gate=None):
    _fresh(tmp, name)
    vid = _video()
    log = []
    stages = [("topic", Stage("topic", log), "pending"), ("script", Stage("script", log), "scripting"),
              ("tts", Voicer(lengths, log), "tts"), ("images", Stage("images", log), "images"),
              ("render", Stage("render", log), "rendering")]
    original = run.STAGES
    run.STAGES = stages
    cfg = {"length_gate": gate} if gate else {}
    try:
        ok = run.process_video(vid, cfg)
    finally:
        run.STAGES = original
    return vid, ok, log


def test_a_script_of_the_wrong_length_is_written_again(tmp):
    print("test_a_script_of_the_wrong_length_is_written_again")
    vid, ok, log = _gate_run(tmp, "short", [25.0, 30.0])
    check(ok, "the run succeeds once a script comes out in the window")
    check(log == ["topic", "script", "tts(25.0)", "script", "tts(30.0)", "images", "render"],
          f"25 minutes is too short, so the script and voice are redone and the topic is not ({log})")
    check(db.get_video(vid)["status"] == "ready", "and the video ends ready")
    check({"script", "tts"} <= db.finished_stages(vid),
          "the redone stages count as finished again, because the newest record says so")


def test_the_window_is_27_to_33_inclusive(tmp):
    print("test_the_window_is_27_to_33_inclusive")
    for minutes, expected in ((26.9, False), (27.0, True), (30.0, True), (33.0, True), (33.1, False)):
        vid, ok, log = _gate_run(tmp, f"edge{minutes}", [minutes, minutes, minutes])
        check(ok == expected, f"{minutes} minutes is {'accepted' if expected else 'rejected'}")


def test_three_bad_scripts_in_a_row_fail_the_video(tmp):
    print("test_three_bad_scripts_in_a_row_fail_the_video")
    vid, ok, log = _gate_run(tmp, "never", [25.0, 24.0, 36.0])
    check(not ok, "the run fails")
    check(log.count("script") == 3 and "images" not in log,
          f"three scripts were written and the pictures were never paid for ({log})")
    video = db.get_video(vid)
    check(video["status"] == "failed" and "gave up after 3 scripts" in video["error"],
          f"the error says why ({video['error']})")


def test_the_gate_reads_its_window_from_config(tmp):
    print("test_the_gate_reads_its_window_from_config")
    vid, ok, log = _gate_run(tmp, "narrow", [30.0], gate={"min_minutes": 31, "max_minutes": 33, "attempts": 1})
    check(not ok, "a 30 minute narration fails a 31-33 window, with one attempt allowed")


# --- the daily message -------------------------------------------------------------------

def _events(vid, *rows):
    for stage, level, message in rows:
        db.log_event(vid, stage, level, message)


def test_the_success_message_says_where_and_when(tmp):
    print("test_the_success_message_says_where_and_when")
    _fresh(tmp, "ok")
    vid = _video(status="published", youtube_id="AbC123", seo_title="10 Stoic Lessons That Matter",
                 published_at="2026-10-13T18:00:00Z", duration_s=1800.0, attempts=1)
    _events(vid, ("images", "info", "70 made, 0 reused, 0 redrawn, 0 duplicated, 0 failed of 70 "
                                   "planned ({}), 3 prompts rewritten, 6720 neurons over 70 Cloudflare calls"))
    video = notify.latest_video()
    text = notify.build_message("success", video, notify.run_events(vid), "https://example/run/1",
                                owner="alice", now=NOW)
    check(text.startswith("@alice "), "it mentions the owner, which is what makes it an email")
    for needle in ("10 Stoic Lessons That Matter", "https://youtu.be/AbC123",
                   "13.10.2026 18:00 UTC", "30.0 min", "6720", "https://example/run/1"):
        check(needle in text, f"contains: {needle}")


def test_the_failure_message_says_which_stage_and_what_happens_next(tmp):
    print("test_the_failure_message_says_which_stage_and_what_happens_next")
    _fresh(tmp, "bad")
    vid = _video(status="failed", error="render: ffmpeg could not join the clips", attempts=1,
                 title="Some Title")
    _events(vid, ("render", "error", "ffmpeg could not join the clips"))
    text = notify.build_message("failure", notify.latest_video(), notify.run_events(vid),
                                "https://example/run/2", owner="alice", now=NOW)
    check("FAILED" in text and "**render**" in text, "it names the failed stage")
    check("ffmpeg could not join the clips" in text, "and the error text")
    check("Attempt 1 of 3" in text and "2 attempt(s) left" in text,
          "and how many attempts remain, with what the next run will do")
    db.update_video(vid, attempts=db.MAX_ATTEMPTS)
    text = notify.build_message("failure", notify.latest_video(), notify.run_events(vid), "u", now=NOW)
    check("will not be retried" in text, "a video out of attempts says it will not be retried")
    check("without a video" in notify.build_message("failure", None, [], "u", now=NOW),
          "a run that recorded nothing says so rather than crashing")


def test_a_run_that_never_started_does_not_describe_an_old_video(tmp):
    print("test_a_run_that_never_started_does_not_describe_an_old_video")
    _fresh(tmp, "nostart")
    _video(status="published", youtube_id="old", seo_title="Last Week's Video", attempts=1)
    text = notify.build_message("not-started", notify.latest_video(), [], "https://example/run/9",
                                owner="alice", now=NOW)
    check("did not start" in text and "preflight" in text, "it says the preflight failed")
    check("Last Week's Video" not in text and "youtu.be" not in text,
          "and does not dress an old video up as this run's result")


def test_neurons_are_summed_across_runs():
    print("test_neurons_are_summed_across_runs")
    events = [{"message": "40 made, 3000 neurons over 31 Cloudflare calls"},
              {"message": "30 reused, 1250 neurons over 13 Cloudflare calls"},
              {"message": "stage finished"}]
    check(notify.neurons_in(events) == 4250, "a video that took two runs reports both")


# --- did it actually go public? ---------------------------------------------------------

import check_published as pub


class FakeOEmbed:
    """urlopen stand-in: 200 with a title for a public id, 403 for a private one."""

    def __init__(self, public_ids, error=None):
        self.public, self.error, self.asked = set(public_ids), error, []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.asked.append(url)
        if self.error:
            raise self.error
        video_id = url.split("watch?v=")[1].split("&")[0]
        if video_id in self.public:
            import io, json as j
            body = j.dumps({"title": f"title of {video_id}"}).encode()

            class R:
                def read(self_inner): return body
                def __enter__(self_inner): return self_inner
                def __exit__(self_inner, *a): return False
            return R()
        import urllib.error
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)


def _scheduled(when, youtube_id, title):
    vid = _video(status="published", youtube_id=youtube_id, seo_title=title,
                 published_at=when.isoformat(timespec="seconds").replace("+00:00", "Z"))
    return vid


def test_a_video_that_went_public_is_confirmed_once(tmp):
    print("test_a_video_that_went_public_is_confirmed_once")
    _fresh(tmp, "pubok")
    vid = _scheduled(NOW - timedelta(hours=2), "goodid", "Published Fine")
    due = pub.due_videos(NOW, grace_minutes=30)
    check([v["id"] for v in due] == [vid], f"a video past its moment is due to be checked ({due})")

    public, detail = pub.is_public("goodid", FakeOEmbed(["goodid"]))
    check(public is True and "title of goodid" in detail, f"oEmbed says public ({detail})")
    db.log_event(vid, "publish", "info", pub.CONFIRMED)
    check(pub.due_videos(NOW, 30) == [], "once confirmed it is never checked again")


def test_a_video_still_private_raises_the_alarm(tmp):
    print("test_a_video_still_private_raises_the_alarm")
    _fresh(tmp, "pubbad")
    _scheduled(NOW - timedelta(hours=2), "privateid", "Should Have Published")
    results = []
    for v in pub.due_videos(NOW, 30):
        ok, detail = pub.is_public(v["youtube_id"], FakeOEmbed([]))
        results.append({"id": v["id"], "youtube_id": v["youtube_id"], "title": v["seo_title"],
                        "published_at": v["published_at"], "public": ok, "detail": detail})
    check(results[0]["public"] is False, "a private video reads as not public")
    report = pub.build_report(results, NOW, owner="alice", run_url="u")
    check(report and "@alice" in report and "Should Have Published" in report,
          "the mail names the video and the owner")
    check("audit" in report and "YouTube Studio" in report,
          "and says the likely cause and what to do about it tonight")
    check("https://youtu.be/privateid" in report, "with a link straight to it")


def test_nothing_is_said_when_all_is_well(tmp):
    print("test_nothing_is_said_when_all_is_well")
    _fresh(tmp, "pubquiet")
    good = [{"id": 1, "youtube_id": "a", "title": "Fine", "published_at": "x",
             "public": True, "detail": "public"}]
    check(pub.build_report(good, NOW) is None,
          "a daily all-is-well mail would be ignored within a week, so none is sent")


def test_a_video_not_yet_due_is_left_alone(tmp):
    print("test_a_video_not_yet_due_is_left_alone")
    _fresh(tmp, "pubearly")
    _scheduled(NOW + timedelta(days=3), "futureid", "Next Week")
    check(pub.due_videos(NOW, 30) == [], "a video whose moment has not come is not checked")
    _scheduled(NOW - timedelta(minutes=5), "justnow", "Five Minutes Ago")
    check(pub.due_videos(NOW, 30) == [],
          "nor one published five minutes ago, inside the grace period")


def test_a_network_failure_is_not_a_failed_publication(tmp):
    print("test_a_network_failure_is_not_a_failed_publication")
    public, detail = pub.is_public("anything", FakeOEmbed([], error=OSError("dns is down")))
    check(public is None and "could not check" in detail,
          f"an unanswerable question is not an answer of no ({detail})")
    results = [{"id": 1, "youtube_id": "a", "title": "Unknown", "published_at": "x",
                "public": None, "detail": detail}]
    report = pub.build_report(results, NOW)
    check(report and "not treated as a failure" in report,
          "it is reported, but not as a video that failed to publish")


def test_a_video_nobody_means_to_publish_can_be_retired(tmp):
    print("test_a_video_nobody_means_to_publish_can_be_retired")
    _fresh(tmp, "pubignore")
    vid = _scheduled(NOW - timedelta(days=2), "testupload", "A Test Upload")
    check(len(pub.due_videos(NOW, 30)) == 1, "a test upload would otherwise be reported nightly")
    db.log_event(vid, "publish", "info", pub.IGNORED)
    check(pub.due_videos(NOW, 30) == [], "marking it retires it from the check for good")


# --- the weekly report -------------------------------------------------------------------

def test_the_weekly_report_counts_the_queue_and_alarms_on_empty(tmp):
    print("test_the_weekly_report_counts_the_queue_and_alarms_on_empty")
    _fresh(tmp, "weekly")
    future = lambda d: (NOW + timedelta(days=d)).isoformat(timespec="seconds").replace("+00:00", "Z")
    past = (NOW - timedelta(days=2)).isoformat(timespec="seconds").replace("+00:00", "Z")
    _video(status="published", youtube_id="a", published_at=future(1), seo_title="Tomorrow")
    _video(status="published", youtube_id="b", published_at=future(3), seo_title="In three days")
    _video(status="published", youtube_id="c", published_at=past, seo_title="Already public")
    _video(status="failed")                                   # not on YouTube: not queued
    text, queued = weekly.build_report(NOW, "alice", "u", (True, "open"))
    check(queued == 2, f"two videos are queued ahead, the public one and the failed one are not ({queued})")
    check("warning: only 2 video(s) queued ahead" in text, "fewer than 3 is a warning")
    check(text.index("Tomorrow") < text.index("In three days"), "soonest first")
    check("Already public" not in text, "a video whose moment has passed is not in the queue")
    check("not asked of YouTube" in text, "and the report admits it reads its own records")

    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.execute("UPDATE videos SET published_at = ? WHERE youtube_id IN ('a','b')", (past,))
        conn.commit()
    text, queued = weekly.build_report(NOW, "", "", (None, "not run"))
    check(queued == 0 and "ALARM" in text, "an empty queue is an alarm")


def test_the_weekly_report_lists_failures_and_the_neurons_we_logged(tmp):
    print("test_the_weekly_report_lists_failures_and_the_neurons_we_logged")
    _fresh(tmp, "weekly2")
    vid = _video(status="failed")
    db.log_event(vid, "tts", "error", "edge-tts got a 403")
    db.log_event(vid, "tts", "error", "edge-tts got a 403 again")
    db.log_event(vid, "images", "info", "50 made, 4800 neurons over 50 Cloudflare calls")
    now = datetime.now(timezone.utc)
    text, _ = weekly.build_report(now, "", "", (False, "BLOCKED (HTTP 429)"))
    check(text.count("edge-tts got a 403") == 1, "a repeated failure at one stage is listed once")
    check("Failed runs in the last 7 days: 1" in text, "and counted once")
    check("4800" in text and "BLOCKED" in text, "Neurons logged in the last day, and the live probe")
    check("lower bound" in text, "and the report says the neuron figure is a lower bound")


# --- not running at all, which is the failure nothing else can see ------------------------

def _finished_upload(vid, when):
    """Mark a video's upload stage as finished at a given moment, the way run.py does."""
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.execute("INSERT INTO events (video_id, stage, level, message, created_at) "
                     "VALUES (?, 'upload', 'info', 'stage finished', ?)",
                     (vid, when.isoformat(timespec="seconds")))
        conn.commit()


def test_the_second_cron_only_works_when_the_first_one_did_not(tmp):
    print("test_the_second_cron_only_works_when_the_first_one_did_not")
    _fresh(tmp, "guard")
    now = datetime.now(timezone.utc)

    run_it, reason = ci_should_run.decide(now)
    check(run_it and "nothing has been finished" in reason,
          f"an empty day is a day to work ({reason})")

    vid = _video(status="published", youtube_id="abc", attempts=1)
    _finished_upload(vid, now)
    run_it, reason = ci_should_run.decide(now)
    check(not run_it, f"with today's video already delivered the second cron stands down ({reason})")

    # Yesterday's delivery is not today's: the schedule must not skip a day on the strength of it.
    _fresh(tmp, "guard2")
    old = _video(status="published", youtube_id="abc", attempts=1)
    _finished_upload(old, now - timedelta(days=1))
    run_it, _ = ci_should_run.decide(now)
    check(run_it, "yesterday's video does not excuse today")

    # A half-built video outranks everything: resuming is what keeps the paid-for work.
    _fresh(tmp, "guard3")
    done = _video(status="published", youtube_id="abc", attempts=1)
    _finished_upload(done, now)
    _video(status="failed", attempts=1)
    run_it, reason = ci_should_run.decide(now)
    check(run_it and "unfinished" in reason,
          f"an unfinished video is picked up even on a day that already delivered ({reason})")


def test_the_heartbeat_speaks_the_day_the_queue_stops_growing(tmp):
    print("test_the_heartbeat_speaks_the_day_the_queue_stops_growing")
    _fresh(tmp, "beat")
    now = datetime.now(timezone.utc)

    # A good day: something was finished, and there is cover ahead.
    vid = _video(status="published", youtube_id="abc", seo_title="Today's video",
                 published_at=(now + timedelta(days=3)).isoformat(timespec="seconds"))
    _finished_upload(vid, now)
    for extra in (4, 5):
        _video(status="published", youtube_id=f"x{extra}", seo_title=f"In {extra}",
               published_at=(now + timedelta(days=extra)).isoformat(timespec="seconds"))
    message, alert = build(now)
    check(not alert, "a day that produced a video and has cover raises nothing")
    check("Today's video" in message, "and names what was added")

    # The same queue, but nothing produced today: that is the alarm, on the day it happens.
    _fresh(tmp, "beat2")
    for extra in (3, 4):
        _video(status="published", youtube_id=f"y{extra}",
               published_at=(now + timedelta(days=extra)).isoformat(timespec="seconds"))
    db.log_event(_video(status="failed"), "script", "error", "outline still breaks the rules")
    message, alert = build(now)
    check(alert, "no new video in the window is an alert the same day")
    check("No new video reached the queue" in message, "and says so plainly")
    check("What to do" in message and "writes a fresh one" in message,
          "with the action for the failure that caused it, not a link to a log")

    # Nothing ran at all: a dropped cron, which the daily workflow itself can never report.
    _fresh(tmp, "beat3")
    _video(status="published", youtube_id="z",
           published_at=(now + timedelta(days=2)).isoformat(timespec="seconds"))
    message, alert = build(now)
    check(alert and "Nothing ran in the last" in message,
          "a schedule that never fired is reported as the schedule failing")

    # An empty queue is the loudest case: the next bad day is a day with no video.
    _fresh(tmp, "beat4")
    message, alert = build(now)
    check(alert and "Queue: **empty**" in message, "an empty queue is called empty")


def build(now):
    """The heartbeat report, without the network check, at a fixed moment."""
    return heartbeat.build_report(now=now, check_token=False)


def test_every_failure_has_an_action_or_heals_itself():
    print("test_every_failure_has_an_action_or_heals_itself")
    from core import failures
    seen = [
        ("images", "cloudflare HTTP 429: quota exceeded", "cloudflare-quota", True),
        ("upload", "HttpError 403 uploadLimitExceeded", "youtube-upload-quota", True),
        ("preflight", "the YouTube token does not work: invalid_grant", "youtube-auth", False),
        ("preflight", "secrets not set in this workflow: GROQ_API_KEY", "secret-missing", False),
        ("tts", "edge-tts cannot speak from this machine: 403", "tts-blocked", False),
        ("script", "outline still breaks the title rules", "script-quality", True),
        ("render", "ffmpeg exited with 1", "render", False),
        ("upload", "cloudflare network error: timed out", "service-down", True),
    ]
    for stage, message, expected, heals in seen:
        kind, self_healing, action = failures.classify(stage, message)
        check(kind == expected, f"{message[:38]!r} is a {expected}")
        check(self_healing is heals, f"and {'heals itself' if heals else 'needs a person'}")
        check(len(action) > 40, "and carries a sentence of instruction, not a shrug")
    kind, _, action = failures.classify("thumbnail", "something nobody has seen")
    check(kind == "unknown" and "core/failures.py" in action,
          "an unrecognised failure says so, rather than being silently mis-sorted")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_next_id_is_the_sequence_not_the_highest_survivor(tmp)
        test_an_unfinished_video_is_resumed_and_the_rest_are_not(tmp)
        test_a_failed_run_resumes_at_the_stage_that_failed(tmp)
        test_a_script_of_the_wrong_length_is_written_again(tmp)
        test_the_window_is_27_to_33_inclusive(tmp)
        test_three_bad_scripts_in_a_row_fail_the_video(tmp)
        test_the_gate_reads_its_window_from_config(tmp)
        test_the_success_message_says_where_and_when(tmp)
        test_the_failure_message_says_which_stage_and_what_happens_next(tmp)
        test_a_run_that_never_started_does_not_describe_an_old_video(tmp)
        test_neurons_are_summed_across_runs()
        test_a_video_that_went_public_is_confirmed_once(tmp)
        test_a_video_still_private_raises_the_alarm(tmp)
        test_nothing_is_said_when_all_is_well(tmp)
        test_a_video_not_yet_due_is_left_alone(tmp)
        test_a_network_failure_is_not_a_failed_publication(tmp)
        test_a_video_nobody_means_to_publish_can_be_retired(tmp)
        test_the_weekly_report_counts_the_queue_and_alarms_on_empty(tmp)
        test_the_weekly_report_lists_failures_and_the_neurons_we_logged(tmp)
        test_the_second_cron_only_works_when_the_first_one_did_not(tmp)
        test_the_heartbeat_speaks_the_day_the_queue_stops_growing(tmp)
        test_every_failure_has_an_action_or_heals_itself()
    print("ALL CI TESTS PASSED")
