"""Offline tests for the upload stage and publish.py. Run: python tests/test_upload.py

No network and no Google account: the YouTube service is replaced by fakes that behave the
way googleapiclient's resumable requests do - a scripted sequence of chunk results,
failures and server answers - so what is tested is this project's own logic: the request it
builds, the two mandatory flags, retrying and resuming, the database writes, the queue.
"""
import json
import sys
import tempfile
from pathlib import Path

import httplib2
from googleapiclient.errors import HttpError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import config, db
import pipeline.upload as up
import publish


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# --- fakes ---------------------------------------------------------------------------

def http_error(status, reason="", message=""):
    content = json.dumps({"error": {"message": message,
                                    "errors": [{"reason": reason}] if reason else []}}).encode()
    return HttpError(httplib2.Response({"status": str(status)}), content)


class Status:
    def __init__(self, fraction):
        self._f = fraction

    def progress(self):
        return self._f


class FakeHttp:
    """Answers the 'how much do you have of this upload?' probe."""

    def __init__(self, status=308, headers=None, body=b""):
        self.answer = (httplib2.Response({"status": str(status), **(headers or {})}), body)
        self.calls = []

    def request(self, uri, method, headers=None):
        self.calls.append((uri, method, headers))
        return self.answer


class FakeInsert:
    """A resumable videos.insert request: each next_chunk() consumes one scripted event."""

    def __init__(self, events, http=None):
        self.events = list(events)
        self.http = http or FakeHttp()
        self.resumable_uri = None
        self.resumable_progress = 0
        self.calls = 0
        self.on_call = None            # optional hook, called with the call number

    def next_chunk(self, num_retries=0):
        self.calls += 1
        if self.on_call:
            self.on_call(self.calls)
        kind, value = self.events.pop(0)
        if self.resumable_uri is None:
            self.resumable_uri = "https://upload.example/session/abc"
        if kind == "raise":
            raise value
        if kind == "progress":
            self.resumable_progress = int(value * 1000)
            return Status(value), None
        return Status(1.0), value                       # ("done", response)


class FakeService:
    def __init__(self, events, response=None, thumbnail_error=None, http=None):
        self.insert_request = FakeInsert(events, http)
        self.insert_kwargs = None
        self.thumbnail_calls = []
        self._thumbnail_error = thumbnail_error

    def videos(self):
        outer = self

        class V:
            def insert(self, **kw):
                outer.insert_kwargs = kw
                return outer.insert_request
        return V()

    def thumbnails(self):
        outer = self

        class T:
            def set(self, **kw):
                outer.thumbnail_calls.append(kw)

                class Exec:
                    def execute(_self):
                        if outer._thumbnail_error:
                            raise outer._thumbnail_error
                        return {}
                return Exec()
        return T()


OK_RESPONSE = {"id": "abc123XYZ", "status": {"privacyStatus": "private",
                                             "selfDeclaredMadeForKids": False,
                                             "containsSyntheticMedia": True}}
# schedule_publish off by default in the tests: most of them are about the upload itself,
# and a publishAt would put a moving timestamp in every comparison. The scheduling tests
# turn it on explicitly.
CFG = {"privacy_status": "private", "schedule_publish": False, "youtube": {"chunk_mb": 8}}
SCHEDULED = {**CFG, "schedule_publish": True, "publish_time": "21:00",
             "timezone": "Europe/Kiev", "publish_days_ahead": 3}


def _video(tmp, name="v", **over):
    media = tmp / f"{name}.mp4"
    media.write_bytes(b"\x00" * 4096)
    thumb = tmp / f"{name}.jpg"
    thumb.write_bytes(b"\xff\xd8\xff" + b"\x00" * 200)
    video = {"seo_title": "10 Stoic Lessons That Matter", "title": "working",
             "description": "Hook.\n\nSummary.", "tags": json.dumps(["stoicism", "stoic philosophy"]),
             "video_path": str(media), "thumbnail_path": str(thumb)}
    video.update(over)
    return video


def _db_video(tmp, name, **over):
    db.DB_PATH = tmp / f"{name}.db"
    config.OUTPUT_DIR = tmp / f"out_{name}"
    db.init_db()
    vid = db.create_video("upload test")
    fields = _video(tmp, name, **over)
    db.update_video(vid, status="ready", seo_title=fields["seo_title"], title=fields["title"],
                    description=fields["description"], tags=fields["tags"],
                    video_path=fields["video_path"], thumbnail_path=fields["thumbnail_path"])
    return vid


def _events(vid):
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM events WHERE video_id = ?", (vid,))]


# --- the request -----------------------------------------------------------------------

def test_body_has_what_was_asked_for(tmp):
    print("test_body_has_what_was_asked_for")
    body = up.build_body(_video(tmp), CFG)
    s, st = body["snippet"], body["status"]
    check(s["title"] == "10 Stoic Lessons That Matter", "the seo title is the YouTube title")
    check(s["description"].startswith("Hook.") and s["tags"] == ["stoicism", "stoic philosophy"],
          "description and tags come from the database (tags decoded from JSON)")
    check(s["categoryId"] == "22" and s["defaultLanguage"] == "en", "category 22, language en")
    check(st["privacyStatus"] == "private", "privacy comes from config.privacy_status")
    check(up.build_body(_video(tmp), {"privacy_status": "unlisted",
                                      "schedule_publish": False})["status"]["privacyStatus"]
          == "unlisted", "and changes with that one line")
    check(up.build_body(_video(tmp, seo_title=None), CFG)["snippet"]["title"] == "working",
          "with no seo title it falls back to the working title")
    try:
        up.build_body(_video(tmp), {"privacy_status": "everyone"})
        check(False, "a nonsense privacy value should raise")
    except up.UploadError:
        check(True, "a nonsense privacy value raises instead of reaching YouTube")


def test_a_scheduled_video_goes_up_private_for_the_right_moment(tmp):
    print("test_a_scheduled_video_goes_up_private_for_the_right_moment")
    from datetime import datetime, timezone as tz
    summer = up.publish_at(SCHEDULED, datetime(2026, 10, 3, 19, 0, tzinfo=tz.utc))
    winter = up.publish_at(SCHEDULED, datetime(2026, 1, 15, 10, 0, tzinfo=tz.utc))
    check(summer == "2026-10-06T18:00:00Z", f"three days ahead, 21:00 Kyiv in summer ({summer})")
    check(winter == "2026-01-18T19:00:00Z",
          f"and in winter, when Kyiv is an hour further from UTC ({winter}) - the same 21:00 "
          f"local either way, which a fixed offset would get wrong for half the year")

    late = up.publish_at(SCHEDULED, datetime(2026, 10, 3, 19, 30, tzinfo=tz.utc))
    check(late == "2026-10-06T18:00:00Z",
          f"running after 21:00 local still schedules three days out ({late})")

    status = up.build_body(_video(tmp), SCHEDULED,
                           datetime(2026, 10, 3, 19, 0, tzinfo=tz.utc))["status"]
    check(status["publishAt"] == summer, "the request carries the moment")
    check(status["privacyStatus"] == "private",
          "and goes up private, which is what YouTube requires of a scheduled video")
    public = up.build_body(_video(tmp), {**SCHEDULED, "privacy_status": "public"},
                           datetime(2026, 10, 3, 19, 0, tzinfo=tz.utc))["status"]
    check(public["privacyStatus"] == "private",
          "asking for public AND a schedule would be rejected, so private wins here")
    check("publishAt" not in up.build_body(_video(tmp), CFG)["status"],
          "with scheduling off there is no publishAt at all")


def test_the_two_flags_cannot_be_switched_off(tmp):
    print("test_the_two_flags_cannot_be_switched_off")
    for cfg in (CFG, {**CFG, "youtube": {"made_for_kids": True, "contains_synthetic_media": False,
                                          "selfDeclaredMadeForKids": True}}):
        st = up.build_body(_video(tmp), cfg)["status"]
        check(st["selfDeclaredMadeForKids"] is False and st["containsSyntheticMedia"] is True,
              "selfDeclaredMadeForKids=false and containsSyntheticMedia=true, whatever config says")
    body = up.build_body(_video(tmp), CFG)
    up.assert_mandatory_flags(body)
    for key, bad in (("selfDeclaredMadeForKids", True), ("containsSyntheticMedia", False)):
        tampered = json.loads(json.dumps(body))
        tampered["status"][key] = bad
        try:
            up.assert_mandatory_flags(tampered)
            check(False, f"{key}={bad} should be refused")
        except up.UploadError as e:
            check(key in str(e), f"{key}={bad} is refused before sending")
    gone = json.loads(json.dumps(body))
    del gone["status"]["containsSyntheticMedia"]
    try:
        up.assert_mandatory_flags(gone)
        check(False, "a missing flag should be refused")
    except up.UploadError:
        check(True, "a missing containsSyntheticMedia is refused too")


def test_metadata_problems_are_found_before_uploading(tmp):
    print("test_metadata_problems_are_found_before_uploading")
    video = _video(tmp)
    ok = up.build_body(video, CFG)
    check(up.check_metadata(ok, video["video_path"], video["thumbnail_path"]) == [], "good metadata passes")
    bad = up.build_body(_video(tmp, seo_title="T" * 101, description="d" * 5001,
                               tags=json.dumps(["x" * 30] * 20)), CFG)
    problems = " | ".join(up.check_metadata(bad, "missing.mp4", video["thumbnail_path"]))
    for needle in ("101 characters", "5000 bytes", "500 characters", "does not exist"):
        check(needle in problems, f"caught: {needle}")
    empty = up.build_body(_video(tmp, seo_title=None, title=None, description=None), CFG)
    check("no title" in " ".join(up.check_metadata(empty, video["video_path"], None)),
          "a video the seo stage never reached is refused with a pointer to it")


# --- retrying and resuming --------------------------------------------------------------------

def test_retries_resume_and_succeed(tmp):
    print("test_retries_resume_and_succeed")
    video = _video(tmp)
    svc = FakeService([("raise", ConnectionError("reset")), ("raise", http_error(503)),
                       ("progress", 0.3), ("raise", httplib2.HttpLib2Error("dns")),
                       ("progress", 0.9), ("done", OK_RESPONSE)])
    sleeps = []
    response, stats = up.upload_video(svc, up.build_body(video, CFG), video["video_path"],
                                      tmp / "s1.json", sleep=sleeps.append)
    check(response == OK_RESPONSE, "the upload finishes with YouTube's response")
    check(stats["retries"] == 3 and len(sleeps) == 3, f"three failures, three waits ({stats['retries']})")
    check(all(0 < w <= up.MAX_BACKOFF_S for w in sleeps) and sleeps[1] > 0, f"waits are bounded ({sleeps})")
    check(svc.insert_request.calls == 6, "every failure was followed by another attempt at the same request, not a new upload")
    check(svc.insert_kwargs["part"] == "snippet,status", "snippet and status parts are sent")
    check(not (tmp / "s1.json").exists(), "the session file is removed once the upload completes")


def test_failures_with_progress_between_them_do_not_add_up(tmp):
    print("test_failures_with_progress_between_them_do_not_add_up")
    video = _video(tmp)
    events = []
    for i in range(1, 16):                      # 15 failures in all, never 2 in a row
        events += [("raise", ConnectionError("blip")), ("progress", i / 20)]
    events.append(("done", OK_RESPONSE))
    svc = FakeService(events)
    response, stats = up.upload_video(svc, up.build_body(video, CFG), video["video_path"],
                                      tmp / "s2.json", sleep=lambda s: None)
    check(response == OK_RESPONSE and stats["retries"] == 15,
          "more than 10 failures in total are fine while the upload keeps moving")


def test_gives_up_after_ten_failures_in_a_row(tmp):
    print("test_gives_up_after_ten_failures_in_a_row")
    video = _video(tmp)
    svc = FakeService([("raise", ConnectionError("down"))] * 30)
    sleeps = []
    try:
        up.upload_video(svc, up.build_body(video, CFG), video["video_path"], tmp / "s3.json",
                        sleep=sleeps.append)
        check(False, "a dead connection should end in an error")
    except up.UploadError as e:
        check("gave up" in str(e) and "resume" in str(e), f"the error says what to do ({e})")
    check(len(sleeps) == up.MAX_RETRIES, f"it waited and retried {up.MAX_RETRIES} times, then stopped")
    check(svc.insert_request.calls == up.MAX_RETRIES + 1, "and made exactly one more attempt than retries")


def test_a_rejected_request_is_not_retried(tmp):
    print("test_a_rejected_request_is_not_retried")
    video = _video(tmp)
    svc = FakeService([("raise", http_error(403, "quotaExceeded", "The request cannot be completed"))])
    try:
        up.upload_video(svc, up.build_body(video, CFG), video["video_path"], tmp / "s4.json",
                        sleep=lambda s: (_ for _ in ()).throw(AssertionError("must not wait")))
        check(False, "quota exhaustion should raise")
    except up.UploadError as e:
        check("quota" in str(e) and "midnight Pacific" in str(e), f"in plain words ({e})")
    check(svc.insert_request.calls == 1, "tried once; retrying a refusal only wastes quota")


def test_session_is_saved_after_the_first_chunk(tmp):
    print("test_session_is_saved_after_the_first_chunk")
    video = _video(tmp)
    state = tmp / "s5.json"
    svc = FakeService([("progress", 0.2), ("progress", 0.6), ("done", OK_RESPONSE)])
    seen = {}
    svc.insert_request.on_call = lambda n: seen.setdefault(n, state.exists())
    up.upload_video(svc, up.build_body(video, CFG), video["video_path"], state, sleep=lambda s: None)
    check(seen[1] is False and seen[2] is True and seen[3] is True,
          "written after the first chunk, so a crash after that point can be resumed")


def test_a_restarted_process_resumes_from_what_youtube_holds(tmp):
    print("test_a_restarted_process_resumes_from_what_youtube_holds")
    video = _video(tmp)
    state = tmp / "s6.json"
    up._save_session(state, video["video_path"], "https://upload.example/session/xyz")
    http = FakeHttp(status=308, headers={"range": "bytes=0-2047"})
    svc = FakeService([("done", OK_RESPONSE)], http=http)
    response, stats = up.upload_video(svc, up.build_body(video, CFG), video["video_path"], state,
                                      sleep=lambda s: None)
    req = svc.insert_request
    check(http.calls[0][0].endswith("xyz") and http.calls[0][2]["Content-Range"] == "bytes */4096",
          "it asked YouTube about the saved session with a bytes */total probe")
    check(req.resumable_uri.endswith("xyz") and req.resumable_progress == 2048,
          "and carried on from the 2048 bytes YouTube reported, not from zero")
    check(stats["resumed_from"] == 2048 and stats["insert_calls"] == 0,
          "resuming is not a second videos.insert request")


def test_a_session_that_had_finished_is_collected_not_redone(tmp):
    print("test_a_session_that_had_finished_is_collected_not_redone")
    video = _video(tmp)
    state = tmp / "s7.json"
    up._save_session(state, video["video_path"], "https://upload.example/session/done")
    http = FakeHttp(status=200, body=json.dumps(OK_RESPONSE).encode())
    svc = FakeService([], http=http)
    response, stats = up.upload_video(svc, up.build_body(video, CFG), video["video_path"], state,
                                      sleep=lambda s: None)
    check(response["id"] == "abc123XYZ" and svc.insert_request.calls == 0,
          "the video id is read from the finished session, and not one byte is sent again")
    check(not state.exists(), "the session file is cleared")


def test_a_forgotten_or_foreign_session_starts_over(tmp):
    print("test_a_forgotten_or_foreign_session_starts_over")
    video = _video(tmp)
    state = tmp / "s8.json"
    up._save_session(state, video["video_path"], "https://upload.example/session/old")
    svc = FakeService([("done", OK_RESPONSE)], http=FakeHttp(status=404))
    _, stats = up.upload_video(svc, up.build_body(video, CFG), video["video_path"], state,
                               sleep=lambda s: None)
    check(stats["resumed_from"] == 0 and stats["insert_calls"] == 1,
          "a session YouTube answers 404 for is discarded and the upload starts fresh")

    state2 = tmp / "s9.json"
    up._save_session(state2, video["video_path"], "https://upload.example/session/other")
    Path(video["video_path"]).write_bytes(b"\x01" * 9999)          # a different file now
    svc2 = FakeService([("done", OK_RESPONSE)], http=FakeHttp(status=308, headers={"range": "bytes=0-99"}))
    _, stats2 = up.upload_video(svc2, up.build_body(video, CFG), video["video_path"], state2,
                                sleep=lambda s: None)
    check(stats2["resumed_from"] == 0 and svc2.insert_request.http.calls == [],
          "a session saved for a different file is ignored without even asking YouTube")


# --- the stage -----------------------------------------------------------------------------

def test_run_end_to_end(tmp):
    print("test_run_end_to_end")
    vid = _db_video(tmp, "e2e")
    svc = FakeService([("progress", 0.5), ("done", OK_RESPONSE)])
    check(up.run(vid, CFG, service=svc, sleep=lambda s: None) is True, "the stage completes")

    sent = svc.insert_kwargs["body"]
    check(sent["status"]["selfDeclaredMadeForKids"] is False and sent["status"]["containsSyntheticMedia"] is True,
          "the request that actually went out carries both flags")
    check(sent["snippet"]["title"] == "10 Stoic Lessons That Matter" and sent["status"]["privacyStatus"] == "private",
          "title from the database, privacy private")

    video = db.get_video(vid)
    check(video["youtube_id"] == "abc123XYZ", "youtube_id is in the database")
    check(video["status"] == "published" and video["published_at"], "status published, with a time")
    check(svc.thumbnail_calls and svc.thumbnail_calls[0]["videoId"] == "abc123XYZ",
          "the thumbnail was set in its own call, on the new video")

    report = json.loads((config.output_dir(vid) / "upload.json").read_text(encoding="utf-8"))
    check(report["quota_costing_requests"] == {"videos.insert": 1, "thumbnails.set": 1},
          f"the quota-costing requests are counted ({report['quota_costing_requests']})")
    check("does not report usage" in report["quota_note"] or "does not report" in report["quota_note"],
          "and the note says the API cannot tell the real spend")


def test_a_video_already_on_youtube_is_not_uploaded_twice(tmp):
    print("test_a_video_already_on_youtube_is_not_uploaded_twice")
    vid = _db_video(tmp, "twice")
    db.update_video(vid, youtube_id="already1", status="uploading")
    svc = FakeService([("raise", AssertionError("must not be called"))])
    check(up.run(vid, CFG, service=svc) is True, "the stage reports success")
    check(svc.insert_request.calls == 0 and svc.insert_kwargs is None, "without touching YouTube")
    check(db.get_video(vid)["status"] == "published", "and leaves the video marked published")


def test_a_failed_thumbnail_does_not_undo_the_upload(tmp):
    print("test_a_failed_thumbnail_does_not_undo_the_upload")
    vid = _db_video(tmp, "thumbfail")
    svc = FakeService([("done", OK_RESPONSE)],
                      thumbnail_error=http_error(403, "forbidden", "custom thumbnails"))
    check(up.run(vid, CFG, service=svc, sleep=lambda s: None) is True, "the stage still completes")
    video = db.get_video(vid)
    check(video["youtube_id"] == "abc123XYZ" and video["status"] == "published",
          "the video is recorded as uploaded")
    warnings = [e["message"] for e in _events(vid) if e["level"] == "warning"]
    check(any("thumbnail not set" in w and "phone-verified" in w for w in warnings),
          "and the missing thumbnail is a warning that says why")


def test_forced_private_is_reported(tmp):
    print("test_forced_private_is_reported")
    vid = _db_video(tmp, "forced")
    svc = FakeService([("done", OK_RESPONSE)])
    up.run(vid, {**CFG, "privacy_status": "public"}, service=svc, sleep=lambda s: None)
    warnings = [e["message"] for e in _events(vid) if e["level"] == "warning"]
    check(any("requested as public" in w and "private" in w and "audit" in w for w in warnings),
          "asking for public and getting private is logged, with the reason")

    vid2 = _db_video(tmp, "flagback")
    bad = {"id": "z", "status": {"privacyStatus": "private", "containsSyntheticMedia": False}}
    up.run(vid2, CFG, service=FakeService([("done", bad)]), sleep=lambda s: None)
    check(any("containsSyntheticMedia" in e["message"] for e in _events(vid2) if e["level"] == "warning"),
          "if YouTube reports the synthetic-media flag as stored false, that is a warning")


def test_a_failed_upload_leaves_no_youtube_id(tmp):
    print("test_a_failed_upload_leaves_no_youtube_id")
    vid = _db_video(tmp, "fail")
    svc = FakeService([("raise", http_error(403, "quotaExceeded"))])
    try:
        up.run(vid, CFG, service=svc, sleep=lambda s: None)
        check(False, "should raise")
    except up.UploadError:
        check(True, "the error propagates")
    check(not db.get_video(vid)["youtube_id"], "no youtube_id was recorded for an upload that did not happen")


# --- publish.py ------------------------------------------------------------------------------

def _publish_setup(tmp, name, n_ready=2):
    db.DB_PATH = tmp / f"{name}.db"
    config.OUTPUT_DIR = tmp / f"out_{name}"
    db.init_db()
    ids = []
    for i in range(n_ready):
        vid = db.create_video(f"topic {i}")
        db.update_video(vid, status="ready")
        ids.append(vid)
    return ids


def _run_publish(cfg, argv=()):
    original_load, original_run = publish.load_config, publish.upload.run
    calls = []
    publish.load_config = lambda: cfg
    return calls, original_load, original_run


def test_publish_is_off_until_the_config_says_so(tmp):
    print("test_publish_is_off_until_the_config_says_so")
    ids = _publish_setup(tmp, "off")
    calls = []
    original = publish.load_config, publish.upload.run
    publish.load_config = lambda: {"youtube_publish": False}
    publish.upload.run = lambda vid, cfg: calls.append(vid)
    try:
        check(publish.main([]) == 0 and calls == [], "youtube_publish false: exit 0, nothing uploaded")
    finally:
        publish.load_config, publish.upload.run = original
    check(all(db.get_video(i)["status"] == "ready" for i in ids), "and the queue is untouched")


def test_publish_takes_the_oldest_ready_video(tmp):
    print("test_publish_takes_the_oldest_ready_video")
    ids = _publish_setup(tmp, "queue", n_ready=3)
    db.update_video(ids[0], status="failed")             # not ready: skipped
    db.update_video(ids[1], status="published", youtube_id="x")
    calls = []
    original = publish.load_config, publish.upload.run
    publish.load_config = lambda: {"youtube_publish": True}
    publish.upload.run = lambda vid, cfg: calls.append(vid) or True
    try:
        check(publish.main([]) == 0, "exit 0")
    finally:
        publish.load_config, publish.upload.run = original
    check(calls == [ids[2]], f"the oldest video that is ready and not yet uploaded was chosen ({calls})")


def test_empty_queue_publishes_nothing(tmp):
    print("test_empty_queue_publishes_nothing")
    _publish_setup(tmp, "empty", n_ready=0)
    calls = []
    original = publish.load_config, publish.upload.run
    publish.load_config = lambda: {"youtube_publish": True}
    publish.upload.run = lambda vid, cfg: calls.append(vid)
    try:
        check(publish.main([]) == 3 and calls == [], "an empty queue exits 3 and uploads nothing")
    finally:
        publish.load_config, publish.upload.run = original


def test_failed_publish_returns_the_video_to_the_queue(tmp):
    print("test_failed_publish_returns_the_video_to_the_queue")
    ids = _publish_setup(tmp, "failq", n_ready=1)
    original = publish.load_config, publish.upload.run

    def boom(vid, cfg):
        raise up.UploadError("the network went away")
    publish.load_config = lambda: {"youtube_publish": True}
    publish.upload.run = boom
    try:
        check(publish.main([]) == 1, "exit 1")
    finally:
        publish.load_config, publish.upload.run = original
    check(db.get_video(ids[0])["status"] == "ready", "the video is back to ready, to be retried (and resumed), not failed")
    check(any(e["level"] == "error" and "network" in e["message"] for e in _events(ids[0])),
          "the reason is in the events table")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_body_has_what_was_asked_for(tmp)
        test_a_scheduled_video_goes_up_private_for_the_right_moment(tmp)
        test_the_two_flags_cannot_be_switched_off(tmp)
        test_metadata_problems_are_found_before_uploading(tmp)
        test_retries_resume_and_succeed(tmp)
        test_failures_with_progress_between_them_do_not_add_up(tmp)
        test_gives_up_after_ten_failures_in_a_row(tmp)
        test_a_rejected_request_is_not_retried(tmp)
        test_session_is_saved_after_the_first_chunk(tmp)
        test_a_restarted_process_resumes_from_what_youtube_holds(tmp)
        test_a_session_that_had_finished_is_collected_not_redone(tmp)
        test_a_forgotten_or_foreign_session_starts_over(tmp)
        test_run_end_to_end(tmp)
        test_a_video_already_on_youtube_is_not_uploaded_twice(tmp)
        test_a_failed_thumbnail_does_not_undo_the_upload(tmp)
        test_forced_private_is_reported(tmp)
        test_a_failed_upload_leaves_no_youtube_id(tmp)
        test_publish_is_off_until_the_config_says_so(tmp)
        test_publish_takes_the_oldest_ready_video(tmp)
        test_empty_queue_publishes_nothing(tmp)
        test_failed_publish_returns_the_video_to_the_queue(tmp)
    print("ALL UPLOAD TESTS PASSED")
