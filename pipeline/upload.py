"""Stage 8: upload the finished video to YouTube (Data API v3), then set its thumbnail.

Not part of the production chain: production ends at status `ready`, and publish.py - a
separate process, run on the publishing schedule - picks the oldest ready video and calls
run() here. See README, "виробництво окремо від публікації".

What is sent. Title, description and tags come from the seo stage (videos.seo_title,
description, tags); category 22 (People & Blogs); default language en; privacy from
config.privacy_status. Two status flags are set every time and are NOT configurable, so no
config edit can switch them off: selfDeclaredMadeForKids = false, and
containsSyntheticMedia = true, which YouTube requires for AI-generated content and sanctions
channels for omitting. They are asserted on the request before it is sent and compared with
what YouTube's response says came back.

The upload is resumable, because the file is hundreds of megabytes and connections drop.
  - In one process, a dropped connection or a 5xx is retried with growing waits and the
    upload carries on from the byte the server has, not from the start.
  - Across processes, the session URI is saved to output/<id>/upload_session.json after the
    first chunk. A restarted stage asks YouTube how many bytes it already holds and sends
    only the rest; if the upload had in fact finished, it collects the video id and does
    nothing more. A session YouTube has forgotten is discarded and the upload starts over.
  - A video that already has a youtube_id is never uploaded again.

Quota. Costs are what Google's quota page says (read 2026-10-02, page updated 2026-09-15):
videos.insert is 1 unit in its own "uploads" bucket of 100 per day, thumbnails.set about 50
of the 10,000 general units. The API does not report what was actually spent, so this stage
logs what it asked for and where to read the real figure (Cloud Console, APIs & Services,
YouTube Data API v3, Quotas) rather than invent one.

The first upload from an unaudited API project is kept private by YouTube whatever is asked
for; after the upload the privacy that came back is compared with the one requested and the
difference is logged.
"""
import http.client
import json
import random
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from core import db
from core.config import output_dir
from core.logger import get_logger

log = get_logger("upload")

# Set on every upload, not read from config: see the module docstring.
MADE_FOR_KIDS = False
CONTAINS_SYNTHETIC_MEDIA = True

CATEGORY_ID = "22"               # People & Blogs
DEFAULT_LANGUAGE = "en"
PRIVACY = ("private", "unlisted", "public")
DEFAULT_CHUNK_MB = 8             # must be a multiple of 256 KiB; 8 MiB is
MAX_RETRIES = 10                 # consecutive failures with no progress before giving up
MAX_BACKOFF_S = 60
RETRIABLE_STATUS = (500, 502, 503, 504)

MAX_TITLE = 100                  # YouTube's limits
MAX_DESCRIPTION_BYTES = 5000
MAX_TAGS_CHARS = 500
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024

QUOTA_NOTE = ("videos.insert: documented as 1 unit in the separate uploads bucket (100 a "
              "day). thumbnails.set: documented as ~50 of the 10,000 general units a day. "
              "The API does not report usage; the real figure is in Google Cloud Console > "
              "APIs & Services > YouTube Data API v3 > Quotas & System limits.")


class UploadError(Exception):
    pass


# --- what is sent ----------------------------------------------------------------

def build_body(video, cfg):
    """The videos.insert request body for this video."""
    privacy = cfg.get("privacy_status", "private")
    if privacy not in PRIVACY:
        raise UploadError(f"privacy_status {privacy!r} in config.json must be one of {PRIVACY}")
    yt = cfg.get("youtube") or {}
    tags = video.get("tags")
    tags = json.loads(tags) if isinstance(tags, str) and tags else (tags or [])
    return {
        "snippet": {
            "title": video.get("seo_title") or video.get("title") or "",
            "description": video.get("description") or "",
            "tags": tags,
            "categoryId": str(yt.get("category_id", CATEGORY_ID)),
            "defaultLanguage": yt.get("default_language", DEFAULT_LANGUAGE),
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": MADE_FOR_KIDS,
            "containsSyntheticMedia": CONTAINS_SYNTHETIC_MEDIA,
        },
    }


def assert_mandatory_flags(body):
    """The two compliance flags, checked on the request itself just before it goes out."""
    status = body.get("status") or {}
    if status.get("selfDeclaredMadeForKids") is not False:
        raise UploadError("status.selfDeclaredMadeForKids must be false on every upload")
    if status.get("containsSyntheticMedia") is not True:
        raise UploadError("status.containsSyntheticMedia must be true on every upload: "
                          "YouTube requires AI-generated content to be declared")


def check_metadata(body, video_path, thumbnail_path):
    """Everything YouTube would reject, found before spending an upload on it."""
    problems = []
    snippet = body["snippet"]
    if not snippet["title"].strip():
        problems.append("the video has no title (did the seo stage run?)")
    if len(snippet["title"]) > MAX_TITLE:
        problems.append(f"title is {len(snippet['title'])} characters, YouTube allows {MAX_TITLE}")
    if not snippet["description"].strip():
        problems.append("the video has no description (did the seo stage run?)")
    if len(snippet["description"].encode("utf-8")) > MAX_DESCRIPTION_BYTES:
        problems.append(f"description is over {MAX_DESCRIPTION_BYTES} bytes")
    if sum(len(t) + 1 for t in snippet["tags"]) > MAX_TAGS_CHARS:
        problems.append(f"tags are over {MAX_TAGS_CHARS} characters in total")
    if not video_path or not Path(video_path).exists():
        problems.append(f"the video file {video_path} does not exist (did render run?)")
    if thumbnail_path and Path(thumbnail_path).exists() and \
            Path(thumbnail_path).stat().st_size > MAX_THUMBNAIL_BYTES:
        problems.append(f"the thumbnail is over {MAX_THUMBNAIL_BYTES // 1024 // 1024} MB")
    return problems


# --- the upload --------------------------------------------------------------------

def _retriable_exceptions():
    import httplib2
    return (httplib2.HttpLib2Error, IOError, socket.timeout, http.client.IncompleteRead,
            http.client.ImproperConnectionState, http.client.CannotSendRequest,
            http.client.CannotSendHeader, http.client.ResponseNotReady,
            http.client.BadStatusLine)


def explain_http_error(error):
    """A sentence a person can act on for the errors an upload actually meets."""
    from googleapiclient.errors import HttpError
    status = getattr(error.resp, "status", "?") if isinstance(error, HttpError) else "?"
    try:
        detail = json.loads(error.content.decode("utf-8"))["error"]
        reasons = [e.get("reason", "") for e in detail.get("errors", [])]
        message = detail.get("message", "")
    except (ValueError, KeyError, AttributeError, TypeError):
        reasons, message = [], str(error)

    if "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons:
        return (f"YouTube says the daily quota is used up (HTTP {status}, {message}). The "
                "uploads bucket resets daily at midnight Pacific time; try again after that.")
    if "uploadLimitExceeded" in reasons:
        return f"YouTube says this channel has reached its upload limit for now ({message})"
    if "youtubeSignupRequired" in reasons:
        return "the Google account has no YouTube channel; create one at youtube.com first"
    if "forbidden" in reasons or status == 403:
        return (f"YouTube refused the request (HTTP 403: {message}). For a thumbnail this "
                "usually means the channel is not phone-verified, which custom thumbnails "
                "and videos over 15 minutes both require.")
    if status == 401:
        return "YouTube rejected the credentials (HTTP 401); run scripts/youtube_auth.py again"
    return f"YouTube returned HTTP {status}: {message}"


def _load_session(state_path, media_path):
    """The saved session if it belongs to this very file, else None."""
    state_path = Path(state_path)
    if not state_path.exists():
        return None
    try:
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        stat = Path(media_path).stat()
        if saved["size"] == stat.st_size and saved["mtime"] == int(stat.st_mtime) and saved["uri"]:
            return saved
    except (ValueError, KeyError, OSError):
        pass
    log.warning("an old upload session exists but does not match the file; ignoring it")
    state_path.unlink(missing_ok=True)
    return None


def _save_session(state_path, media_path, uri):
    stat = Path(media_path).stat()
    Path(state_path).write_text(json.dumps(
        {"uri": uri, "size": stat.st_size, "mtime": int(stat.st_mtime)}), encoding="utf-8")


def probe_session(http, uri, size):
    """Ask YouTube what it holds of an upload session. Returns ("done", response_body),
    ("partial", bytes_received) or ("gone", None)."""
    resp, content = http.request(uri, "PUT",
                                 headers={"Content-Range": f"bytes */{size}",
                                          "Content-Length": "0"})
    if resp.status in (200, 201):
        return "done", json.loads(content)
    if resp.status == 308:
        received = resp.get("range")             # "bytes=0-1048575", absent if nothing yet
        return "partial", (int(received.split("-")[1]) + 1 if received else 0)
    if resp.status in (404, 410):
        return "gone", None
    raise UploadError(f"YouTube answered HTTP {resp.status} when asked about the upload "
                      "session")


def upload_video(service, body, media_path, state_path, chunk_mb=DEFAULT_CHUNK_MB,
                 sleep=time.sleep):
    """Send the file and return (response, stats). Retries and resumes as described in the
    module docstring."""
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    size = Path(media_path).stat().st_size
    media = MediaFileUpload(str(media_path), chunksize=chunk_mb * 1024 * 1024, resumable=True)
    try:
        return _upload(service, body, media, media_path, state_path, size, sleep)
    finally:
        # MediaFileUpload keeps the video open until it is garbage collected; closing it
        # here means a failed upload does not hold the file (and on Windows, lock it).
        handle = getattr(media, "_fd", None)
        if handle:
            handle.close()


def _upload(service, body, media, media_path, state_path, size, sleep):
    from googleapiclient.errors import HttpError

    request = service.videos().insert(part="snippet,status", body=body, media_body=media)
    stats = {"bytes": size, "resumed_from": 0, "retries": 0, "chunks": 0,
             "insert_calls": 1, "started": time.time()}

    saved = _load_session(state_path, media_path)
    if saved:
        kind, value = probe_session(request.http, saved["uri"], size)
        if kind == "done":
            log.info("the saved upload session had already completed; collecting its result")
            stats["resumed_from"] = size
            Path(state_path).unlink(missing_ok=True)
            return value, stats
        if kind == "partial":
            request.resumable_uri = saved["uri"]
            request.resumable_progress = value
            stats["resumed_from"] = value
            stats["insert_calls"] = 0                  # no second videos.insert request
            log.info("resuming the upload from byte %d of %d (%.0f%%)", value, size,
                     100 * value / size)
        else:
            log.warning("YouTube no longer knows the saved upload session; starting over")
            Path(state_path).unlink(missing_ok=True)

    retriable = _retriable_exceptions()
    response, failures, last_logged = None, 0, -1
    while response is None:
        before = request.resumable_progress
        try:
            status, response = request.next_chunk(num_retries=2)
        except HttpError as e:
            if e.resp.status not in RETRIABLE_STATUS:
                raise UploadError(explain_http_error(e)) from e
            error = e
        except retriable as e:
            error = e
        else:
            failures = 0
            stats["chunks"] += 1
            if request.resumable_uri and saved is None:
                _save_session(state_path, media_path, request.resumable_uri)
                saved = {"uri": request.resumable_uri}
            if status:
                percent = int(status.progress() * 100)
                if percent // 10 != last_logged // 10:
                    log.info("uploaded %d%%", percent)
                    last_logged = percent
            continue

        # a retriable failure: wait, then carry on from what the server has
        if request.resumable_progress > before:
            failures = 0
        failures += 1
        stats["retries"] += 1
        if failures > MAX_RETRIES:
            raise UploadError(f"the upload kept failing ({error}); gave up after "
                              f"{MAX_RETRIES} attempts in a row. Run publish again to "
                              "resume from where it stopped.") from error
        wait = min(MAX_BACKOFF_S, 2 ** failures) * random.uniform(0.5, 1.0)
        log.warning("upload hit %s (attempt %d/%d), waiting %.1fs then resuming",
                    type(error).__name__, failures, MAX_RETRIES, wait)
        sleep(wait)

    Path(state_path).unlink(missing_ok=True)
    stats["seconds"] = round(time.time() - stats["started"], 1)
    return response, stats


def set_thumbnail(service, youtube_id, thumbnail_path):
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    media = MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg")
    try:
        service.thumbnails().set(videoId=youtube_id, media_body=media).execute()
    except HttpError as e:
        raise UploadError(explain_http_error(e)) from e
    finally:
        handle = getattr(media, "_fd", None)
        if handle:
            handle.close()


# --- the stage -----------------------------------------------------------------------

def run(video_id, cfg, service=None, sleep=time.sleep):
    video = db.get_video(video_id) or {}
    if video.get("youtube_id"):
        log.info("video %s is already on YouTube as %s; not uploading it again",
                 video_id, video["youtube_id"])
        db.update_video(video_id, status="published")
        return True

    body = build_body(video, cfg)
    assert_mandatory_flags(body)
    media_path, thumb_path = video.get("video_path"), video.get("thumbnail_path")
    problems = check_metadata(body, media_path, thumb_path)
    if problems:
        raise UploadError("; ".join(problems))

    if service is None:
        from core import youtube
        try:
            service = youtube.build_service(youtube.load_credentials())
        except youtube.AuthError as e:
            raise UploadError(str(e)) from e

    out = output_dir(video_id)
    privacy = body["status"]["privacyStatus"]
    size_mb = Path(media_path).stat().st_size / 1e6
    log.info("uploading %s (%.0f MB) as %r, privacy %s", Path(media_path).name, size_mb,
             body["snippet"]["title"], privacy)

    chunk_mb = int((cfg.get("youtube") or {}).get("chunk_mb", DEFAULT_CHUNK_MB))
    response, stats = upload_video(service, body, media_path, out / "upload_session.json",
                                   chunk_mb, sleep)
    youtube_id = response.get("id")
    if not youtube_id:
        raise UploadError(f"YouTube accepted the upload but returned no video id: {response}")
    # Written before anything else can go wrong: this is what stops a second upload.
    db.update_video(video_id, youtube_id=youtube_id)
    log.info("uploaded: https://youtu.be/%s in %ss (%d chunks, %d retries%s)", youtube_id,
             stats.get("seconds", "?"), stats["chunks"], stats["retries"],
             f", resumed from byte {stats['resumed_from']}" if stats["resumed_from"] else "")

    _compare_with_response(video_id, response, privacy)

    thumb_calls = 0
    if thumb_path and Path(thumb_path).exists():
        thumb_calls = 1
        try:
            set_thumbnail(service, youtube_id, thumb_path)
            log.info("thumbnail set")
        except UploadError as e:
            # The video is up; a missing thumbnail is fixable by hand and not worth failing
            # the stage over, which would make the next run look like an unfinished upload.
            log.warning("the video is uploaded but its thumbnail was not set: %s", e)
            db.log_event(video_id, "upload", "warning", f"thumbnail not set: {e}")
    else:
        log.warning("no thumbnail file to set")

    db.update_video(video_id, status="published",
                    published_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    spent = {"videos.insert": stats["insert_calls"], "thumbnails.set": thumb_calls}
    log.info("quota: requests that cost quota this run %s. %s", spent, QUOTA_NOTE)
    (out / "upload.json").write_text(json.dumps({
        "youtube_id": youtube_id, "privacy_requested": privacy,
        "privacy_returned": (response.get("status") or {}).get("privacyStatus"),
        "bytes": stats["bytes"], "seconds": stats.get("seconds"), "chunks": stats["chunks"],
        "retries": stats["retries"], "resumed_from": stats["resumed_from"],
        "quota_costing_requests": spent, "quota_note": QUOTA_NOTE}, indent=2), encoding="utf-8")
    db.log_event(video_id, "upload", "info",
                 f"https://youtu.be/{youtube_id}, privacy {privacy}, {size_mb:.0f} MB, "
                 f"{stats['retries']} retries; quota requests {spent}")
    return True


def _compare_with_response(video_id, response, requested_privacy):
    """What YouTube says it stored, against what was asked for."""
    status = response.get("status") or {}
    problems = []
    if status.get("privacyStatus") and status["privacyStatus"] != requested_privacy:
        problems.append(f"privacy was requested as {requested_privacy} but YouTube set it to "
                        f"{status['privacyStatus']} (an unaudited API project is forced to "
                        "private; publish by hand in YouTube Studio until it passes audit)")
    if "selfDeclaredMadeForKids" in status and status["selfDeclaredMadeForKids"] is not MADE_FOR_KIDS:
        problems.append("YouTube stored selfDeclaredMadeForKids as "
                        f"{status['selfDeclaredMadeForKids']}, not false")
    if "containsSyntheticMedia" in status and status["containsSyntheticMedia"] is not CONTAINS_SYNTHETIC_MEDIA:
        problems.append("YouTube stored containsSyntheticMedia as "
                        f"{status['containsSyntheticMedia']}, not true - fix it in YouTube Studio")
    elif "containsSyntheticMedia" not in status:
        log.warning("YouTube's response did not echo containsSyntheticMedia, so it could "
                    "not be confirmed; check the video's 'altered content' setting in Studio")
    for p in problems:
        log.warning("%s", p)
        db.log_event(video_id, "upload", "warning", p)
