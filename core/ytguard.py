"""One guard for every channel's uploads (owner 17.11: "однаково в усіх репо каналів").

The same file lives in every channel repository (yt-factory, yt-kids, archive-channel,
prehistoric-channel, whatif-channel, closed-tabs, HugeToScale, map-timelapse). It holds the rules,
the uploaders call it:

    yt = service_from_env()                       # the channel's own token (YT_TOKEN_JSON)
    check_channel(yt, expected_id)                # the token belongs to THIS channel, or Doubt
    videos = all_uploads(yt, known_ids)           # every video, every page, retried, or Doubt
    same = find_existing(videos, ref, title)      # the episode is already up: do not upload
    when = next_free_slot(slot_fn, videos, taken) # a slot no video of the channel holds
    description += "\\n\\n" + ref                  # [ref: <channel>/<slug>] in every new video
    ... upload ...
    persist([...state files...], "state: ...")    # in git at once, or the run fails and mails

Doubt means "not sure": nothing is uploaded, the owner gets one mail, the episode waits. A video found
by its marker (or its exact title, for the ones made before the markers) is never uploaded again.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


class Doubt(RuntimeError):
    """Not sure the channel was read in full or is the right one: no upload."""


def ref(channel, slug):
    return f"[ref: {channel}/{slug}]"


def service_from_env(env="YT_TOKEN_JSON", token_file=None):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    raw = os.environ.get(env)
    d = json.loads(raw) if raw else json.loads(Path(token_file).read_text(encoding="utf-8"))
    cred = Credentials(token=None, refresh_token=d["refresh_token"], client_id=d["client_id"],
                       client_secret=d["client_secret"],
                       token_uri=d.get("token_uri", "https://oauth2.googleapis.com/token"),
                       scopes=d.get("scopes"))
    return build("youtube", "v3", credentials=cred, cache_discovery=False)


def _retry(fn, what, tries=4):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                                       # noqa: BLE001
            last = e
            print(f"ytguard: {what} failed ({i + 1}/{tries}): {str(e)[:160]}", file=sys.stderr)
            time.sleep(2 * 3 ** i)
    raise Doubt(f"{what} failed {tries} times: {str(last)[:200]}")


def channel_id(yt):
    items = _retry(lambda: yt.channels().list(part="id,snippet,contentDetails", mine=True).execute(),
                   "channel")["items"]
    if not items:
        raise Doubt("the token has no channel")
    return items[0]


def check_channel(yt, expected_id):
    """The token must belong to this channel (a token of another channel once uploaded to the wrong one)."""
    ch = channel_id(yt)
    if not expected_id:
        raise Doubt(f"no expected channel id in the config; the token is {ch['id']} ({ch['snippet']['title']})")
    if ch["id"] != expected_id:
        raise Doubt(f"the token is channel {ch['id']} ({ch['snippet']['title']}), not {expected_id}")
    return ch


def all_uploads(yt, known_ids=()):
    """Every video of the channel (private and scheduled too), each page retried. Doubt when the count
    is short of what YouTube reports or an id the state already knows is missing."""
    ch = channel_id(yt)
    pl = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    ids, tok, total = [], None, None
    while True:
        r = _retry(lambda t=tok: yt.playlistItems().list(part="contentDetails", playlistId=pl, maxResults=50,
                                                         pageToken=t).execute(), "uploads page")
        ids += [i["contentDetails"]["videoId"] for i in r.get("items", [])]
        total = (r.get("pageInfo") or {}).get("totalResults", total)
        tok = r.get("nextPageToken")
        if not tok:
            break
    if total is not None and len(ids) < total:
        raise Doubt(f"the uploads list has {len(ids)} of {total} videos")
    vids = []
    for k in range(0, len(ids), 50):
        part = ids[k:k + 50]
        vids += _retry(lambda p=part: yt.videos().list(part="snippet,status", id=",".join(p)).execute(),
                       "video details")["items"]
    seen = {v["id"] for v in vids}
    missing = [i for i in known_ids if i and i not in seen]
    if missing:
        raise Doubt(f"the state knows videos the channel list lacks: {missing[:5]}")
    return [{"id": v["id"], "title": v["snippet"].get("title", ""),
             "description": v["snippet"].get("description", ""),
             "tags": v["snippet"].get("tags") or [],
             "privacy": v["status"].get("privacyStatus"), "publish_at": v["status"].get("publishAt"),
             "published_at": v["snippet"].get("publishedAt")} for v in vids]


def find_existing(videos, marker, title=None):
    for v in videos:
        if marker and marker in v["description"]:
            return v
    if title:
        for v in videos:
            if v["title"].strip() == title.strip():
                return v
    return None


def channel_dates(videos):
    """The moments the channel already holds: publishAt of scheduled ones, the publish time of public
    and unlisted ones."""
    out = []
    for v in videos:
        t = v["publish_at"] or (v["published_at"] if v["privacy"] in ("public", "unlisted") else None)
        if t:
            out.append(t)
    return out


def taken_days(videos, extra=(), tz=None):
    """Calendar days (in the channel's own time zone) that already have a video: one video per slot."""
    from datetime import datetime
    days = set()
    for t in list(channel_dates(videos)) + [x for x in extra if x]:
        d = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        days.add((d.astimezone(tz) if tz else d).date())
    return days


def notify(subject, body, repo=None):
    """A mail to the owner: a comment that mentions them on the open "run-log" issue (GitHub mails it)."""
    print(f"NOTIFY (mail is off (owner 18.11): the text stays in the Actions log, nobody is written to): {subject}\n{body}", file=sys.stderr)
    return
    repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        print(f"NOTIFY: {subject}\n{body}", file=sys.stderr)
        return

    def gh(*a):
        return subprocess.run(["gh", *a, "-R", repo], capture_output=True, text=True).stdout.strip()
    owner = repo.split("/")[0]
    found = json.loads(gh("issue", "list", "--label", "run-log", "--state", "open", "--json", "number") or "[]")
    if found:
        n = str(found[0]["number"])
    else:
        gh("label", "create", "run-log", "--color", "b60205")
        n = gh("issue", "create", "--title", "Run log", "--label", "run-log",
               "--body", "A comment here mentioning the owner arrives as an email.").rsplit("/", 1)[-1]
    gh("issue", "comment", n, "--body", f"@{owner} **{subject}**\n\n{body}")


def persist(paths, message, branch="main", root=None):
    """Commit the state at once (owner 17.11): every path added on its own (`git add a b || true` once
    added nothing when b was missing), a conflict resolved in favour of this run's own lines, five
    tries; a state that cannot be committed fails the run and mails the owner."""
    if not os.environ.get("GITHUB_ACTIONS") or os.environ.get("NO_PERSIST") or os.environ.get("DRY_RUN"):
        return False
    cwd = str(root or Path.cwd())

    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd)
    if not git("config", "user.email").stdout.strip():
        git("config", "user.name", "github-actions[bot]")
        git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    for p in paths:
        if Path(cwd, p).exists():
            git("add", "--", str(p))
    if git("diff", "--cached", "--quiet").returncode == 0:
        return True
    r = git("commit", "-q", "-m", message)
    if r.returncode:
        raise RuntimeError(f"commit failed: {r.stderr[-300:]}")
    last = ""
    for i in range(5):
        r = git("pull", "-q", "--rebase", "-X", "theirs", "--autostash", "origin", branch)
        if r.returncode:
            git("rebase", "--abort")
        else:
            r = git("push", "-q", "origin", f"HEAD:{branch}")
            if r.returncode == 0:
                print(f"ytguard: state committed: {message}")
                return True
        last = (r.stderr or r.stdout or "")[-300:]
        time.sleep(8 * (i + 1))
    notify("The state of an upload could not be saved to git", f"{message}\n{last}\nThe run fails.")
    raise RuntimeError(f"the state could not be committed: {last}")
