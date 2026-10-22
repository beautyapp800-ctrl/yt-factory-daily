"""Offline tests for the bust thumbnail. Run: python tests/test_bust_thumbnail.py

Cloudflare, pollinations and Gemini are all stubbed: a synthetic "bust" (a light head and
shoulders on dark grey) stands in for the generated picture, so the checks are exact and
the suite spends no Neurons and no Gemini requests.
"""
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import config, db, neurons
import core.images as image_api
import pipeline.bust_thumbnail as bust
import pipeline.bust_library as library
library.LIB_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-busts-"))
import pipeline.phrase as phrase
import pipeline.thumbnail as thumb
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'scripts_refresh', Path(__file__).resolve().parent.parent / 'scripts' / 'refresh_thumbnails.py')
scripts_refresh = importlib.util.module_from_spec(_spec)
sys.modules['scripts_refresh'] = scripts_refresh
_spec.loader.exec_module(scripts_refresh)


def _forbidden(*a, **k):
    raise AssertionError("this test reached the network")


image_api.synthesize = _forbidden
image_api.synthesize_cloudflare = _forbidden
image_api.synthesize_pollinations = _forbidden
bust.vision_json = _forbidden
bust.complete_json = _forbidden
phrase.complete_json = _forbidden


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def fake_bust(path, cx=0.5, bg=52):
    img = Image.new("RGB", (1024, 1024), (bg, bg + 1, bg + 4))
    d = ImageDraw.Draw(img)
    x = int(1024 * cx)
    d.ellipse((x - 190, 170, x + 190, 640), fill=(205, 200, 192))
    d.rectangle((x - 330, 600, x + 330, 1024), fill=(180, 176, 170))
    arr = np.asarray(img, dtype=np.float64) + np.random.default_rng(1).normal(0, 12, (1024, 1024, 1))
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(2)).save(path, "JPEG", quality=92)
    return path


CFG = {"image_style": "x", "images": {"provider": "cloudflare",
                                      "cloudflare": {"model": "m", "steps": 2}},
       "thumbnail": {"style": "bust", "candidates": 4, "attempts": 3, "steps": 8,
                     "neurons_per_image_guess": 192, "reserve_neurons": 300}}


def _stub(cf_costs=96, centre=0.5, vision=None, cf_error=None, steps_seen=None):
    """Cloudflare that writes a fake bust and reports `cf_costs` Neurons."""
    calls = []

    def cloudflare(prompt, path, seed, cfg):
        steps = cfg["images"]["cloudflare"]["steps"]
        if steps_seen is not None:
            steps_seen.append(steps)
        if cf_error:
            raise image_api.ImageError(cf_error(steps))
        calls.append(prompt)
        fake_bust(path, cx=centre)
        return path, cf_costs

    image_api.cloudflare_available = lambda: True
    bust.image_api.synthesize_cloudflare = cloudflare
    bust.vision_json = vision or (lambda prompt, imgs, model=None: {"images": [
        {"index": i, "marble": 8, "face_ok": True, "text": False, "composition": 8,
         "likeness": 7} for i in range(len(imgs))], "best": 0})
    phrase.complete_json = lambda *a, **k: {"phrase": "DISCIPLINE BUILDS FREEDOM"}
    return calls


def _restore():
    bust.image_api.synthesize_cloudflare = _forbidden
    bust.vision_json = _forbidden
    phrase.complete_json = _forbidden


def test_the_bust_is_chosen_by_the_script_but_nobody_is_named():
    key, _ = bust.philosopher_for("Marcus Aurelius wrote. " * 7 + "Seneca said. " * 3)
    check(key == "marcus", "seven mentions against three pick Marcus Aurelius's likeness")
    key, _ = bust.philosopher_for("Seneca. " * 5 + "Marcus Aurelius. " * 5)
    check(key is None, "a tie picks nobody")
    key, _ = bust.philosopher_for("Epictetus taught. " * 2)
    check(key is None, "two passing mentions pick nobody")
    check("Seneca" in bust.bust_prompt("seneca") and "marble bust" in bust.bust_prompt(None),
          "the bust drawn is the philosopher the script leans on")
    prompt = bust.bust_prompt(None)
    check("black background" in prompt and "no hands" in prompt and "no text" in prompt,
          "the bust is asked for on black, with no hands and no lettering")
    check(not hasattr(bust, "attribution"),
          "no name is put under the phrase: the model wrote it, not Seneca")


def test_the_words_come_from_phrase_py():
    answers = iter([{"phrase": "ANGER COSTS"}, {"phrase": "ANGER COSTS MORE"}])
    asked = []

    def model(prompt, **k):
        asked.append(prompt)
        return next(answers)
    phrase.complete_json = model
    try:
        got = bust.headline_for("10 Ways to Stay Calm", "anger")
    finally:
        phrase.complete_json = _forbidden
    check(got == ("ANGER COSTS MORE", "phrase"),
          f"a two-word phrase is sent back and the three-word one is used ({got})")
    check("THREE to FOUR words" in asked[0], "the model is asked for three or four words")

    def broken(*a, **k):
        raise RuntimeError("no model")
    phrase.complete_json = broken
    try:
        got = bust.headline_for("10 Stoic Lessons to Master the Pause", "pause")
        bad = bust.headline_for("10 Morning Habits To Stop Right Now For Real Calm", "calm")
    finally:
        phrase.complete_json = _forbidden
    check(got == ("MASTER THE PAUSE", "title"), f"with no model the title is used whole ({got})")
    check(bad == (bust.FALLBACK_HEADLINE, "fallback"),
          f"a title too long to stack gives the fixed line ({bad})")
    check(bust.headline_from_title("10 Ways to Strip Emotion From Criticism")
          == "STRIP EMOTION FROM CRITICISM", "the title's small words are not dropped")
    check(not bust.fits("ANGER COSTS") and not bust.fits("ONE TWO THREE FOUR FIVE")
          and bust.fits("SILENCE OUTLASTS EVERY INSULT"), "fits() wants three or four words")


def test_a_bust_thumbnail_is_made_checked_and_costed(tmp):
    calls = _stub()
    try:
        d = bust.make(5, "10 Stoic Lessons on Discipline", "discipline",
                      "Marcus Aurelius. " * 8, tmp / "a", CFG, budget=1400)
    finally:
        _restore()
    check(Path(d["path"]).exists() and Image.open(d["path"]).size == (1280, 720),
          "a 1280x720 thumbnail is written")
    check(d["passed"], f"it passes every check ({d['chosen']['problems']})")
    check(len(calls) == 4, "four candidates on the first attempt, and no second attempt")
    check(d["neurons"] == 384 and d["neurons_per_image"] == 96,
          "the Neurons are the ones Cloudflare reported")
    check(d["philosopher"] == "marcus" and d["headline"] == "DISCIPLINE BUILDS FREEDOM"
          and d["headline_source"] == "phrase", "Marcus's bust, under the phrase from phrase.py")
    line = bust.summary(d)
    check("384 neurons" in line, "the event line carries the spend for core.neurons")
    check(f"[layout {bust.LAYOUT}]" in line, "and the layout, so the refresh can tell")
    with Image.open(d["path"]) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.int32)
    lum = rgb @ np.array([2126, 7152, 722]) / 10000
    check(lum[:, :500].std() > lum[:, 900:].std() * 0.5, "the bust is on the left")
    right = rgb[:, round(1280 * bust.TEXT_LEFT):]
    check(np.median(lum[:, round(1280 * bust.TEXT_LEFT):]) < bust.MAX_GROUND_LUM,
          "the ground behind the type is almost black")
    r, g, b = right[..., 0], right[..., 1], right[..., 2]
    yellow = ((r > 220) & (g > 170) & (b < 80)).sum()
    white = ((r > 225) & (g > 225) & (b > 225)).sum()
    red = ((r > 170) & (g < 70) & (b < 70)).sum()
    check(yellow > 5000 and white > 5000, f"yellow and white type ({yellow}, {white} px)")
    check(500 < red < 6000, f"a thin red rule ({red} px)")
    with Image.open(d["small"]) as small:
        check(small.size == (210, 118), "thumb_small.jpg is the 210x118 the feed shows")


def test_a_small_budget_buys_fewer_candidates(tmp):
    calls = _stub()
    try:
        d = bust.make(6, "t", "t", "", tmp / "b", CFG, budget=400)
    finally:
        _restore()
    check(bust.neurons_per_image(8) == 326.4 and bust.neurons_per_image(2) == 96,
          "the cost per image follows the measured 19 + 38 a step")
    check(len(calls) == 1, f"400 Neurons buy one 8-step candidate ({len(calls)})")
    check(d["neurons"] <= 400, "and the budget is not exceeded")


def test_no_budget_means_no_bust_and_no_pollinations(tmp):
    calls = _stub()
    try:
        bust.make(7, "t", "t", "", tmp / "c", CFG, budget=100)
        raised = False
    except RuntimeError:
        raised = True
    finally:
        _restore()
    check(raised and not calls, "with too few Neurons nothing is drawn and make() raises, so "
                                "the caller falls back (pollinations is never asked)")


def test_a_failed_check_is_drawn_again_then_the_best_is_kept(tmp):
    calls = _stub(centre=1.0)                  # the bust lands at the edge, under the fade
    try:
        d = bust.make(9, "t", "t", "", tmp / "e", CFG, budget=5000)
    finally:
        _restore()
    check(len(calls) == 12, "three attempts of four candidates")
    check(not d["passed"] and Path(d["path"]).exists(),
          "after three failed attempts the best one is still written")
    check(any("bust" in p for p in d["chosen"]["problems"]), "and the reason is recorded")


def test_gemini_rejections_and_gemini_absence(tmp):
    def picky(prompt, imgs, model=None):
        return {"images": [{"index": i, "marble": 9 if i == 2 else 3, "face_ok": i == 2,
                            "text": False, "composition": 8, "likeness": 8}
                           for i in range(len(imgs))]}
    _stub(vision=picky)
    try:
        d = bust.make(10, "t", "t", "", tmp / "f", CFG, budget=5000)
    finally:
        _restore()
    check(d["chosen"]["source"] == "bust_02.jpg", "Gemini's acceptable candidate is the one used")

    def broken(prompt, imgs, model=None):
        raise bust.LLMError("quota")
    _stub(vision=broken)
    try:
        d = bust.make(11, "t", "t", "", tmp / "g", CFG, budget=5000)
    finally:
        _restore()
    check(d["passed"] and d["chosen"]["judged_by"] == "measurement",
          "without Gemini the pick falls to measurement and still passes")


def test_eight_steps_refused_falls_back_to_four(tmp):
    steps = []
    calls = _stub(steps_seen=steps)
    orig = bust.image_api.synthesize_cloudflare

    def refuses_eight(prompt, path, seed, cfg):
        if cfg["images"]["cloudflare"]["steps"] > 4:
            steps.append(cfg["images"]["cloudflare"]["steps"])
            raise image_api.ImageError("cloudflare HTTP 400: b'steps must be <= 4'")
        return orig(prompt, path, seed, cfg)

    bust.image_api.synthesize_cloudflare = refuses_eight
    try:
        d = bust.make(12, "t", "t", "", tmp / "h", CFG, budget=5000)
    finally:
        _restore()
    check(d["steps"] == 4 and steps[0] == 8 and len(calls) == 4,
          "a refused step count is lowered once, not given up on")


def test_neurons_count_a_rolling_day_and_hold_frames_back(tmp):
    now = datetime(2026, 10, 19, 14, 0, tzinfo=timezone.utc)
    check(neurons.window_start(now) == datetime(2026, 10, 18, 14, 0, tzinfo=timezone.utc),
          "the window is the last 24 hours, not a calendar day")

    db.DB_PATH = tmp / "n.db"
    db.init_db()
    vid = db.create_video("t")
    ledger = tmp / "ledger.json"
    ledger.write_text(json.dumps({"spend": [
        {"at": "2026-10-18T18:08:00+00:00", "neurons": 979}]}), encoding="utf-8")
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.execute("INSERT INTO events (video_id, stage, level, message, created_at) VALUES "
                     "(?, 'images', 'info', ?, ?)",
                     (vid, "64 made, 32 redrawn for lettering, 9216 neurons over 96 Cloudflare "
                           "calls", "2026-10-19T06:23:38+00:00"))
        conn.commit()
    # What the probe saw on 19 October: shut at 14:42, open at 18:29 UTC.
    shut = datetime(2026, 10, 19, 14, 42, tzinfo=timezone.utc)
    spent = neurons.spent_today(shut, ledger)
    check(spent == 9216 + 979, f"at 14:42 the evening's 979 and the morning's frames are both "
                               f"inside the window, over 10,000 with the probe ({spent})")
    check(neurons.available(300, shut, ledger) == 0, "so nothing is free then")
    opened = datetime(2026, 10, 19, 18, 29, tzinfo=timezone.utc)
    check(neurons.spent_today(opened, ledger) == 9216,
          "at 18:29 the 979 from 18:08 the day before has dropped out, as Cloudflare showed")
    check(neurons.available(0, opened, ledger) == 784 and neurons.frames_done(opened),
          "leaving 784, and the frames inside the window hold nothing back")
    next_day = datetime(2026, 10, 20, 7, 0, tzinfo=timezone.utc)
    check(neurons.spent_today(next_day, ledger) == 0 and not neurons.frames_done(next_day),
          "a day after the frames the window is empty and the next frames are still to come")
    check(neurons.available(0, next_day, ledger, frames_reserve=9250) == 750,
          "so 9,250 is held for them and 750 is free")


def test_the_corner_stays_empty_and_the_check_says_so(tmp):
    for headline in ("ANGER COSTS MORE", "SILENCE OUTLASTS EVERY INSULT",
                     "FEAR SHRINKS IN DAYLIGHT", "STRIP EMOTION FROM CRITICISM",
                     bust.FALLBACK_HEADLINE):
        path = fake_bust(tmp / "corner.jpg")
        img, res = bust.compose(path, headline, 1)
        check(res["ok"], f"{headline}: every check passes ({res['problems']})")
        check(3 <= res["lines"] <= 4 and res["head_cap_px"] >= bust.MIN_HEAD_CAP_PX,
              f"{headline}: {res['lines']} lines, capitals {res['head_cap_px']}px at 210x118")
        # The drawn pixels in the corner are the backdrop's own: no white, no yellow.
        corner = np.asarray(img.convert("RGB"))[bust.CORNER[1]:, bust.CORNER[0]:]
        check(corner.max() < 140, f"{headline}: the bottom-right corner holds no type")
    # A layout forced into the corner is caught by the measurement, not just avoided.
    layout = bust.layout_text("ANGER COSTS MORE")
    layout["top"] = 400
    with __import__("PIL.Image", fromlist=["Image"]).open(tmp / "corner.jpg") as im:
        bg = bust.place_bust(im.convert("RGB"), bust.ground_colour(im.convert("RGB")), 1)
    final, masks, inks = bust.draw_text(bg, layout)
    res = bust.check(bg, final, masks, inks, layout, np.ones((720, 1280), dtype=bool))
    check(any("corner" in p for p in res["problems"]), "type pushed into the corner fails")


def test_a_video_without_a_bust_is_done_first_and_old_layouts_again(tmp):
    import scripts_refresh as r
    db.DB_PATH = tmp / "plan.db"
    db.init_db()
    now = datetime(2026, 10, 19, 14, 0, tzinfo=timezone.utc)
    ids = {}
    for name, when in (("no_bust", "2026-10-20T18:00:00Z"), ("old_layout", "2026-10-19T18:00:00Z"),
                       ("current", "2026-10-21T18:00:00Z"), ("public", "2026-10-15T18:00:00Z")):
        vid = db.create_video(name)
        db.update_video(vid, youtube_id="yt" + name, published_at=when)
        ids[name] = vid
    ledger = {"done": {str(ids["old_layout"]): {"layout": 1},
                       str(ids["current"]): {"layout": r.LAYOUT}}, "spend": [], "set_log": []}
    urgent, old = r.plan(ledger, now, 2)
    check([v["id"] for v in urgent] == [ids["no_bust"], ids["old_layout"]],
          "the video with no bust comes before the redo of an old layout")
    check([v["id"] for v in old] == [ids["public"]] and ids["current"] not in
          [v["id"] for v in urgent + old], "a public one is later, and a current one is left alone")


def _events(stage):
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        return [r[0] for r in conn.execute("SELECT message FROM events WHERE stage=?", (stage,))]


def test_the_bust_is_drawn_before_the_frames_and_finished_after(tmp):
    import run as runner
    import pipeline.bust as bust_stage
    names = [n for n, _, _ in runner.STAGES]
    check(names.index("bust") < names.index("images") < names.index("thumbnail"),
          f"the bust is drawn before the frames, the type set after the title ({names})")

    db.DB_PATH = tmp / "stage.db"
    db.init_db()
    config.OUTPUT_DIR = tmp / "out"
    vid = db.create_video("discipline")
    db.update_video(vid, seo_title="10 Stoic Lessons on Discipline")
    calls = _stub()
    real_available = neurons.available
    neurons.available = lambda *a, **k: 0          # the estimate is not asked any more
    try:
        check(bust_stage.run(vid, CFG), "the bust stage succeeds")
        drawn = len(calls)
        check(drawn == CFG["thumbnail"]["candidates"],
              f"Cloudflare is asked even when the estimate says nothing is free ({drawn})")
        check(bust_stage.run(vid, CFG) and len(calls) == drawn,
              "a second run of the stage does not pay for the bust again")
        bust.image_api.synthesize_cloudflare = _forbidden
        check(thumb.run(vid, CFG), "the thumbnail stage sets the type on the kept bust")
    finally:
        neurons.available = real_available
        _restore()
    check(any("neurons over" in m for m in _events("bust")),
          "the bust stage's spend is on the record")
    check(any(m.startswith("bust:") for m in _events("thumbnail")),
          "the thumbnail event says bust")
    video = db.get_video(vid)
    check(video["thumbnail_path"].endswith("thumbnail.jpg"), "the bust thumbnail is recorded")


REFUSED = ("cloudflare HTTP 429: you have used up your daily free allocation of 10,000 "
           "neurons")


def test_no_bust_and_nothing_kept_means_no_thumbnail_yet_never_a_substitute(tmp):
    import pipeline.bust as bust_stage
    db.DB_PATH = tmp / "wait.db"
    db.init_db()
    config.OUTPUT_DIR = tmp / "out_wait"
    vid = db.create_video("anger")
    db.update_video(vid, seo_title="10 Ways to Win Arguments Through Silence")
    _stub(cf_error=lambda steps: REFUSED)
    used = {}
    real_classic, real_draw, real_lib = (thumb.make_thumbnail, thumb.cover_image.draw,
                                         library.LIB_DIR)
    library.LIB_DIR = tmp / "empty_lib"                 # nothing kept yet
    thumb.make_thumbnail = lambda *a, **k: used.setdefault("classic", True)
    thumb.cover_image.draw = _forbidden
    try:
        check(bust_stage.run(vid, CFG), "a refused bust does not stop the run")
        check(thumb.run(vid, CFG), "nor does the thumbnail stage: the video goes ahead")
    finally:
        thumb.make_thumbnail, thumb.cover_image.draw = real_classic, real_draw
        library.LIB_DIR = real_lib
        _restore()
    check("classic" not in used, "the frame-cut thumbnail is never made in its place")
    check(not db.get_video(vid).get("thumbnail_path"),
          "with nothing kept no file is recorded, so the upload sets no thumbnail of ours")
    check(any(m.startswith("bust pending") for m in _events("thumbnail")),
          "and the event marks the bust as pending")


def test_a_kept_bust_is_the_cover_from_the_first_minute(tmp):
    import pipeline.bust as bust_stage
    import pipeline.upload as upload
    db.DB_PATH = tmp / "temp.db"
    db.init_db()
    config.OUTPUT_DIR = tmp / "out_temp"
    lib = tmp / "temp_lib"
    real_lib = library.LIB_DIR
    library.LIB_DIR = lib
    old = library.add(fake_bust(tmp / "old.jpg", cx=0.45), 20, "seneca", "gemini", "ok",
                      "2026-10-01T08:00:00+00:00")
    newer = library.add(fake_bust(tmp / "newer.jpg", cx=0.5), 21, "marcus", "gemini", "ok",
                        "2026-10-05T08:00:00+00:00")
    names = []
    for n in range(2):
        vid = db.create_video(f"anger{n}")
        db.update_video(vid, seo_title="10 Ways to Win Arguments Through Silence")
        _stub(cf_error=lambda steps: REFUSED)
        try:
            check(bust_stage.run(vid, CFG) and thumb.run(vid, CFG),
                  f"video {n + 1}: the stages complete while Cloudflare refuses")
        finally:
            _restore()
        video = db.get_video(vid)
        check(video["thumbnail_path"].endswith("thumbnail.jpg")
              and Path(video["thumbnail_path"]).exists(),
              f"video {n + 1}: it has a thumbnail file to upload")
        msg = next(m for m in _events("thumbnail") if m.startswith("bust temporary")
                   and m not in names)
        names.append(msg)
    library.LIB_DIR = real_lib
    check(old in names[0] and newer in names[1],
          "the second video gets the other bust, not the one just used")
    check(all(not m.startswith("bust:") for m in _events("thumbnail")),
          "and neither is recorded as a bust of its own")
    check([u["file"] for u in library.uses_from_events()] == [old, newer],
          "the reuse is read back from the database for the next pick")
    check(any("bust temporary" in m for m in _events("thumbnail")), "the event is on record")
    check(upload.set_thumbnail is not None, "and the upload stage sends thumbnail_path as it did")


def _pending_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bust_pending", Path(__file__).resolve().parent.parent / "scripts" / "bust_pending.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pending_videos_get_their_bust_before_publication(tmp):
    import sqlite3
    from contextlib import closing
    pend = _pending_module()
    db.DB_PATH = tmp / "pending.db"
    db.init_db()
    config.OUTPUT_DIR = tmp / "out_pending"
    pend.SAMPLES = tmp / "samples"
    pend.LEDGER = tmp / "bust_pending.json"
    now = datetime.now(timezone.utc)
    ids = {}
    for name, hours, message in (
            ("frame_cut", 30, "SILENCE / BEATS ARGUMENTS in amber over scene_027_00.jpg"),
            ("has_bust", 6, f"bust: SILENCE ENDS DEBATE [layout {bust.LAYOUT}] (words ...)"),
            ("pending", 4, "bust pending: no bust thumbnail yet"),
            ("published", -2, "bust pending: no bust thumbnail yet")):
        vid = db.create_video(name)
        db.update_video(vid, youtube_id="yt_" + name, seo_title="10 Ways to Stay Calm",
                        published_at=(now + timedelta(hours=hours)).isoformat())
        with closing(sqlite3.connect(db.DB_PATH)) as conn:
            conn.execute("INSERT INTO events (video_id, stage, level, message, created_at) "
                         "VALUES (?, 'thumbnail', 'info', ?, ?)", (vid, message, now.isoformat()))
            conn.commit()
        ids[name] = vid
    ledger = pend.load_ledger(pend.LEDGER)
    got = [r["id"] for r in pend.pending(ledger, now)]
    check(got == [ids["pending"], ids["frame_cut"]],
          f"the frame-cut and the pending one are due, soonest first; not the bust, not the "
          f"published one ({got})")

    sent = []
    import pipeline.upload as upload
    import core.youtube as youtube
    real = (upload.set_thumbnail, youtube.build_service, youtube.load_credentials)
    upload.set_thumbnail = lambda service, yt, path: sent.append((yt, Path(path).name))
    youtube.build_service = lambda creds: "service"
    youtube.load_credentials = lambda: "creds"
    pend.load_config = lambda: CFG
    _stub(cf_error=lambda steps: REFUSED)
    try:
        check(pend.main() == 0 and not sent, "while Cloudflare refuses, nothing is sent")
        _stub()
        check(pend.main() == 0, "the next run draws")
    finally:
        upload.set_thumbnail, youtube.build_service, youtube.load_credentials = real
        _restore()
    check(sent == [("yt_pending", "thumbnail.jpg"), ("yt_frame_cut", "thumbnail.jpg")],
          f"and sets the bust on both through thumbnails.set ({sent})")
    ledger = pend.load_ledger(pend.LEDGER)
    check(set(ledger["done"]) == {str(ids["pending"]), str(ids["frame_cut"])}
          and not pend.pending(ledger, now), "after which nothing is pending")


def test_the_library_keeps_every_bust_and_picks_the_longest_unused(tmp):
    lib = tmp / "lib"
    names = []
    for vid, day in ((30, 1), (31, 2), (32, 3)):
        names.append(library.add(fake_bust(tmp / f"b{vid}.jpg", cx=0.4 + vid / 100), vid,
                                 "seneca", "gemini", "ok", f"2026-10-{day:02d}T08:00:00+00:00",
                                 lib_dir=lib))
    check(library.add(tmp / "b30.jpg", 30, lib_dir=lib) == names[0]
          and len(library.entries(lib)) == 3, "a bust already kept is not kept twice")
    check(library.pick([], lib).name == names[0], "with no reuse the oldest drawn is picked")
    uses = [{"file": names[0], "at": "2026-10-10T00:00:00+00:00"}]
    check(library.pick(uses, lib).name == names[1],
          "once it has been reused, the next longest out of use is")
    check(library.pick([], tmp / "empty") is None, "an empty library picks nothing")


def test_two_hours_before_publication_a_kept_bust_is_used(tmp):
    import sqlite3
    from contextlib import closing
    pend = _pending_module()
    db.DB_PATH = tmp / "fallback.db"
    db.init_db()
    config.OUTPUT_DIR = tmp / "out_fallback"
    pend.SAMPLES = tmp / "samples_fallback"
    pend.LEDGER = tmp / "fallback.json"
    lib = tmp / "fallback_lib"
    real_lib = library.LIB_DIR
    library.LIB_DIR = lib
    kept = library.add(fake_bust(tmp / "kept.jpg"), 20, "seneca", "gemini", "ok",
                       "2026-10-01T08:00:00+00:00")
    now = datetime.now(timezone.utc)
    ids = {}
    for name, hours in (("soon", 1.5), ("later", 20)):
        vid = db.create_video(name)
        db.update_video(vid, youtube_id="yt_" + name, seo_title="10 Ways to Stay Calm",
                        published_at=(now + timedelta(hours=hours)).isoformat())
        with closing(sqlite3.connect(db.DB_PATH)) as conn:
            conn.execute("INSERT INTO events (video_id, stage, level, message, created_at) "
                         "VALUES (?, 'thumbnail', 'warning', 'bust pending: none', ?)",
                         (vid, now.isoformat()))
            conn.commit()
        ids[name] = vid
    sent = []
    import pipeline.upload as upload
    import core.youtube as youtube
    real = (upload.set_thumbnail, youtube.build_service, youtube.load_credentials)
    upload.set_thumbnail = lambda service, yt, path: sent.append((yt, Path(path).name))
    youtube.build_service = lambda creds: "service"
    youtube.load_credentials = lambda: "creds"
    pend.load_config = lambda: CFG
    _stub(cf_error=lambda steps: REFUSED)
    try:
        check(pend.main() == 0, "the run completes while Cloudflare refuses")
    finally:
        upload.set_thumbnail, youtube.build_service, youtube.load_credentials = real
        library.LIB_DIR = real_lib
        _restore()
    check(sent == [("yt_soon", "thumbnail.jpg")],
          f"the video due within two hours gets a kept bust, the later one waits ({sent})")
    ledger = pend.load_ledger(pend.LEDGER)
    done = ledger["done"].get(str(ids["soon"]), {})
    check(done.get("source") == "library" and str(ids["later"]) not in ledger["done"],
          "the ledger says it came from the library")
    check([u["file"] for u in ledger["library_uses"]] == [kept],
          "and which kept bust was used, so it is not the next one picked")
    with Image.open(config.OUTPUT_DIR / str(ids["soon"]) / "thumbnail.jpg") as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.int32)
    right = rgb[:, round(1280 * bust.TEXT_LEFT):]
    yellow = ((right[..., 0] > 220) & (right[..., 1] > 170) & (right[..., 2] < 80)).sum()
    check(yellow > 5000, "the kept bust carries this video's phrase in the channel's layout")


def test_the_own_bust_replaces_the_temporary_one_through_thumbnails_set(tmp):
    import sqlite3
    from contextlib import closing
    pend = _pending_module()
    db.DB_PATH = tmp / "replace.db"
    db.init_db()
    config.OUTPUT_DIR = tmp / "out_replace"
    pend.SAMPLES = tmp / "samples_replace"
    pend.LEDGER = tmp / "replace.json"
    real_lib = library.LIB_DIR
    library.LIB_DIR = tmp / "replace_lib"
    kept = library.add(fake_bust(tmp / "k.jpg"), 20, "seneca", "gemini", "ok",
                       "2026-10-01T08:00:00+00:00")
    now = datetime.now(timezone.utc)
    vid = db.create_video("temp")
    db.update_video(vid, youtube_id="yt_temp", seo_title="10 Ways to Stay Calm",
                    published_at=(now + timedelta(hours=30)).isoformat())
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.execute("INSERT INTO events (video_id, stage, level, message, created_at) VALUES "
                     "(?, 'thumbnail', 'warning', ?, ?)",
                     (vid, f"bust temporary: no bust of its own yet (x); kept bust {kept} set "
                           f"until it is drawn", now.isoformat()))
        conn.commit()
    check([r["id"] for r in pend.pending(pend.load_ledger(pend.LEDGER), now)] == [vid],
          "a video wearing a stand-in is still waiting for its own bust")
    sent = []
    import pipeline.upload as upload
    import core.youtube as youtube
    real = (upload.set_thumbnail, youtube.build_service, youtube.load_credentials)
    upload.set_thumbnail = lambda service, yt, path: sent.append((yt, Path(path).name))
    youtube.build_service = lambda creds: "service"
    youtube.load_credentials = lambda: "creds"
    pend.load_config = lambda: CFG
    _stub(cf_error=lambda steps: REFUSED)
    try:
        check(pend.main() == 0 and not sent,
              "while Cloudflare refuses the stand-in stays and nothing more is sent")
        _stub()
        check(pend.main() == 0, "when it answers the run draws")
    finally:
        upload.set_thumbnail, youtube.build_service, youtube.load_credentials = real
        library.LIB_DIR = real_lib
        _restore()
    check(sent == [("yt_temp", "thumbnail.jpg")], f"its own bust is sent over it ({sent})")
    ledger = pend.load_ledger(pend.LEDGER)
    check(ledger["done"][str(vid)]["source"] == "drawn" and not ledger["library_uses"],
          "recorded as the video's own, not a reuse")
    check(not pend.pending(ledger, now), "after which it is no longer pending")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_the_bust_is_chosen_by_the_script_but_nobody_is_named()
        test_the_words_come_from_phrase_py()
        test_a_bust_thumbnail_is_made_checked_and_costed(tmp)
        test_a_small_budget_buys_fewer_candidates(tmp)
        test_no_budget_means_no_bust_and_no_pollinations(tmp)
        test_a_failed_check_is_drawn_again_then_the_best_is_kept(tmp)
        test_gemini_rejections_and_gemini_absence(tmp)
        test_eight_steps_refused_falls_back_to_four(tmp)
        test_neurons_count_a_rolling_day_and_hold_frames_back(tmp)
        test_the_corner_stays_empty_and_the_check_says_so(tmp)
        test_a_video_without_a_bust_is_done_first_and_old_layouts_again(tmp)
        test_the_bust_is_drawn_before_the_frames_and_finished_after(tmp)
        test_no_bust_and_nothing_kept_means_no_thumbnail_yet_never_a_substitute(tmp)
        test_a_kept_bust_is_the_cover_from_the_first_minute(tmp)
        test_pending_videos_get_their_bust_before_publication(tmp)
        test_the_library_keeps_every_bust_and_picks_the_longest_unused(tmp)
        test_two_hours_before_publication_a_kept_bust_is_used(tmp)
        test_the_own_bust_replaces_the_temporary_one_through_thumbnails_set(tmp)
    print("ALL BUST THUMBNAIL TESTS PASSED")
