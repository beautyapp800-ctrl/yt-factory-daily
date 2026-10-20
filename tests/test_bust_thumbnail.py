"""Offline tests for the bust thumbnail. Run: python tests/test_bust_thumbnail.py

Cloudflare, pollinations and Gemini are all stubbed: a synthetic "bust" (a light head and
shoulders on dark grey) stands in for the generated picture, so the checks are exact and
the suite spends no Neurons and no Gemini requests.
"""
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import config, db, neurons
import core.images as image_api
import pipeline.bust_thumbnail as bust
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


def test_the_stage_uses_the_bust_and_falls_back(tmp):
    db.DB_PATH = tmp / "stage.db"
    db.init_db()
    config.OUTPUT_DIR = tmp / "out"
    vid = db.create_video("discipline")
    db.update_video(vid, seo_title="10 Stoic Lessons on Discipline")
    _stub()
    real_available = neurons.available
    neurons.available = lambda *a, **k: 1500
    try:
        check(thumb.run(vid, CFG), "the stage succeeds")
    finally:
        neurons.available = real_available
        _restore()
    video = db.get_video(vid)
    check(video["thumbnail_path"].endswith("thumbnail.jpg"), "the bust thumbnail is recorded")
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        msgs = [r[0] for r in conn.execute("SELECT message FROM events WHERE stage='thumbnail'")]
    check(any(m.startswith("bust:") and "neurons" in m for m in msgs),
          "the event says bust and carries the spend")

    # Now the bust cannot be made at all: the older thumbnail is built from the narration
    # frames, and no covers are bought.
    def boom(*a, **k):
        raise RuntimeError("no bust could be drawn")
    real_make = bust.make
    bust.make = boom
    seen = {}

    def classic(candidates, title, out_dir, **k):
        seen["candidates"] = candidates
        return {"path": Path(out_dir) / "thumbnail.jpg", "small": None, "first": ["A"],
                "second": ["B"], "source": "x.jpg", "font_size": 1, "small_font_size": 1,
                "block_height": 1, "block_width": 1, "block_fraction": 0.4, "accent": "a",
                "accent_rgb": [1, 1, 1], "layouts_tried": 1, "word_options": 1,
                "readability": {"ok": True, "cap_px": 9, "contrast_first": 5,
                                "contrast_second": 5, "ocr_share": None},
                "contrast": 1, "rim": 1, "left": 1, "detail": 1, "rank": 1, "candidates": 1,
                "rejected_for_text": [], "focus": True}
    real_classic, real_phrase = thumb.make_thumbnail, thumb._phrase_for
    real_draw = thumb.cover_image.draw
    thumb.make_thumbnail = classic
    thumb._phrase_for = lambda *a: ([(["A"], ["B"])], "A B", "")
    thumb.cover_image.draw = _forbidden
    try:
        check(thumb.run(vid, CFG), "the stage still succeeds when the bust fails")
    finally:
        bust.make, thumb.make_thumbnail, thumb._phrase_for = real_make, real_classic, real_phrase
        thumb.cover_image.draw = real_draw
    check("candidates" in seen, "and the older thumbnail was made, without buying covers")


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
        test_the_stage_uses_the_bust_and_falls_back(tmp)
    print("ALL BUST THUMBNAIL TESTS PASSED")
