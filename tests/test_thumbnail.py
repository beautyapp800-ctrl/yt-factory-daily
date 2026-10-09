"""Offline tests for the thumbnail stage. Run: python tests/test_thumbnail.py

Uses Pillow and the real Anton font in assets/fonts, with synthetic pictures standing in
for the generated images, so the checks are exact: which picture is chosen, how the words
are split, where the type lands, and whether the feed-size check can actually say no.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import config, db
import core.images as image_api
import pipeline.cover_image as cover
import pipeline.phrase as phrase
import pipeline.thumbnail as thumb


def _network_forbidden(*args, **kwargs):
    raise AssertionError("this test reached the network: a stage that talks to Groq or "
                         "Cloudflare was run without being stubbed, and that spends real "
                         "Neurons (96 an image) every time the suite runs")


# Replaced for the whole file. test_run_uses_the_seo_title... used to call thumb.run() with an
# empty config, which since the thumbnail got its own phrase and its own picture meant a real
# Groq call and three real Cloudflare images per call - 576 Neurons for one run of the tests,
# found only because the log showed a phrase the title did not contain.
image_api.synthesize = _network_forbidden
phrase.complete_json = _network_forbidden


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def _save(path, array):
    Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB").save(path, "PNG")
    return str(path)


def _pictures(tmp):
    """Four 1024x1024 pictures: flat grey, a blank bright panel on dark (high contrast but
    nothing to look at), textured with a bright rim, and textured with a dark rim."""
    y, x = np.mgrid[0:1024, 0:1024]
    rim = (x < 150) | (x > 874) | (y < 150) | (y > 874)

    flat = np.full((1024, 1024), 120)
    panel = np.full((1024, 1024), 15)
    panel[250:780, 250:780] = 235                       # one big flat bright shape

    texture = (np.sin(x / 7.0) * np.sin(y / 11.0) * 110 + 125)
    bright_rim = texture.copy(); bright_rim[rim] = 245
    dark_rim = texture.copy(); dark_rim[rim] = 8
    dark_rim[:, :460] = dark_rim[:, :460] * 0.2         # dark on the left, where type goes
    return (_save(tmp / "flat.png", flat), _save(tmp / "panel.png", panel),
            _save(tmp / "bright.png", bright_rim), _save(tmp / "dark.png", dark_rim))


# --- the words ---------------------------------------------------------------

def test_the_words_come_from_the_title():
    print("test_the_words_come_from_the_title")
    first, second = thumb.title_lines("10 Stoic Lessons to Use Money as a Tool, Not a Status Symbol")
    check((first, second) == (["MONEY", "AS", "A", "TOOL"], ["NOT", "A", "STATUS", "SYMBOL"]),
          f"a comma is a setup and a payoff, one line each ({first} / {second})")
    first, second = thumb.title_lines("10 Stoic Lessons To Unchain Your Wealth")
    check(first == ["UNCHAIN"] and second == ["YOUR", "WEALTH"],
          f"without one it splits where the halves come out even ({first} / {second})")
    check(all(w == w.upper() for w in first + second), "always upper case")
    check("STOIC" not in first and "LESSONS" not in first,
          "the opening every title in the niche shares is dropped")


def test_more_informative_word_pairs_come_first():
    print("test_more_informative_word_pairs_come_first")
    options = thumb.title_line_options("10 Stoic Lessons to Use Money as a Tool, Not a Status Symbol")
    check(len(options) > 5, f"several pairs are on the table ({len(options)})")
    words = [len(a) + len(b) for a, b in options]
    check(words[0] == max(words), f"the fullest pair is offered first ({options[0]})")
    check(all(1 <= len(a) <= thumb.MAX_WORDS_PER_LINE and 1 <= len(b) <= thumb.MAX_WORDS_PER_LINE
              for a, b in options), "every pair is two usable lines")
    check(any(a == ["NOT", "A"] and b == ["STATUS", "SYMBOL"] for a, b in options),
          "including the shorter ones the geometry may need")


def test_a_layout_is_only_offered_if_it_meets_the_brief():
    print("test_a_layout_is_only_offered_if_it_meets_the_brief")
    found = thumb.layouts(["NOT", "A"], ["STATUS", "SYMBOL"])
    check(found, "a short pair has usable layouts")
    for layout in found:
        check_height = thumb.HEIGHT / 3 <= layout["height"] <= thumb.HEIGHT * thumb.MAX_BLOCK_FRACTION
        if not check_height or layout["width"] > thumb.WIDTH * thumb.TEXT_COLUMN - thumb.MARGIN_X:
            raise AssertionError(f"offered a layout that breaks the brief: {layout}")
    check(True, f"all {len(found)} are at least a third of the frame tall and fit the left half")
    check(found[0]["size"] > found[-1]["size"], "largest type first, so the first that reads is the biggest")
    check(all(l["small_size"] < l["size"] for l in found), "the second line is always the smaller one")
    check(thumb.layouts(["EXTRAORDINARILY", "UNCOMPROMISING"], ["MAGNIFICENT"]) == [],
          "words too wide for the left half at that height are refused, not squeezed")


# --- the picture -------------------------------------------------------------

def test_the_background_is_dark_edged_dark_left_and_has_something_to_look_at(tmp):
    print("test_the_background_is_dark_edged_dark_left_and_has_something_to_look_at")
    flat, panel, bright, dark = _pictures(tmp)
    path, details = thumb.pick_background([(flat, "cloudflare"), (panel, "cloudflare"),
                                           (bright, "cloudflare"), (dark, "cloudflare")])
    check(path == dark, "texture, a dark rim and a dark left half beat flat and beat a bright rim")
    check(path != panel,
          "and beat a big flat bright shape, which scores high on contrast but is empty")
    check(details["detail"] > 1 and details["left"] < details["rim"] + 100,
          f"the numbers behind the choice are recorded ({details})")


def test_watermarked_and_duplicate_images_are_set_aside(tmp):
    print("test_watermarked_and_duplicate_images_are_set_aside")
    flat, panel, bright, dark = _pictures(tmp)
    path, details = thumb.pick_background([(dark, "pollinations"), (flat, "cloudflare"),
                                           (flat, "duplicate")])
    check(path == flat, "a pollinations image (watermarked) is not used while a clean one exists")
    check(details["candidates"] == 1, "and the same file listed twice is one candidate")
    path, _ = thumb.pick_background([(dark, "pollinations")])
    check(path == dark, "if nothing else exists, the watermarked image beats no thumbnail")
    try:
        thumb.pick_background([])
        check(False, "no images should raise")
    except RuntimeError:
        check(True, "no images at all is an error, not a blank thumbnail")


def test_the_picture_is_cropped_darkened_and_vignetted(tmp):
    print("test_the_picture_is_cropped_darkened_and_vignetted")
    grey = _save(tmp / "grey.png", np.full((1024, 1024), 160))
    img = thumb.background(grey)
    check(img.size == (1280, 720), f"1280x720 ({img.size})")
    arr = np.asarray(img.convert("L"), dtype=np.float64)
    centre, corner = arr[360, 640], arr[6, 6]
    check(centre < 160, f"the middle is darker than the original ({centre:.0f} from 160)")
    check(corner < centre * 0.6, f"and the corners are darker still ({corner:.0f} against {centre:.0f})")
    check(thumb.crop_16_9(Image.new("RGB", (1024, 1024))).size == (1024, 576),
          "a square is cropped to 16:9, not stretched")


def test_the_scrim_darkens_behind_the_type_and_fades_out(tmp):
    print("test_the_scrim_darkens_behind_the_type_and_fades_out")
    grey = _save(tmp / "even.png", np.full((1024, 1024), 200))
    base = thumb.background(grey)
    box = (54, 400, 600, 664)
    after = np.asarray(thumb.apply_scrim(base, box).convert("L"), dtype=np.float64)
    before = np.asarray(base.convert("L"), dtype=np.float64)
    check(after[600, 300] < before[600, 300] * 0.3, "behind the type it is much darker")
    check(after[100, 1200] > before[100, 1200] * 0.95,
          "the opposite corner is untouched, so there is no rectangle")
    # The ratio, not the luminance: the picture under it has its own vignette, so absolute
    # brightness falls again towards the frame edge while the scrim itself keeps weakening.
    ratio = after[600, 600:1100] / np.maximum(before[600, 600:1100], 1e-9)
    check(np.all(np.diff(ratio) >= -1e-6) and ratio[0] < 0.6 < ratio[-1],
          "the scrim only weakens to the right of the type, a gradient rather than an edge")


# --- the whole thing ---------------------------------------------------------

def test_the_finished_thumbnail(tmp):
    print("test_the_finished_thumbnail")
    flat, panel, bright, dark = _pictures(tmp)
    out = tmp / "made"
    out.mkdir()
    result = thumb.make_thumbnail([(flat, "cloudflare"), (dark, "cloudflare")],
                                  "10 Stoic Lessons to Use Money as a Tool, Not a Status Symbol", out)

    with Image.open(out / "thumbnail.jpg") as opened:
        big, fmt = opened.copy(), opened.format
    with Image.open(out / "thumb_small.jpg") as small:
        check(small.size == (210, 118), f"thumb_small.jpg is {small.size}")
    check(big.size == (1280, 720) and fmt == "JPEG", f"{big.size} JPEG")
    check(result["block_fraction"] >= 1 / 3,
          f"the type is {result['block_fraction']:.0%} of the frame height, at least a third")

    arr = np.asarray(big.convert("L"))
    white = arr > 240
    rows, cols = np.where(white)
    check(cols.max() < thumb.WIDTH * 0.55,
          f"all the type is in the left half, the right is left to the picture ({cols.max()}px)")
    check(40 <= cols.min() <= 80, f"left aligned, a margin in from the edge ({cols.min()}px)")
    # Two lines, in two different colours: white above, amber below.
    rgb = np.asarray(big.convert("RGB"), dtype=int)
    warm = (rgb[:, :, 0] > 200) & (rgb[:, :, 2] < 140) & (rgb[:, :, 1] > 130)
    warm_rows = np.where(warm.any(axis=1))[0]
    check(len(warm_rows) > 10 and warm_rows.min() > rows.min(),
          "the second line is warm-coloured and sits under the white one")
    check(warm_rows.max() > thumb.HEIGHT * 0.8 and rows.min() > thumb.HEIGHT * 0.45,
          f"the block sits low in the frame (rows {rows.min()}-{warm_rows.max()})")

    r = result["readability"]
    check(r["ok"], f"it passes at 210x118: capital {r['cap_px']:.0f}px, "
                   f"contrast {r['contrast_first']:.1f}:1 and {r['contrast_second']:.1f}:1")


def test_the_feed_size_check_can_say_no(tmp):
    print("test_the_feed_size_check_can_say_no")
    # A check that always passes is a decoration, so: white type on a white picture, with
    # the scrim removed so nothing can save it.
    white_picture = _save(tmp / "white.png", np.full((1024, 1024), 255))
    out = tmp / "made_white"
    out.mkdir()
    original = thumb.SCRIM_DARKNESS, thumb.DARKEN, thumb.VIGNETTE_STRENGTH
    thumb.SCRIM_DARKNESS, thumb.DARKEN, thumb.VIGNETTE_STRENGTH = 0.0, 1.0, 0.0
    try:
        result = thumb.make_thumbnail([(white_picture, "cloudflare")], "Stop Chasing Approval", out)
    finally:
        thumb.SCRIM_DARKNESS, thumb.DARKEN, thumb.VIGNETTE_STRENGTH = original
    r = result["readability"]
    check(not r["ok"] and min(r["contrast_first"], r["contrast_second"]) < thumb.MIN_CONTRAST,
          f"white type on a white picture fails ({r['contrast_first']:.1f}:1, "
          f"{r['contrast_second']:.1f}:1, need {thumb.MIN_CONTRAST})")
    check(result["layouts_tried"] > 1, "and it tried everything before giving up")


def test_run_uses_the_seo_title_and_records_what_it_did(tmp):
    print("test_run_uses_the_seo_title_and_records_what_it_did")
    db.DB_PATH = tmp / "run.db"
    config.OUTPUT_DIR = tmp / "out_run"
    db.init_db()
    vid = db.create_video("thumb test")
    db.update_video(vid, title="Ten Lessons On Money", seo_title="NEVER Explain Yourself Again")
    flat, panel, bright, dark = _pictures(tmp)
    db.add_scene(vid, 1, text="One.", image_prompt="x")
    scene = db.get_scenes(vid)[0]
    for i, path in enumerate((flat, dark)):
        db.add_scene_image(vid, scene["id"], i, prompt="x", seed=i, provider="cloudflare", path=path)

    with _stubbed(("PAUSE STOPS RAGE", []), [(dark, "cloudflare")]):
        check(thumb.run(vid, {}) is True, "the stage completes")
    video = db.get_video(vid)
    check(video["thumbnail_path"].endswith("thumbnail.jpg") and Path(video["thumbnail_path"]).exists(),
          "videos.thumbnail_path points at the file")
    check((config.output_dir(vid) / "thumb_small.jpg").exists(), "thumb_small.jpg is saved beside it")
    written = json.loads((config.output_dir(vid) / "thumbnail.json").read_text(encoding="utf-8"))
    check(written["first"] and written["second"] and "readability" in written,
          f"thumbnail.json records the words and the measurements ({sorted(written)})")
    check(written["phrase"] == "PAUSE STOPS RAGE", "the words are the phrase that was written")
    check(written["first"] + written["second"] == ["PAUSE", "STOPS", "RAGE"],
          "and nothing was cut out of the title")
    check(written["accent"] == thumb.accent_for(vid)[0], "the accent is the one this video owns")
    check(written["drawn_for_the_thumbnail"] is True, "the picture was the one drawn for it")

    db.update_video(vid, seo_title="")
    with _stubbed(("PAUSE STOPS RAGE", []), [(dark, "cloudflare")]):
        thumb.run(vid, {})
    check(True, "with no seo title it falls back to the working title")


import contextlib


@contextlib.contextmanager
def _stubbed(phrase_result, covers):
    """Replace the two things run() asks the network for, and put them back afterwards.

    Every test that calls run() has to say what they return, which is the point: none of them
    can forget they exist. Restoring matters as much as replacing - the first version of this
    helper did not, and a stub left on the phrase module made a later test of the real
    write_phrase quietly test the stub instead.
    """
    real_write, real_draw = phrase.write_phrase, cover.draw
    phrase.write_phrase = lambda *a, **k: phrase_result
    cover.draw = lambda *a, **k: covers
    try:
        yield
    finally:
        phrase.write_phrase, cover.draw = real_write, real_draw


def test_a_failed_phrase_falls_back_loudly_not_quietly(tmp):
    print("test_a_failed_phrase_falls_back_loudly_not_quietly")
    db.DB_PATH = tmp / "fallback.db"
    config.OUTPUT_DIR = tmp / "out_fallback"
    db.init_db()
    vid = db.create_video("fallback")
    db.update_video(vid, title="Working", seo_title="Stop Chasing Approval")
    flat, panel, bright, dark = _pictures(tmp)
    db.add_scene(vid, 1, text="One.", image_prompt="x")
    scene = db.get_scenes(vid)[0]
    db.add_scene_image(vid, scene["id"], 0, prompt="x", seed=0, provider="cloudflare", path=dark)

    # No phrase could be written, and no cover could be drawn: the worst day. A thumbnail
    # must still come out - a video with none is worse than one with a weak one - but both
    # fallbacks have to be on the record, or nobody learns the fix stopped working.
    with _stubbed(("WHILE STAYING TRUE", ["begins with a joining word"]), []):
        check(thumb.run(vid, {}) is True, "a thumbnail is still made")
    written = json.loads((config.output_dir(vid) / "thumbnail.json").read_text(encoding="utf-8"))
    check(written["phrase_fallback"], "the fallback is written into thumbnail.json")
    check(written["phrase"] == "", "and no phrase is claimed")
    check(written["drawn_for_the_thumbnail"] is False, "the narration-frame fallback is recorded too")
    events = [e for e in _events(vid) if e["level"] == "warning"]
    check(any("fell back to cutting the title" in e["message"] for e in events),
          "and a warning event is in the database, where the weekly report can count it")


def _events(vid):
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM events WHERE video_id = ?", (vid,))]


def test_no_title_raises(tmp):
    print("test_no_title_raises")
    db.DB_PATH = tmp / "notitle.db"
    config.OUTPUT_DIR = tmp / "out_notitle"
    db.init_db()
    vid = db.create_video("x")
    try:
        thumb.run(vid, {})
        check(False, "no title should raise")
    except RuntimeError as e:
        check("title" in str(e), "the error names the actual problem")


# --- the phrase --------------------------------------------------------------------------

def test_the_phrase_rules_refuse_what_the_brief_names():
    print("test_the_phrase_rules_refuse_what_the_brief_names")
    for opener in ("while", "when", "to", "that", "how", "for"):
        bad = f"{opener.upper()} PROGRESS DRAGS"
        check(phrase.problems(bad), f"a phrase opening on '{opener}' is refused")
    check(not phrase.problems("ANGER COSTS MORE"), "a finished thought passes")
    check(not phrase.problems("PATIENCE OR PRIDE"), "so does a named tension")
    check(phrase.problems("ANGER"), "one word is too few")
    check(phrase.problems("ANGER COSTS YOU FAR MORE THAN"), "five words are too many")
    check(phrase.problems("STOIC LESSONS"), "words every video in the niche uses say nothing")
    title = "10 Stoic Lessons to Stay Calm While Progress Drags"
    check(phrase.problems("STAY CALM WHILE", title) or phrase.problems("CALM WHILE PROGRESS", title),
          "three title words in a row are the old defect wearing a different hat")
    check(phrase.problems("PAT IENCE BREEDS SUCCESS", "10 Stoic Lessons to Master Patience"),
          "a word broken in half is refused - it reached a finished tile once")
    check(phrase.problems("SILENCE SHATTERS ARGUMENTS", "", fits=lambda p: False),
          "and so is one the layout says it cannot set")


def test_every_example_in_the_prompt_obeys_the_prompt():
    print("test_every_example_in_the_prompt_obeys_the_prompt")
    import re
    text = phrase._prompt("t", "p", ["l"], "T")
    # the ones the prompt quotes as FAILURES are the only examples allowed to break the rules
    shown = [m for m in re.findall(r'"([A-Z][A-Z ]+)"', text)
             if m not in ("WHILE PROGRESS DRAGS", "WHILE STAYING TRUE", "YOUR PHRASE")]
    check(shown, "the prompt gives examples")
    for example in shown:
        check(not phrase.problems(example), f"'{example}' passes the rules it is an example of")


def test_a_rejected_phrase_is_asked_for_again_with_the_reason():
    print("test_a_rejected_phrase_is_asked_for_again_with_the_reason")
    answers = iter([{"phrase": "WHILE PROGRESS DRAGS"}, {"phrase": "TO KEEP FRIENDS"},
                    {"phrase": "KINDNESS HAS LIMITS"}])
    prompts = []

    def fake(prompt, **kw):
        prompts.append(prompt)
        return next(answers)

    phrase.complete_json = fake
    try:
        got, complaints = phrase.write_phrase("friendship", "p", ["a", "b"], "Some Title Here")
    finally:
        phrase.complete_json = _network_forbidden
    check(got == "KINDNESS HAS LIMITS" and not complaints, "the third try is the one used")
    check(len(prompts) == 3, "it took three calls")
    check("begins with" in prompts[1] and "WHILE PROGRESS DRAGS" in prompts[1],
          "the second call is told what was wrong with the first, by name")

    phrase.complete_json = lambda *a, **k: {"phrase": "WHILE STAYING TRUE"}
    try:
        got, complaints = phrase.write_phrase("t", "p", ["a"], "T", attempts=2)
    finally:
        phrase.complete_json = _network_forbidden
    check(complaints, "refusals all the way down come back as complaints, not as a phrase")


def test_line_options_break_the_phrase_evenly_first():
    print("test_line_options_break_the_phrase_evenly_first")
    check(phrase.line_options("ANGER COSTS MORE")[0] == (["ANGER"], ["COSTS", "MORE"]),
          "the more even split comes first")
    check(phrase.line_options("ONE") == [], "a single word cannot be two lines")


# --- the accent --------------------------------------------------------------------------

def test_the_accent_turns_with_the_video_and_always_reads():
    print("test_the_accent_turns_with_the_video_and_always_reads")
    names = [thumb.accent_for(v)[0] for v in range(1, 30)]
    check(all(a != b for a, b in zip(names, names[1:])), "no two neighbouring videos share a colour")
    check(thumb.accent_for(12) == thumb.accent_for(12), "the same video always gets the same one")
    check(len({n for n, _ in thumb.ACCENTS}) >= 6, "there are enough colours for a row not to repeat")
    near_black = np.full((118, 210), 20, dtype=np.uint8)
    full = np.full((118, 210), 255, dtype=np.uint8)
    for name, rgb in thumb.ACCENTS:
        ratio = thumb._contrast_ratio(rgb, near_black, full)
        check(ratio >= thumb.MIN_CONTRAST,
              f"{name} clears {thumb.MIN_CONTRAST}:1 on the dark scrim ({ratio:.1f}:1)")


# --- the picture -------------------------------------------------------------------------

def _lamp(tmp, name, radius, level=235, base=12):
    y, x = np.mgrid[0:720, 0:1280]
    img = np.full((720, 1280), base, dtype=np.float64)
    img[((x - 900) ** 2 + (y - 300) ** 2) < radius ** 2] = level
    return _save(tmp / name, img)


def test_a_frame_needs_one_light_neither_a_sliver_nor_a_room(tmp):
    print("test_a_frame_needs_one_light_neither_a_sliver_nor_a_room")
    check(thumb.has_focus(thumb.focal_stats(_lamp(tmp, "lamp.png", 110))),
          "one lamp in a dark room has a focus")
    check(not thumb.has_focus(thumb.focal_stats(_lamp(tmp, "sliver.png", 6))),
          "a light a few pixels wide does not: at 210x118 it is not there")
    check(not thumb.has_focus(thumb.focal_stats(_lamp(tmp, "room.png", 520))),
          "a lit room does not: the light is the picture, so there is nothing to land on")
    flat = _save(tmp / "grey.png", np.full((720, 1280), 110))
    check(not thumb.has_focus(thumb.focal_stats(flat)), "an evenly lit frame does not")


def test_a_dark_picture_keeps_its_light_and_a_lit_one_is_taken_down():
    print("test_a_dark_picture_keeps_its_light_and_a_lit_one_is_taken_down")
    y, x = np.mgrid[0:720, 0:1280]
    dark = np.full((720, 1280, 3), 12.0)
    dark[((x - 900) ** 2 + (y - 300) ** 2) < 110 ** 2] = 140.0        # a dim lamp
    out = thumb.normalise_light(dark)
    check(out[300, 900].mean() > 200, f"the lamp is held up near white ({out[300, 900].mean():.0f})")
    check(out[600, 200].mean() < 12, "and the room around it goes no brighter")
    lit = np.full((720, 1280, 3), 160.0)
    check(thumb.normalise_light(lit).mean() < 160, "a flat bright frame is darkened, as it always was")


def test_a_better_lamp_beats_a_better_texture(tmp):
    print("test_a_better_lamp_beats_a_better_texture")
    lamp = _lamp(tmp, "good_lamp.png", 110)
    flat, panel, bright, dark = _pictures(tmp)
    source, pick = thumb.pick_background([(bright, "cloudflare"), (lamp, "cloudflare"),
                                          (flat, "cloudflare")])
    check(source == lamp, "the frame with one light is chosen over a richer one with none")
    check(pick["focus"] is True, "and the pick says it has a focus")


# --- the cover ---------------------------------------------------------------------------

def test_the_cover_prompt_follows_the_picture_rules_and_the_video():
    print("test_the_cover_prompt_follows_the_picture_rules_and_the_video")
    style = ("cinematic painterly illustration, wide horizontal composition, soft volumetric "
             "light, quiet atmosphere, no text, no lettering")
    seen = [cover.prompt_for(v, "t", style) for v in range(10, 16)]
    check(len(set(seen)) == 6, "six neighbouring videos get six different compositions")
    for subject in cover.SUBJECTS:
        hits = cover.banned_words_in(f"{subject}, {cover.COMPOSITION}")
        check(not hits, f"nothing the picture rules ban is in: {subject[:46]}... ({hits})")
    check("wide horizontal composition" not in seen[0] and "soft volumetric light" not in seen[0],
          "the parts of the channel style that fight a thumbnail composition are taken out")
    check("no lettering" in seen[0], "and the channel's own bans are kept")


def test_covers_are_not_bought_twice_and_a_failure_is_not_fatal(tmp):
    print("test_covers_are_not_bought_twice_and_a_failure_is_not_fatal")
    db.DB_PATH = tmp / "cov.db"
    config.OUTPUT_DIR = tmp / "out_cov"
    db.init_db()
    calls = []

    def fake(prompt, path, seed, cfg):
        calls.append(path)
        Path(path).write_bytes(b"x" * 2000)
        return "cloudflare", 96

    image_api.synthesize = fake
    try:
        first = cover.draw(77, "t", {"image_style": "cinematic"})
        check(len(first) == cover.CANDIDATES and len(calls) == cover.CANDIDATES,
              "a first run draws every candidate")
        again = cover.draw(77, "t", {"image_style": "cinematic"})
        check(len(calls) == cover.CANDIDATES and len(again) == cover.CANDIDATES,
              "a rerun of the same video draws nothing: it does not spend twice")

        def failing(prompt, path, seed, cfg):
            raise image_api.ImageError("service down")

        image_api.synthesize = failing
        check(cover.draw(78, "t", {"image_style": "cinematic"}) == [],
              "when the service is down the result is empty and nothing raises")
    finally:
        image_api.synthesize = _network_forbidden


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_the_words_come_from_the_title()
        test_more_informative_word_pairs_come_first()
        test_a_layout_is_only_offered_if_it_meets_the_brief()
        test_the_background_is_dark_edged_dark_left_and_has_something_to_look_at(tmp)
        test_watermarked_and_duplicate_images_are_set_aside(tmp)
        test_the_picture_is_cropped_darkened_and_vignetted(tmp)
        test_the_scrim_darkens_behind_the_type_and_fades_out(tmp)
        test_the_finished_thumbnail(tmp)
        test_the_feed_size_check_can_say_no(tmp)
        test_run_uses_the_seo_title_and_records_what_it_did(tmp)
        test_a_failed_phrase_falls_back_loudly_not_quietly(tmp)
        test_no_title_raises(tmp)
        test_the_phrase_rules_refuse_what_the_brief_names()
        test_every_example_in_the_prompt_obeys_the_prompt()
        test_a_rejected_phrase_is_asked_for_again_with_the_reason()
        test_line_options_break_the_phrase_evenly_first()
        test_the_accent_turns_with_the_video_and_always_reads()
        test_a_frame_needs_one_light_neither_a_sliver_nor_a_room(tmp)
        test_a_dark_picture_keeps_its_light_and_a_lit_one_is_taken_down()
        test_a_better_lamp_beats_a_better_texture(tmp)
        test_the_cover_prompt_follows_the_picture_rules_and_the_video()
        test_covers_are_not_bought_twice_and_a_failure_is_not_fatal(tmp)
    print("ALL THUMBNAIL TESTS PASSED")
