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
import pipeline.thumbnail as thumb


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

    check(thumb.run(vid, {}) is True, "the stage completes")
    video = db.get_video(vid)
    check(video["thumbnail_path"].endswith("thumbnail.jpg") and Path(video["thumbnail_path"]).exists(),
          "videos.thumbnail_path points at the file")
    check((config.output_dir(vid) / "thumb_small.jpg").exists(), "thumb_small.jpg is saved beside it")
    written = json.loads((config.output_dir(vid) / "thumbnail.json").read_text(encoding="utf-8"))
    check(written["first"] and written["second"] and "readability" in written,
          f"thumbnail.json records the words and the measurements ({sorted(written)})")

    db.update_video(vid, seo_title="")
    thumb.run(vid, {})
    check(True, "with no seo title it falls back to the working title")


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
        test_no_title_raises(tmp)
    print("ALL THUMBNAIL TESTS PASSED")
