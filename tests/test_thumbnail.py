"""Offline tests for the thumbnail stage. Run: python tests/test_thumbnail.py

Uses Pillow and the real Anton font in assets/fonts, with synthetic pictures standing in
for the generated images, so the checks are exact: which picture is chosen, how dark it
is made, where the text lands, and whether the readability check can actually say no.
"""
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
    """Three 1024x1024 pictures: flat grey, high contrast with a bright rim, and high
    contrast with a dark rim. Only the last one should win."""
    flat = np.full((1024, 1024), 120)
    y, x = np.mgrid[0:1024, 0:1024]
    stripes = np.where((x // 64) % 2 == 0, 20, 230)          # strong contrast everywhere
    bright_rim = stripes.copy()
    dark_rim = stripes.copy()
    rim = (x < 150) | (x > 874) | (y < 150) | (y > 874)
    bright_rim[rim] = 245
    dark_rim[rim] = 8
    return (_save(tmp / "flat.png", flat), _save(tmp / "bright.png", bright_rim),
            _save(tmp / "dark.png", dark_rim))


def test_words_from_the_title():
    print("test_words_from_the_title")
    cases = {
        "10 Stoic Lessons To Unchain Your Wealth": ["UNCHAIN", "YOUR", "WEALTH"],
        "10 Stoic Lessons That Will Make You Mentally Unbreakable":
            ["MAKE", "YOU", "MENTALLY", "UNBREAKABLE"],
        "NEVER Explain Yourself Again": ["NEVER", "EXPLAIN", "YOURSELF", "AGAIN"],
        "Stop Chasing Approval": ["STOP", "CHASING", "APPROVAL"],
        # setup, comma, payoff: the payoff is the thumbnail, not a window across the comma
        "10 Stoic Lessons to Use Money as a Tool, Not a Status Symbol":
            ["NOT", "A", "STATUS", "SYMBOL"],
    }
    for title, expected in cases.items():
        got = thumb.words_from_title(title)
        check(got == expected, f"{title!r} -> {' '.join(got)}")
        check(thumb.MIN_WORDS <= len(got) <= thumb.MAX_WORDS, "3 to 5 words")
    check(all(w == w.upper() for w in thumb.words_from_title("a quiet mind wins")),
          "always upper case")
    check(thumb.words_from_title("7 Stoic Habits for a Calm Mind") == ["CALM", "MIND"],
          "a promise shorter than three words is cut clean rather than padded with a fragment")


def test_background_is_the_contrasty_dark_edged_one(tmp):
    print("test_background_is_the_contrasty_dark_edged_one")
    flat, bright, dark = _pictures(tmp)
    path, details = thumb.pick_background([(flat, "cloudflare"), (bright, "cloudflare"),
                                           (dark, "cloudflare")])
    check(path == dark, "high contrast and a dark rim beats flat, and beats a bright rim")
    check(details["candidates"] == 3 and details["rim"] < 60, f"measured, not guessed ({details})")


def test_watermarked_and_duplicate_images_are_set_aside(tmp):
    print("test_watermarked_and_duplicate_images_are_set_aside")
    flat, bright, dark = _pictures(tmp)
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


def test_background_is_16_9_darker_and_vignetted(tmp):
    print("test_background_is_16_9_darker_and_vignetted")
    grey = _save(tmp / "grey.png", np.full((1024, 1024), 200))
    img = thumb.background(grey)
    check(img.size == (1280, 720), f"1280x720 ({img.size})")
    arr = np.asarray(img.convert("L"), dtype=np.float64)
    centre = arr[360, 640]
    check(abs(centre / 200 - 0.65) < 0.02,
          f"the middle is darkened by 35% ({centre / 200:.3f} of the original)")
    check(arr[5, 5] < centre * 0.7 and arr[714, 1274] < centre * 0.7,
          f"the corners are darker still, the vignette ({arr[5, 5]:.0f} against {centre:.0f})")
    square = Image.new("RGB", (1024, 1024))
    check(thumb.crop_16_9(square).size == (1024, 576), "a square is cropped to 16:9, not stretched")


def test_text_layout_is_a_third_of_the_frame_in_the_bottom_third():
    print("test_text_layout_is_a_third_of_the_frame_in_the_bottom_third")
    for words in (["UNCHAIN", "YOUR", "WEALTH"], ["MAKE", "YOU", "MENTALLY", "UNBREAKABLE"],
                  ["NEVER", "EXPLAIN", "YOURSELF", "AGAIN"], ["CALM", "MIND"]):
        lines, size = thumb.best_layout(words)
        check(" ".join(lines) == " ".join(words), f"{lines}: the words are kept in order")
        font, boxes, gap, width, height = thumb._measure(lines, size, thumb.FONT_PATH)
        check(width <= 1280 - 2 * thumb.MARGIN_X, f"{words}: fits the width ({width}px)")
        check(160 <= height <= thumb.MAX_TEXT_HEIGHT or words == ["CALM", "MIND"],
              f"{words}: about a third of the 720px frame tall ({height}px)")


def test_the_finished_thumbnail(tmp):
    print("test_the_finished_thumbnail")
    flat, bright, dark = _pictures(tmp)
    out = tmp / "made"
    out.mkdir()
    result = thumb.make_thumbnail([(flat, "cloudflare"), (dark, "cloudflare")],
                                  "10 Stoic Lessons To Unchain Your Wealth", out)

    big, small = Image.open(out / "thumbnail.jpg"), Image.open(out / "thumb_small.jpg")
    check(big.size == (1280, 720) and big.format == "JPEG", f"{big.size} JPEG")
    check(small.size == (210, 118) and small.format == "JPEG", f"thumb_small.jpg is {small.size}")
    check(result["words"] == ["UNCHAIN", "YOUR", "WEALTH"], "the words come from the title")

    # Where the white text is: the pixels that are almost pure white on a darkened picture.
    arr = np.asarray(big.convert("L"))
    white = arr > 240
    rows, cols = np.where(white)
    check(rows.min() >= 460, f"all the text is in the bottom third ({rows.min()}px from the top)")
    check(40 <= cols.min() <= 80, f"left aligned, a margin in from the edge ({cols.min()}px)")
    check(cols.max() < 1280 * 0.6, "and the block does not run across the whole frame")
    check(not white[:400].any(), "nothing white in the top half of the frame")

    # The 4px black outline: just outside a white letter edge the pixel is near black.
    top_row = rows.min()
    check(arr[top_row - 3, cols[rows == top_row][0]] < 40,
          "the lettering has a black outline round it")

    # A dark-rimmed contrasty picture under 35% darkening and white type is legible.
    r = result["readability"]
    check(r["ok"] and r["cap_px"] >= thumb.MIN_SMALL_CAP_PX and r["contrast"] >= thumb.MIN_CONTRAST,
          f"readable at 210x118: capitals {r['cap_px']:.0f}px, contrast {r['contrast']:.1f}:1")


def test_the_readability_check_can_say_no(tmp):
    print("test_the_readability_check_can_say_no")
    # A check that always passes is a decoration. White type over a white picture, even
    # after the 35% darkening, must fail it.
    white_picture = _save(tmp / "white.png", np.full((1024, 1024), 255))
    out = tmp / "made_white"
    out.mkdir()
    result = thumb.make_thumbnail([(white_picture, "cloudflare")], "Stop Chasing Approval", out)
    r = result["readability"]
    check(not r["ok"] and r["contrast"] < thumb.MIN_CONTRAST,
          f"white lettering on a bright picture fails ({r['contrast']:.1f}:1 against {thumb.MIN_CONTRAST})")


def test_run_uses_the_seo_title_and_records_the_path(tmp):
    print("test_run_uses_the_seo_title_and_records_the_path")
    db.DB_PATH = tmp / "run.db"
    config.OUTPUT_DIR = tmp / "out_run"
    db.init_db()
    vid = db.create_video("thumb test")
    db.update_video(vid, title="Ten Lessons On Money",
                    seo_title="NEVER Explain Yourself Again")
    flat, bright, dark = _pictures(tmp)
    db.add_scene(vid, 1, text="One.", image_prompt="x")
    scene = db.get_scenes(vid)[0]
    for i, path in enumerate((flat, dark)):
        db.add_scene_image(vid, scene["id"], i, prompt="x", seed=i, provider="cloudflare", path=path)

    check(thumb.run(vid, {}) is True, "the stage completes")
    video = db.get_video(vid)
    check(video["thumbnail_path"].endswith("thumbnail.jpg") and Path(video["thumbnail_path"]).exists(),
          "videos.thumbnail_path points at the file")
    check((config.output_dir(vid) / "thumb_small.jpg").exists(), "thumb_small.jpg is saved beside it")

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
        test_words_from_the_title()
        test_background_is_the_contrasty_dark_edged_one(tmp)
        test_watermarked_and_duplicate_images_are_set_aside(tmp)
        test_background_is_16_9_darker_and_vignetted(tmp)
        test_text_layout_is_a_third_of_the_frame_in_the_bottom_third()
        test_the_finished_thumbnail(tmp)
        test_the_readability_check_can_say_no(tmp)
        test_run_uses_the_seo_title_and_records_the_path(tmp)
        test_no_title_raises(tmp)
    print("ALL THUMBNAIL TESTS PASSED")
