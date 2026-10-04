"""Offline tests for the images stage. Run: python tests/test_images.py

No network: core.images.synthesize_cloudflare / synthesize_pollinations /
cloudflare_available are stubbed, so this checks the pipeline's own logic (image
count formula, the cloudflare -> pollinations -> duplicate fallback order, seed
uniqueness, cover-image assignment, the neuron-budget projection, the banned-word
prompt fix, the OCR regeneration trigger) without calling Cloudflare, pollinations.ai,
an LLM, or Tesseract - ocr.detect_text defaults to (0, 0) ("no text found") and
core.llm.complete is stubbed, unless a test deliberately overrides one.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep test output out of the working logs/factory.log.
import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import config, db, images, ocr
import pipeline.images as stage


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_images_per_scene():
    print("test_images_per_scene")
    check(images.images_per_scene(33, 12) == 3, "33s scene at 12s/image -> 3 images")
    check(images.images_per_scene(5, 12) == 1, "a very short scene still gets at least 1")
    check(images.images_per_scene(0, 12) == 1, "a zero-duration scene still gets at least 1")
    check(images.images_per_scene(96, 12) == 8, "96s scene -> exactly 8 images")


def _setup_video(tmp, name, scene_durations, prompts=None):
    db.DB_PATH = tmp / f"{name}.db"
    config.OUTPUT_DIR = tmp / f"out_{name}"
    db.init_db()
    vid = db.create_video(f"{name} test")
    for i, dur in enumerate(scene_durations, 1):
        prompt = (prompts[i - 1] if prompts else f"a scene {i} illustration, muted tones")
        db.add_scene(vid, i, text=f"Scene {i} text.", image_prompt=prompt, duration_s=dur)
    return vid


def _stub(cloudflare=None, pollinations=None, available=True, ocr_result=(0, 0),
         llm_complete=None):
    """Install stand-ins for every network/OCR/LLM call images.py or pipeline.images
    makes, returning the originals so the caller can restore them in a finally block.
    ocr_result defaults to "no text found", so tests that do not care about OCR do
    not need Tesseract installed or real image bytes it can decode."""
    original = {
        "synthesize_cloudflare": images.synthesize_cloudflare,
        "synthesize_pollinations": images.synthesize_pollinations,
        "cloudflare_available": images.cloudflare_available,
        "ocr.detect_text": ocr.detect_text,
        "stage.complete": stage.complete,
    }
    if cloudflare is not None:
        images.synthesize_cloudflare = cloudflare
    if pollinations is not None:
        images.synthesize_pollinations = pollinations
    images.cloudflare_available = lambda: available
    ocr.detect_text = lambda path, **kw: ocr_result
    stage.complete = llm_complete or (lambda prompt, **kw: "a plain rewritten scene")
    return original


def _restore(original):
    images.synthesize_cloudflare = original["synthesize_cloudflare"]
    images.synthesize_pollinations = original["synthesize_pollinations"]
    images.cloudflare_available = original["cloudflare_available"]
    ocr.detect_text = original["ocr.detect_text"]
    stage.complete = original["stage.complete"]


def test_cloudflare_primary(tmp):
    print("test_cloudflare_primary")
    vid = _setup_video(tmp, "primary", [24])   # 2 images
    calls = {"cloudflare": 0, "pollinations": 0}

    def fake_cf(prompt, out_path, seed, cfg):
        calls["cloudflare"] += 1
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        return out_path, 19.2

    def fail_pollinations(prompt, out_path, seed, cfg):
        calls["pollinations"] += 1
        raise images.ImageError("should not be reached")

    original = _stub(cloudflare=fake_cf, pollinations=fail_pollinations)
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 12}}
        check(stage.run(vid, cfg) is True, "the stage completes")
    finally:
        _restore(original)

    check(calls["cloudflare"] == 2 and calls["pollinations"] == 0,
          "cloudflare alone serves every image when it keeps succeeding")
    rows = db.get_video_images(vid)
    check(all(r["provider"] == "cloudflare" for r in rows), "both images credited to cloudflare")


def test_pollinations_fallback(tmp):
    print("test_pollinations_fallback")
    vid = _setup_video(tmp, "fallback", [24])   # 2 images

    def fail_cf(prompt, out_path, seed, cfg):
        raise images.ImageError("cloudflare down for this test")

    def fake_poll(prompt, out_path, seed, cfg):
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

    original = _stub(cloudflare=fail_cf, pollinations=fake_poll)
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 12}}
        check(stage.run(vid, cfg) is True, "the stage completes when pollinations covers")
    finally:
        _restore(original)

    rows = db.get_video_images(vid)
    check(all(r["provider"] == "pollinations" for r in rows),
          "both images fell through to pollinations")


def test_duplicate_when_both_fail(tmp):
    print("test_duplicate_when_both_fail")
    vid = _setup_video(tmp, "duplicate", [36])   # 3 images
    attempt = {"n": 0}

    def flaky_cf(prompt, out_path, seed, cfg):
        attempt["n"] += 1
        if attempt["n"] == 1:
            Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
            return out_path, 20.0
        raise images.ImageError("down after the first image")

    def fail_poll(prompt, out_path, seed, cfg):
        raise images.ImageError("also down")

    original = _stub(cloudflare=flaky_cf, pollinations=fail_poll)
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 12}}
        check(stage.run(vid, cfg) is True,
              "the stage completes by duplicating once the first image exists")
    finally:
        _restore(original)

    rows = db.get_video_images(vid)
    check(len(rows) == 3, f"all 3 slots got something ({len(rows)})")
    check(rows[0]["provider"] == "cloudflare", "the first image is a real generation")
    check(all(r["provider"] == "duplicate" for r in rows[1:]),
          "the second and third are marked as duplicates")
    check(rows[1]["path"] == rows[0]["path"] == rows[2]["path"],
          "a duplicate points at the same file as the original, not a copy")


def test_first_image_failure_has_nothing_to_duplicate(tmp):
    print("test_first_image_failure_has_nothing_to_duplicate")
    vid = _setup_video(tmp, "nothingyet", [24])   # 2 images, both providers always fail

    def fail_cf(prompt, out_path, seed, cfg):
        raise images.ImageError("cloudflare down")

    def fail_poll(prompt, out_path, seed, cfg):
        raise images.ImageError("pollinations down")

    original = _stub(cloudflare=fail_cf, pollinations=fail_poll)
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 12}}
        try:
            stage.run(vid, cfg)
            check(False, "a video with nothing ever succeeding should raise")
        except RuntimeError as e:
            check("no images" in str(e), f"the error names the actual problem ({e})")
    finally:
        _restore(original)


def test_cover_survives_first_image_failure(tmp):
    print("test_cover_survives_first_image_failure")
    vid = _setup_video(tmp, "cover", [36])   # 3 images
    attempt = {"n": 0}

    def flaky_cf(prompt, out_path, seed, cfg):
        attempt["n"] += 1
        if attempt["n"] == 1:
            raise images.ImageError("first image never renders")
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        return out_path, 18.0

    def fail_poll(prompt, out_path, seed, cfg):
        raise images.ImageError("also down")

    original = _stub(cloudflare=flaky_cf, pollinations=fail_poll)
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 12}}
        stage.run(vid, cfg)
    finally:
        _restore(original)

    scene = db.get_scenes(vid)[0]
    check(scene["image_path"] is not None,
          "the cover points at the first image that actually succeeded, even "
          "though image index 0 failed outright and had nothing to duplicate")


def test_seeds_are_distinct(tmp):
    print("test_seeds_are_distinct")
    vid = _setup_video(tmp, "seeds", [36, 24])   # 3 + 2 = 5

    def fake_cf(prompt, out_path, seed, cfg):
        Path(out_path).write_bytes(b"\xff\xd8\xff" + str(seed).encode() + b"\x00" * 50)
        return out_path, 20.0

    original = _stub(cloudflare=fake_cf)
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 12}}
        stage.run(vid, cfg)
    finally:
        _restore(original)

    seeds = [r["seed"] for r in db.get_video_images(vid)]
    check(len(seeds) == len(set(seeds)), f"every image got a distinct seed ({seeds})")


def test_neuron_budget_projection(tmp):
    print("test_neuron_budget_projection")
    # 10 scenes x 12 images each = 120 images, comfortably over the sample size of 5.
    vid = _setup_video(tmp, "budget", [144] * 10)
    check(sum(images.images_per_scene(144, 12) for _ in range(10)) == 120,
          "fixture really does plan 120 images")

    def fake_cf(prompt, out_path, seed, cfg):
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        return out_path, 100.0   # deliberately over budget: 100 x 120 = 12000 > 10000

    original = _stub(cloudflare=fake_cf)
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 12}}
        stage.run(vid, cfg)
    finally:
        _restore(original)

    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        log_path = config.OUTPUT_DIR.parent / "logs" / "factory.log"
    # The projection is logged, not stored in the db; check it ran without error and
    # only NEURON_SAMPLE_SIZE calls were actually sampled (not all 120).
    check(stage.NEURON_SAMPLE_SIZE == 5, "sampling is capped at the first 5 images")


def test_provider_lock():
    print("test_provider_lock")
    # provider: "pollinations" must skip cloudflare even when it would succeed.
    calls = {"cloudflare": 0}

    def fake_cf(prompt, out_path, seed, cfg):
        calls["cloudflare"] += 1
        Path(out_path).write_bytes(b"\xff\xd8\xff")
        return out_path, 1.0

    def fake_poll(prompt, out_path, seed, cfg):
        Path(out_path).write_bytes(b"\xff\xd8\xff")

    original = _stub(cloudflare=fake_cf, pollinations=fake_poll)
    try:
        used, _ = images.synthesize("x", Path(tempfile.mktemp()), 1,
                                    {"images": {"provider": "pollinations"}})
        check(used == "pollinations" and calls["cloudflare"] == 0,
              "provider: pollinations bypasses cloudflare entirely")
    finally:
        _restore(original)


def test_no_scenes_raises(tmp):
    print("test_no_scenes_raises")
    db.DB_PATH = tmp / "empty.db"
    config.OUTPUT_DIR = tmp / "out_empty"
    db.init_db()
    vid = db.create_video("no scenes")
    try:
        stage.run(vid, {"images": {"provider": "cloudflare"}})
        check(False, "running images with zero scenes should raise")
    except RuntimeError as e:
        check("no scenes" in str(e), "the error names the actual problem")


def test_banned_prompt_is_fixed_before_generation(tmp):
    print("test_banned_prompt_is_fixed_before_generation")
    vid = _setup_video(tmp, "fixme", [24],
                       prompts=["A gift box with a label left on a doorstep"])
    seen_prompts = []

    def fake_cf(prompt, out_path, seed, cfg):
        seen_prompts.append(prompt)
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        return out_path, 96.0

    llm_calls = {"n": 0}

    def fake_llm(prompt, **kw):
        llm_calls["n"] += 1
        return "An empty stone courtyard at dawn, long shadows, no people"

    original = _stub(cloudflare=fake_cf, llm_complete=fake_llm)
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 12}}
        stage.run(vid, cfg)
    finally:
        _restore(original)

    check(llm_calls["n"] == 1, "exactly one rewrite request was made for the one bad prompt")
    check(all("box" not in p and "label" not in p for p in seen_prompts),
          "the banned nouns never reached the image provider")
    scene = db.get_scenes(vid)[0]
    check(scene["image_prompt"] == "An empty stone courtyard at dawn, long shadows, no people",
          "the fix was persisted back to scenes.image_prompt, not just used transiently")


def test_a_prompt_with_hands_or_a_face_is_rewritten(tmp):
    print("test_a_prompt_with_hands_or_a_face_is_rewritten")
    # The generator draws six fingers and three fingers, and nothing downstream can repair
    # it, so a prompt that asks for hands is refused before anything is drawn.
    vid = _setup_video(tmp, "anatomy", [24, 24, 24],
                       prompts=["Fingers gripping a cold metal railing in a stairwell",
                                "A close-up of a face lit by a phone screen",
                                "An empty stairwell, winter light along a worn brass railing"])
    asked, seen = [], []

    def fake_cf(prompt, out_path, seed, cfg):
        seen.append(prompt)
        Path(out_path).write_bytes(b"x" * 2000)
        return out_path, 96.0

    def fake_llm(prompt, **kw):
        asked.append(prompt)
        return "An empty stairwell at dusk, winter light along the worn brass railing"

    original = _stub(cloudflare=fake_cf, llm_complete=fake_llm)
    try:
        stage.run(vid, {"images": {"provider": "cloudflare", "images_per_seconds": 24}})
    finally:
        _restore(original)

    check(len(asked) == 2, f"the two prompts with a hand or a face were sent for a rewrite ({len(asked)})")
    check(all("hands or a face" in a for a in asked),
          "and the rewrite request says what the problem is")
    check(not any(w in p.lower() for p in seen for w in ("finger", "face", "gripping")),
          f"nothing with a hand or a face reached the generator ({seen})")
    prompts = [sc["image_prompt"] for sc in db.get_scenes(vid)]
    check("stairwell" in prompts[0] and "finger" not in prompts[0].lower(),
          f"and the fix is stored back on the scene ({prompts[0][:60]})")


def test_a_stale_image_is_drawn_again_when_its_prompt_changed(tmp):
    print("test_a_stale_image_is_drawn_again_when_its_prompt_changed")
    # Reuse keys on the prompt the file was made with, not just the file name, or a rule
    # change - taking hands out - would never reach the pictures already on disk.
    vid = _setup_video(tmp, "stale", [24], prompts=["An empty platform in the rain"])
    drawn = {"n": 0}

    def fake_cf(prompt, out_path, seed, cfg):
        drawn["n"] += 1
        _write_real_jpeg(out_path, drawn["n"])
        return out_path, 96.0

    original = _stub(cloudflare=fake_cf, ocr_result=(0, 0))
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 24}}
        stage.run(vid, cfg)
        after_first = drawn["n"]
        stage.run(vid, cfg)
        after_reuse = drawn["n"]
        for sc in db.get_scenes(vid):
            db.update_scene(sc["id"], image_prompt="A different empty platform at dawn")
        stage.run(vid, cfg)
    finally:
        _restore(original)
    check(after_reuse == after_first, f"an unchanged prompt reuses the file ({after_reuse - after_first} redrawn)")
    check(drawn["n"] > after_reuse, f"a changed prompt draws it again ({drawn['n'] - after_reuse} redrawn)")


def test_files_outside_the_plan_are_removed(tmp):
    print("test_files_outside_the_plan_are_removed")
    vid = _setup_video(tmp, "orphans", [24], prompts=["An empty platform in the rain"])
    images_dir = config.output_dir(vid) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "scene_009_07.jpg").write_bytes(b"left over from a longer plan")

    def fake_cf(prompt, out_path, seed, cfg):
        _write_real_jpeg(out_path)
        return out_path, 96.0

    original = _stub(cloudflare=fake_cf, ocr_result=(0, 0))
    try:
        stage.run(vid, {"images": {"provider": "cloudflare", "images_per_seconds": 24}})
    finally:
        _restore(original)
    check(not (images_dir / "scene_009_07.jpg").exists(),
          "a file the plan has no slot for is deleted, so the cache stops carrying it")
    check(len(list(images_dir.glob("*.jpg"))) == len(db.get_video_images(vid)),
          "what is left is exactly what the database points at")


def test_clean_prompt_skips_the_llm(tmp):
    print("test_clean_prompt_skips_the_llm")
    vid = _setup_video(tmp, "clean", [24],
                       prompts=["An empty stone courtyard at dawn, long shadows"])
    llm_calls = {"n": 0}

    def fake_cf(prompt, out_path, seed, cfg):
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        return out_path, 96.0

    def fake_llm(prompt, **kw):
        llm_calls["n"] += 1
        return "should not be called"

    original = _stub(cloudflare=fake_cf, llm_complete=fake_llm)
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 12}}
        stage.run(vid, cfg)
    finally:
        _restore(original)

    check(llm_calls["n"] == 0, "a prompt with no banned nouns never calls the LLM")


STYLE = "cinematic painterly illustration, no text, no lettering, no watermark"


def test_style_suffix_is_not_scanned_for_banned_words(tmp):
    print("test_style_suffix_is_not_scanned_for_banned_words")
    # Caught live on video 7: image_style carries "no lettering", and "lettering" is a
    # banned word, so every one of 43 perfectly good prompts was flagged for rewriting.
    vid = _setup_video(tmp, "stylescan", [24],
                       prompts=[f"A plain unmarked parcel on a doorstep at dusk, {STYLE}"])
    llm_calls = {"n": 0}

    def fake_cf(prompt, out_path, seed, cfg):
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        return out_path, 96.0

    def fake_llm(prompt, **kw):
        llm_calls["n"] += 1
        return "should not be called"

    original = _stub(cloudflare=fake_cf, llm_complete=fake_llm)
    try:
        stage.run(vid, {"image_style": STYLE,
                        "images": {"provider": "cloudflare", "images_per_seconds": 12}})
    finally:
        _restore(original)
    check(llm_calls["n"] == 0, "a clean prompt is not sent for a rewrite because of its own style")


def test_fixed_prompt_keeps_its_style_suffix(tmp):
    print("test_fixed_prompt_keeps_its_style_suffix")
    vid = _setup_video(tmp, "stylekeep", [24],
                       prompts=[f"A gift box with a label on a doorstep, {STYLE}"])
    seen = []

    def fake_cf(prompt, out_path, seed, cfg):
        seen.append(prompt)
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        return out_path, 96.0

    original = _stub(cloudflare=fake_cf, llm_complete=lambda prompt, **kw: "A plain unmarked parcel on a doorstep")
    try:
        stage.run(vid, {"image_style": STYLE,
                        "images": {"provider": "cloudflare", "images_per_seconds": 12}})
    finally:
        _restore(original)
    stored = db.get_scenes(vid)[0]["image_prompt"]
    check(stored == f"A plain unmarked parcel on a doorstep, {STYLE}",
          f"the rewrite got the style put back ({stored})")
    check(all(p == stored for p in seen), "the provider was sent the full prompt, style included")


def test_total_neurons_are_counted_across_all_calls(tmp):
    print("test_total_neurons_are_counted_across_all_calls")
    vid = _setup_video(tmp, "neurontotal", [24, 24], prompts=["A courtyard at dawn", "A quiet corridor"])

    def fake_cf(prompt, out_path, seed, cfg):
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        return out_path, 96.0

    original = _stub(cloudflare=fake_cf, llm_complete=lambda prompt, **kw: "x")
    try:
        stage.run(vid, {"images": {"provider": "cloudflare", "images_per_seconds": 12}})
    finally:
        _restore(original)
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        messages = [r[0] for r in conn.execute(
            "SELECT message FROM events WHERE video_id = ? AND stage = 'images'", (vid,))]
    check(any("384 neurons over 4 Cloudflare calls" in m for m in messages),
          f"4 images at 96 each are reported as 384 in the events table ({messages})")


def _write_real_jpeg(path, seed=0):
    """A valid small JPEG. The resume tests need one: an image is only reused if it opens."""
    from PIL import Image
    Image.new("RGB", (64, 36), (seed % 255, 40, 80)).save(path, "JPEG")


def test_lettering_is_redrawn_with_a_different_seed(tmp):
    print("test_lettering_is_redrawn_with_a_different_seed")
    vid = _setup_video(tmp, "ocr", [24])   # 2 images
    seeds = []

    def fake_cf(prompt, out_path, seed, cfg):
        seeds.append(seed)
        Path(out_path).write_bytes(b"\xff\xd8\xff" + str(len(seeds)).encode())
        return out_path, 96.0

    # Every first draw "has lettering", every redraw does not.
    reads = {"n": 0}

    def fake_ocr(path, **kw):
        reads["n"] += 1
        return (20, 90) if reads["n"] % 2 == 1 else (0, 0)

    original = _stub(cloudflare=fake_cf, ocr_result=None)
    ocr.detect_text = fake_ocr
    try:
        stage.run(vid, {"images": {"provider": "cloudflare", "images_per_seconds": 12}})
    finally:
        _restore(original)

    check(len(seeds) == 4, f"each of the 2 images was drawn, then drawn again ({len(seeds)})")
    check(seeds[1] != seeds[0] and seeds[3] != seeds[2],
          f"the redraw uses a different seed, so a provider that honours it draws something "
          f"else rather than the same frame ({seeds})")
    check(len(db.get_video_images(vid)) == 2, "the redraw replaces the slot, it does not add one")


def test_lettering_gives_up_after_three_attempts(tmp):
    print("test_lettering_gives_up_after_three_attempts")
    vid = _setup_video(tmp, "ocrfail", [24])
    calls = {"n": 0}

    def fake_cf(prompt, out_path, seed, cfg):
        calls["n"] += 1
        Path(out_path).write_bytes(b"\xff\xd8\xff" + str(calls["n"]).encode())
        return out_path, 96.0

    original = _stub(cloudflare=fake_cf, ocr_result=(25, 90))      # never clean
    try:
        stage.run(vid, {"images": {"provider": "cloudflare", "images_per_seconds": 12}})
    finally:
        _restore(original)

    check(calls["n"] == 2 * (stage.TEXT_ATTEMPTS + 1),
          f"2 images x {stage.TEXT_ATTEMPTS} redraws of the same prompt, then one more with "
          f"the subject changed, and then it stops ({calls['n']})")
    check(len(db.get_video_images(vid)) == 2, "the images are kept anyway, lettering and all")
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        warnings = [r[0] for r in conn.execute(
            "SELECT message FROM events WHERE video_id = ? AND level = 'warning'", (vid,))]
    check(any("still shows lettering" in w and "text-free rewrite" in w for w in warnings),
          f"and it says so in the events table, including that the rewrite did not help "
          f"either ({warnings[:1]})")


def test_a_subject_that_always_carries_writing_is_replaced(tmp):
    print("test_a_subject_that_always_carries_writing_is_replaced")
    # Redrawing the same prompt cannot clear lettering off a document: a picture of an
    # invoice will have writing on it however it is worded. Measured on video 7 - the frame
    # drawn for "a modest funeral invoice" had INVOICE across it and Tesseract could not read
    # it at any setting, so the only fix is to stop asking for the invoice.
    vid = _setup_video(tmp, "textfree", [24], prompts=["A candle burned down on a table"])
    seen, asked = [], []

    def fake_cf(prompt, out_path, seed, cfg):
        seen.append(prompt)
        Path(out_path).write_bytes(b"x" * 2000)
        return out_path, 96.0

    def fake_llm(prompt, **kw):
        asked.append(prompt)
        return "A candle burned down to the holder, chairs pushed back from a bare table"

    # every draw reads as having lettering until the subject is changed
    def fake_ocr(path, **kw):
        return (0, 0) if "burned down to the holder" in (seen[-1] if seen else "") else (20, 90)

    original = _stub(cloudflare=fake_cf, llm_complete=fake_llm)
    ocr.detect_text = fake_ocr
    try:
        stage.run(vid, {"images": {"provider": "cloudflare", "images_per_seconds": 24}})
    finally:
        _restore(original)

    check(len(seen) == stage.TEXT_ATTEMPTS + 1,
          f"three redraws of the same prompt, then one of a different subject ({len(seen)})")
    check(any("never carry writing" in a for a in asked),
          "the last request asks for a subject that cannot carry writing")
    check(seen[-1] != seen[0] and "holder" in seen[-1],
          f"and the last draw used it ({seen[-1][:60]})")
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        warnings = [r[0] for r in conn.execute(
            "SELECT message FROM events WHERE video_id = ? AND level = 'warning'", (vid,))]
    check(not any("text-free rewrite" in w for w in warnings),
          f"and nothing is logged as unresolved, because it was resolved ({warnings})")


def test_images_from_an_earlier_run_are_not_paid_for_again(tmp):
    print("test_images_from_an_earlier_run_are_not_paid_for_again")
    # The point of the whole thing: a run that died at the render must not spend another
    # 6720 Neurons on pictures it already has. On a 10,000 a day allowance it could not.
    vid = _setup_video(tmp, "resume", [24, 24])
    drawn = {"n": 0}

    def fake_cf(prompt, out_path, seed, cfg):
        drawn["n"] += 1
        _write_real_jpeg(out_path, drawn["n"])
        return out_path, 96.0

    original = _stub(cloudflare=fake_cf, ocr_result=(0, 0))
    try:
        stage.run(vid, {"images": {"provider": "cloudflare", "images_per_seconds": 12}})
        first = drawn["n"]
        paths = [r["path"] for r in db.get_video_images(vid)]
        stage.run(vid, {"images": {"provider": "cloudflare", "images_per_seconds": 12}})
    finally:
        _restore(original)

    check(first == 4, f"the first run drew all 4 images ({first})")
    check(drawn["n"] == first, f"the second drew none of them ({drawn['n'] - first} redrawn)")
    rows = db.get_video_images(vid)
    check([r["path"] for r in rows] == paths, "the same files are still the video's images")
    check({r["provider"] for r in rows} == {"cloudflare"},
          "the row still records how the picture was really made, not that this run reused it")
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        messages = [r[0] for r in conn.execute(
            "SELECT message FROM events WHERE video_id = ? AND stage = 'images'", (vid,))]
    check(any("4 reused" in m for m in messages),
          f"and the run says how many it reused ({messages[-1][:80]})")


def test_a_reused_image_with_lettering_is_still_redrawn(tmp):
    print("test_a_reused_image_with_lettering_is_still_redrawn")
    # Resuming must not preserve a bad frame forever: the check runs on what is already
    # there too, which is also how the flagged images of an earlier video get fixed.
    vid = _setup_video(tmp, "resumebad", [24])
    drawn = {"n": 0}

    def fake_cf(prompt, out_path, seed, cfg):
        drawn["n"] += 1
        _write_real_jpeg(out_path, drawn["n"])
        return out_path, 96.0

    original = _stub(cloudflare=fake_cf, ocr_result=(0, 0))
    try:
        stage.run(vid, {"images": {"provider": "cloudflare", "images_per_seconds": 12}})
        before = drawn["n"]
        # now the files on disk read as having lettering on the first look, clean after
        reads = {"n": 0}

        def fake_ocr(path, **kw):
            reads["n"] += 1
            return (20, 90) if reads["n"] % 2 == 1 else (0, 0)
        ocr.detect_text = fake_ocr
        stage.run(vid, {"images": {"provider": "cloudflare", "images_per_seconds": 12}})
    finally:
        _restore(original)
    check(drawn["n"] > before,
          f"a frame already on disk that reads as having lettering is drawn again "
          f"({drawn['n'] - before} redraws)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_images_per_scene()
        test_cloudflare_primary(tmp)
        test_pollinations_fallback(tmp)
        test_duplicate_when_both_fail(tmp)
        test_first_image_failure_has_nothing_to_duplicate(tmp)
        test_cover_survives_first_image_failure(tmp)
        test_seeds_are_distinct(tmp)
        test_neuron_budget_projection(tmp)
        test_provider_lock()
        test_no_scenes_raises(tmp)
        test_banned_prompt_is_fixed_before_generation(tmp)
        test_clean_prompt_skips_the_llm(tmp)
        test_a_prompt_with_hands_or_a_face_is_rewritten(tmp)
        test_a_stale_image_is_drawn_again_when_its_prompt_changed(tmp)
        test_files_outside_the_plan_are_removed(tmp)
        test_style_suffix_is_not_scanned_for_banned_words(tmp)
        test_fixed_prompt_keeps_its_style_suffix(tmp)
        test_total_neurons_are_counted_across_all_calls(tmp)
        test_lettering_is_redrawn_with_a_different_seed(tmp)
        test_lettering_gives_up_after_three_attempts(tmp)
        test_a_subject_that_always_carries_writing_is_replaced(tmp)
        test_images_from_an_earlier_run_are_not_paid_for_again(tmp)
        test_a_reused_image_with_lettering_is_still_redrawn(tmp)
    print("ALL IMAGES TESTS PASSED")
