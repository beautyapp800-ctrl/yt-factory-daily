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
    ocr.detect_text = lambda path, min_confidence=60: ocr_result
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
                       prompts=["A delivery courier hands over a gift box with a label"])
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
                       prompts=[f"Hands holding a plain unmarked parcel, {STYLE}"])
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
                       prompts=[f"A courier hands over a gift box with a label, {STYLE}"])
    seen = []

    def fake_cf(prompt, out_path, seed, cfg):
        seen.append(prompt)
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        return out_path, 96.0

    original = _stub(cloudflare=fake_cf, llm_complete=lambda prompt, **kw: "Hands holding a plain unmarked parcel")
    try:
        stage.run(vid, {"image_style": STYLE,
                        "images": {"provider": "cloudflare", "images_per_seconds": 12}})
    finally:
        _restore(original)
    stored = db.get_scenes(vid)[0]["image_prompt"]
    check(stored == f"Hands holding a plain unmarked parcel, {STYLE}",
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


def test_ocr_triggers_one_regeneration(tmp):
    print("test_ocr_triggers_one_regeneration")
    vid = _setup_video(tmp, "ocr", [24])   # 2 images
    calls = {"n": 0}

    def fake_cf(prompt, out_path, seed, cfg):
        calls["n"] += 1
        Path(out_path).write_bytes(b"\xff\xd8\xff" + str(calls["n"]).encode())
        return out_path, 96.0

    # First call for each image "has text" (simulated), the regeneration "doesn't".
    ocr_calls = {"n": 0}

    def fake_ocr(path, min_confidence=60):
        ocr_calls["n"] += 1
        return (20, 90) if ocr_calls["n"] % 2 == 1 else (0, 0)

    original = _stub(cloudflare=fake_cf, ocr_result=None)
    ocr.detect_text = fake_ocr
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 12}}
        stage.run(vid, cfg)
    finally:
        _restore(original)

    check(calls["n"] == 4, f"each of 2 images was generated, then regenerated once ({calls['n']})")
    rows = db.get_video_images(vid)
    check(len(rows) == 2, "regeneration replaces the same slot, not an extra row")


def test_ocr_gives_up_after_one_retry(tmp):
    print("test_ocr_gives_up_after_one_retry")
    vid = _setup_video(tmp, "ocrstuck", [24])
    vid_scenes = db.get_scenes(vid)[:1]   # only exercise scene 1's first image really
    calls = {"n": 0}

    def fake_cf(prompt, out_path, seed, cfg):
        calls["n"] += 1
        Path(out_path).write_bytes(b"\xff\xd8\xff")
        return out_path, 96.0

    original = _stub(cloudflare=fake_cf, ocr_result=(20, 95))   # always "finds text"
    try:
        cfg = {"images": {"provider": "cloudflare", "images_per_seconds": 24}}  # 1 image
        check(stage.run(vid, cfg) is True,
              "a persistently flagged image is kept after one retry, not looped forever")
    finally:
        _restore(original)
    check(calls["n"] == 2, f"generated once, regenerated once, then stopped ({calls['n']})")


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
        test_style_suffix_is_not_scanned_for_banned_words(tmp)
        test_fixed_prompt_keeps_its_style_suffix(tmp)
        test_total_neurons_are_counted_across_all_calls(tmp)
        test_ocr_triggers_one_regeneration(tmp)
        test_ocr_gives_up_after_one_retry(tmp)
    print("ALL IMAGES TESTS PASSED")
