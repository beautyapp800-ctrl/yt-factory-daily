"""Stage 4: illustrate the script.

A scene can be long - video 7 averages 33s - so each scene gets
max(1, round(duration_s / images_per_seconds)) images rather than one, all sharing
the scene's own image_prompt but rendered with a different seed each, so the same
description still produces visibly different frames.

images_per_seconds is the only thing that sets the Cloudflare bill, and since the
render stage shows each image once, it is also the length of a shot: at 22 it means
63 images and a new picture every 22 seconds for a 23 minute video. Shortening it
buys a faster cut at a proportionally higher Neuron cost - 96 Neurons per image
against 10,000 a day.

Fallback order per image: Cloudflare Workers AI, then pollinations.ai, then - only
if both fail - duplicate whichever image most recently succeeded anywhere in this
video, rather than leave that slot with nothing. The very first image of the video
has nothing to duplicate yet, so a failure there is a genuine gap, logged as such.

Before any generation, every scene's image_prompt is checked against
core.prompts.IMAGE_BANNED_WORDS (phone, box, label, logo...) and rewritten by the
model if it names one, since a flux-schnell-class model barely follows a negative
instruction like "no text" once the prompt itself describes a text-bearing object.
The fix is persisted back to scenes.image_prompt, not just used for this run.

Optionally, after generation, Tesseract OCR gets one look at the image; real,
legible text at high confidence triggers one regeneration attempt. This is a
last-resort net, not the primary defense - see core/ocr.py for why it demonstrably
misses an image model's fake lettering most of the time - and is skipped outright,
not an error, when Tesseract is not installed.

The first NEURON_SAMPLE_SIZE Cloudflare images are used to measure real neuron
cost per image (not assumed from blog posts, which is what burned the pollinations
"nologo=true" assumption earlier) and project whether the free 10000/day plan
actually covers the whole video; the projection is logged plainly either way.
"""
import json
import re

from core import db, images, ocr
from core.config import output_dir
from core.llm import complete
from core.logger import get_logger
from core.prompts import IMAGE_SYSTEM, find_banned_image_words, image_prompt_fix_request
from core.text import clean as clean_text

log = get_logger("images")

DEFAULT_IMAGES_PER_SECONDS = 12
NEURON_SAMPLE_SIZE = 5
DAILY_FREE_NEURONS = 10000
PROMPT_FIX_ATTEMPTS = 2
OCR_MIN_CHARS = 8
OCR_REGENERATE_ATTEMPTS = 1


def _project_neuron_budget(samples, total_planned):
    """What the measured per-image neuron cost implies for the whole video."""
    if not samples:
        log.warning("no neuron figures were readable from any Cloudflare response; "
                    "Cloudflare's own docs do not commit to a fixed number for this "
                    "model, so the daily budget cannot be checked in advance")
        return
    avg = sum(samples) / len(samples)
    projected = avg * total_planned
    log.info("neuron measurement: %s (n=%d) -> avg %.1f/image, projected %.0f for "
             "%d images (daily free budget %d)", samples, len(samples), avg,
             projected, total_planned, DAILY_FREE_NEURONS)
    if projected > DAILY_FREE_NEURONS:
        log.warning("projected %.0f neurons EXCEEDS the %d/day free budget by %.0f; "
                    "cut images_per_seconds, or split across two days, or lean on "
                    "the pollinations fallback more", projected, DAILY_FREE_NEURONS,
                    projected - DAILY_FREE_NEURONS)
    else:
        headroom = DAILY_FREE_NEURONS - projected
        log.info("projected %.0f neurons fits the %d/day free budget, %.0f to spare",
                 projected, DAILY_FREE_NEURONS, headroom)


def _unwrap_json_prompt(text):
    """The fix-request explicitly asks for plain text, no JSON - but this model
    follows negative instructions about as reliably for text as it does for images,
    and occasionally answers {"prompt": "..."} anyway. Caught live: two of video 7's
    first 43 rewrites came back wrapped like this. Unwrap it rather than storing the
    braces as if they were part of the scene description.
    """
    text = (text or "").strip()
    if not text.startswith("{"):
        return text
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("prompt"), str):
            return data["prompt"]
    except json.JSONDecodeError:
        pass
    match = re.search(r'"prompt"\s*:\s*"([^"]*)', text)
    return match.group(1) if match else text


def _split_style(prompt, cfg):
    """(scene part, style suffix) of a stored image prompt. The script stage stores
    '<what the scene shows>, <config image_style>'; only the first part is the LLM's to
    answer for. The style carries its own 'no text, no lettering, no watermark', which
    would trip the banned-word check on every single prompt if it were scanned too -
    caught live on video 7: all 43 prompts flagged, every one queued for an LLM rewrite
    that would then have dropped the style off the end."""
    suffix = f", {cfg.get('image_style') or ''}"
    if len(suffix) > 2 and prompt.endswith(suffix):
        return prompt[:-len(suffix)], suffix
    return prompt, ""


def _fix_prompt_if_needed(scene, cfg):
    """Rewrite scene['image_prompt'] in place (db + the in-memory dict) if its scene part
    names a banned word. The style suffix is set aside while that happens and put back
    after. Returns the (possibly unchanged) full prompt text."""
    full = scene["image_prompt"] or ""
    prompt, suffix = _split_style(full, cfg)
    hits = find_banned_image_words(prompt)
    if not hits:
        return full

    for attempt in range(1, PROMPT_FIX_ATTEMPTS + 1):
        log.warning("scene %d image_prompt names %s, rewriting (attempt %d/%d)",
                    scene["idx"], hits, attempt, PROMPT_FIX_ATTEMPTS)
        try:
            fixed = complete(image_prompt_fix_request(prompt, hits), system=IMAGE_SYSTEM,
                             max_tokens=150, temperature=0.8)
        except Exception as e:
            log.warning("scene %d prompt fix request failed: %s", scene["idx"], e)
            break
        fixed = clean_text(_unwrap_json_prompt(fixed)).strip(" \"'")
        hits = find_banned_image_words(fixed)
        if fixed and not hits:
            log.info("scene %d image_prompt fixed: %s", scene["idx"], fixed[:80])
            db.update_scene(scene["id"], image_prompt=fixed + suffix)
            scene["image_prompt"] = fixed + suffix
            return fixed + suffix
        prompt = fixed or prompt

    log.warning("scene %d image_prompt still names %s after %d attempts, using as-is",
               scene["idx"], hits, PROMPT_FIX_ATTEMPTS)
    if prompt + suffix != scene["image_prompt"]:
        db.update_scene(scene["id"], image_prompt=prompt + suffix)
        scene["image_prompt"] = prompt + suffix
    return prompt + suffix


def run(video_id, cfg):
    scenes = db.get_scenes(video_id)
    if not scenes:
        raise RuntimeError("no scenes to illustrate; did the script stage run?")

    images_per_seconds = (cfg.get("images") or {}).get(
        "images_per_seconds", DEFAULT_IMAGES_PER_SECONDS)
    cf_note = "available" if images.cloudflare_available() else \
        "unavailable (CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN not set)"
    ocr_note = "available" if ocr.available() else "unavailable (Tesseract not installed)"
    log.info("cloudflare %s, ocr check %s, %.0fs per image", cf_note, ocr_note,
             images_per_seconds)

    log.info("checking %d image_prompts for banned nouns before generation", len(scenes))
    fixed_count = 0
    for scene in scenes:
        before = scene["image_prompt"]
        after = _fix_prompt_if_needed(scene, cfg)
        if after != before:
            fixed_count += 1
    if fixed_count:
        log.info("%d/%d prompts rewritten", fixed_count, len(scenes))

    plan = []
    for scene in scenes:
        duration = scene["duration_s"] or 0
        count = images.images_per_scene(duration, images_per_seconds)
        plan.append((scene, count))
    total_planned = sum(c for _, c in plan)
    log.info("%d scenes -> %d images planned", len(scenes), total_planned)

    image_dir = output_dir(video_id) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    db.clear_scene_images(video_id)
    made, duplicated, failed, ocr_regenerated = 0, 0, 0, 0
    by_provider = {}
    neuron_samples = []
    last_success_path = None

    for scene, count in plan:
        prompt = scene["image_prompt"] or ""
        if not prompt.strip():
            log.warning("scene %d has no image_prompt, skipping its %d image(s)",
                        scene["idx"], count)
            failed += count
            continue

        cover_set = False
        for i in range(count):
            # A fixed seed per (scene, image) so a re-run reproduces the same frame
            # rather than drawing a new one every time. +500 on an OCR retry so a
            # provider that does honour seed (pollinations) draws something different.
            seed = scene["idx"] * 1000 + i
            out_path = image_dir / f"scene_{scene['idx']:03d}_{i:02d}.jpg"

            try:
                used, neurons = images.synthesize(prompt, out_path, seed, cfg)
            except images.ImageError as e:
                if last_success_path is not None:
                    # Cloudflare and pollinations both failed: show the most recent
                    # successful frame again rather than leave a gap.
                    db.add_scene_image(video_id, scene["id"], i, prompt=prompt, seed=seed,
                                       provider="duplicate", path=str(last_success_path))
                    by_provider["duplicate"] = by_provider.get("duplicate", 0) + 1
                    duplicated += 1
                    cover_path = last_success_path
                    log.info("scene %d image %d/%d: both providers failed (%s), "
                            "duplicating %s", scene["idx"], i + 1, count, e,
                            last_success_path.name)
                else:
                    log.warning("scene %d image %d/%d failed with nothing yet to "
                                "duplicate: %s", scene["idx"], i + 1, count, e)
                    db.log_event(video_id, "images", "warning",
                                f"scene {scene['idx']} image {i + 1}/{count} failed: {e}")
                    failed += 1
                    continue
            else:
                if used == "cloudflare" and neurons is not None and \
                        len(neuron_samples) < NEURON_SAMPLE_SIZE:
                    neuron_samples.append(neurons)
                    log.info("neuron sample %d/%d: %.1f", len(neuron_samples),
                             NEURON_SAMPLE_SIZE, neurons)

                chars, mean_conf = ocr.detect_text(out_path)
                if chars > OCR_MIN_CHARS:
                    log.warning("scene %d image %d/%d: OCR read %d confident "
                                "characters (%.0f%% conf), regenerating once",
                                scene["idx"], i + 1, count, chars, mean_conf)
                    try:
                        used2, neurons2 = images.synthesize(prompt, out_path, seed + 500, cfg)
                        used, neurons = used2, neurons2
                        ocr_regenerated += 1
                        chars2, _ = ocr.detect_text(out_path)
                        if chars2 > OCR_MIN_CHARS:
                            log.warning("scene %d image %d/%d: still %d characters "
                                        "after regenerating, keeping it anyway",
                                        scene["idx"], i + 1, count, chars2)
                    except images.ImageError as e:
                        log.warning("scene %d image %d/%d: OCR regeneration failed "
                                    "(%s), keeping the original", scene["idx"], i + 1,
                                    count, e)

                db.add_scene_image(video_id, scene["id"], i, prompt=prompt, seed=seed,
                                   provider=used, path=str(out_path))
                by_provider[used] = by_provider.get(used, 0) + 1
                made += 1
                last_success_path = out_path
                cover_path = out_path

            if not cover_set:
                # The scene's own single-image field becomes its cover/first frame,
                # for anything downstream that only wants one representative image.
                # Whichever image succeeds first (real or duplicated), not
                # necessarily index 0.
                db.update_scene(scene["id"], image_path=str(cover_path))
                cover_set = True

    _project_neuron_budget(neuron_samples, total_planned)

    log.info("done: %d made, %d duplicated, %d failed of %d planned (%s), "
             "%d OCR regenerations", made, duplicated, failed, total_planned,
             ", ".join(f"{k}={v}" for k, v in by_provider.items()) or "none",
             ocr_regenerated)
    db.log_event(video_id, "images", "info",
                 f"{made} made, {duplicated} duplicated, {failed} failed of "
                 f"{total_planned} planned ({by_provider}), {ocr_regenerated} "
                 f"OCR regenerations, {fixed_count} prompts rewritten")

    if made == 0 and duplicated == 0:
        raise RuntimeError(f"no images were produced for any of {len(scenes)} scenes")
    return True
