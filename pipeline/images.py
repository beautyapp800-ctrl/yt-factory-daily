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
from pathlib import Path

from core import db, images, ocr
from core.config import output_dir
from core.llm import complete
from core.logger import get_logger
from core.prompts import (IMAGE_SYSTEM, anatomy_fix_request, find_banned_anatomy_words,
                          find_banned_image_words, image_prompt_fix_request)
from core.text import clean as clean_text

log = get_logger("images")

DEFAULT_IMAGES_PER_SECONDS = 12
NEURON_SAMPLE_SIZE = 5
DAILY_FREE_NEURONS = 10000
# Leave this much of the daily allowance unspent. Retries are what would eat it: a run that
# spends its last Neurons regenerating a picture that is merely imperfect has nothing left
# for the pictures that do not exist yet.
NEURON_RESERVE = 300

# Checking a finished image for rendered lettering. Strict on purpose - see core.ocr - and
# a flagged image is drawn again with a different seed, so a false alarm costs 96 Neurons
# while a miss puts garbled words in the video. Measured on video 7: 15 of 72 flagged.
TEXT_OCR = {"min_confidence": 50, "min_word_len": 3, "upscale": 2, "contrast": 1.6}
TEXT_MIN_CHARS = 3
TEXT_ATTEMPTS = 3          # first draw plus two redraws
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


def _prompt_problems(prompt):
    """(kind, hits) for the first rule this prompt breaks, or (None, []).

    Two rules, both about things the generator renders badly enough that no later stage can
    repair them: writing, which comes back as garbled pseudo-text, and hands and faces, which
    come back with six fingers or an expression that is almost right. Anatomy is checked
    first because it is the one that spoils a frame outright.
    """
    hits = find_banned_anatomy_words(prompt)
    if hits:
        return "anatomy", hits
    hits = find_banned_image_words(prompt)
    if hits:
        return "lettering", hits
    return None, []


def _fix_prompt_if_needed(scene, cfg):
    """Rewrite scene['image_prompt'] in place (db + the in-memory dict) if its scene part
    breaks a rule. The style suffix is set aside while that happens and put back after.
    Returns the (possibly unchanged) full prompt text."""
    full = scene["image_prompt"] or ""
    prompt, suffix = _split_style(full, cfg)
    kind, hits = _prompt_problems(prompt)
    if not kind:
        return full

    for attempt in range(1, PROMPT_FIX_ATTEMPTS + 1):
        log.warning("scene %d image_prompt has %s (%s), rewriting (attempt %d/%d)",
                    scene["idx"], kind, hits, attempt, PROMPT_FIX_ATTEMPTS)
        request = (anatomy_fix_request(prompt, hits) if kind == "anatomy"
                   else image_prompt_fix_request(prompt, hits))
        try:
            fixed = complete(request, system=IMAGE_SYSTEM, max_tokens=150, temperature=0.8)
        except Exception as e:
            log.warning("scene %d prompt fix request failed: %s", scene["idx"], e)
            break
        fixed = clean_text(_unwrap_json_prompt(fixed)).strip(" \"'")
        kind, hits = _prompt_problems(fixed) if fixed else (kind, hits)
        if fixed and not kind:
            log.info("scene %d image_prompt fixed: %s", scene["idx"], fixed[:80])
            db.update_scene(scene["id"], image_prompt=fixed + suffix)
            scene["image_prompt"] = fixed + suffix
            return fixed + suffix
        prompt = fixed or prompt

    log.warning("scene %d image_prompt still has %s (%s) after %d attempts, using as-is",
               scene["idx"], kind, hits, PROMPT_FIX_ATTEMPTS)
    if prompt + suffix != scene["image_prompt"]:
        db.update_scene(scene["id"], image_prompt=prompt + suffix)
        scene["image_prompt"] = prompt + suffix
    return prompt + suffix


def _usable_image(path):
    """Whether a file left over from an earlier run is a real image worth keeping."""
    from PIL import Image
    try:
        with Image.open(path) as img:
            width, height = img.size
            img.verify()            # catches a file truncated by a run that was killed
        return width >= 32 and height >= 32
    except Exception:
        return False


def _reads_as_text(path):
    """(flagged, characters) for rendered lettering in a finished image."""
    if not ocr.available():
        return False, 0
    chars, _ = ocr.detect_text(path, **TEXT_OCR)
    return chars >= TEXT_MIN_CHARS, chars


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

    log.info("checking %d image_prompts for hands, faces and lettering before generation",
             len(scenes))
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

    # What an earlier run of this video already produced. The files are what cost Neurons,
    # so they are kept and the rows rebuilt around them: a run that died at the render must
    # not pay for 70 pictures again, and on a 10,000 Neuron allowance it could not.
    previous = {(row["scene_id"], row["idx"]): row for row in db.get_video_images(video_id)}
    db.clear_scene_images(video_id)

    made, reused, redrawn, duplicated, failed = 0, 0, 0, 0, 0
    by_provider = {}
    neuron_samples = []
    neurons_total, cf_calls = 0.0, 0     # every Cloudflare call, redraws included
    last_success_path = None
    done = 0                             # image slots settled, for the budget guard

    for scene, count in plan:
        prompt = scene["image_prompt"] or ""
        if not prompt.strip():
            log.warning("scene %d has no image_prompt, skipping its %d image(s)",
                        scene["idx"], count)
            failed += count
            done += count
            continue

        cover_set = False
        for i in range(count):
            out_path = image_dir / f"scene_{scene['idx']:03d}_{i:02d}.jpg"
            base_seed = scene["idx"] * 1000 + i
            old = previous.get((scene["id"], i))
            # An image already on disk starts as attempt 0: it is still checked for
            # lettering, so the flagged ones from an earlier run are redrawn now, but a
            # clean one costs nothing.
            #
            # It is only the same image if it was drawn from the same prompt. scene_images
            # stores the prompt each file was made with, so a prompt the rules have since
            # rewritten - hands taken out of it, say - makes the file on disk stale, and it
            # is drawn again. Without this the rules could change and nothing would follow.
            from_disk = (old is not None and old["path"] == str(out_path)
                         and (old["prompt"] or "") == prompt
                         and out_path.exists() and _usable_image(out_path))

            provider = old["provider"] if from_disk else None
            neurons, error, generated = None, None, False
            flagged, chars = False, 0
            for attempt in range(TEXT_ATTEMPTS):
                if attempt == 0 and from_disk:
                    pass                                     # nothing to generate yet
                else:
                    if attempt > 0:
                        remaining = total_planned - done - 1
                        per_image = (sum(neuron_samples) / len(neuron_samples)
                                     if neuron_samples else 96)
                        if neurons_total + per_image * (remaining + 1) > \
                                DAILY_FREE_NEURONS - NEURON_RESERVE:
                            log.warning("scene %d image %d: keeping a frame with lettering "
                                        "(%d characters); redrawing it would leave too "
                                        "little of the daily allowance for the %d images "
                                        "still to make", scene["idx"], i + 1, chars, remaining)
                            break
                    try:
                        # A different seed each attempt, so a provider that honours seed
                        # draws something else rather than the same frame again.
                        used, neurons = images.synthesize(
                            prompt, out_path, base_seed + 500 * attempt, cfg)
                    except images.ImageError as e:
                        error = e
                        break
                    provider, error, generated = used, None, True
                    if used == "cloudflare":
                        cf_calls += 1
                        neurons_total += neurons or 0
                        if neurons is not None and len(neuron_samples) < NEURON_SAMPLE_SIZE:
                            neuron_samples.append(neurons)
                            log.info("neuron sample %d/%d: %.1f", len(neuron_samples),
                                     NEURON_SAMPLE_SIZE, neurons)
                    if attempt > 0:
                        redrawn += 1

                flagged, chars = _reads_as_text(out_path)
                if not flagged:
                    break
                log.warning("scene %d image %d: OCR reads %d characters of lettering on it "
                            "(attempt %d/%d)", scene["idx"], i + 1, chars, attempt + 1,
                            TEXT_ATTEMPTS)
            else:
                db.log_event(video_id, "images", "warning",
                             f"scene {scene['idx']} image {i + 1} still shows lettering "
                             f"after {TEXT_ATTEMPTS} attempts")

            done += 1
            if error is not None and not from_disk:
                if last_success_path is not None:
                    db.add_scene_image(video_id, scene["id"], i, prompt=prompt, seed=base_seed,
                                       provider="duplicate", path=str(last_success_path))
                    by_provider["duplicate"] = by_provider.get("duplicate", 0) + 1
                    duplicated += 1
                    cover_path = last_success_path
                    log.info("scene %d image %d/%d: both providers failed (%s), duplicating %s",
                             scene["idx"], i + 1, count, error, last_success_path.name)
                else:
                    log.warning("scene %d image %d/%d failed with nothing yet to duplicate: %s",
                                scene["idx"], i + 1, count, error)
                    db.log_event(video_id, "images", "warning",
                                 f"scene {scene['idx']} image {i + 1}/{count} failed: {error}")
                    failed += 1
                    continue
            else:
                db.add_scene_image(video_id, scene["id"], i, prompt=prompt, seed=base_seed,
                                   provider=provider or "reused", path=str(out_path))
                by_provider[provider or "reused"] = by_provider.get(provider or "reused", 0) + 1
                if generated:
                    made += 1
                else:
                    reused += 1
                last_success_path = out_path
                cover_path = out_path

            if not cover_set:
                # The scene's own single-image field becomes its cover/first frame, for
                # anything downstream that only wants one representative image.
                db.update_scene(scene["id"], image_path=str(cover_path))
                cover_set = True

    # Files the plan no longer has a slot for. They are dead weight that the Actions cache
    # would otherwise carry between runs for ever: lowering images_per_seconds from 22 to 26
    # left 23 of video 7's 72 files orphaned in one go.
    kept = {Path(r["path"]).resolve() for r in db.get_video_images(video_id)}
    orphans = [f for f in image_dir.glob("*.jpg") if f.resolve() not in kept]
    for f in orphans:
        f.unlink(missing_ok=True)
    if orphans:
        log.info("removed %d image file(s) the current plan has no slot for", len(orphans))

    _project_neuron_budget(neuron_samples, total_planned)

    log.info("done: %d generated, %d reused from an earlier run, %d redrawn for lettering, "
             "%d duplicated, %d failed of %d planned (%s)", made, reused, redrawn, duplicated,
             failed, total_planned,
             ", ".join(f"{k}={v}" for k, v in by_provider.items()) or "none")
    if cf_calls:
        log.info("neurons actually used: %.0f over %d Cloudflare calls (%.1f each), %.0f%% of "
                 "the %d/day free allowance", neurons_total, cf_calls,
                 neurons_total / cf_calls, 100 * neurons_total / DAILY_FREE_NEURONS,
                 DAILY_FREE_NEURONS)
    db.log_event(video_id, "images", "info",
                 f"{made} made, {reused} reused, {redrawn} redrawn for lettering, "
                 f"{duplicated} duplicated, {failed} failed of {total_planned} planned "
                 f"({by_provider}), {fixed_count} prompts rewritten, "
                 f"{neurons_total:.0f} neurons over {cf_calls} Cloudflare calls")

    if made == 0 and reused == 0 and duplicated == 0:
        raise RuntimeError(f"no images were produced for any of {len(scenes)} scenes")
    return True
