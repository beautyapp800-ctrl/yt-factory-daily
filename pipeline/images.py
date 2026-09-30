"""Stage 4: illustrate the script.

One still image per scene is too long on screen: video 7 averages 33s a scene, and
channels in this niche change the image every 8-15 seconds. So each scene gets
max(1, round(duration_s / images_per_seconds)) images instead of one, all sharing
the scene's own image_prompt but rendered with a different seed each, so the same
description still produces visibly different frames.

Fallback order per image: Cloudflare Workers AI, then pollinations.ai, then - only
if both fail - duplicate whichever image most recently succeeded anywhere in this
video, rather than leave that slot with nothing. The very first image of the video
has nothing to duplicate yet, so a failure there is a genuine gap, logged as such.

The first NEURON_SAMPLE_SIZE Cloudflare images are used to measure real neuron
cost per image (not assumed from blog posts, which is what burned the pollinations
"nologo=true" assumption earlier) and project whether the free 10000/day plan
actually covers the whole video; the projection is logged plainly either way.
"""
from core import db, images
from core.config import output_dir
from core.logger import get_logger

log = get_logger("images")

DEFAULT_IMAGES_PER_SECONDS = 12
NEURON_SAMPLE_SIZE = 5
DAILY_FREE_NEURONS = 10000


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


def run(video_id, cfg):
    scenes = db.get_scenes(video_id)
    if not scenes:
        raise RuntimeError("no scenes to illustrate; did the script stage run?")

    images_per_seconds = (cfg.get("images") or {}).get(
        "images_per_seconds", DEFAULT_IMAGES_PER_SECONDS)
    cf_note = "available" if images.cloudflare_available() else \
        "unavailable (CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN not set)"
    log.info("cloudflare %s, %.0fs per image", cf_note, images_per_seconds)

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
    made, duplicated, failed = 0, 0, 0
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
            # rather than drawing a new one every time.
            seed = scene["idx"] * 1000 + i
            out_path = image_dir / f"scene_{scene['idx']:03d}_{i:02d}.jpg"

            try:
                used, neurons = images.synthesize(prompt, out_path, seed, cfg)
                if used == "cloudflare" and neurons is not None and \
                        len(neuron_samples) < NEURON_SAMPLE_SIZE:
                    neuron_samples.append(neurons)
                    log.info("neuron sample %d/%d: %.1f", len(neuron_samples),
                             NEURON_SAMPLE_SIZE, neurons)
                db.add_scene_image(video_id, scene["id"], i, prompt=prompt, seed=seed,
                                   provider=used, path=str(out_path))
                by_provider[used] = by_provider.get(used, 0) + 1
                made += 1
                last_success_path = out_path
                cover_path = out_path
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

            if not cover_set:
                # The scene's own single-image field becomes its cover/first frame,
                # for anything downstream that only wants one representative image.
                # Whichever image succeeds first (real or duplicated), not
                # necessarily index 0.
                db.update_scene(scene["id"], image_path=str(cover_path))
                cover_set = True

    _project_neuron_budget(neuron_samples, total_planned)

    log.info("done: %d made, %d duplicated, %d failed of %d planned (%s)",
             made, duplicated, failed, total_planned,
             ", ".join(f"{k}={v}" for k, v in by_provider.items()) or "none")
    db.log_event(video_id, "images", "info",
                 f"{made} made, {duplicated} duplicated, {failed} failed of "
                 f"{total_planned} planned ({by_provider})")

    if made == 0 and duplicated == 0:
        raise RuntimeError(f"no images were produced for any of {len(scenes)} scenes")
    return True
