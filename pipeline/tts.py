"""Stage 3: voice the script, scene by scene.

Each scene is synthesized to its own wav file under output/<video_id>/audio/, so a
later render stage can pace images to the real spoken duration rather than an
estimate. The scenes are then joined into one narration.wav for a full listen.

After synthesis, the actual words-per-minute is measured from real audio and checked
against config.words_per_minute; more than 15% off is logged as a warning, since it is
a signal the config's length_scale or rate needs tuning, not a reason to fail the video.
"""
from core import db, text as txt, tts
from core.config import output_dir
from core.logger import get_logger

log = get_logger("tts")

PACE_TOLERANCE = 0.15


def run(video_id, cfg):
    scenes = db.get_scenes(video_id)
    if not scenes:
        raise RuntimeError("no scenes to voice; did the script stage run?")

    provider, settings = tts.provider_settings(cfg)
    log.info("voicing %d scenes with %s (%s)", len(scenes), provider,
             settings.get("voice"))

    audio_dir = output_dir(video_id) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    wav_paths, total_words, total_seconds = [], 0, 0.0
    for i, scene in enumerate(scenes, 1):
        out_path = audio_dir / f"scene_{scene['idx']:03d}.wav"
        duration = tts.synthesize(scene["text"], out_path, cfg)
        words = txt.word_count(scene["text"])
        db.update_scene(scene["id"], audio_path=str(out_path), duration_s=round(duration, 2))
        wav_paths.append(out_path)
        total_words += words
        total_seconds += duration
        if i % 10 == 0 or i == len(scenes):
            log.info("voiced %d/%d scenes", i, len(scenes))

    narration_path = output_dir(video_id) / "narration.wav"
    full_duration = tts.concatenate(wav_paths, narration_path)

    actual_wpm = (total_words / total_seconds) * 60 if total_seconds else 0
    target_wpm = cfg["words_per_minute"]
    off_by = abs(actual_wpm - target_wpm) / target_wpm if target_wpm else 0
    log.info("narration: %d words in %.1fs (%.1f min), %.1f words/min (target %d)",
             total_words, full_duration, full_duration / 60, actual_wpm, target_wpm)
    if off_by > PACE_TOLERANCE:
        log.warning("actual pace %.1f wpm is %.0f%% off the %d wpm target; tune "
                    "tts.%s.%s in config.json", actual_wpm, off_by * 100, target_wpm,
                    provider, "length_scale" if provider == "piper" else "rate")
        db.log_event(video_id, "tts", "warning",
                     f"pace {actual_wpm:.1f} wpm vs target {target_wpm} wpm "
                     f"({off_by * 100:.0f}% off)")

    db.update_video(video_id, duration_s=round(full_duration, 1))
    db.log_event(video_id, "tts", "info",
                 f"{len(scenes)} scenes voiced with {provider}, "
                 f"{full_duration / 60:.1f} min, {actual_wpm:.1f} wpm")
    return True
