"""Stage 3: voice the script, sentence by sentence.

Every sentence is synthesized on its own, so its real spoken duration is known and
sentences can carry silence between them: 350ms within a scene, 700ms between scenes,
1400ms wherever a new part of the script begins (hook -> lesson 1, lesson N -> lesson
N+1, lesson 10 -> outro). Sentences belonging to one scene are then concatenated into
that scene's own wav (scenes.audio_path / duration_s, unchanged in shape from before),
and all scenes plus their connecting silences are concatenated into one raw track.

The raw track is mastered once, as a whole: -14 LUFS (YouTube's loudness target),
an 80Hz high-pass to cut rumble, upsampled to 48kHz, exported as 192kbps mono
output/<video_id>/voice.mp3. Normalising per scene instead would flatten the natural
loud/quiet variation between scenes rather than the track as a whole, so it happens
exactly once, on the final concatenation.

output/<video_id>/timings.json holds every sentence's {scene_idx, sentence_idx, text,
start_s, end_s}, timed from the very start of the video, for the subtitle stage later.
output/<video_id>/sample_90s.mp3 is the first 90 seconds of the mastered track, cut
from voice.mp3, for a quick listen without sitting through 30 minutes.
"""
import json

from core import db, text as txt, tts
from core.config import output_dir
from core.logger import get_logger

log = get_logger("tts")

MIN_SCENE_S, MAX_SCENE_S = 15, 90
SAMPLE_SECONDS = 90
# Quieter than this and a file that rendered without error still reads as silence,
# which is what a provider returning an empty utterance looks like.
SILENCE_THRESHOLD_DBFS = -50
PACE_TOLERANCE = 0.15
PROGRESS_EVERY = 20


def _lesson_boundary_indices(video_id, total_sentences):
    """Global sentence indices (0-based, into every scene's sentences end to end) at
    which a new part of the script begins - the hook itself does not count, but going
    from the hook into lesson 1, lesson N into lesson N+1, and lesson 10 into the
    outro all do.

    Reconstructed from script.txt and outline.json rather than carried over from the
    script stage, since both files hold the exact final text the scenes were split
    from. Returns an empty set - falling back to ordinary scene pauses everywhere -
    if that reconstruction does not check out, rather than guessing.
    """
    out = output_dir(video_id)
    script_path, outline_path = out / "script.txt", out / "outline.json"
    if not script_path.exists() or not outline_path.exists():
        log.warning("script.txt or outline.json missing; lesson-boundary pauses "
                    "will fall back to ordinary scene pauses")
        return set()

    outline = json.loads(outline_path.read_text(encoding="utf-8"))
    lesson_count = len(outline.get("lessons", []))
    body = script_path.read_text(encoding="utf-8").split("\n", 1)[1].strip()
    blocks = [b.strip() for b in body.split("\n\n") if b.strip()]
    if len(blocks) != lesson_count + 2:
        log.warning("script.txt has %d parts, expected %d; lesson-boundary pauses "
                    "will fall back to ordinary scene pauses", len(blocks), lesson_count + 2)
        return set()

    boundaries, cursor = set(), 0
    for i, block in enumerate(blocks):
        if i > 0:                                  # the hook's own start is not a boundary
            boundaries.add(cursor)
        cursor += len(txt.split_sentences(block))

    if cursor != total_sentences:
        log.warning("parts account for %d sentences but scenes hold %d; lesson-boundary "
                    "pauses will fall back to ordinary scene pauses", cursor, total_sentences)
        return set()
    return boundaries


def _voice_sentences(video_id, scenes, cfg, pauses, boundaries, parts_dir):
    """Synthesize every sentence, inserting the right silence before each one.

    Returns (timings, scene_wav_plans, narration_segments, spoken_seconds):
      timings            - the full list to write as timings.json
      scene_wav_plans     - {scene_id: (duration_s, [wav paths making up that scene])}
      narration_segments - ordered list of paths/silence-durations for the full track
      spoken_seconds      - total time actually speaking, pauses excluded (for pace)
    """
    sentence_gap = pauses["sentence_ms"] / 1000
    scene_gap = pauses["scene_ms"] / 1000
    lesson_gap = pauses["lesson_ms"] / 1000

    timings = []
    scene_wav_plans = {}
    narration_segments = []          # list of ("file", path) or ("silence", seconds)
    global_index = 0
    global_time = 0.0
    spoken_seconds = 0.0
    total_sentences = sum(len(txt.split_sentences(s["text"])) or 1 for s in scenes)

    for scene in scenes:
        sentences = txt.split_sentences(scene["text"]) or [scene["text"]]
        scene_segments, scene_start = [], None

        for local_idx, sentence in enumerate(sentences):
            if global_index > 0:
                if global_index in boundaries:
                    gap = lesson_gap
                elif local_idx == 0:
                    gap = scene_gap
                else:
                    gap = sentence_gap
                if gap > 0:
                    if local_idx == 0:
                        # Between scenes: belongs to the full track, not this scene's own file.
                        narration_segments.append(("silence", gap))
                    else:
                        sil_path = parts_dir / f"sil_{scene['idx']:03d}_{local_idx:02d}.wav"
                        tts.silence(gap, sil_path)
                        scene_segments.append(sil_path)
                global_time += gap

            if local_idx == 0:
                scene_start = global_time

            sent_path = parts_dir / f"scene_{scene['idx']:03d}_sent_{local_idx:02d}.wav"
            duration = tts.synthesize(sentence, sent_path, cfg)
            start_s, global_time = global_time, global_time + duration
            timings.append({"scene_idx": scene["idx"], "sentence_idx": local_idx,
                            "text": sentence, "start_s": round(start_s, 3),
                            "end_s": round(global_time, 3)})
            scene_segments.append(sent_path)
            spoken_seconds += duration
            global_index += 1
            if global_index % PROGRESS_EVERY == 0 or global_index == total_sentences:
                log.info("voiced %d/%d sentences", global_index, total_sentences)

        scene_wav_plans[scene["id"]] = (round(global_time - scene_start, 2), scene_segments)
        narration_segments.append(("scene", scene["id"]))

    return timings, scene_wav_plans, narration_segments, spoken_seconds


def run(video_id, cfg):
    scenes = db.get_scenes(video_id)
    if not scenes:
        raise RuntimeError("no scenes to voice; did the script stage run?")

    provider, settings = tts.provider_settings(cfg)
    pauses = tts.pause_settings(cfg)
    log.info("voicing %d scenes with %s (%s); pauses %dms/%dms/%dms "
             "(sentence/scene/lesson)", len(scenes), provider, settings.get("voice"),
             pauses["sentence_ms"], pauses["scene_ms"], pauses["lesson_ms"])

    tts.require_ffmpeg()

    out_dir = output_dir(video_id)
    audio_dir = out_dir / "audio"
    parts_dir = audio_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    total_sentences = sum(len(txt.split_sentences(s["text"])) or 1 for s in scenes)
    boundaries = _lesson_boundary_indices(video_id, total_sentences)
    log.info("%d lesson-boundary pauses found in %d sentences", len(boundaries), total_sentences)

    timings, scene_plans, narration_segments, spoken_seconds = _voice_sentences(
        video_id, scenes, cfg, pauses, boundaries, parts_dir)

    # One wav per scene, built from its own sentence + internal-pause pieces.
    total_words = 0
    for scene in scenes:
        duration, segments = scene_plans[scene["id"]]
        scene_wav = audio_dir / f"scene_{scene['idx']:03d}.wav"
        tts.concatenate(segments, scene_wav)
        db.update_scene(scene["id"], audio_path=str(scene_wav), duration_s=duration)
        total_words += txt.word_count(scene["text"])
        if not MIN_SCENE_S <= duration <= MAX_SCENE_S:
            log.warning("scene %d is %.1fs, outside the %d-%ds range",
                        scene["idx"], duration, MIN_SCENE_S, MAX_SCENE_S)
            db.log_event(video_id, "tts", "warning",
                         f"scene {scene['idx']} duration {duration:.1f}s outside "
                         f"{MIN_SCENE_S}-{MAX_SCENE_S}s")

    # The full track: scene wavs and the silences between them, in order.
    full_segments = []
    for kind, value in narration_segments:
        if kind == "silence":
            sil_path = parts_dir / f"gap_{len(full_segments):04d}.wav"
            tts.silence(value, sil_path)
            full_segments.append(sil_path)
        else:
            full_segments.append(audio_dir / f"scene_{next(s for s in scenes if s['id'] == value)['idx']:03d}.wav")

    narration_raw = out_dir / "narration.wav"
    real_duration = tts.concatenate(full_segments, narration_raw)

    log.info("mastering: -14 LUFS, 80Hz high-pass, 48kHz, 192kbps mono")
    voice_mp3 = tts.master(narration_raw, out_dir / "voice.mp3", cfg)

    # Existence, non-empty, and not-actually-silence checks on the deliverable.
    if not voice_mp3.exists() or voice_mp3.stat().st_size == 0:
        raise RuntimeError(f"{voice_mp3} was not produced or is empty")
    mean_db, _ = tts.measure_volume(voice_mp3)
    if mean_db < SILENCE_THRESHOLD_DBFS:
        raise RuntimeError(f"{voice_mp3} measures {mean_db:.1f} dBFS mean volume, "
                           f"which is silence, not narration")
    log.info("voice.mp3: %.1f dBFS mean volume (not silence)", mean_db)

    sample_path = tts.extract_sample(voice_mp3, out_dir / "sample_90s.mp3", SAMPLE_SECONDS)

    (out_dir / "timings.json").write_text(
        json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8")

    actual_wpm = (total_words / spoken_seconds) * 60 if spoken_seconds else 0
    target_wpm = cfg["words_per_minute"]
    off_by = abs(actual_wpm - target_wpm) / target_wpm if target_wpm else 0
    log.info("narration: %d words, %.1f min spoken + pauses = %.1f min total, "
             "%.1f words/min speaking pace (target %d)", total_words, spoken_seconds / 60,
             real_duration / 60, actual_wpm, target_wpm)
    if off_by > PACE_TOLERANCE:
        tune = "length_scale" if provider == "piper" else "rate"
        log.warning("speaking pace %.1f wpm is %.0f%% off the %d wpm target; tune "
                    "tts.%s.%s in config.json", actual_wpm, off_by * 100, target_wpm,
                    provider, tune)
        db.log_event(video_id, "tts", "warning",
                     f"pace {actual_wpm:.1f} wpm vs target {target_wpm} wpm "
                     f"({off_by * 100:.0f}% off)")

    # Intermediate sentence and silence pieces have all been folded into scene wavs
    # and narration.wav; nothing downstream needs them kept around.
    for f in parts_dir.glob("*.wav"):
        f.unlink(missing_ok=True)
    parts_dir.rmdir()

    db.update_video(video_id, duration_s=round(real_duration, 1))
    db.log_event(video_id, "tts", "info",
                 f"{len(scenes)} scenes / {total_sentences} sentences voiced with "
                 f"{provider}, {real_duration / 60:.1f} min total, {actual_wpm:.1f} wpm, "
                 f"sample at {sample_path.name}")
    return True
