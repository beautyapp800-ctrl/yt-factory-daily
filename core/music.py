"""The background music bed: pick a track, loop it to length, set its level under the voice.

Tracks live in assets/music/ and are all CC0 (assets/music/CREDITS.json records where each
came from and under what licence). The repository is public, so CC0 is not a preference but
a requirement: a licence that forbids redistributing the file on its own - most "free, no
attribution" stock licences do - would be broken by committing the file here at all.

Three things this has to get right, and all three are measured rather than assumed:

Level. "26 dB below the voice" is a statement about loudness, so both are measured with
ffmpeg's EBU R128 meter and the music is given exactly the gain that lands it that far under.
Picking a fixed -26 dB gain instead would leave a quiet track inaudible and a loud one
competing with the narration.

Seams. A three minute track under a thirty minute video is looped ten times, and a plain cut
back to the start clicks, because the waveform jumps. So the track is first turned into a
block that loops perfectly: its last few seconds are crossfaded onto its first few, which
makes the block's end and its start the same material, already faded into each other. That
block is then simply repeated. The join is continuous by construction - there is no
sample-level discontinuity to click - and it is built once rather than once per repeat.

The first version crossfaded N copies of the track in a single filter graph, which decoded
the file N times. It worked and it was slow: 309s for a 25 minute bed on an idle machine,
and over 1800s - the timeout, which killed the render - while a render was using the CPU.

Length. The bed is built to the voice track's own duration, then faded in and out, so it can
be mixed with duration=first and never truncate or extend the video.
"""
import json
import math
import re
import subprocess
from pathlib import Path

from core.logger import get_logger

log = get_logger("music")

MUSIC_DIR = Path(__file__).resolve().parent.parent / "assets" / "music"
EXTENSIONS = (".mp3", ".ogg", ".flac", ".wav", ".m4a", ".opus")

DEFAULTS = {
    "enabled": True,
    "gain_below_voice_db": 26,
    "fade_in_s": 3.0,
    "fade_out_s": 5.0,
    "loop_crossfade_s": 4.0,
}


class MusicError(Exception):
    pass


def settings(cfg):
    block = (cfg.get("music") or {})
    return {**DEFAULTS, **{k: v for k, v in block.items() if k in DEFAULTS}}


def available_tracks(music_dir=MUSIC_DIR):
    """Every usable track, sorted by name so the choice does not depend on the filesystem."""
    directory = Path(music_dir)
    if not directory.exists():
        return []
    return sorted((p for p in directory.iterdir()
                   if p.suffix.lower() in EXTENSIONS and p.stat().st_size > 10_000),
                  key=lambda p: p.name)


def pick_track(video_id, tracks):
    """Which track this video gets. Deterministic in the video id, and it walks the list one
    at a time, so consecutive videos never repeat until the whole set has been used."""
    if not tracks:
        return None
    return tracks[int(video_id) % len(tracks)]


def _ffmpeg(args, context, timeout=900):
    try:
        result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stats"] + args,
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # A MusicError, not a crash: the caller drops the music and renders the video.
        raise MusicError(f"{context}: gave up after {timeout}s") from e
    if result.returncode != 0:
        raise MusicError(f"{context}: {result.stderr.strip()[-400:]}")
    return result


def seamless_block(track, out_path, crossfade):
    """A block of `track` that can be repeated end to end without a click.

    The last `crossfade` seconds are mixed onto the first `crossfade` seconds and that
    becomes the block's head, followed by the untouched middle. The block therefore ends
    where its own head begins, through material that is already faded, so repeating it is
    continuous. Length is the track's length minus one crossfade.
    """
    length = duration(track)
    if length <= 3 * crossfade:
        raise MusicError(f"{Path(track).name} is only {length:.0f}s, too short to loop with "
                         f"a {crossfade:.0f}s crossfade")
    head_and_tail = (f"[0:a]atrim=start={length - crossfade:.3f},asetpts=N/SR/TB[tail];"
                     f"[1:a]atrim=0:{crossfade:.3f},asetpts=N/SR/TB[head];"
                     f"[tail][head]acrossfade=d={crossfade:.3f}:c1=tri:c2=tri[join];"
                     f"[2:a]atrim={crossfade:.3f}:{length - crossfade:.3f},asetpts=N/SR/TB[body];"
                     f"[join][body]concat=n=2:v=0:a=1[out]")
    _ffmpeg(["-i", str(track), "-i", str(track), "-i", str(track),
             "-filter_complex", head_and_tail, "-map", "[out]",
             "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(out_path)],
            f"ffmpeg could not build a seamless loop of {Path(track).name}")
    return out_path


def duration(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=nw=1:nk=1", str(path)],
                            capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise MusicError(f"ffprobe could not read the duration of {path}")
    return float(result.stdout.strip())


def loudness(path):
    """Integrated loudness in LUFS, measured with ffmpeg's EBU R128 meter."""
    result = subprocess.run(
        ["ffmpeg", "-nostats", "-i", str(path), "-filter_complex", "ebur128", "-f", "null", "-"],
        capture_output=True, text=True, timeout=900)
    matches = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", result.stderr)
    if not matches:
        raise MusicError(f"could not measure the loudness of {path}: "
                         f"{result.stderr.strip()[-300:]}")
    return float(matches[-1])


def build_bed(track, seconds, out_path, cfg, voice_lufs, keep_block=False):
    """A music bed of exactly `seconds`, looped seamlessly, faded, and sitting
    gain_below_voice_db under `voice_lufs`. Returns (path, details)."""
    s = settings(cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    track_s = duration(track)
    crossfade = min(s["loop_crossfade_s"], max(0.5, track_s / 4))
    block = out_path.with_name(out_path.stem + "_loop.wav")
    seamless_block(track, block, crossfade)
    block_s = duration(block)
    copies = max(1, math.ceil(seconds / block_s))

    target_lufs = voice_lufs - s["gain_below_voice_db"]
    # Measured on the loop block, not the source track: the block already contains the
    # crossfade, and summing two uncorrelated signals at half amplitude there costs about
    # 3 dB that the track on its own cannot predict. It is also 96 seconds to scan instead
    # of 25 minutes, and the bed is only this block repeated, so the two have the same
    # loudness.
    gain_db = target_lufs - loudness(block)
    fade_out_at = max(0.0, seconds - s["fade_out_s"])
    try:
        _ffmpeg(["-stream_loop", str(copies - 1), "-i", str(block),
                 "-af", (f"volume={gain_db:.2f}dB,"
                         f"afade=t=in:st=0:d={s['fade_in_s']},"
                         f"afade=t=out:st={fade_out_at:.3f}:d={s['fade_out_s']},"
                         f"atrim=0:{seconds:.3f}"),
                 "-ac", "1", "-ar", "48000", "-b:a", "192k", str(out_path)],
                f"ffmpeg could not build a music bed from {Path(track).name}")
    finally:
        if not keep_block:
            block.unlink(missing_ok=True)

    # Measure what was actually built and correct it. Predicting the gain from the source
    # track alone comes out about 3 dB low, because each crossfade sums two uncorrelated
    # signals at half amplitude and the fades trim the ends - effects too small to model
    # honestly and too large to ignore. One cheap pass over a mono file fixes it exactly.
    built_lufs = loudness(out_path)
    correction = target_lufs - built_lufs
    if abs(correction) > 0.3:
        corrected = out_path.with_name(out_path.stem + "_corrected" + out_path.suffix)
        _ffmpeg(["-i", str(out_path), "-af", f"volume={correction:.2f}dB",
                 "-ac", "1", "-ar", "48000", "-b:a", "192k", str(corrected)],
                "ffmpeg could not correct the music bed's level")
        corrected.replace(out_path)
        gain_db += correction
        built_lufs = loudness(out_path)

    details = {"track": Path(track).name, "track_seconds": round(track_s, 1),
               "loop_block_seconds": round(block_s, 1), "copies": copies,
               "crossfade_s": round(crossfade, 1), "gain_db": round(gain_db, 2),
               "target_lufs": round(target_lufs, 1), "measured_lufs": round(built_lufs, 1),
               "correction_db": round(correction, 2),
               "bed_seconds": round(duration(out_path), 1)}
    log.info("music bed: %s (%.0fs) looped %d times with %.1fs crossfades, %+.1f dB total to "
             "reach %.1f LUFS (asked %.1f, %.0f dB under the voice)", details["track"], track_s,
             copies, crossfade, gain_db, built_lufs, target_lufs, s["gain_below_voice_db"])
    return out_path, details


def credits(music_dir=MUSIC_DIR):
    path = Path(music_dir) / "CREDITS.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
