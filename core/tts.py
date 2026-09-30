"""Text-to-speech: two interchangeable providers, picked by config['tts']['provider'].

piper runs fully offline from local voice models in assets/voices/ (not in git: see
README for the download command). edge-tts calls Microsoft's free cloud endpoint,
which some server/cloud IPs see answer with a 403; that case is caught and logged as
a plain explanation, not a stack trace, because the fix is switching provider, not
retrying harder.

Both providers converge on the same output: a 22050 Hz mono wav file, so the render
stage never has to know which one made it.
"""
import asyncio
import re
import shutil
import subprocess
import wave
from pathlib import Path

from core.logger import get_logger
from core.retry import retry

log = get_logger("tts")

VOICES_DIR = Path(__file__).resolve().parent.parent / "assets" / "voices"

FFMPEG_INSTALL_HINT = (
    "ffmpeg was not found on PATH. On Windows, the easiest fix is:\n"
    "  winget install Gyan.FFmpeg\n"
    "then open a new terminal so PATH picks it up. Manual builds are at "
    "https://www.gyan.dev/ffmpeg/builds/ (download the \"full\" build and add its "
    "bin\\ folder to PATH)."
)

DEFAULT_PIPER_VOICE = "en_US-ryan-high"
DEFAULT_PIPER_LENGTH_SCALE = 1.08
DEFAULT_EDGE_VOICE = "en-US-GuyNeural"
DEFAULT_EDGE_RATE = "-8%"

# Candidates downloaded / offered for the by-ear comparison. Not all of these need
# to be the one config.json picks; they are what scripts/tts_sample.py renders.
PIPER_VOICES = ["en_US-ryan-high", "en_GB-alan-medium", "en_US-joe-medium"]
EDGE_VOICES = ["en-US-GuyNeural", "en-GB-RyanNeural", "en-US-BrianNeural",
               "en-US-ChristopherNeural"]

SAMPLE_RATE = 22050


class TTSError(Exception):
    pass


_ffmpeg_checked = False


def require_ffmpeg():
    """Raise a plain, actionable error instead of letting a subprocess call fail
    somewhere deep in a stack trace."""
    global _ffmpeg_checked
    if _ffmpeg_checked:
        return
    if shutil.which("ffmpeg") is None:
        raise TTSError(FFMPEG_INSTALL_HINT)
    _ffmpeg_checked = True


def _run_ffmpeg(args, error_context):
    require_ffmpeg()
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise TTSError(f"{error_context}: {result.stderr.strip()[-400:]}")
    return result


# --- piper -------------------------------------------------------------------

_piper_cache = {}


def _load_piper_voice(name):
    if name not in _piper_cache:
        from piper import PiperVoice
        model_path = VOICES_DIR / f"{name}.onnx"
        if not model_path.exists():
            raise TTSError(
                f"piper voice '{name}' is not downloaded (looked in {model_path}). Get it "
                f"with:\n  python -m piper.download_voices --download-dir {VOICES_DIR} {name}")
        _piper_cache[name] = PiperVoice.load(str(model_path))
    return _piper_cache[name]


def synthesize_piper(text, out_path, voice=DEFAULT_PIPER_VOICE,
                     length_scale=DEFAULT_PIPER_LENGTH_SCALE):
    """Render text to a wav file with a local piper voice model."""
    from piper.config import SynthesisConfig
    pv = _load_piper_voice(voice)
    syn_cfg = SynthesisConfig(length_scale=length_scale)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as f:
        pv.synthesize_wav(text, f, syn_config=syn_cfg)


# --- edge-tts ------------------------------------------------------------------

def _is_403(error):
    msg = str(error)
    return "403" in msg or "Forbidden" in msg or "NoAudioReceived" in msg


@retry(attempts=2, base_delay=3)
def synthesize_edge(text, out_path, voice=DEFAULT_EDGE_VOICE, rate=DEFAULT_EDGE_RATE):
    """Render text to a wav file via Microsoft's free cloud TTS, converted from mp3."""
    import edge_tts
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path = out_path.with_suffix(".mp3")

    async def _run():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(str(mp3_path))

    try:
        asyncio.run(_run())
    except Exception as e:
        if _is_403(e):
            # Not worth retrying: a 403 here means the calling IP is blocked, which
            # will not change between attempts.
            raise TTSError(
                "edge-tts got a 403 from Microsoft's endpoint. Some server or cloud IPs "
                "are blocked from this free service; it typically works from a home "
                "connection. Switch config tts.provider to \"piper\" if this keeps "
                "happening on this machine.") from e
        raise TTSError(f"edge-tts request failed: {e}") from e

    if not mp3_path.exists() or mp3_path.stat().st_size == 0:
        raise TTSError("edge-tts produced no audio (empty or missing file)")

    _mp3_to_wav(mp3_path, out_path)
    mp3_path.unlink(missing_ok=True)


def _mp3_to_wav(src, dst):
    _run_ffmpeg(["-i", str(src), "-ar", str(SAMPLE_RATE), "-ac", "1", str(dst)],
               "ffmpeg could not convert edge-tts output to wav")


# --- shared --------------------------------------------------------------------

def wav_duration(path):
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def provider_settings(cfg):
    """The provider name plus its own settings block, filled in with defaults."""
    tts_cfg = cfg.get("tts") or {}
    provider = tts_cfg.get("provider", "piper")
    if provider == "piper":
        block = tts_cfg.get("piper") or {}
        return provider, {"voice": block.get("voice", DEFAULT_PIPER_VOICE),
                          "length_scale": block.get("length_scale", DEFAULT_PIPER_LENGTH_SCALE)}
    if provider == "edge":
        block = tts_cfg.get("edge") or {}
        return provider, {"voice": block.get("voice", DEFAULT_EDGE_VOICE),
                          "rate": block.get("rate", DEFAULT_EDGE_RATE)}
    raise TTSError(f"unknown tts provider: {provider!r} (use \"piper\" or \"edge\")")


def synthesize(text, out_path, cfg):
    """Render one piece of narration to a wav file. Returns its duration in seconds."""
    provider, settings = provider_settings(cfg)
    if provider == "piper":
        synthesize_piper(text, out_path, settings["voice"], settings["length_scale"])
    else:
        synthesize_edge(text, out_path, settings["voice"], settings["rate"])
    return wav_duration(out_path)


def concatenate(wav_paths, out_path):
    """Join wav files (scene wavs, sentence wavs, silence segments - anything at the
    same sample rate/channels) into one track with ffmpeg's concat demuxer. -c copy
    requires that match, which holds here because everything is rendered or generated
    at SAMPLE_RATE mono."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text(
        "\n".join(f"file '{Path(p).resolve().as_posix()}'" for p in wav_paths), encoding="utf-8")
    try:
        _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy",
                    str(out_path)],
                   f"ffmpeg could not concatenate {len(wav_paths)} audio segments")
    finally:
        list_path.unlink(missing_ok=True)
    return wav_duration(out_path)


def silence(duration_s, out_path):
    """A wav file of exactly duration_s of digital silence, at SAMPLE_RATE mono, so
    it concatenates cleanly with rendered speech via -c copy."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(["-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
                "-t", f"{duration_s:.3f}", str(out_path)],
               f"ffmpeg could not generate {duration_s:.3f}s of silence")
    return out_path


def measure_volume(path):
    """Mean and max volume in dBFS, via ffmpeg's volumedetect filter. Used to catch a
    file that rendered but is effectively silent (a synthesis failure that still
    produced a valid, non-empty wav)."""
    require_ffmpeg()
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True)
    mean = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", result.stderr)
    peak = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", result.stderr)
    if not mean:
        raise TTSError(f"ffmpeg volumedetect gave no reading for {path}: "
                       f"{result.stderr.strip()[-300:]}")
    return float(mean.group(1)), float(peak.group(1)) if peak else None


def master(in_wav, out_mp3, cfg):
    """Loudness-normalise to YouTube's -14 LUFS, cut the sub-80Hz rumble, upsample to
    48kHz (Piper's native 22050Hz is too low for YouTube), and export as 192kbps mono
    mp3. One pass over the whole track, not per scene: normalising each scene on its
    own would flatten the natural loud/quiet variation between scenes instead of the
    track as a whole."""
    settings = mastering_settings(cfg)
    out_mp3 = Path(out_mp3)
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    filters = (f"highpass=f={settings['highpass_hz']},"
              f"loudnorm=I={settings['target_lufs']}:TP=-1.5:LRA=11")
    _run_ffmpeg(["-i", str(in_wav), "-af", filters,
                "-ar", str(settings["sample_rate"]), "-ac", "1",
                "-b:a", f"{settings['bitrate_kbps']}k", str(out_mp3)],
               "ffmpeg could not master the narration track")
    return out_mp3


def extract_sample(in_audio, out_path, seconds):
    """The first `seconds` of a track, re-encoded (stream copy can misplace the cut
    point on a compressed format, and the sample has to end exactly on time)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(["-i", str(in_audio), "-t", str(seconds), "-ar", "48000", "-ac", "1",
                "-b:a", "192k", str(out_path)],
               f"ffmpeg could not cut a {seconds}s sample")
    return out_path


def mastering_settings(cfg):
    block = (cfg.get("tts") or {}).get("mastering") or {}
    return {
        "target_lufs": block.get("target_lufs", -14),
        "highpass_hz": block.get("highpass_hz", 80),
        "sample_rate": block.get("sample_rate", 48000),
        "bitrate_kbps": block.get("bitrate_kbps", 192),
    }


def pause_settings(cfg):
    block = (cfg.get("tts") or {}).get("pauses") or {}
    return {
        "sentence_ms": block.get("sentence_ms", 350),
        "scene_ms": block.get("scene_ms", 700),
        "lesson_ms": block.get("lesson_ms", 1400),
    }
