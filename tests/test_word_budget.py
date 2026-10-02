"""Offline tests for the script word budget. Run: python tests/test_word_budget.py

Pure arithmetic: how many words fill target_duration_min once the pauses are counted.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from pipeline import script

CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def _cfg(**over):
    cfg = json.loads(json.dumps(CFG))
    cfg.update(over)
    return cfg


def test_config_carries_the_chosen_voice_and_pauses():
    print("test_config_carries_the_chosen_voice_and_pauses")
    p = CFG["tts"]["pauses"]
    check((p["sentence_ms"], p["scene_ms"], p["lesson_ms"]) == (220, 450, 900),
          "config.json carries 220/450/900ms")
    check(CFG["tts"]["edge"]["voice"] == "en-US-AndrewMultilingualNeural",
          "Andrew is the configured voice")


def test_word_budget_inverts_the_runtime_model():
    print("test_word_budget_inverts_the_runtime_model")
    for minutes in (20, 30, 45):
        cfg = _cfg(target_duration_min=minutes)
        words = script.word_budget(cfg)
        back = script.estimated_runtime_s(words, cfg) / 60
        check(abs(back - minutes) < 0.02, f"{minutes} min -> {words} words -> {back:.2f} min")


def test_word_budget_follows_pace_and_pauses():
    print("test_word_budget_follows_pace_and_pauses")
    slow = script.word_budget(_cfg(words_per_minute=150))
    fast = script.word_budget(_cfg(words_per_minute=180))
    check(fast > slow, f"a faster voice needs more words for the same runtime ({slow} -> {fast})")
    short, long_ = _cfg(), _cfg()
    short["tts"]["pauses"] = {"sentence_ms": 100, "scene_ms": 200, "lesson_ms": 300}
    long_["tts"]["pauses"] = {"sentence_ms": 600, "scene_ms": 1200, "lesson_ms": 2400}
    check(script.word_budget(short) > script.word_budget(_cfg()) > script.word_budget(long_),
          "longer pauses leave fewer words, because silence takes runtime")


def test_the_window_is_28_to_32_minutes():
    print("test_the_window_is_28_to_32_minutes")
    cfg = _cfg(target_duration_min=30)
    budget = script.word_budget(cfg)
    lowest = script.estimated_runtime_s(budget * (1 - script.WORD_TOLERANCE), cfg) / 60
    highest = script.estimated_runtime_s(budget * (1 + script.WORD_TOLERANCE), cfg) / 60
    check(28 <= lowest and highest <= 32,
          f"a script accepted at either edge of the tolerance runs {lowest:.1f}-{highest:.1f} min")


def test_the_window_survives_the_measured_pace_spread():
    print("test_the_window_survives_the_measured_pace_spread")
    # The same voice measured 169.8 wpm on one whole video and 162.7 on another. The budget
    # is computed at the configured pace; the video is then spoken at whichever it gets.
    cfg = _cfg(target_duration_min=30)
    budget = script.word_budget(cfg)
    for pace in (162.7, 169.8):
        real = _cfg(target_duration_min=30, words_per_minute=pace)
        lowest = script.estimated_runtime_s(budget * (1 - script.WORD_TOLERANCE), real) / 60
        highest = script.estimated_runtime_s(budget * (1 + script.WORD_TOLERANCE), real) / 60
        check(28 <= lowest and highest <= 32,
              f"spoken at {pace} wpm, an accepted script runs {lowest:.1f}-{highest:.1f} min")


if __name__ == "__main__":
    test_config_carries_the_chosen_voice_and_pauses()
    test_word_budget_inverts_the_runtime_model()
    test_word_budget_follows_pace_and_pauses()
    test_the_window_is_28_to_32_minutes()
    test_the_window_survives_the_measured_pace_spread()
    print("ALL WORD BUDGET TESTS PASSED")
