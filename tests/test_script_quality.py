"""Offline tests for the script-quality guards. Run: python tests/test_script_quality.py

No network: the model is stubbed. These lock in the four defects found in video 3,
so a later change cannot quietly bring them back.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.text as txt
import pipeline.script as sc


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_title_ban():
    print("test_title_ban")
    # The two titles that made video 3 read like a time-management video.
    check(sc.title_problems("Use a Daily Thought Log"), "'Daily Thought Log' is rejected")
    check(sc.title_problems("Review and Reset Weekly"), "'Review and Reset Weekly' is rejected")
    check(sc.title_problems("Build a Resilience Framework"), "'Framework' is rejected")
    check(sc.title_problems("Stop It"), "a 2 word title is rejected")
    check(sc.title_problems("Stop Explaining Every Single Choice You Have Ever Made Again"),
          "a 10 word title is rejected")
    check(sc.title_problems("Stop Explaining: Your Choices"), "a colon is rejected")
    check(not sc.title_problems("Stop Explaining Your Decisions to Spectators"),
          "a well formed refusal title passes")
    check(not sc.title_problems("10 Things You Stop Waiting For", strict=False),
          "the video title is exempt from the 4-8 word rule")


def test_trimming():
    print("test_trimming")
    body = "This sentence carries exactly seven words here. " * 80
    limit = int(395 * sc.TRIM_RATIO)
    trimmed = txt.trim_to_words(body, limit)
    check(txt.word_count(trimmed) <= limit, f"trimmed to {txt.word_count(trimmed)} <= {limit}")
    check(trimmed.rstrip().endswith("."), "trimming stops on a sentence boundary")
    check(txt.word_count(txt.trim_to_words("Short text here.", 100)) == 3,
          "text already inside the limit is untouched")
    check(sc.ordered_words(395) == 340,
          "395 wanted words are ordered as 340, to absorb the overshoot")


def test_trim_to_budget():
    print("test_trim_to_budget")
    parts = {("hook", 0): "word " * 200, ("outro", 0): "word " * 200}
    for i in range(10):
        parts[("lesson", i)] = "This sentence has exactly six words. " * (110 if i < 5 else 40)
    before = sc._total_words(parts)
    check(before > 4350, f"the fixture really is over budget at {before} words")
    sc.trim_to_budget(parts, 10, 4350)
    after = sc._total_words(parts)
    check(after < before, f"{before} words were actually cut down to {after}")
    check(after <= 4350, f"{after} words is inside the 4350 budget")
    short = [txt.word_count(parts[("lesson", i)]) for i in range(5, 10)]
    check(all(n == 240 for n in short), "the short lessons were left alone")


def test_hook_preview_detector():
    print("test_hook_preview_detector")
    titles = ["Stop Explaining Your Decisions to Spectators",
              "Quit Auditioning for People Who Already Decided",
              "Drop the Grudge You Feed Every Morning"]
    previewing = ("It begins with explaining your decisions to spectators, then auditioning "
                  "for people who decided, then the grudge you feed every morning.")
    clean = ("You stand in the pharmacy queue and your jaw tightens. There is another way "
             "to stand inside that hour.")
    check(len(txt.titles_present_in(previewing, titles)) > sc.MAX_HOOK_TITLE_HITS,
          "a hook reciting the lessons is caught")
    check(len(txt.titles_present_in(clean, titles)) == 0,
          "a hook that opens on a scene is not flagged")


def test_duplicate_openings():
    print("test_duplicate_openings")
    parts = {("hook", 0): "You sit at the table and wait. Second sentence.",
             ("lesson", 0): "You sit at the table again. Second sentence.",
             ("lesson", 1): "The pharmacy queue has not moved. Second sentence."}
    order = [("hook", 0), ("lesson", 0), ("lesson", 1)]
    rewritten = []

    def regenerate(key, openings):
        rewritten.append(key)
        return "A different place entirely now. Second sentence."

    sc.fix_duplicate_openings(parts, order, regenerate)
    check(rewritten == [("lesson", 0)], "only the later of the two clashing parts is rewritten")
    openers = [txt.opening_words(parts[k], 4) for k in order]
    check(len(set(openers)) == 3, "all three parts now open differently")

    # Digits are dropped on purpose: two openings differing only by a time still clash.
    check(txt.opening_words("You wake at 6 sharp") == txt.opening_words("You wake at 7 sharp"),
          "openings differing only by a number count as the same")


if __name__ == "__main__":
    for test in (test_title_ban, test_trimming, test_trim_to_budget,
                 test_hook_preview_detector, test_duplicate_openings):
        test()
    print("ALL SCRIPT QUALITY TESTS PASSED")
