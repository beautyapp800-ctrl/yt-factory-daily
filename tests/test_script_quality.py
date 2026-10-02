"""Offline tests for the script-quality guards. Run: python tests/test_script_quality.py

No network: the model is stubbed. These lock in the four defects found in video 3,
so a later change cannot quietly bring them back.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the log at a temp dir before anything calls get_logger(), so a test run does
# not write into the working logs/factory.log. Left unset, test lines land in the real
# log and any tool grepping it for failures sees them as production errors.
import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

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
    # Over-long is trimmed for free, short has to be bought back with rewrites, so the
    # order aims at the full target rather than under it.
    check(sc.ordered_words(395) == 395, "the word order is not discounted below the target")
    check(sc.ordered_words(10) == 180, "a tiny target is floored at 180 words")


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


def test_domain_diversity():
    print("test_domain_diversity")
    digital = [{"title": f"Stop Checking Your Phone Again {i}",
                "focus": "About scrolling the feed and answering email."} for i in range(5)]
    problems = sc.domain_complaints(digital)
    check(problems, "five lessons in one domain are rejected")
    check("digital habits" in problems[0], "the complaint names the crowded domain")
    check("work" in problems[0] and "family" in problems[0],
          "the complaint lists the domains to use instead")
    spread = [{"title": "Stop Explaining Your Work", "focus": "About your boss at the office."},
              {"title": "Stop Carrying Family Debt", "focus": "About your mother and the past."},
              {"title": "Drop the Money Story", "focus": "About rent and savings."}]
    check(not sc.domain_complaints(spread), "a plan spread across domains is accepted")


def test_forbidden_opening_prefixes():
    print("test_forbidden_opening_prefixes")
    frag = sc._avoid_openings(["You stand in the kitchen and wait.", "You sit at the desk."])
    check('do not begin with "you stand in"' in frag, "the banned word sequence is stated")
    check('do not begin with "you sit at"' in frag, "each clashing prefix is listed")
    check(sc._avoid_openings([]) == "", "no openings means no fragment")


def test_name_tics():
    print("test_name_tics")
    parts = {("hook", 0): "Maya waits. Then Maya speaks. Maya again.",
             ("lesson", 0): "Later Maya calls. Tomas says nothing."}
    sc.swap_overused_names(parts, avoid=["Nadia"])
    joined = " ".join(parts.values())
    check("Maya" not in joined, "a name used four times is replaced")
    check(joined.count("Tomas") == 1,
          "the replacement does not collide with a name already in the text")
    check("Nadia" not in joined, "a name recent videos used is not chosen")

    kept = {("hook", 0): "You wait. Then Tomas speaks once."}
    sc.swap_overused_names(kept, avoid=[])
    check("Tomas" in kept[("hook", 0)], "a name used once is left alone")

    twice = "You wait in a teal coat. A teal door opens. Tomas nods twice at Tomas."
    check(("colour", "teal") in sc.collect_tics(twice),
          "a colour word used twice is recorded as a tic")
    check(("colour", "teal") not in sc.collect_tics("One teal coat by the door."),
          "a word used once is not a tic, which is what keeps ordinary capitals out")


def test_outro_target():
    print("test_outro_target")
    check(sc.OUTRO_WORDS == 140, "the outro target matches what the model actually returns")
    check(sc.OPENING_REWRITES == 2, "a clashing opening gets two attempts")


def test_clean_spaces_punctuation_without_corrupting_it():
    print("test_clean_spaces_punctuation_without_corrupting_it")
    # The replacement for "comma with no space after it" had been committed as a control
    # character instead of a backreference, so "Wait,what" came out as "Wait<SOH> what".
    out = txt.clean("Wait,what happened;really:no 6:15 and 1,000 ok")
    check(out == "Wait, what happened; really: no 6:15 and 1,000 ok", f"punctuation is spaced ({out!r})")
    check(not any(ord(c) < 32 for c in out), "and nothing unprintable is inserted")


if __name__ == "__main__":
    for test in (test_title_ban, test_trimming, test_trim_to_budget,
                 test_hook_preview_detector, test_duplicate_openings,
                 test_domain_diversity, test_forbidden_opening_prefixes,
                 test_name_tics, test_outro_target,
                 test_clean_spaces_punctuation_without_corrupting_it):
        test()
    print("ALL SCRIPT QUALITY TESTS PASSED")
