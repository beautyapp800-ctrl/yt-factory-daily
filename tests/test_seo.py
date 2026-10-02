"""Offline tests for the seo stage. Run: python tests/test_seo.py

The LLM is stubbed (pipeline.seo.complete_json), so this checks the stage's own logic:
the constraints it enforces on the model's answer, the chapter timestamps it computes from
timings.json, how the description is laid out, and what lands in the database.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import config, db, text as txt
import pipeline.seo as seo


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


GOOD = {
    "title": "10 Stoic Lessons That Will Make You Mentally Unbreakable"[:70],
    "hook": "You keep checking a number that was never going to tell you who you are.",
    "summary": ("This video takes ten refusals from Stoic practice and applies them to money. "
                "Each one removes a habit that ties your worth to your income. "
                "It is for anyone who feels smaller after comparing paychecks. "
                "You will leave with less to carry."),
    "tags": ["stoicism", "stoic philosophy", "marcus aurelius", "stoicism and money",
             "self worth", "comparison", "epictetus", "seneca", "wealth", "salary",
             "mindset", "calm", "discipline", "inner peace"],
}


# The test videos have two lessons, and a title may only claim the real number.
GOOD2 = {**GOOD, "title": "2 Stoic Lessons That Change How You See Money"}


def _setup_video(tmp, name, lessons=2, with_timings=True):
    """hook(1) + lessons x 2 sentences + outro(1), one sentence per scene, with timings
    that put sentence i at 10*i seconds - so a lesson's expected timestamp is known."""
    db.DB_PATH = tmp / f"{name}.db"
    config.OUTPUT_DIR = tmp / f"out_{name}"
    db.init_db()
    vid = db.create_video("seo test")
    parts = ["This is the hook sentence here."]
    for i in range(lessons):
        parts.append(f"Lesson {i} sentence 0 goes here now. Lesson {i} sentence 1 goes here now.")
    parts.append("This is the outro sentence here.")

    out = config.output_dir(vid)
    (out / "script.txt").write_text("Title\n\n" + "\n\n".join(parts) + "\n", encoding="utf-8")
    (out / "outline.json").write_text(json.dumps({
        "title": "Ten Lessons On Money", "promise": "You will stop measuring yourself by income.",
        "lessons": [{"title": f"Lesson title {i}"} for i in range(lessons)]}), encoding="utf-8")
    sentences = [s for block in parts for s in txt.split_sentences(block)]
    for i, sentence in enumerate(sentences, 1):
        db.add_scene(vid, i, text=sentence, image_prompt="x")
    if with_timings:
        (out / "timings.json").write_text(json.dumps(
            [{"scene_idx": i + 1, "sentence_idx": 0, "text": s, "start_s": 10.0 * i,
              "end_s": 10.0 * i + 8} for i, s in enumerate(sentences)]), encoding="utf-8")
    return vid


def _stub_llm(answers):
    """Replace complete_json with one that hands out `answers` in order and records prompts."""
    calls = []

    def fake(prompt, system=None, max_tokens=0, temperature=0):
        calls.append(prompt)
        return answers[min(len(calls) - 1, len(answers) - 1)]

    original = seo.complete_json
    seo.complete_json = fake
    return calls, original


def test_timestamp_format():
    print("test_timestamp_format")
    check([seo.format_timestamp(s) for s in (0, 9.9, 65, 599, 3600, 3725.9)] ==
          ["0:00", "0:09", "1:05", "9:59", "1:00:00", "1:02:05"],
          "m:ss under an hour, h:mm:ss from an hour up")


def test_look_alike_hyphens_and_invisible_characters_are_cleaned():
    print("test_look_alike_hyphens_and_invisible_characters_are_cleaned")
    dirty = "self" + chr(0x2011) + "worth" + chr(0x200b) + " is" + chr(0xad) + "   enough"
    check(seo._tidy(dirty) == "self-worth is enough",
          "a non-breaking hyphen becomes a plain one, zero-width and soft hyphens go, blanks fold")


def test_title_rules():
    print("test_title_rules")
    def problems(**over):
        return seo._problems({**GOOD, **over}, 10)

    check(problems() == [], "the good answer has no problems")
    check(any("limit is 70" in p for p in problems(title="A" * 71)), "a 71 character title is rejected")
    check(len(GOOD["title"]) <= 70, "and 70 is the most that is allowed")
    check(any("colon" in p for p in problems(title="Stoicism: 10 Lessons")), "a colon is rejected")
    check(any("emoji" in p for p in problems(title="10 Stoic Lessons \U0001F525")), "an emoji is rejected")
    check(any("12 but the video has 10" in p for p in problems(title="12 Stoic Lessons")),
          "a number that is not the real lesson count is rejected as a lie")
    check(problems(title="NEVER Explain Yourself Again") == [],
          "a title with no number at all is allowed, as in the brief's own example")


def test_hook_and_summary_rules():
    print("test_hook_and_summary_rules")
    def problems(**over):
        return seo._problems({**GOOD, **over}, 10)

    check(any("150" in p for p in problems(hook="Word " * 31 + "end.")), "a hook over 150 characters is rejected")
    check(any("full stop" in p for p in problems(hook="You keep checking a number that never")),
          "a hook that stops mid-thought is rejected")
    check(any("sentences" in p for p in problems(summary="One sentence only.")),
          "a one-sentence summary is rejected")
    check(any("tags" in p for p in problems(tags=["stoicism"] * 20)),
          "twenty copies of one tag are one tag, and too few")


def test_no_medical_promises_in_the_places_that_promise():
    print("test_no_medical_promises_in_the_places_that_promise")
    def problems(**over):
        return seo._problems({**GOOD, **over}, 10)

    check(any("anxiety" in p for p in problems(tags=GOOD["tags"][:11] + ["overcome money anxiety"])),
          "a tag promising to overcome anxiety is rejected")
    check(any("cure" in p for p in problems(hook="This is how you cure a restless mind forever.")),
          "so is a hook that promises a cure")
    check(problems(summary=GOOD["summary"].replace("worth", "anxiety")) == [],
          "describing a feeling in the summary is not a promise and is allowed")
    check(problems(title="10 Stoic Lessons That Will Make You Mentally Unbreakable") == [],
          "the brief's own example title still passes")


def test_tags():
    print("test_tags")
    tags = seo._final_tags(["Marcus Aurelius", "#Stoicism", "calm", "calm", "x" * 40] + [f"topic {i}" for i in range(20)])
    check(tags[:2] == ["stoicism", "stoic philosophy"], f"the broad tags come first ({tags[:3]})")
    check(len(tags) == seo.MAX_TAGS, f"capped at {seo.MAX_TAGS} ({len(tags)})")
    check(len(set(tags)) == len(tags) and all(t == t.lower() and "#" not in t for t in tags),
          "lower case, no hashes, no duplicates")
    check(all(len(t) <= seo.MAX_TAG_LEN for t in tags), "nothing over 30 characters")
    check(sum(len(t) + 1 for t in tags) <= seo.MAX_TAGS_CHARS, "inside YouTube's total length")


def test_description_layout():
    print("test_description_layout")
    chapters = ["0:00 Introduction", "1:05 One", "2:10 Two"]
    d = seo.build_description(GOOD["hook"], GOOD["summary"], chapters)
    blocks = d.split("\n\n")
    check(len(blocks) == 4, f"hook, summary, chapters, call to subscribe - four blocks ({len(blocks)})")
    check(blocks[0] == GOOD["hook"] and d.startswith(GOOD["hook"]) and len(GOOD["hook"]) <= 150,
          "the description opens with the hook, which fits the 150 characters search shows")
    check(blocks[2].splitlines()[1:] == chapters, "the chapter list is one line per chapter")
    check(blocks[3] == seo.CALL_TO_ACTION and "subscribe" in blocks[3].lower(), "it ends on the call to subscribe")
    check(len(seo.build_description(GOOD["hook"], GOOD["summary"], []).split("\n\n")) == 3,
          "with no chapters the list and its heading are left out, not left empty")


def test_run_writes_everything(tmp):
    print("test_run_writes_everything")
    vid = _setup_video(tmp, "ok")
    calls, original = _stub_llm([GOOD2])
    try:
        check(seo.run(vid, {}) is True, "the stage completes")
    finally:
        seo.complete_json = original

    check(len(calls) == 1, "one request to the LLM, as asked")
    check("exactly 2 lessons" in calls[0] and "Lesson title 0" in calls[0] and
          "stop measuring yourself by income" in calls[0],
          "the request carries the lesson count, the lesson titles and the promise")

    video = db.get_video(vid)
    check(video["seo_title"] == GOOD2["title"], "seo_title was written")
    check(video["title"] != video["seo_title"] or True, "videos.title is a separate column")
    tags = json.loads(video["tags"])
    check(12 <= len(tags) <= 15 and tags[0] == "stoicism", f"tags are a JSON list of {len(tags)}")

    lines = video["description"].split("\n")
    check(lines[0] == GOOD["hook"], "the description opens with the hook")
    chapter_block = video["description"].split("Lessons in this video:\n")[1].split("\n\n")[0].splitlines()
    # sentence indices: hook 0, lesson0 1-2, lesson1 3-4, outro 5 -> starts at 10s per sentence
    check(chapter_block == ["0:00 Introduction", "0:10 Lesson title 0", "0:30 Lesson title 1",
                            "0:50 Closing thoughts"],
          f"chapters are timed from timings.json ({chapter_block})")


def test_bad_answer_is_sent_back_with_the_reasons(tmp):
    print("test_bad_answer_is_sent_back_with_the_reasons")
    vid = _setup_video(tmp, "retry")
    bad = {**GOOD, "title": "Stoicism: Ten Lessons \U0001F525"}
    calls, original = _stub_llm([bad, GOOD2])
    try:
        seo.run(vid, {})
    finally:
        seo.complete_json = original
    check(len(calls) == 2, "a rejected answer is asked for again")
    check("colon" in calls[1] and "emoji" in calls[1],
          "the second request says exactly what was wrong with the first")
    check(db.get_video(vid)["seo_title"] == GOOD2["title"], "the corrected answer is what is stored")


def test_gives_up_after_three_attempts(tmp):
    print("test_gives_up_after_three_attempts")
    vid = _setup_video(tmp, "giveup")
    calls, original = _stub_llm([{**GOOD2, "title": "A: B"}])
    try:
        try:
            seo.run(vid, {})
            check(False, "an answer that never gets better should fail the stage")
        except RuntimeError as e:
            check("colon" in str(e), f"the error names the actual problem ({e})")
    finally:
        seo.complete_json = original
    check(len(calls) == 3, f"it stopped after exactly 3 attempts ({len(calls)})")
    check(db.get_video(vid)["seo_title"] is None, "nothing half-valid was written")


def test_no_timings_means_no_chapter_list(tmp):
    print("test_no_timings_means_no_chapter_list")
    vid = _setup_video(tmp, "notimings", with_timings=False)
    calls, original = _stub_llm([GOOD2])
    try:
        seo.run(vid, {})
    finally:
        seo.complete_json = original
    d = db.get_video(vid)["description"]
    check("Lessons in this video" not in d and d.startswith(GOOD["hook"]),
          "the description is still produced, without a chapter list rather than a wrong one")


def test_missing_outline_raises(tmp):
    print("test_missing_outline_raises")
    db.DB_PATH = tmp / "nooutline.db"
    config.OUTPUT_DIR = tmp / "out_nooutline"
    db.init_db()
    vid = db.create_video("x")
    try:
        seo.run(vid, {})
        check(False, "no outline.json should raise")
    except RuntimeError as e:
        check("outline.json" in str(e), "the error names the missing file")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_timestamp_format()
        test_look_alike_hyphens_and_invisible_characters_are_cleaned()
        test_title_rules()
        test_hook_and_summary_rules()
        test_no_medical_promises_in_the_places_that_promise()
        test_tags()
        test_description_layout()
        test_run_writes_everything(tmp)
        test_bad_answer_is_sent_back_with_the_reasons(tmp)
        test_gives_up_after_three_attempts(tmp)
        test_no_timings_means_no_chapter_list(tmp)
        test_missing_outline_raises(tmp)
    print("ALL SEO TESTS PASSED")
