"""Offline tests for search-led titles and the daily Short. Run: python tests/test_search_and_shorts.py

No network, no model. What is checked is what a viewer or YouTube would see: whether a title
opens with the search, whether two videos are really one, whether captions break mid-sentence,
whether the Short goes public before the video it points to.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core.logger  # noqa: F401,E402
from core import db  # noqa: E402
from pipeline import search_topics as st, seo, short  # noqa: E402

failures = []


def check(ok, msg):
    print(f"  {'ok' if ok else 'FAIL'}: {msg}")
    if not ok:
        failures.append(msg)


def test_what_autocomplete_noise_is_thrown_out():
    print("test_what_autocomplete_noise_is_thrown_out")
    for phrase in ("seneca village", "seneca crane death", "stoichiometry chemistry",
                   "marcus aurelius gladiator", "how to control emotions in trading",
                   "stoicism audiobook", "seneca high school", "stoicism"):
        check(bool(st.why_not(phrase)), f"'{phrase}' is out ({st.why_not(phrase)})")
    for phrase in ("stoicism for beginners", "how to stop overthinking everything",
                   "how to control emotions at work", "marcus aurelius philosophy"):
        check(not st.why_not(phrase), f"'{phrase}' is a possible video")


def test_two_phrases_that_are_one_video():
    print("test_two_phrases_that_are_one_video")
    check(st.same_video("stoic philosophy", "stoicism philosophy"), "stoic = stoicism")
    check(st.same_video("how to control emotions", "how to control your emotions"),
          "'your' changes nothing")
    check(st.same_video("how to control emotions and anger", "how to control emotions"),
          "a phrase that only extends a used one is the same video")
    check(not st.same_video("how to stop overthinking", "how to control emotions"),
          "different problems are different videos")


def test_the_choice_spreads_across_the_bases():
    print("test_the_choice_spreads_across_the_bases")
    found = {p: {"score": 10 - i * 0.1, "lists": 2, "best": 0, "bases": [], "head": True}
             for i, p in enumerate(["how to stop overthinking everything",
                                    "how to stop overthinking at night",
                                    "how to stop overthinking decisions",
                                    "stoicism for beginners", "marcus aurelius philosophy"])}
    saved = st.intents
    st.intents = lambda phrases: {p: "stoic-lessons" for p in phrases}
    try:
        picked = [p for p, _r in st.choose(found=found, count=3)]
    finally:
        st.intents = saved
    bases = [st.base_of(p) for p in picked]
    check(len(set(bases)) == 3, f"three videos, three bases: {picked}")


def test_the_title_must_open_with_the_search():
    print("test_the_title_must_open_with_the_search")
    phrase = "how to stop overthinking"
    good = {"title": "How to Stop Overthinking - 10 Stoic Habits for a Quiet Mind",
            "hook": "A quiet mind is a habit.", "summary": "One. Two. Three.",
            "tags": [f"t{i}" for i in range(12)]}
    check(not [c for c in seo._problems(good, 10, phrase) if "begin with" in c],
          "a title that opens with the search passes")
    bad = {**good, "title": "10 Stoic Habits - How to Stop Overthinking"}
    check(any("begin with" in c for c in seo._problems(bad, 10, phrase)),
          "the same words later in the title are refused")
    reworded = {**good, "title": "How to Quit Overthinking - 10 Stoic Habits"}
    check(any("begin with" in c for c in seo._problems(reworded, 10, phrase)),
          "a reworded search is refused: it is not what people type")
    check(st.display("how to stop overthinking in a relationship")
          == "How to Stop Overthinking in a Relationship", "small words stay small")


def test_a_short_script_is_checked():
    print("test_a_short_script_is_checked")
    body = ["word"] * 110
    lines = ["Why do you brace for blows that never land?", " ".join(body) + ".",
             "The full video on letting go is on this channel."]
    check(not short.script_problems(lines, "t"), "a good script passes")
    check(short.script_problems(["In this video we look at fear."] + lines[1:], "t"),
          "an opening like an introduction is refused")
    check(short.script_problems(lines[:2] + ["Subscribe for more."], "t"),
          "a script that does not send the viewer to the full video is refused")
    check(short.script_problems(["Why?", "Short.", "The full video is on this channel."], "t"),
          "too few words is refused")
    check(short.plain("a five‑year plan — “now”") == 'a five-year plan - "now"',
          "look-alike hyphens and quotes become plain ones the font has")


def test_a_close_short_is_kept_not_dropped():
    print("test_a_close_short_is_kept_not_dropped")
    # 2026-11-06/07: three Shorts lost to scripts of 99 words, 88 words, 33 s, all otherwise
    # fine. Stub everything but the retry loop and see what it does with them.
    import pipeline.seo  # noqa: F401
    tmp = Path(tempfile.mkdtemp())
    real = {k: getattr(short, k) for k in ("lessons_of", "choose_lesson", "write_script", "speak",
                                           "duration", "soundtrack", "encode", "frames",
                                           "caption_chunks")}
    hook = "Why do you brace for blows that never land?"
    close = "The full video on letting go is on this channel."

    def script(n):
        return {"hook": hook, "idea": "One idea.", "moment": " ".join(["word"] * (n - 20)) + ".",
                "takeaway": "Do this.", "closing": close}

    try:
        cur = [0.0]
        order = iter([34.0, 36.0, 38.0])

        def speak(text, path, **k):
            cur[0] = next(order) - short.LEAD - short.TAIL
            Path(path).write_bytes(b"mp3-" + str(cur[0]).encode())
            return [(0.0, 0.2, "w")]
        short.speak = speak
        short.duration = lambda path: cur[0]
        short.lessons_of = lambda v: [("Lesson", [{"text": "t", "image_path": None}])]
        short.choose_lesson = lambda l, t: (0, "test")
        replies = iter([99, 104, 108])
        short.write_script = lambda *a, **k: script(next(replies))
        short.soundtrack = lambda *a, **k: tmp / "a.m4a"
        short.encode = lambda it, audio, out, **k: Path(out).write_bytes(b"x") or out
        short.frames = lambda *a, **k: iter(())
        short.caption_chunks = lambda *a, **k: []
        report = short.build(1, {}, pictures_for=lambda v, s: [object()], work=tmp)
        check(abs(report["seconds"] - 38.0) < 0.1,
              f"three scripts all a little short: the closest, 38 s, is kept ({report['seconds']})")
        order = iter([20.0, 21.0, 22.0])
        replies = iter([104, 104, 104])
        try:
            short.build(1, {}, pictures_for=lambda v, s: [object()], work=tmp)
            check(False, "a 22 s Short is still refused")
        except RuntimeError:
            check(True, "a 22 s Short is still refused")
    finally:
        for k, v in real.items():
            setattr(short, k, v)


def test_captions_never_cross_a_sentence():
    print("test_captions_never_cross_a_sentence")
    script = "Ever notice how you tense up before the next crisis? You are tired. Breathe."
    words = []
    t = 0.0
    for w in script.split():
        bare = w.strip(".,?!")
        words.append((t, t + 0.3, bare))
        t += 0.35
    chunks = short.caption_chunks(words, script=script)
    shown = [" ".join(w[2] for w in c) for _s, _e, c in chunks]
    check("crisis You" not in " | ".join(shown) and not any("crisis You" in s for s in shown),
          f"no caption holds the end of one sentence and the start of the next: {shown}")
    check(all(1 <= len(c) <= 3 for _s, _e, c in chunks), "never more than three words")
    check(min(len(c) for _s, _e, c in chunks[:3]) >= 2,
          "ten words go 3-3-2-2, not 3-3-3-1")


def test_the_short_goes_public_with_its_video():
    print("test_the_short_goes_public_with_its_video")
    video = {"youtube_id": "abc123", "seo_title": "How to Stop Overthinking - 10 Lessons",
             "published_at": "2099-11-01T19:00:00Z", "search_phrase": "how to stop overthinking"}
    report = {"lines": ["Why do you brace for blows that never land?", "One.", "Two.",
                        "The full video is on this channel."], "seconds": 45, "lesson": "x"}
    body = short.short_body(video, report, {"privacy_status": "private"})
    check(body["status"]["publishAt"] == video["published_at"],
          "the Short is scheduled for the moment its full video goes public, never before")
    check(body["status"]["selfDeclaredMadeForKids"] is False, "not made for kids")
    check("https://youtu.be/abc123" in body["snippet"]["description"],
          "the description links the full video")
    check(body["snippet"]["title"].endswith("#shorts") and len(body["snippet"]["title"]) <= 100,
          f"title fits: {body['snippet']['title']}")
    check("how to stop overthinking" in body["snippet"]["tags"], "the search phrase is a tag")


def test_the_new_columns_can_be_written():
    print("test_the_new_columns_can_be_written")
    for column in ("search_phrase", "short_youtube_id"):
        check(column in db._VIDEO_FIELDS and any(c == column for _t, c, _d in db._MIGRATIONS),
              f"videos.{column} is migrated and writable")
    with tempfile.TemporaryDirectory() as tmp:
        db.DB_PATH = Path(tmp) / "t.db"
        db.init_db()
        vid = db.create_video("t")
        db.update_video(vid, search_phrase="stoicism for beginners", short_youtube_id="s1")
        check(db.used_search_phrases() == {"stoicism for beginners"},
              "a used phrase is remembered")


if __name__ == "__main__":
    test_what_autocomplete_noise_is_thrown_out()
    test_two_phrases_that_are_one_video()
    test_the_choice_spreads_across_the_bases()
    test_the_title_must_open_with_the_search()
    test_a_short_script_is_checked()
    test_a_close_short_is_kept_not_dropped()
    test_captions_never_cross_a_sentence()
    test_the_short_goes_public_with_its_video()
    test_the_new_columns_can_be_written()
    if failures:
        print(f"\n{len(failures)} FAILED")
        sys.exit(1)
    print("\nALL SEARCH AND SHORTS TESTS PASSED")
