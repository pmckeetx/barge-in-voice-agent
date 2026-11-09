"""One scripted call against the real API. Runs only with POLYAI_LIVE=1 and a POLY_API_KEY.

    POLYAI_LIVE=1 uv run pytest -m live
"""
import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("POLYAI_LIVE") != "1", reason="set POLYAI_LIVE=1 to call the real API"),
]


async def test_interrupted_reply_is_cut_and_recorded_as_heard():
    from caller import run

    room = await run("check-in")
    lines = room.desk.lines
    cuts = [text for _, who, text in lines if text.startswith("(barge-in")]
    cut_replies = [text for _, who, text in lines if who == "Agent" and text.endswith("[cut off]")]
    callers = [text.lower() for _, who, text in lines if who == "Caller"]

    assert cuts, "the caller's interruption never stopped the reply"
    assert cut_replies and cut_replies[0].rstrip(" [cutof]").endswith("-")
    assert not any("check-in is from" in c for c in callers)  # the agent never heard itself
    assert len([t for _, who, t in lines if who == "Agent"]) >= 3  # it answered the interruption and "go on"


async def test_first_words_are_heard_and_the_goodbye_plays_in_full():
    from pickup import run, summarise, words

    s = summarise(await run())

    assert words(s["first_line"])[:2] == ["hi", "yes"], s["first_line"]  # said before the session was ready
    assert s["heard"] >= s["of"] - 2, s["first_line"]  # allow a word or two transcribed differently
    assert s["goodbye_finished"] and not s["capped"]
    assert s["goodbye_played_s"] == s["goodbye_received_s"] > 1.0  # it hung up once the whole goodbye had played
