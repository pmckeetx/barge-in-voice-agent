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


async def test_a_recorded_barge_in_is_scored_as_stopped(tmp_path):
    from bargein import analyse
    from caller import run
    from record import FrontDesk, recorded
    from recorder import StereoRecorder

    recorder = StereoRecorder()
    await run("check-in", client=recorded(FrontDesk, recorder), recorder=recorder)
    recorder.save(tmp_path / "call.wav", script="check-in")

    [o] = [o for o in analyse(tmp_path / "call.wav") if o.line == "parking"]
    assert o.verdict == "stopped" and 0.2 < o.talked_s < 2.0
    heard = next(e["t_s"] for e in recorder.events if e["type"] == "speech_started" and e["t_s"] >= o.onset_s)
    # The recording agrees with the server's event: the stop lands on the tick after speech_started, or earlier
    # when the agent was between words at that moment.
    assert -0.4 < o.onset_s + o.talked_s - heard < 0.1


async def test_first_words_are_heard_and_the_goodbye_plays_in_full():
    from pickup import run, summarise

    s = summarise(await run())

    assert s["lost"] == 0, s["first_line"]  # "Hi, yes" was said before the session was ready, and it was heard
    assert s["goodbye_finished"] and not s["capped"]
    assert s["goodbye_played_s"] == s["goodbye_received_s"] > 1.0  # it hung up once the whole goodbye had played
