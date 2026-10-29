"""The front desk against a fake realtime server: no key, no network."""
import asyncio

import numpy as np
import pytest

import agent
from agent import FrontDesk, connect
from audio import BYTES_PER_MS, Speaker
from fake_realtime import FakeRealtime

CHECK_IN = "Check-in is from 3 pm. Please bring your photo ID and the card you booked with. There's a 100 dollar deposit."
TALK = np.full(480, 4000, np.int16).tobytes()  # 20 ms of a loud caller
QUIET = bytes(960)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("POLY_API_KEY", "test-key")


async def until(condition, timeout=3.0):
    end = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > end:
            raise TimeoutError
        await asyncio.sleep(0.01)


async def open_desk(fake, monkeypatch, speaker):
    monkeypatch.setattr(agent, "URL", fake.url)
    ws = await connect()
    desk = FrontDesk(ws, speaker)
    await desk.start()
    listener = asyncio.create_task(desk.listen())
    return ws, desk, listener


async def caller_says(desk, chunks=10):
    for _ in range(chunks):
        await desk.send_audio(TALK)
    for _ in range(20):
        await desk.send_audio(QUIET)


async def test_barge_in_stops_playback_and_truncates_to_what_played(monkeypatch):
    async with FakeRealtime([("say", CHECK_IN, 3000), ("say", "Valet only.", 1000)]) as fake:
        speaker = Speaker()
        ws, desk, listener = await open_desk(fake, monkeypatch, speaker)
        await caller_says(desk)  # "tell me about checking in"
        await until(lambda: len(speaker.queue) and len(speaker.queue[0][1]) >= 2000 * BYTES_PER_MS)
        speaker.pull(1200 * BYTES_PER_MS)  # 1.2 s reaches the caller's ear
        await desk.send_audio(TALK)  # and they talk over it
        await until(lambda: fake.sent("conversation.item.truncate"))

        assert not speaker.busy  # playback stopped at once
        truncate = fake.sent("conversation.item.truncate")[0]
        assert truncate["item_id"] == "item_1"
        assert truncate["audio_end_ms"] == 1200
        listener.cancel()
        await ws.close()


async def test_cut_off_reply_is_recorded_as_heard_not_as_written(monkeypatch):
    async with FakeRealtime([("say", CHECK_IN, 3000)]) as fake:
        speaker = Speaker()
        ws, desk, listener = await open_desk(fake, monkeypatch, speaker)
        await caller_says(desk)
        await until(lambda: len(speaker.queue) and len(speaker.queue[0][1]) >= 2000 * BYTES_PER_MS)
        speaker.pull(1000 * BYTES_PER_MS)
        await desk.send_audio(TALK)
        await until(lambda: any(who == "Agent" for _, who, _ in desk.lines))

        said = [text for _, who, text in desk.lines if who == "Agent"]
        assert said == [CHECK_IN[: round(len(CHECK_IN) / 3)].rstrip() + "-  [cut off]"]
        assert desk.written["item_1"] == CHECK_IN  # the full text arrived, and was never logged as said
        listener.cancel()
        await ws.close()


async def test_caller_speaking_while_the_agent_is_quiet_truncates_nothing(monkeypatch):
    async with FakeRealtime([("say", "Hello.", 500)]) as fake:
        speaker = Speaker()
        ws, desk, listener = await open_desk(fake, monkeypatch, speaker)
        await caller_says(desk)
        await until(lambda: fake.sent("conversation.item.truncate") or any(w == "Agent" for _, w, _ in desk.lines))
        assert fake.sent("conversation.item.truncate") == []
        listener.cancel()
        await ws.close()


async def test_tool_result_gets_a_reply_when_the_caller_waited(monkeypatch):
    monkeypatch.setattr(agent, "look_up_booking", lambda surname: asyncio.sleep(0.05, {"room": "214"}))
    async with FakeRealtime([("tool", "look_up_booking", {"surname": "Okafor", "say_while_waiting": "One moment."}),
                             ("say", "You're in room 214.", 500)]) as fake:
        speaker = Speaker()
        ws, desk, listener = await open_desk(fake, monkeypatch, speaker)
        await caller_says(desk)
        await until(lambda: fake.sent("response.create"))
        outputs = fake.sent("conversation.item.create")
        assert outputs[0]["item"]["type"] == "function_call_output"
        assert '"room": "214"' in outputs[0]["item"]["output"]
        listener.cancel()
        await ws.close()


async def test_late_tool_result_waits_when_the_caller_moved_on(monkeypatch):
    release = asyncio.Event()

    async def slow_lookup(surname):
        await release.wait()
        return {"room": "214"}

    monkeypatch.setattr(agent, "look_up_booking", slow_lookup)
    async with FakeRealtime([("tool", "look_up_booking", {"surname": "Okafor", "say_while_waiting": "One moment."}),
                             ("say", "Breakfast is 7 to 10.", 500)]) as fake:
        speaker = Speaker()
        ws, desk, listener = await open_desk(fake, monkeypatch, speaker)
        await caller_says(desk)  # "look up my booking"
        await until(lambda: desk.tasks)
        await caller_says(desk)  # "never mind, is breakfast included?" while the lookup runs
        await until(lambda: desk.caller_turns == 2)
        release.set()
        await until(lambda: fake.sent("conversation.item.create"))
        await asyncio.sleep(0.1)

        assert fake.sent("conversation.item.create")[0]["item"]["type"] == "function_call_output"
        assert fake.sent("response.create") == []  # the result waits in history; it doesn't talk over the new question
        listener.cancel()
        await ws.close()


async def test_refused_session_update_raises(monkeypatch):
    class Refuses(FakeRealtime):
        async def handler(self, ws):
            await ws.send('{"type": "session.created", "session": {}}')
            await ws.recv()
            await ws.send('{"type": "error", "error": {"message": "unknown parameter"}}')
            await asyncio.sleep(0.1)

    async with Refuses([]) as fake:
        monkeypatch.setattr(agent, "URL", fake.url)
        async with connect() as ws:
            with pytest.raises(RuntimeError, match="unknown parameter"):
                await FrontDesk(ws, Speaker()).start()
