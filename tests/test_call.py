"""Both edges of a call against a fake server that is slow to get ready and talks at speaking pace."""
import asyncio

import numpy as np
import pytest

import agent
import call
from agent import connect
from audio import BYTES_PER_MS, Speaker
from call import Call
from pickup import NaiveCall

GOODBYE = "Thank you, Ms Okafor. Enjoy your stay with us on Friday, and goodbye."
TALK = np.full(480, 4000, np.int16).tobytes()  # 20 ms of a loud caller
QUIET = bytes(960)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("POLY_API_KEY", "test-key")


def frame(n: int) -> bytes:
    """20 ms of quiet audio with its number written into every sample, so the test can see order and loss."""
    return np.full(480, n + 1, np.int16).tobytes()


def number(pcm: bytes) -> int:
    return int(np.frombuffer(pcm, np.int16)[0]) - 1


async def talk_from_pickup(desk: Call):
    """A microphone that starts the moment the line opens: one numbered frame every 20 ms."""
    n = 0
    while True:
        await desk.send_audio(frame(n))
        n += 1
        await asyncio.sleep(0.02)


async def play(speaker: Speaker):
    """A sound card: plays 20 ms of the agent's voice every 20 ms."""
    while True:
        speaker.pull(20 * BYTES_PER_MS)
        await asyncio.sleep(0.02)


async def until(condition, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > end:
            raise TimeoutError
        await asyncio.sleep(0.01)


async def caller_says(desk):
    for _ in range(10):
        await desk.send_audio(TALK)
    for _ in range(20):
        await desk.send_audio(QUIET)


class DrainCall(Call):
    """Sleeps a second after end_call, then hangs up as soon as the speaker is empty."""

    async def hang_up_after_goodbye(self):
        await asyncio.sleep(1.0)
        await self.speaker.drain()
        await self.hang_up()


async def open_call(fake, monkeypatch, speaker, client=Call):
    monkeypatch.setattr(agent, "URL", fake.url)
    desk = client(None, speaker)
    desk.ws = await connect()
    await desk.start()
    return desk, asyncio.create_task(desk.listen())


@pytest.mark.parametrize("client", [Call, NaiveCall])
async def test_caller_audio_from_pickup_reaches_the_server_in_order(monkeypatch, client):
    from fake_realtime import FakeRealtime

    async with FakeRealtime([], ready_after=0.3) as fake:  # 0.6 s from pickup to session.updated
        monkeypatch.setattr(agent, "URL", fake.url)
        desk = client(None, Speaker())
        mic = asyncio.create_task(talk_from_pickup(desk))
        async with connect() as ws:
            desk.ws = ws
            await desk.start()
            await asyncio.sleep(0.2)
            mic.cancel()

    numbers = [number(pcm) for _, pcm in fake.audio]
    assert all(ready for ready, _ in fake.audio)  # nothing reached the server before session.updated
    if client is Call:
        assert numbers == list(range(len(numbers)))  # from the first frame, in order, none lost
    else:
        assert numbers[0] >= 25  # the naive client lost the first half second or more
        assert numbers == list(range(numbers[0], numbers[0] + len(numbers)))


async def test_buffer_keeps_the_last_five_seconds(monkeypatch):
    from fake_realtime import FakeRealtime

    desk = Call(None, Speaker())
    for n in range(300):  # 6 s of the caller, held before there's a connection
        await desk.send_audio(frame(n))
    async with FakeRealtime([]) as fake:
        monkeypatch.setattr(agent, "URL", fake.url)
        async with connect() as ws:
            desk.ws = ws
            await desk.start()
            await asyncio.sleep(0.1)

    assert [number(pcm) for _, pcm in fake.audio] == list(range(50, 300))  # the oldest second went


async def test_hang_up_waits_until_the_goodbye_has_played(monkeypatch):
    from fake_realtime import FakeRealtime

    async with FakeRealtime([("goodbye", GOODBYE, 2000)], pace=True) as fake:
        speaker = Speaker()
        desk, listener = await open_call(fake, monkeypatch, speaker)
        card = asyncio.create_task(play(speaker))
        await caller_says(desk)  # "that's everything, thanks"
        await asyncio.wait_for(desk.hung_up.wait(), 5)
        card.cancel()

        assert speaker.played["item_1"] == 2000 * BYTES_PER_MS  # every millisecond of the goodbye played
        assert not speaker.busy
        assert fake.sent("response.create") == []  # nothing new was asked for after the goodbye
        await until(lambda: listener.done())  # the socket is closed, so listen() has returned


async def test_a_fixed_sleep_cuts_the_goodbye(monkeypatch):
    from fake_realtime import FakeRealtime

    async with FakeRealtime([("goodbye", GOODBYE, 4000)], pace=True) as fake:
        speaker = Speaker()
        desk, listener = await open_call(fake, monkeypatch, speaker, NaiveCall)
        card = asyncio.create_task(play(speaker))
        await caller_says(desk)
        await asyncio.wait_for(desk.hung_up.wait(), 5)
        card.cancel()

        assert speaker.played["item_1"] < 1500 * BYTES_PER_MS  # about a second of a four-second goodbye


@pytest.mark.parametrize("client", [Call, DrainCall])
async def test_a_pause_in_the_stream_isnt_the_end_of_the_goodbye(monkeypatch, client):
    from fake_realtime import FakeRealtime

    async with FakeRealtime([("goodbye", GOODBYE, 3000)], pace=True, stall=(1500, 0.5)) as fake:
        speaker = Speaker()
        desk, listener = await open_call(fake, monkeypatch, speaker, client)
        card = asyncio.create_task(play(speaker))
        await caller_says(desk)
        await asyncio.wait_for(desk.hung_up.wait(), 6)
        card.cancel()

        played = speaker.played["item_1"] // BYTES_PER_MS
        if client is Call:
            assert played == 3000  # it waited for response.done, so the pause didn't matter
        else:
            assert played < 2000  # the speaker ran dry during the pause, and that looked like the end


async def test_hang_up_gives_up_when_the_goodbye_never_finishes(monkeypatch):
    from fake_realtime import FakeRealtime

    monkeypatch.setattr(call, "HANG_UP_CAP_S", 0.5)
    async with FakeRealtime([("goodbye-stalls", GOODBYE, 300)]) as fake:
        desk, listener = await open_call(fake, monkeypatch, Speaker())
        await caller_says(desk)
        await until(lambda: desk.goodbye)
        await asyncio.wait_for(desk.hung_up.wait(), 1.5)

        assert any("hanging up anyway" in text for _, _, text in desk.lines)
        listener.cancel()


async def test_talking_over_the_goodbye_keeps_the_line_open(monkeypatch):
    from fake_realtime import FakeRealtime

    async with FakeRealtime([("goodbye", GOODBYE, 2000), ("say", "Breakfast is 7 to 10 in the Atrium.", 500)], pace=True) as fake:
        speaker = Speaker()
        desk, listener = await open_call(fake, monkeypatch, speaker)
        card = asyncio.create_task(play(speaker))
        await caller_says(desk)  # "that's everything, thanks"
        await until(lambda: speaker.played.get("item_1", 0) >= 500 * BYTES_PER_MS)
        await caller_says(desk)  # "oh, wait, one more thing"
        await until(lambda: fake.sent("conversation.item.create"))
        await asyncio.sleep(0.3)
        card.cancel()

        output = fake.sent("conversation.item.create")[0]["item"]
        assert output["type"] == "function_call_output" and output["call_id"] == "call_1"
        assert '"hung_up": false' in output["output"]
        assert desk.hanging_up.cancelled()
        assert not desk.hung_up.is_set()
        assert fake.sent("response.create") == []  # the caller's new turn gets its own reply
        listener.cancel()
        await desk.ws.close()
