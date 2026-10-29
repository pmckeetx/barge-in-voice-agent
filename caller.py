"""A scripted caller for the front desk: no microphone, no speakers, a real call to Dialog-RSN-1.

It plays the WAVs in samples/ as the caller and interrupts the agent mid-reply. The agent's audio plays into a
simulated room at real time, and --echo mixes it back into the caller's microphone at that volume, the way laptop
speakers leak into a laptop mic.

    uv run caller.py                         # the agent with barge-in handled
    uv run caller.py --echo 0.3 --before     # speakers leaking into the mic, nothing handled
    uv run caller.py --echo 0.3              # the same room, with the barge-in handling and the echo canceller
    uv run caller.py --script booking        # interrupt the agent while a tool is running
"""
import argparse
import asyncio
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from agent import FrontDesk, connect
from audio import RATE, EchoCanceller, Speaker

SAMPLES = Path(__file__).parent / "samples"
TICK = 0.02
CHUNK = int(RATE * TICK) * 2
ECHO_DELAY_MS = 40  # speaker to mic, through the device buffers and the air

# Each step: the line the caller says, and when. None waits for the agent to finish its reply first; a number
# interrupts that reply that many seconds after it starts.
SCRIPTS = {
    "check-in": [("check-in", None), ("parking", 4.0), ("go-on", None)],
    "booking": [("booking", None), ("never-mind", 0.3)],
}


class Before(FrontDesk):
    """A client with no barge-in handling: it keeps playing, never truncates, and logs every reply in full."""

    async def cut_off(self):
        pass

    async def record(self, response: dict):
        for item in response.get("output", []):
            if item["type"] == "message":
                self.log("Agent", self.written.get(item["id"], self.text_of(item)))


def read(name: str) -> bytes:
    with wave.open(str(SAMPLES / f"{name}.wav")) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1
        return w.readframes(w.getnframes())


class Room:
    """Plays the agent's audio at real time and builds what the caller's microphone hears."""

    def __init__(self, desk: FrontDesk, speaker: Speaker, echo_gain: float):
        self.desk, self.speaker, self.echo_gain = desk, speaker, echo_gain
        self.voice = bytearray()  # the caller's line, still to say
        self.air = deque([np.zeros(CHUNK // 2)] * (ECHO_DELAY_MS // 20))  # sound takes a moment to reach the mic
        self.last_heard = 0.0  # when the agent was last audible
        self.recording = bytearray()  # both voices as the room heard them, for --save

    async def run(self):
        start, n = time.monotonic(), 0
        while True:
            played = np.frombuffer(self.speaker.pull(CHUNK), dtype=np.int16)
            if np.abs(played).max() > 200:
                self.last_heard = time.monotonic()
            said, self.voice[:] = bytes(self.voice[:CHUNK]).ljust(CHUNK, b"\0"), self.voice[CHUNK:]
            self.air.append(played.astype(np.float32))
            mic = np.frombuffer(said, dtype=np.int16).astype(np.float32) + self.air.popleft() * self.echo_gain
            await self.desk.send_audio(mic.clip(-32768, 32767).astype(np.int16).tobytes())
            both = played.astype(np.float32) + np.frombuffer(said, dtype=np.int16)
            self.recording.extend(both.clip(-32768, 32767).astype(np.int16).tobytes())
            n += 1
            await asyncio.sleep(max(0, start + n * TICK - time.monotonic()))  # keep real time

    def say(self, name: str):
        self.desk.log("", f"(caller says {name}.wav)")
        self.voice.extend(read(name))

    async def until_agent_speaks(self, timeout=30):
        end = time.monotonic() + timeout
        while not self.speaker.busy and time.monotonic() < end:
            await asyncio.sleep(TICK)

    async def until_agent_done(self, timeout=60):
        end = time.monotonic() + timeout
        await self.until_agent_speaks()
        while time.monotonic() < end and (self.desk.responding or self.speaker.busy or self.desk.tasks or time.monotonic() - self.last_heard < 2.0):
            await asyncio.sleep(TICK)


async def run(script: str, echo_gain: float = 0.0, before: bool = False, aec: bool | None = None) -> Room:
    """Play one script against the real API and return the room, with everything it heard.

    The echo canceller is on when the room has echo (the agent is on speakers), unless aec says otherwise.
    """
    speaker = Speaker()
    aec = echo_gain > 0 if aec is None else aec
    echo = EchoCanceller() if aec and not before else None
    if echo:
        speaker.on_played = echo.played
    async with connect() as ws:
        desk = (Before if before else FrontDesk)(ws, speaker, echo)
        await desk.start()
        room = Room(desk, speaker, echo_gain)
        tasks = [asyncio.create_task(room.run()), asyncio.create_task(desk.listen())]
        await asyncio.sleep(0.5)
        for i, (line, interrupt_after) in enumerate(SCRIPTS[script]):
            if i == 0:
                pass  # the agent doesn't greet: the caller speaks first
            elif interrupt_after is None:
                await room.until_agent_done()
            else:
                await room.until_agent_speaks()
                await asyncio.sleep(interrupt_after)
            room.say(line)
            await asyncio.sleep(len(read(line)) / (RATE * 2))
        await room.until_agent_done()
        for task in tasks:
            task.cancel()
    return room


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--script", choices=SCRIPTS, default="check-in")
    parser.add_argument("--echo", type=float, default=0.0, help="how loud the agent leaks back into the mic, 0 to 1")
    parser.add_argument("--before", action="store_true", help="no echo canceller and no barge-in handling")
    parser.add_argument("--save", help="write the call as the room heard it, both voices, to this WAV")
    args = parser.parse_args()

    room = await run(args.script, args.echo, args.before)
    if args.save:
        with wave.open(args.save, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(bytes(room.recording))


if __name__ == "__main__":
    asyncio.run(main())
