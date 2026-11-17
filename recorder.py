"""Record a call as a stereo WAV on the playback clock: the caller on the left, the agent on the right.

Both channels are written from the same 20 ms tick of the room, so a sample on the left and the sample beside it
on the right happened at the same moment for the caller. A JSON file next to the WAV says what the caller said,
when, and what they meant by it.
"""
import json
import wave
from pathlib import Path

import numpy as np


class StereoRecorder:
    def __init__(self, rate: int = 24000, room_rate: int = 24000):
        self.rate, self.step = rate, room_rate // rate  # step 3 writes 24 kHz room audio as an 8 kHz phone call
        self.frames = bytearray()
        self.lines: list[dict] = []  # what the caller said, when, and why
        self.events: list[dict] = []  # server events, on the same clock

    def add(self, caller: bytes, agent: bytes):
        """One tick: what the agent heard from the caller, and what the caller heard from the agent."""
        pair = np.column_stack([np.frombuffer(caller, np.int16), np.frombuffer(agent, np.int16)])
        if self.step > 1:
            pair = pair.reshape(-1, self.step, 2).mean(axis=1).astype(np.int16)
        self.frames.extend(pair.tobytes())

    def now(self) -> float:
        """Seconds of call recorded so far: the recording's clock."""
        return len(self.frames) / 4 / self.rate

    def label(self, line: str, intent: str | None):
        self.lines.append({"line": line, "onset_s": round(self.now(), 3), "intent": intent})

    def mark(self, event: str):
        self.events.append({"type": event, "t_s": round(self.now(), 3)})

    def save(self, path: Path, **info):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(self.rate)
            w.writeframes(bytes(self.frames))
        sidecar = {**info, "rate": self.rate, "channels": ["caller", "agent"], "lines": self.lines, "events": self.events}
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
