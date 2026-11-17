"""Record scripted calls as stereo WAVs on the playback clock, for bargein.py to score.

    uv run record.py --script check-in --runs 20      # runs/check-in-01.wav, its labels in runs/check-in-01.json
    uv run record.py --script mm-hmm --runs 10
    uv run record.py --script check-in --format pcmu  # the caller on a phone line: 8 kHz G.711 mu-law
    uv run bargein.py runs/*.wav

After each call it prints the stop the recording shows next to when speech_started arrived, on the same clock, so
you can check the recording against the server's events.
"""
import argparse
import asyncio
import base64
from pathlib import Path

import numpy as np
import websockets

from agent import FrontDesk, refused
from bargein import analyse
from caller import SCRIPTS, run
from recorder import StereoRecorder

RUNS = Path(__file__).parent / "runs"


def to_pcmu(pcm: np.ndarray) -> bytes:
    """16-bit PCM to G.711 mu-law, the encoding a phone trunk or a Twilio media stream carries."""
    x = pcm.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(x), 32635) + 0x84
    exponent = np.maximum(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


class PhoneDesk(FrontDesk):
    """The front desk on a phone line: the caller's audio goes in as 8 kHz mu-law. Replies still come back as PCM."""

    INPUT_FORMAT = {"type": "audio/pcmu"}

    async def send_audio(self, pcm: bytes):
        narrowband = np.frombuffer(pcm, np.int16).reshape(-1, 3).mean(axis=1)  # 24 kHz to 8 kHz
        await self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(to_pcmu(narrowband)).decode()})


def recorded(desk: type[FrontDesk], recorder: StereoRecorder) -> type[FrontDesk]:
    """The same desk, noting on the recording's clock when each speech_started arrives."""

    class Recorded(desk):
        async def handle(self, event: dict):
            if event["type"] == "input_audio_buffer.speech_started":
                recorder.mark("speech_started")
            await super().handle(event)

    return Recorded


def next_path(name: str) -> Path:
    RUNS.mkdir(exist_ok=True)
    n = 1
    while (RUNS / f"{name}-{n:02}.wav").exists():
        n += 1
    return RUNS / f"{name}-{n:02}.wav"


def compare(wav: Path, recorder: StereoRecorder) -> list[str]:
    """For every overlap, the recording's stop beside the first speech_started after the caller started."""
    starts = [e["t_s"] for e in recorder.events if e["type"] == "speech_started"]
    out = []
    for o in analyse(wav):
        heard = next((t for t in starts if t >= o.onset_s - 0.1), None)
        what = f"{o.line} ({o.intent})" if o.intent else "caller"
        how = {"stopped": "stopped after", "talked on": "talked on for", "ended": "ended after"}[o.result]
        line = f"  {what}: {how} {o.talked_s:.2f} s"
        if heard is None:
            line += "; no speech_started"
        else:
            line += f"; speech_started arrived after {heard - o.onset_s:.2f} s (recording minus events {o.onset_s + o.talked_s - heard:+.2f} s)"
        out.append(line)
    return out


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--script", choices=SCRIPTS, default="check-in")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--format", choices=["pcm", "pcmu"], default="pcm", help="pcmu: the caller on a phone line")
    args = parser.parse_args()
    phone = args.format == "pcmu"
    for _ in range(args.runs):
        recorder = StereoRecorder(rate=8000 if phone else 24000)
        await run(args.script, client=recorded(PhoneDesk if phone else FrontDesk, recorder), recorder=recorder)
        wav = next_path(args.script + ("-pcmu" if phone else ""))
        recorder.save(wav, script=args.script, format=args.format)
        print(wav.relative_to(Path.cwd()) if wav.is_relative_to(Path.cwd()) else wav, flush=True)
        print("\n".join(compare(wav, recorder)) or "  no overlaps", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except websockets.InvalidStatus as error:
        raise SystemExit(refused(error))
