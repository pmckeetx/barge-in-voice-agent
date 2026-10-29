"""Run a script several times against the real API and summarise how barge-in went.

    uv run measure.py                        # 5 runs of the check-in call, barge-in handled
    uv run measure.py --echo 0.3 --before    # the same with speaker echo and no handling

For each run: how long after the caller started talking over the reply the playback stopped, where the reply was
cut, how often the agent's own voice came back as a caller turn, and whether its answer to "go on" started over.
"""
import argparse
import asyncio
import statistics
import time

import agent
from caller import SCRIPTS, run

starts: list[float] = []  # when each speech_started arrived
_handle = agent.FrontDesk.handle


async def handle(self, event):
    if event["type"] == "input_audio_buffer.speech_started":
        starts.append(time.monotonic())
    await _handle(self, event)


agent.FrontDesk.handle = handle


def summarise(room) -> dict:
    lines = room.desk.lines
    interrupt = next(t for t, who, text in lines if text.startswith("(caller says") and "check-in" not in text and "booking" not in text)
    stops = [text for _, who, text in lines if text.startswith("(barge-in")]
    replies = [text for _, who, text in lines if who == "Agent"]
    callers = [text for _, who, text in lines if who == "Caller"]
    heard = next((t for t in starts if t > interrupt), None)
    return {
        "stopped_after_s": round(heard - interrupt, 2) if heard else None,
        "cuts": [s.split("after ")[1].rstrip(")") for s in stops],
        "own_voice_as_caller": sum("check-in is" in c.lower() or "valet" in c.lower() or "parking is" in c.lower() for c in callers),
        "last_reply_starts_over": bool(replies) and "check-in is from" in replies[-1].lower(),
        "last_reply": replies[-1][:120] if replies else "",
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--script", choices=SCRIPTS, default="check-in")
    parser.add_argument("--echo", type=float, default=0.0)
    parser.add_argument("--before", action="store_true")
    parser.add_argument("--aec", choices=["auto", "on", "off"], default="auto", help="the echo canceller: auto is on when --echo > 0")
    args = parser.parse_args()
    delays = []
    for i in range(args.runs):
        starts.clear()
        room = await run(args.script, args.echo, args.before, {"auto": None, "on": True, "off": False}[args.aec])
        s = summarise(room)
        print(f"RUN {i + 1}: {s}", flush=True)
        if s["stopped_after_s"] is not None:
            delays.append(s["stopped_after_s"])
    if delays:
        print(f"stopped after the caller started talking: median {statistics.median(delays):.2f} s, range {min(delays):.2f} to {max(delays):.2f} s, {len(delays)} runs")


if __name__ == "__main__":
    asyncio.run(main())
