"""Run a script several times against the real API and summarise how barge-in went.

    uv run measure.py                              # 5 runs of the check-in call, barge-in handled
    uv run measure.py --runs 3 --no-truncate       # stop playback but never truncate: the server's own trim only
    uv run measure.py --echo 0.3 --before          # speaker echo and no handling
    uv run measure.py --aec on                     # the echo canceller with no echo, as on headphones

For each run: how long after the caller started talking over the reply the playback stopped, where the reply was
cut, how often the agent's own voice came back as a caller turn, whether the answer to the interruption repeated
words the caller had already heard, and whether the answer to "go on" started over.
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


class NoTruncate(agent.FrontDesk):
    """Stops playback on a barge-in but never sends conversation.item.truncate."""

    async def cut_off(self):
        for item_id, played_ms in self.speaker.stop().items():
            self.log("", f"(barge-in: stopped the reply after {played_ms / 1000:.1f} s, no truncate)")


def summarise(room) -> dict:
    lines = room.desk.lines
    interrupt = next(t for t, who, text in lines if text.startswith("(caller says") and "check-in" not in text and "booking" not in text)
    stops = [text for _, who, text in lines if text.startswith("(barge-in")]
    replies = [text for _, who, text in lines if who == "Agent"]
    callers = [text for _, who, text in lines if who == "Caller"]
    heard = next((t for t in starts if t > interrupt), None)
    cut = next((i for i, r in enumerate(replies) if r.endswith("[cut off]")), None)
    answer = next((r for r in replies[cut + 1:] if not r.endswith("[cut off]")), "") if cut is not None else ""
    return {
        "stopped_after_s": round(heard - interrupt, 2) if heard else None,
        "cuts": [s.split("after ")[1].split(" s")[0] + " s" for s in stops],
        "own_voice_as_caller": sum("check-in is" in c.lower() or "valet" in c.lower() or "parking is" in c.lower() for c in callers),
        "answer_restates": any(p in answer.lower() for p in ("check-in is", "photo id", "card you booked")),
        "last_reply_starts_over": bool(replies) and "check-in is from" in replies[-1].lower(),
        "last_reply": replies[-1][:80] if replies else "",
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--script", choices=SCRIPTS, default="check-in")
    parser.add_argument("--echo", type=float, default=0.0)
    parser.add_argument("--before", action="store_true")
    parser.add_argument("--no-truncate", action="store_true", help="stop playback, but never truncate")
    parser.add_argument("--aec", choices=["auto", "on", "off"], default="auto", help="the echo canceller: auto is on when --echo > 0")
    args = parser.parse_args()
    delays, restated = [], 0
    for i in range(args.runs):
        starts.clear()
        room = await run(args.script, args.echo, args.before, {"auto": None, "on": True, "off": False}[args.aec],
                         client=NoTruncate if args.no_truncate else None)
        s = summarise(room)
        print(f"RUN {i + 1}: {s}", flush=True)
        restated += s["answer_restates"]
        if s["stopped_after_s"] is not None:
            delays.append(s["stopped_after_s"])
    if delays:
        print(f"stopped after the caller started talking: median {statistics.median(delays):.2f} s, "
              f"range {min(delays):.2f} to {max(delays):.2f} s, {len(delays)} runs; "
              f"answer repeated what was heard in {restated} of {args.runs}")


if __name__ == "__main__":
    asyncio.run(main())
