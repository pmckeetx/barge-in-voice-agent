"""A scripted call that tests both edges: the caller starts talking the moment the line opens, and says goodbye
at the end. No microphone or speakers: it plays samples/ through the simulated room from caller.py, on a real call.

    uv run pickup.py                # buffer from pickup, hang up once the goodbye has played
    uv run pickup.py --before       # drop audio until the session is ready, hang up a second into the goodbye
    uv run pickup.py --talk-over    # the caller cuts in half a second into the goodbye
    uv run pickup.py --runs 5       # repeat the call and count what was heard at each edge
"""
import argparse
import asyncio
import base64
import re
import statistics
import time

import websockets

from agent import FrontDesk, connect, refused
from audio import BYTES_PER_MS, Speaker
from call import Call
from caller import Room

LINE = "Hi, yes, I'm calling about my booking for Friday. The surname's Okafor."


class NaiveCall(Call):
    """The usual client: drops the caller's audio until the session is ready, and hangs up a second after the
    goodbye starts, once the speaker has played what it has."""

    async def send_audio(self, pcm: bytes):
        if self.live and not self.hung_up.is_set():
            await FrontDesk.send_audio(self, pcm)

    async def hang_up_after_goodbye(self):
        await asyncio.sleep(1.0)
        await self.speaker.drain()
        await self.hang_up()


class Counted:
    """Counts the goodbye's audio as it arrives, so the summary can say how much of it played."""

    goodbye_item = None
    goodbye_received = 0

    async def handle(self, event: dict):
        if event["type"] == "response.function_call_arguments.done" and event["name"] == "end_call":
            self.goodbye_item, self.goodbye_received = None, 0  # a new goodbye, after one the caller talked over
        if event["type"] == "response.output_audio.delta" and self.goodbye and event["response_id"] == self.goodbye["response_id"]:
            self.goodbye_item = event["item_id"]
            self.goodbye_received += len(base64.b64decode(event["delta"]))
        await super().handle(event)


class CountedCall(Counted, Call):
    pass


class CountedNaiveCall(Counted, NaiveCall):
    pass


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower().replace("’", "'"))


async def run(before: bool = False, talk_over: bool = False) -> Call:
    speaker = Speaker()
    desk = (CountedNaiveCall if before else CountedCall)(None, speaker)
    room = Room(desk, speaker, echo_gain=0.0)
    pickup = time.monotonic()
    tasks = [asyncio.create_task(room.run())]
    room.say("pickup")  # the caller starts talking the moment the line opens
    async with connect() as ws:
        desk.ws = ws
        desk.timings = {"open": time.monotonic() - pickup}
        await desk.start()
        desk.timings["ready"] = time.monotonic() - pickup
        tasks.append(asyncio.create_task(desk.listen()))
        await room.until_agent_done()
        room.say("thats-all")
        if talk_over:
            while not desk.goodbye_item:
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.5)
            room.say("one-more-thing")
            await asyncio.sleep(3.5)
            await room.until_agent_done()
            room.say("thats-all")
        await asyncio.wait_for(desk.hung_up.wait(), 60)
    for task in tasks:
        task.cancel()
    return desk


def summarise(desk: Call) -> dict:
    callers = [text for _, who, text in desk.lines if who == "Caller"]
    played = desk.speaker.played.get(desk.goodbye_item, 0) // BYTES_PER_MS
    return {
        "ready_s": round(desk.timings["ready"], 2),
        "first_line": callers[0] if callers else "",
        "heard": sum(w in set(words(callers[0] if callers else "")) for w in words(LINE)),
        "of": len(words(LINE)),
        "goodbye_played_s": round(played / 1000, 2),
        "goodbye_received_s": round(desk.goodbye_received // BYTES_PER_MS / 1000, 2),
        "goodbye_finished": desk.goodbye_spoken.is_set(),
        "capped": any("hanging up anyway" in text for _, _, text in desk.lines),
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", action="store_true", help="drop audio until ready, hang up after a fixed second")
    parser.add_argument("--talk-over", action="store_true", help="talk over the goodbye")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    results = []
    for i in range(args.runs):
        if args.runs > 1:
            print(f"\n=== run {i + 1} ===", flush=True)
        s = summarise(await run(args.before, args.talk_over))
        results.append(s)
        print(f"first line: heard {s['heard']} of {s['of']} words (session ready {s['ready_s']:.2f} s after pickup); "
              f"goodbye: played {s['goodbye_played_s']:.1f} s, "
              + ("all of it" if s["goodbye_finished"] else "then hung up while the server was still sending it")
              + (", capped" if s["capped"] else ""), flush=True)
    if args.runs > 1:
        heard = [s["heard"] for s in results]
        print(f"\n{args.runs} runs: first line heard in full in {sum(h == results[0]['of'] for h in heard)}, "
              f"median {statistics.median(heard)} of {results[0]['of']} words; "
              f"session ready a median {statistics.median(s['ready_s'] for s in results):.2f} s after pickup; "
              f"goodbye played in full in {sum(s['goodbye_finished'] and s['goodbye_played_s'] == s['goodbye_received_s'] for s in results)}, "
              f"median {statistics.median(s['goodbye_played_s'] for s in results):.1f} s played")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except websockets.InvalidStatus as error:
        raise SystemExit(refused(error))
