"""The front desk at the Harborview Inn: a voice agent on Dialog-RSN-1 that handles barge-in properly.

When the caller talks over it, it stops playback as soon as the server confirms it, tells the server how much of the reply was heard, keeps
its own record from what was heard, and doesn't let a tool that finishes late talk over the caller's new question.

    uv run agent.py              # talk to it on your microphone, with headphones on
    uv run agent.py --speakers   # on laptop speakers: cancel the echo first
    uv run caller.py         # a scripted caller that interrupts it, no microphone needed
"""
import argparse
import asyncio
import base64
import json
import os
import time

import websockets
from dotenv import load_dotenv

from audio import EchoCanceller, SoundCard, Speaker

load_dotenv()

URL = os.environ.get("DIALOG_REALTIME_URL", "wss://api.studio.poly.ai/v1/realtime")

INSTRUCTIONS = """You are the front desk at the Harborview Inn. Everything you say is spoken aloud, so use no lists or symbols.
When asked about checking in, give the whole rundown in one reply: check-in is from 3 pm, bring photo ID and the card
you booked with, there's a 100 dollar deposit, breakfast is 7 to 10 in the Atrium, the pool is on floor 2, and the
wifi is HarborGuest with no password. Parking is valet only, 40 dollars a night.
Use look_up_booking for anything about a guest's booking. Keep other replies to one or two sentences."""

TOOLS = [{
    "type": "function",
    "name": "look_up_booking",
    "description": "Find a guest's booking by surname.",
    "parameters": {
        "type": "object",
        "properties": {
            "surname": {"type": "string"},
            "say_while_waiting": {
                "type": "string",
                "poly_spoken": True,  # the server speaks this argument while the tool runs
                "description": "A sentence to say while you look, like 'Let me pull up your booking, the system can take a moment to load.'",
            },
        },
        "required": ["surname", "say_while_waiting"],
    },
}]


async def look_up_booking(surname: str) -> dict:
    await asyncio.sleep(3)  # a slow booking system
    return {"surname": surname, "room": "214", "nights": 2, "check_in": "Friday"}


class FrontDesk:
    def __init__(self, ws, speaker: Speaker, echo: EchoCanceller | None = None):
        self.ws = ws
        self.speaker = speaker
        self.echo = echo
        self.written: dict[str, str] = {}  # item_id -> the full text of a reply, heard or not
        self.caller_turns = 0  # bumped every time the caller starts talking
        self.responding = False
        self.tasks: set[asyncio.Task] = set()
        self.lines: list[tuple[float, str, str]] = []  # (time, who, text): everything log() printed

    async def send(self, event: dict):
        await self.ws.send(json.dumps(event))

    async def start(self):
        await self.ws.recv()  # session.created
        await self.send({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": INSTRUCTIONS,
                "tools": TOOLS,
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": {"type": "server_vad"}}},
            },
        })
        updated = json.loads(await self.ws.recv())
        if updated["type"] != "session.updated":
            raise RuntimeError(updated.get("error", {}).get("message", f"unexpected {updated['type']}"))

    async def send_audio(self, pcm: bytes):
        if self.echo:
            pcm = self.echo.clean(pcm)
        if pcm:
            await self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})

    async def cut_off(self):
        """Stop the agent's voice now, then tell the server how much of each reply the caller heard."""
        for item_id, played_ms in self.speaker.stop().items():
            await self.send({
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": played_ms,
            })
            self.log("", f"(barge-in: stopped the reply after {played_ms / 1000:.1f} s)")

    async def handle(self, event: dict):
        match event["type"]:
            case "response.output_audio.delta":
                self.speaker.play(event["item_id"], base64.b64decode(event["delta"]))
            case "input_audio_buffer.speech_started":
                self.caller_turns += 1
                await self.cut_off()  # if the agent was quiet, there's nothing to stop
            case "conversation.item.input_audio_transcription.completed":
                self.log("Caller", event["transcript"])
            case "response.output_audio_transcript.done":
                self.written[event["item_id"]] = event["transcript"]
            case "response.created":
                self.responding = True
            case "response.done":
                self.responding = False
                if event["response"]["status"] == "cancelled":
                    await self.cut_off()  # this can arrive before speech_started
                await self.record(event["response"])
            case "conversation.item.retrieved" if self.text_of(event["item"]):
                self.log("Agent", self.text_of(event["item"]) + "  [cut off]")
            case "response.function_call_arguments.done":
                task = asyncio.create_task(self.run_tool(event))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
            case "error":
                self.log("Error", event["error"]["message"])

    async def record(self, response: dict):
        """Log what the caller heard. A cut-off reply's full text never reached them, so ask the server for the heard part."""
        for item in response.get("output", []):
            if item["type"] != "message":
                continue
            if response["status"] == "cancelled":
                await self.send({"type": "conversation.item.retrieve", "item_id": item["id"]})
            else:
                self.log("Agent", self.written.get(item["id"], self.text_of(item)))

    async def run_tool(self, call: dict):
        turns_before = self.caller_turns
        args = json.loads(call["arguments"])
        result = await look_up_booking(args["surname"])
        self.log("", f"(look_up_booking returned for {args['surname']})")
        await self.send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(result)},
        })
        if self.caller_turns == turns_before:
            await self.send({"type": "response.create"})  # nobody spoke while it ran: answer with the result
        # Otherwise the caller has moved on. The result stays in history, and their new turn gets its own reply.

    @staticmethod
    def text_of(item: dict) -> str:
        return " ".join(c.get("transcript") or c.get("text") or "" for c in item.get("content", []))

    def log(self, who: str, text: str):
        self.lines.append((time.monotonic(), who, text))
        print(f"{who + ':' if who else '':8}{text}", flush=True)

    async def listen(self):
        async for raw in self.ws:
            await self.handle(json.loads(raw))


def connect():
    key = os.environ.get("POLY_API_KEY")
    if not key:
        raise SystemExit("POLY_API_KEY isn't set. Copy .env.example to .env and add your key from studio.poly.ai.")
    return websockets.connect(URL, additional_headers={"Authorization": f"Bearer {key}"})


async def call(speakers: bool):
    """A live call on your microphone. Ctrl+C to hang up."""
    speaker = Speaker()
    echo = EchoCanceller() if speakers else None  # on headphones there's no echo to cancel, and it would hide the caller
    if echo:
        speaker.on_played = echo.played  # the canceller has to know what the speaker played
    async with connect() as ws:
        with SoundCard(speaker) as card:
            desk = FrontDesk(ws, speaker, echo)
            await desk.start()
            print("Connected to the Harborview Inn. Ask about checking in, then talk over the reply. Ctrl+C to hang up.")

            async def stream_mic():
                while True:
                    await desk.send_audio(await card.queue.get())

            await asyncio.gather(stream_mic(), desk.listen())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--speakers", action="store_true", help="you're on speakers, not headphones: cancel the echo")
    try:
        asyncio.run(call(parser.parse_args().speakers))
    except KeyboardInterrupt:
        print("\nHung up.")
