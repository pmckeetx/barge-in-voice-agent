"""The Harborview Inn front desk with both edges of the call handled: it hears the caller's first words, and it
hangs up only once its goodbye has played.

    uv run call.py              # a call on your microphone, with headphones on: start talking as soon as it runs
    uv run call.py --speakers   # on laptop speakers, with the echo canceller on
    uv run pickup.py            # a scripted caller that talks from the first frame, no microphone needed
"""
import argparse
import asyncio
import json
from collections import deque

import websockets

from agent import INSTRUCTIONS, TOOLS, FrontDesk, connect, refused
from audio import EchoCanceller, SoundCard, Speaker

FRAME_S = 0.02  # the sound card and the scripted caller both hand over 20 ms frames
BUFFER_CAP_S = 5.0  # the most of the caller's audio we hold while the session gets ready
HANG_UP_CAP_S = 8.0  # the longest we wait for a goodbye to finish before hanging up anyway

END_CALL = {
    "type": "function",
    "name": "end_call",
    "description": "Hang up. Use it when the caller has everything they need and is saying goodbye.",
    "parameters": {
        "type": "object",
        "properties": {
            "goodbye": {
                "type": "string",
                "poly_spoken": True,  # the server speaks this argument, in the same response as the call
                "description": "The last thing you say, as you hang up: thank the caller by name, wish them a good stay, and say goodbye.",
            },
            "reason": {"type": "string", "description": "Why the call is ending, for the log."},
        },
        "required": ["goodbye", "reason"],
    },
}
CALL_INSTRUCTIONS = INSTRUCTIONS + "\nWhen the caller has everything they need, end the call with end_call."


class Call(FrontDesk):
    """A FrontDesk that listens from pickup and hangs up after its goodbye.

    Create it before the socket exists and feed it audio from the first frame: send_audio() holds the frames until
    the session is configured, then sends them in order ahead of the live audio. Set .ws once connected.
    """

    def __init__(self, ws, speaker: Speaker, echo: EchoCanceller | None = None):
        super().__init__(ws, speaker, echo)
        self.held: deque[bytes] = deque(maxlen=round(BUFFER_CAP_S / FRAME_S))  # past the cap, the oldest frame goes
        self.live = False  # True once the held audio has gone out and frames can go straight to the server
        self.goodbye: dict | None = None  # the end_call call, while its goodbye is being spoken
        self.goodbye_spoken = asyncio.Event()
        self.hanging_up: asyncio.Task | None = None
        self.hung_up = asyncio.Event()

    async def send_audio(self, pcm: bytes):
        if self.hung_up.is_set():
            return
        if self.live:
            await super().send_audio(pcm)
        else:
            self.held.append(pcm)

    async def start(self):
        await self.ws.recv()  # session.created: the session is open, but still on its default settings
        await self.send({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": CALL_INSTRUCTIONS,
                "tools": TOOLS + [END_CALL],
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": {"type": "server_vad"}}},
            },
        })
        updated = json.loads(await self.ws.recv())
        if updated["type"] != "session.updated":
            raise RuntimeError(updated.get("error", {}).get("message", f"unexpected {updated['type']}"))
        held = len(self.held)
        while self.held:  # frames that arrive during the flush join the end of the queue, so order holds
            await super().send_audio(self.held.popleft())
        self.live = True
        self.log("", f"(session ready: sent {held * FRAME_S:.2f} s of held audio, now live)")

    async def handle(self, event: dict):
        match event["type"]:
            case "response.function_call_arguments.done" if event["name"] == "end_call":
                self.goodbye = event
                self.log("", f"(end_call: {json.loads(event['arguments']).get('reason', '')})")
                self.hanging_up = asyncio.create_task(self.hang_up_after_goodbye())
                return  # not a booking lookup: FrontDesk's tool runner never sees it
            case "response.done" if self.goodbye and event["response"]["id"] == self.goodbye["response_id"]:
                if event["response"]["status"] == "completed":
                    self.goodbye_spoken.set()  # the server has sent the whole goodbye
                else:
                    await self.keep_line_open()
        await super().handle(event)

    async def hang_up_after_goodbye(self):
        try:
            async with asyncio.timeout(HANG_UP_CAP_S):
                await self.goodbye_spoken.wait()  # every word of the goodbye has arrived
                await self.speaker.drain()  # and the speaker has played the last of it
        except TimeoutError:
            self.log("", f"(the goodbye didn't finish within {HANG_UP_CAP_S:.0f} s: hanging up anyway)")
        await self.hang_up()

    async def keep_line_open(self):
        """The caller talked over the goodbye, so the server cancelled it. Stay on the line and tell the model."""
        self.hanging_up.cancel()
        call, self.goodbye = self.goodbye, None
        self.log("", "(the caller talked over the goodbye: staying on the line)")
        await self.send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(
                {"hung_up": False, "why": "The caller started talking over your goodbye, so the call is still open."})},
        })
        # No response.create: the caller's new turn gets its own reply.

    async def hang_up(self):
        self.hung_up.set()
        self.log("", "(hangs up)")
        await self.ws.close()


async def call(speakers: bool):
    """A live call on your microphone. The microphone opens first, then the connection."""
    speaker = Speaker()
    echo = EchoCanceller() if speakers else None
    if echo:
        speaker.on_played = echo.played
    desk = Call(None, speaker, echo)
    with SoundCard(speaker) as card:  # pickup: from here on, everything the caller says is kept

        async def stream_mic():
            while True:
                await desk.send_audio(await card.queue.get())

        mic = asyncio.create_task(stream_mic())
        print("Line open. Start talking whenever you like; say goodbye when you're done.")
        async with connect() as ws:
            desk.ws = ws
            await desk.start()
            await desk.listen()  # returns when the hang-up closes the socket
        mic.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--speakers", action="store_true", help="you're on speakers, not headphones: cancel the echo")
    try:
        asyncio.run(call(parser.parse_args().speakers))
    except KeyboardInterrupt:
        print("\nHung up.")
    except websockets.InvalidStatus as error:
        raise SystemExit(refused(error))
