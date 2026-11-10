"""A small stand-in for the Dialog-RSN-1 realtime endpoint, so the tests run with no key and no network.

It behaves like the real server where barge-in is concerned (checked on api.studio.poly.ai, October 2026):

- Loud audio while a reply is playing is a barge-in: speech_started, then the reply's full transcript, then
  response.done with status "cancelled". The server keeps only the heard part of the reply in history.
- conversation.item.truncate cuts that history to audio_end_ms and answers conversation.item.truncated.
- conversation.item.retrieve returns the item as history holds it: a cut reply ends with "-".
- A caller turn ends after 300 ms of quiet, and gets the next scripted step.
- A tool call's response completes as soon as the call is sent; the output comes back whenever the client sends it.
- An end_call whose goodbye is a poly_spoken argument: the goodbye's audio and the call arrive together in one
  response (the first audio just ahead of the call), and response.done only once the goodbye has been sent.
  Talking over the goodbye cancels that response (checked 10 November 2026).

Replies stream faster than real time, so the test decides how much "plays" by pulling from the Speaker itself. With
pace=True they stream at speaking pace instead, a little ahead of playback the way the real server sends them, and
ready_after makes the server slow to send session.created and session.updated, the way a real connection is.
"""
import asyncio
import base64
import json

import numpy as np
import websockets

LOUD = 1000
QUIET_CHUNKS_TO_END = 15  # 15 x 20 ms
# Paced timing, from 10 goodbyes on api.studio.poly.ai, 10 November 2026: the server stays 83 to 303 ms ahead of
# playback, and response.done comes 312 to 477 ms after the last audio delta.
LEAD_MS = 300
DONE_AFTER_S = 0.4


class FakeRealtime:
    def __init__(self, steps, caller_lines=None, ready_after=0.0, pace=False, stall=None):
        """steps: what the server does for each response, in order: ("say", text, ms), ("tool", name, args),
        ("goodbye", text, ms) for an end_call with a spoken goodbye, or ("goodbye-stalls", text, ms) for one whose
        response never finishes. stall=(ms, seconds) pauses every reply's audio that far in."""
        self.steps = list(steps)
        self.caller_lines = list(caller_lines or ["(caller)"] * 10)
        self.ready_after = ready_after  # seconds before session.created, and again before session.updated
        self.pace = pace
        self.stall = stall
        self.ready = False  # True once session.updated has gone out
        self.audio: list[tuple[bool, bytes]] = []  # every input_audio_buffer.append: (was the session ready?, pcm)
        self.received: list[dict] = []
        self.history: dict[str, dict] = {}  # item_id -> {"text", "ms", "heard_ms"}
        self.url = None

    def sent(self, event_type: str) -> list[dict]:
        return [e for e in self.received if e["type"] == event_type]

    async def __aenter__(self):
        self.server = await websockets.serve(self.handler, "127.0.0.1", 0, max_size=None)
        self.url = f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/v1/realtime"
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    async def handler(self, ws):
        send = lambda event: ws.send(json.dumps(event))  # noqa: E731
        state = {"speaking": False, "quiet": 0, "reply": None, "n": 0}
        await asyncio.sleep(self.ready_after)
        await send({"type": "session.created", "session": {"poly_session_id": "conversation_test"}})

        async def respond():
            if not self.steps:
                return
            kind, *rest = self.steps.pop(0)
            state["n"] += 1
            rid, iid = f"resp_{state['n']}", f"item_{state['n']}"
            await send({"type": "response.created", "response": {"id": rid, "status": "in_progress", "output": []}})
            if kind == "tool":
                name, args = rest
                call = {"type": "function_call", "id": iid, "call_id": f"call_{state['n']}", "name": name, "arguments": json.dumps(args)}
                await send({"type": "response.function_call_arguments.done", "response_id": rid, "item_id": iid,
                            "call_id": call["call_id"], "name": name, "arguments": call["arguments"]})
                await send({"type": "response.done", "response": {"id": rid, "status": "completed", "output": [call]}})
                return
            text, ms = rest
            before = []  # items ahead of the spoken message in this response
            fc = None
            if kind.startswith("goodbye"):
                fc = {"type": "function_call", "id": f"fc_{state['n']}", "call_id": f"call_{state['n']}", "name": "end_call",
                      "arguments": json.dumps({"goodbye": text, "reason": "the caller is done"})}
                before = [fc]
            self.history[iid] = {"text": text, "ms": ms, "heard_ms": ms}
            state["reply"] = {"rid": rid, "iid": iid, "text": text, "before": before}
            for n in range(ms // 100):  # 100 ms of audio per delta
                if state["reply"] is None or state["reply"]["rid"] != rid:
                    return  # cancelled by a barge-in
                await send({"type": "response.output_audio.delta", "response_id": rid, "item_id": iid,
                            "delta": base64.b64encode(np.full(2400, 500, np.int16).tobytes()).decode()})
                if n == 0 and fc:  # the goodbye's first audio arrives just ahead of the call itself
                    await send({"type": "response.function_call_arguments.done", "response_id": rid, "item_id": fc["id"],
                                "call_id": fc["call_id"], "name": "end_call", "arguments": fc["arguments"]})
                if self.stall and (n + 1) * 100 == self.stall[0]:
                    await asyncio.sleep(self.stall[1])  # the stream pauses, as on a slow network
                if not self.pace:
                    await asyncio.sleep(0.005)
                elif n >= LEAD_MS // 100 - 1:
                    await asyncio.sleep(0.1)
            if kind == "goodbye-stalls":
                return  # the reply never finishes: no transcript, no response.done
            if self.pace:
                await asyncio.sleep(DONE_AFTER_S)
                if state["reply"] is None or state["reply"]["rid"] != rid:
                    return
            state["reply"] = None
            await send({"type": "response.output_audio_transcript.done", "response_id": rid, "item_id": iid, "transcript": text})
            await send({"type": "response.done", "response": {"id": rid, "status": "completed", "output": before + [self.item(iid)]}})

        async def caller_audio(pcm: bytes):
            loud = np.abs(np.frombuffer(pcm, dtype=np.int16)).max(initial=0) > LOUD
            if loud and not state["speaking"]:
                state["speaking"], state["quiet"] = True, 0
                await send({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})
                if reply := state["reply"]:
                    state["reply"] = None
                    await send({"type": "response.output_audio_transcript.done", "response_id": reply["rid"], "item_id": reply["iid"], "transcript": reply["text"]})
                    await send({"type": "response.done", "response": {"id": reply["rid"], "status": "cancelled", "output": reply["before"] + [self.item(reply["iid"])]}})
            elif state["speaking"]:
                state["quiet"] = 0 if loud else state["quiet"] + 1
                if state["quiet"] >= QUIET_CHUNKS_TO_END:
                    state["speaking"] = False
                    await send({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 0})
                    await send({"type": "conversation.item.input_audio_transcription.completed", "item_id": "item_caller", "transcript": self.caller_lines.pop(0)})
                    asyncio.create_task(respond())

        async def updated(session: dict):
            await asyncio.sleep(self.ready_after)
            self.ready = True
            await send({"type": "session.updated", "session": session})

        async for raw in ws:
            event = json.loads(raw)
            if event["type"] != "input_audio_buffer.append":
                self.received.append(event)
            match event["type"]:
                case "session.update":
                    asyncio.create_task(updated(event["session"]))  # keep reading: audio can arrive meanwhile
                case "input_audio_buffer.append":
                    self.audio.append((self.ready, base64.b64decode(event["audio"])))
                    await caller_audio(self.audio[-1][1])
                case "conversation.item.truncate":
                    self.history[event["item_id"]]["heard_ms"] = event["audio_end_ms"]
                    await send({"type": "conversation.item.truncated", "item_id": event["item_id"], "content_index": 0, "audio_end_ms": event["audio_end_ms"]})
                case "conversation.item.retrieve":
                    await send({"type": "conversation.item.retrieved", "item": self.item(event["item_id"])})
                case "response.create":
                    asyncio.create_task(respond())

    def item(self, iid: str) -> dict:
        h = self.history[iid]
        heard = h["text"]
        if h["heard_ms"] < h["ms"]:
            heard = h["text"][: round(len(h["text"]) * h["heard_ms"] / h["ms"])].rstrip() + "-"
        return {"id": iid, "type": "message", "role": "assistant", "content": [{"type": "output_audio", "transcript": heard}]}
