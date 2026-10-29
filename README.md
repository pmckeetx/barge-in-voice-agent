# Barge-in voice agent

A Python voice agent on PolyAI's Dialog-RSN-1 that handles barge-in properly. When the caller talks over it, it
stops its voice at once, tells the server how much of the reply was heard, keeps its own record from what was
heard, and doesn't let a slow tool talk over the caller's next question. On speakers, it cancels its own echo so it
never hears itself and starts over.

The walkthrough is the post: [Handle barge-in without your voice agent repeating itself](POST_URL).

## What's here

`agent.py` is the front desk at the Harborview Inn, and the file the post walks through. `audio.py` has the speaker
that counts how much of each reply played, the echo canceller and the sound card. `caller.py` is a scripted caller
that interrupts the agent on a real call, in a simulated room, so you can see barge-in work without a microphone.
`measure.py` runs a script several times and summarises how barge-in went.

## Prerequisites

Python 3.13 and [uv](https://docs.astral.sh/uv/). A Dialog-RSN-1 API key: sign in at
[studio.poly.ai](https://studio.poly.ai) and create one in the **API Keys** tab. For `agent.py` you need a
microphone, and headphones unless you pass `--speakers`.

## Set up

```bash
git clone https://github.com/pmckeetx/barge-in-voice-agent.git
cd barge-in-voice-agent
cp .env.example .env    # then add your POLY_API_KEY
uv sync
```

## Run

```bash
uv run caller.py                        # a scripted call: the caller interrupts, the agent stops and carries on
uv run caller.py --echo 0.3 --before    # speakers leaking into the mic, with no barge-in handling
uv run caller.py --echo 0.3             # the same room, handled
uv run caller.py --script booking       # interrupt the agent while a tool is running
uv run agent.py                         # talk to it yourself, with headphones on
uv run agent.py --speakers              # on laptop speakers, with the echo canceller on
```

A variable set in your shell beats `.env`. If you get `HTTP 401`, check `echo $POLY_API_KEY`.

## Test

```bash
uv run pytest                           # offline: a fake realtime server, no key needed
POLYAI_LIVE=1 uv run pytest -m live     # one scripted call against the real API
```

## License

Apache 2.0. See [LICENSE](LICENSE).
