# Barge-in voice agent

A Python voice agent on PolyAI's Dialog-RSN-1 that handles barge-in properly. When the caller talks over it, it
stops its voice as soon as the server confirms the interruption, tells the server how much of the reply was heard, keeps its own record from what was
heard, and doesn't let a slow tool talk over the caller's next question. On speakers, it cancels its own echo so it
never hears itself and starts over.

The walkthrough is the post: [Handle barge-in without your voice agent repeating itself](https://www.petermckee.dev/blog/handle-barge-in-without-your-voice-agent-repeating-itself).

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
uv run measure.py --runs 3              # repeat the call and summarise how barge-in went (--no-truncate, --aec on)
uv run agent.py                         # talk to it yourself, with headphones on
uv run agent.py --speakers              # on laptop speakers, with the echo canceller on
```

A variable set in your shell beats `.env`. If the server refuses the connection (HTTP 401), check `echo $POLY_API_KEY`, and set `DIALOG_REALTIME_URL` if your key is for another PolyAI host.

## Test

```bash
uv run pytest                           # offline: a fake realtime server, no key needed
POLYAI_LIVE=1 uv run pytest -m live     # scripted calls against the real API
```

## Start and end a call without clipping a word

The second post, [Start and end a voice agent call in Python without clipping a word](https://dev.to/pmckeetx/start-and-end-a-voice-agent-call-in-python-without-clipping-a-word-330i),
adds two files and leaves the ones above alone (the barge-in post's code is at the tag `barge-in-post`).

`call.py` is the front desk with both edges of a call handled. It holds the caller's audio from the moment the line
opens (up to 5 s) and sends it, in order, once `session.updated` arrives. It gives the agent an `end_call` tool
whose goodbye the server speaks, and it hangs up only when that goodbye's response is done and the speaker has
played the last of it, or after 8 s if it never finishes. If the caller talks over the goodbye, the line stays open.
`pickup.py` is a scripted caller that starts talking on the first frame, through the same simulated room as
`caller.py`, and counts what was heard at each edge.

```bash
uv run pickup.py                        # buffer from pickup, hang up once the goodbye has played
uv run pickup.py --before               # drop audio until the session is ready, hang up a second into the goodbye
uv run pickup.py --talk-over            # the caller cuts in half a second into the goodbye
uv run pickup.py --runs 5               # repeat it and summarise both edges
uv run pickup.py --save call.out.wav    # also save the call as the room heard it, both voices
uv run call.py                          # a call on your microphone: start talking as soon as it runs
uv run pytest tests/test_call.py        # the offline tests for both edges
```

## Measure barge-in from a recording

The third post, [Measure barge-in stop time from a stereo call recording in Python](https://dev.to/pmckeetx/measure-barge-in-stop-time-from-a-stereo-call-recording-in-python-4pk0), adds three files
and changes `caller.py` only to feed the recorder and give each line an intent (the second post's code is at the
tag `start-and-end-post`).

`recorder.py` writes a scripted call as a stereo WAV on the playback clock: the caller on the left, as the agent
heard them, and the agent on the right, as the caller heard it, both from the same 20 ms tick. A JSON file beside
it says what the caller said, when, and what they meant: a barge-in, a backchannel or background noise. `record.py`
runs a script through that recorder and prints, for each overlap, the stop the recording shows beside when
`speech_started` arrived. `bargein.py` scores any 16-bit two-channel WAV, with the standard library only: for each
time the caller starts while the agent is talking, how long the agent kept talking, and, with labels, whether that
was a stop, a missed barge-in, a backchannel ignored or a false stop.

```bash
uv run record.py --script check-in --runs 20    # "Is there parking?" 4 s into the check-in rundown
uv run record.py --script stop --runs 10        # "Stop." (also wait, mm-hmm and tv)
uv run record.py --format pcmu                  # the caller on a phone line: 8 kHz G.711 mu-law
uv run bargein.py runs/*.wav                    # score every recording, with a table per line
uv run bargein.py call.wav --caller right       # someone else's recording, with the agent on the left
uv run bargein.py call.wav --labels call.csv    # with your own labels: onset_s,intent
uv run pytest tests/test_bargein.py tests/test_recorder.py
```

`make_samples.sh` shows how the caller's lines in `samples/` were made with macOS `say`. `mm-hmm.wav` and `tv.wav`
come from [refund-voice-agent](https://github.com/pmckeetx/refund-voice-agent).

## License

Apache 2.0. See [LICENSE](LICENSE).
