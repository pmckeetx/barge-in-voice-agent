"""Score barge-in from a two-channel call recording: how long the agent kept talking after the caller cut in.

    uv run bargein.py runs/*.wav                      # caller on the left, labels from each WAV's .json
    uv run bargein.py call.wav --caller right         # a recording with the agent on the left
    uv run bargein.py call.wav --labels call.csv      # your own labels: onset_s,intent
    uv run bargein.py runs/*.wav --json               # one JSON object per overlap

It uses the standard library only, so you can copy it next to any stack's recordings.
"""
import argparse
import csv
import json
import math
import re
import statistics
import sys
import wave
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path

FRAME_S = 0.02  # 20 ms frames
FLOOR_DB = -70.0  # treat anything quieter than this as silence, even on a perfectly clean channel
ON_DB = 16  # speech starts on a frame this far above the channel's noise floor
OFF_DB = 10  # and ends only once frames fall below this lower level
# How much quiet ends a channel's speech. The agent's has to outlast its longest pause between sentences and stay
# shorter than its quickest answer after a stop. On api.studio.poly.ai's server voice (17 November 2026) the
# longest pause inside a reply was 0.36 s, and the shortest quiet after a cut was 0.72 s.
HANGOVER_S = {"caller": 0.4, "agent": 0.5}
LEAD_S = 0.2  # the agent must have been talking this long when the caller starts, or it's the agent cutting in
REACT_S = 0.15  # no detector reacts faster: an agent that goes quiet sooner was finishing anyway
MISS_S = 2.0  # still talking this long after the caller finished: it didn't stop
STOP = {"barge-in", "turn"}  # the intents that should stop the agent; backchannel and background should not


@dataclass
class Overlap:
    file: str
    onset_s: float  # when the caller started, on the recording's clock
    line: str | None  # the label's name for what the caller said
    intent: str | None  # barge-in, backchannel, background, or None with no label
    talked_s: float  # how long the agent kept talking after the caller started
    result: str  # stopped, talked on, or ended (the reply finished as the caller started)
    verdict: str | None  # stopped, missed, ignored or false stop, once the intent is known


def frames_db(path: Path, caller: str = "left") -> tuple[list[float], list[float]]:
    """Read a 16-bit stereo WAV and return the level of every 20 ms frame, in dBFS: (caller, agent)."""
    with wave.open(str(path)) as w:
        if w.getnchannels() != 2 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: needs a 16-bit two-channel WAV, not {w.getnchannels()} channel(s)")
        rate, pcm = w.getframerate(), array("h", w.readframes(w.getnframes()))
    if sys.byteorder == "big":
        pcm.byteswap()
    n = round(rate * FRAME_S)

    def levels(channel: array) -> list[float]:
        out = []
        for i in range(0, len(channel) - n + 1, n):
            frame = channel[i:i + n]
            rms = math.sqrt(math.sumprod(frame, frame) / n)
            out.append(20 * math.log10(rms / 32768) if rms else -120.0)
        return out

    left, right = levels(pcm[0::2]), levels(pcm[1::2])
    return (left, right) if caller == "left" else (right, left)


def segments(db: list[float], hangover_s: float) -> list[tuple[float, float]]:
    """Where a channel is speaking, as (start_s, end_s), with gaps between words held together.

    Speech starts above the on level and stops only after hangover_s below the off level. The end is the start
    of that quiet run: the moment the listener last heard it.
    """
    floor = max(sorted(db)[len(db) // 10], FLOOR_DB) if db else FLOOR_DB
    on, off, hang = floor + ON_DB, floor + OFF_DB, round(hangover_s / FRAME_S)
    runs, start, quiet = [], None, 0
    for i, level in enumerate(db):
        if start is None:
            if level >= on:
                start, quiet = i, 0
        elif level < off:
            quiet += 1
            if quiet == hang:
                runs.append((start * FRAME_S, (i + 1 - hang) * FRAME_S))
                start = None
        else:
            quiet = 0
    if start is not None:
        runs.append((start * FRAME_S, (len(db) - quiet) * FRAME_S))
    return [(round(a, 2), round(b, 2)) for a, b in runs]


def score(file: str, caller: list[tuple[float, float]], agent: list[tuple[float, float]], labels: list[dict]) -> list[Overlap]:
    """One Overlap for every time the caller started while the agent was already talking."""
    out, unused = [], sorted(labels, key=lambda label: label["onset_s"])
    for onset, end in caller:
        talking = next(((a, b) for a, b in agent if a + LEAD_S <= onset < b), None)
        if not talking:
            continue
        talked = round(talking[1] - onset, 2)
        result = "ended" if talked < REACT_S else "stopped" if talking[1] <= end + MISS_S else "talked on"
        # A label belongs to the first caller burst that starts at or up to a second after it.
        label = next((x for x in unused if x["onset_s"] - FRAME_S <= onset <= x["onset_s"] + 1.0), None)
        if label:
            unused.remove(label)
        intent = label and label.get("intent")
        verdict = None
        if intent and result != "ended":
            should_stop, stopped = intent in STOP, result == "stopped"
            verdict = {(True, True): "stopped", (True, False): "missed",
                       (False, False): "ignored", (False, True): "false stop"}[should_stop, stopped]
        out.append(Overlap(file, onset, label and label.get("line"), intent, talked, result, verdict))
    return out


def load_labels(wav: Path, csv_path: str | None) -> list[dict]:
    """Labels from a CSV you wrote (onset_s,intent), or the JSON record.py writes next to each WAV, or none."""
    if csv_path:
        with open(csv_path) as f:
            return [{"onset_s": float(r["onset_s"]), "intent": r["intent"], "line": r.get("line")} for r in csv.DictReader(f)]
    sidecar = wav.with_suffix(".json")
    return json.loads(sidecar.read_text())["lines"] if sidecar.exists() else []


def analyse(wav: Path, caller: str = "left", labels_csv: str | None = None, agent_hangover: float | None = None) -> list[Overlap]:
    caller_db, agent_db = frames_db(wav, caller)
    agent = segments(agent_db, agent_hangover or HANGOVER_S["agent"])
    return score(str(wav), segments(caller_db, HANGOVER_S["caller"]), agent, load_labels(wav, labels_csv))


def table(overlaps: list[Overlap]) -> str:
    """One row per kind of line: how many stopped the agent, how fast, and how many shouldn't have."""
    rows = [f"{'recordings':16}{'line':10}{'intent':13}{'calls':>6}{'stopped':>9}  {'median':>8}  {'range':16}"
            f"{'false stops':>12}{'missed':>8}"]
    groups: dict[tuple, list[Overlap]] = {}
    for o in overlaps:
        name = re.sub(r"-\d+$", "", Path(o.file).stem)  # runs/stop-03.wav and runs/stop-04.wav are both "stop"
        groups.setdefault((name, o.line or "?", o.intent or "?"), []).append(o)
    for (name, line, intent), group in groups.items():
        times = [o.talked_s for o in group if o.result == "stopped"]
        median = f"{statistics.median(times):.2f} s" if times else "-"
        spread = f"{min(times):.2f} to {max(times):.2f} s" if times else "-"
        count = lambda v: sum(o.verdict == v for o in group)  # noqa: E731
        rows.append(f"{name:16}{line:10}{intent:13}{len(group):>6}{len(times):>9}  {median:>8}  {spread:16}"
                    f"{count('false stop'):>12}{count('missed'):>8}")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wavs", nargs="+", type=Path)
    parser.add_argument("--caller", choices=["left", "right"], default="left", help="which channel is the caller")
    parser.add_argument("--labels", help="a CSV of onset_s,intent for one WAV, instead of its .json")
    parser.add_argument("--agent-hangover", type=float, help=f"seconds of quiet that end the agent's speech (default {HANGOVER_S['agent']})")
    parser.add_argument("--json", action="store_true", help="print one JSON object per overlap")
    args = parser.parse_args()
    overlaps = [o for wav in args.wavs for o in analyse(wav, args.caller, args.labels, args.agent_hangover)]
    if not overlaps:
        print("No overlaps: the caller never started while the agent was talking. If the agent is on the left, "
              "pass --caller right.")
        return
    if args.labels:
        for label in load_labels(args.wavs[0], args.labels):
            if not any(o.intent == label["intent"] and 0 <= o.onset_s - label["onset_s"] <= 1.0 for o in overlaps):
                print(f"Label at {label['onset_s']} s ({label['intent']}) matched no caller burst over the agent: "
                      f"run without --labels to see where each burst starts.")
    for o in overlaps:
        if args.json:
            print(json.dumps(asdict(o)))
        else:
            what = f"{o.line or 'caller'} ({o.intent})" if o.intent else (o.line or "caller")
            how = {"stopped": "stopped after", "talked on": "talked on for", "ended": "ended after"}[o.result]
            print(f"{o.file}  {o.onset_s:7.2f} s  {what:24} {how} {o.talked_s:.2f} s"
                  + (f"  [{o.verdict}]" if o.verdict else ""))
    if not args.json and overlaps:
        print()
        print(table(overlaps))


if __name__ == "__main__":
    main()
