"""The stereo recorder in the scripted room, against the fake realtime server: no key, no network.

Each test runs a real-time call of about eight seconds.
"""
import json
import wave

import numpy as np
import pytest

import agent
import caller
from bargein import analyse
from fake_realtime import FakeRealtime
from record import FrontDesk, recorded, to_pcmu
from recorder import StereoRecorder

RUNDOWN = "Check-in is from 3 pm. Please bring your photo ID and the card you booked with."


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("POLY_API_KEY", "test-key")


async def record_call(tmp_path, monkeypatch, steps):
    monkeypatch.setitem(caller.SCRIPTS, "test", steps)
    async with FakeRealtime([("say", RUNDOWN, 5000), ("say", "Valet only.", 1000)], pace=True, confirm_ms=780) as fake:
        monkeypatch.setattr(agent, "URL", fake.url)
        recorder = StereoRecorder()
        await caller.run("test", client=recorded(FrontDesk, recorder), recorder=recorder)
    wav = tmp_path / "call.wav"
    recorder.save(wav, script="test")
    return wav, recorder


async def test_a_barge_in_is_recorded_on_one_clock(tmp_path, monkeypatch):
    wav, recorder = await record_call(tmp_path, monkeypatch, [("wait", None, "turn"), ("parking", 1.0, "barge-in")])

    with wave.open(str(wav)) as w:
        assert (w.getnchannels(), w.getframerate()) == (2, 24000)
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)
    labels = json.loads(wav.with_suffix(".json").read_text())
    assert [(x["line"], x["intent"]) for x in labels["lines"]] == [("wait", "turn"), ("parking", "barge-in")]
    assert set(np.unique(pcm[:, 1])) <= {0, 500}  # the right channel is the fake's reply audio, nothing else
    assert np.abs(pcm[:, 0]).max() > 5000  # the left channel is the caller's clips

    [o] = analyse(wav)
    assert (o.line, o.verdict) == ("parking", "stopped")
    assert o.talked_s == pytest.approx(0.78, abs=0.06)  # the fake's confirmation time
    [heard] = [e["t_s"] for e in recorder.events if e["type"] == "speech_started" and e["t_s"] > o.onset_s]
    assert o.onset_s + o.talked_s - heard == pytest.approx(0, abs=0.04)  # the recording agrees with the event


async def test_a_backchannel_is_talked_through(tmp_path, monkeypatch):
    wav, recorder = await record_call(tmp_path, monkeypatch, [("wait", None, "turn"), ("mm-hmm", 1.2, "backchannel")])

    [o] = analyse(wav)
    assert (o.line, o.result, o.verdict) == ("mm-hmm", "talked on", "ignored")
    assert not [e for e in recorder.events if e["type"] == "speech_started" and e["t_s"] > o.onset_s]


def test_phone_audio_is_mu_law():
    pcm = np.array([0, 1000, -1000, 32767, -32768], np.int16)
    assert to_pcmu(pcm) == bytes([0xFF, 0xCE, 0x4E, 0x80, 0x00])
    assert len(to_pcmu(np.zeros(160, np.int16))) == 160  # one byte a sample: 20 ms at 8 kHz


def test_a_phone_recording_is_8_khz(tmp_path):
    recorder = StereoRecorder(rate=8000)
    recorder.add(np.full(480, 300, np.int16).tobytes(), np.full(480, -300, np.int16).tobytes())  # one 20 ms tick
    recorder.save(tmp_path / "phone.wav")
    with wave.open(str(tmp_path / "phone.wav")) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (2, 8000, 160)
        assert list(np.frombuffer(w.readframes(1), np.int16)) == [300, -300]
