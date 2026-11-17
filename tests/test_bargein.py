"""bargein.py on synthetic calls with known answers: no key, no network."""
import json
import wave

import numpy as np
import pytest

from bargein import analyse, frames_db, segments

RATE = 24000


def synth(path, agent, caller, gaps=0.15, agent_left=False, labels=None):
    """A stereo WAV: the agent talks in 0.5 s words with gaps between them over each (start, end) in agent, and the
    caller makes a sound over each (start, end) in caller. Noise, not speech: the analyzer only hears energy."""
    end = max(b for _, b in agent + caller) + 2.0
    rng = np.random.default_rng(0)
    left, right = np.zeros(int(end * RATE)), np.zeros(int(end * RATE))
    for a, b in caller:
        left[int(a * RATE):int(b * RATE)] = rng.normal(0, 3000, int(b * RATE) - int(a * RATE))
    for a, b in agent:
        t = a
        while t < b:
            word_end = min(t + 0.5, b)
            right[int(t * RATE):int(word_end * RATE)] = rng.normal(0, 3000, int(word_end * RATE) - int(t * RATE))
            t = word_end + gaps
    channels = (right, left) if agent_left else (left, right)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes(np.column_stack(channels).astype(np.int16).tobytes())
    if labels:
        path.with_suffix(".json").write_text(json.dumps({"lines": labels}))
    return path


def test_a_known_stop_time_comes_back_within_a_frame(tmp_path):
    # The caller cuts in at 3.0 s and the agent stops 0.78 s later.
    wav = synth(tmp_path / "call.wav", agent=[(1.0, 3.78)], caller=[(3.0, 4.5)],
                labels=[{"line": "parking", "onset_s": 3.0, "intent": "barge-in"}])
    [o] = analyse(wav)
    assert o.onset_s == pytest.approx(3.0, abs=0.02)
    assert o.talked_s == pytest.approx(0.78, abs=0.02)
    assert (o.result, o.verdict) == ("stopped", "stopped")


def test_gaps_between_words_are_not_stops(tmp_path):
    wav = synth(tmp_path / "call.wav", agent=[(0.5, 8.0)], caller=[], gaps=0.3)
    db = frames_db(wav)[1]
    assert segments(db, hangover_s=0.5) == [(0.5, 8.0)]
    assert len(segments(db, hangover_s=0.2)) > 5  # a short hangover splits the reply at every gap


def test_a_quick_answer_after_a_stop_is_not_more_of_the_reply(tmp_path):
    # "Stop." at 3.0 s, the reply cut 0.78 s later, and the answer 0.72 s after that, as on api.studio.
    wav = synth(tmp_path / "call.wav", agent=[(1.0, 3.78), (4.5, 7.0)], caller=[(3.0, 3.5)],
                labels=[{"line": "stop", "onset_s": 3.0, "intent": "barge-in"}])
    [o] = analyse(wav)
    assert (o.talked_s, o.verdict) == (pytest.approx(0.78, abs=0.02), "stopped")


def test_a_stop_during_a_long_sound_counts_from_when_it_ends(tmp_path):
    # A TV in the room for 4.6 s; the agent gives way 2.06 s in, while it's still playing.
    wav = synth(tmp_path / "call.wav", agent=[(1.0, 4.26), (11.0, 13.0)], caller=[(2.2, 6.8)],
                labels=[{"line": "tv", "onset_s": 2.2, "intent": "background"}])
    [o] = analyse(wav)
    assert (o.talked_s, o.verdict) == (pytest.approx(2.06, abs=0.02), "false stop")


def test_a_backchannel_the_agent_talks_through_is_ignored(tmp_path):
    wav = synth(tmp_path / "call.wav", agent=[(1.0, 9.0)], caller=[(2.2, 2.7)],
                labels=[{"line": "mm-hmm", "onset_s": 2.2, "intent": "backchannel"}])
    [o] = analyse(wav)
    assert (o.result, o.verdict) == ("talked on", "ignored")
    assert o.talked_s == pytest.approx(6.8, abs=0.02)


def test_stopping_for_a_backchannel_is_a_false_stop(tmp_path):
    wav = synth(tmp_path / "call.wav", agent=[(1.0, 3.0), (4.5, 6.0)], caller=[(2.2, 2.7)],
                labels=[{"line": "mm-hmm", "onset_s": 2.2, "intent": "backchannel"}])
    [o] = analyse(wav)
    assert (o.result, o.verdict) == ("stopped", "false stop")


def test_talking_on_through_a_barge_in_is_a_miss(tmp_path):
    wav = synth(tmp_path / "call.wav", agent=[(1.0, 9.0)], caller=[(3.0, 4.0)],
                labels=[{"line": "stop", "onset_s": 3.0, "intent": "barge-in"}])
    [o] = analyse(wav)
    assert (o.result, o.verdict) == ("talked on", "missed")


def test_a_reply_that_ends_as_the_caller_starts_is_not_a_stop(tmp_path):
    # The agent finishes at 4.0 s on its own; the caller starts 60 ms before that.
    wav = synth(tmp_path / "call.wav", agent=[(1.0, 4.0)], caller=[(3.94, 5.0)],
                labels=[{"line": "go-on", "onset_s": 3.94, "intent": "barge-in"}])
    [o] = analyse(wav)
    assert (o.result, o.verdict) == ("ended", None)


def test_a_caller_already_talking_when_the_agent_starts_is_not_an_overlap(tmp_path):
    wav = synth(tmp_path / "call.wav", agent=[(2.0, 5.0)], caller=[(1.95, 3.0)])
    assert analyse(wav) == []


def test_caller_on_the_right(tmp_path):
    wav = synth(tmp_path / "call.wav", agent=[(1.0, 3.78)], caller=[(3.0, 4.5)], agent_left=True)
    assert analyse(wav) == []  # read the wrong way round, the "caller" is the agent and nobody overlaps
    [o] = analyse(wav, caller="right")
    assert o.talked_s == pytest.approx(0.78, abs=0.02)
    assert (o.result, o.verdict) == ("stopped", None)  # no labels: a result, no verdict


def test_labels_from_a_csv(tmp_path):
    wav = synth(tmp_path / "call.wav", agent=[(1.0, 9.0)], caller=[(2.2, 2.7), (5.0, 5.4)])
    labels = tmp_path / "labels.csv"
    labels.write_text("onset_s,intent\n2.2,backchannel\n5.0,barge-in\n")
    first, second = analyse(wav, labels_csv=str(labels))
    assert first.verdict == "ignored" and second.verdict == "missed"


def test_a_mono_file_is_refused(tmp_path):
    path = tmp_path / "mono.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes(bytes(RATE * 2))
    with pytest.raises(SystemExit, match="two-channel"):
        analyse(path)
