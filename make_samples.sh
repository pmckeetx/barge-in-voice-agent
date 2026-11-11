#!/bin/sh
# The caller's lines for pickup.py, spoken by macOS `say` and saved as 24 kHz 16-bit mono WAVs.
# You don't need this to run anything: the clips are in samples/. It's here so you can make your own.
set -e
cd "$(dirname "$0")/samples"
line() {
    say -v Samantha -o "$1.aiff" "$2"
    afconvert -f WAVE -d LEI16@24000 -c 1 "$1.aiff" "$1.wav"
    rm "$1.aiff"
    echo "samples/$1.wav  $2"
}
line pickup "Hi, yes, I'm calling about my booking for Friday. The surname's Okafor."
line thats-all "Great, that's everything I needed. Thanks very much."
line one-more-thing "Oh, wait, one more thing. Is breakfast included?"
