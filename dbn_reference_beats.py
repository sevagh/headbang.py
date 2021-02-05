#!/usr/bin/env python3

import argparse
import sys
import json
import numpy
import librosa
from madmom.io.audio import load_audio_file, write_wave_file
from headbang.util import load_wav
import madmom


def main():
    parser = argparse.ArgumentParser(
        prog="mir.py",
        description="Apply beat tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("wav_in", help="input wav file")
    parser.add_argument("beat_wav_out", help="output beat wav file")

    args = parser.parse_args()

    print("Loading file {0} with 44100 sampling rate".format(args.wav_in))
    x = load_wav(args.wav_in)

    proc_beat = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act_beat = madmom.features.beats.RNNBeatProcessor()(x)

    beat_times = proc_beat(act_beat)

    print("Overlaying clicks at beat locations")

    beat_clicks = librosa.clicks(beat_times, sr=44100, length=len(x))

    beat_waveform = (x + beat_clicks).astype(numpy.single)

    print("Writing outputs with clicks to {0}".format(args.beat_wav_out))
    write_wave_file(beat_waveform, args.beat_wav_out, sample_rate=44100)


if __name__ == "__main__":
    main()
