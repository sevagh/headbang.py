#!/usr/bin/env python3

import argparse
import sys
import json
import numpy
import librosa
from madmom.io.audio import load_audio_file, write_wave_file
from headbang.util import load_wav
from headbang.beattrack import apply_single_beat_tracker, algo_names
from headbang.percussive_transients import kick_snare_filter
import madmom


def main():
    parser = argparse.ArgumentParser(
        prog="reference_beats.py",
        description="Apply beat tracking - part of headbang.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--algorithm", type=int, default=1, help="which single algorithm to use"
    )
    parser.add_argument(
        "--pre-kick-snare-filter",
        action="store_true",
        help="apply kick/snare filtering (0-100, 200-500) before",
    )

    parser.add_argument(
        "--filter-order", type=int, default=2, help="butter filter order"
    )
    parser.add_argument("wav_in", help="input wav file")
    parser.add_argument("beat_wav_out", help="output beat wav file")

    args = parser.parse_args()

    print("Loading file {0} with 44100 sampling rate".format(args.wav_in))
    x = load_wav(args.wav_in)

    print("Applying algorithm: {0}".format(algo_names[args.algorithm]))
    if args.pre_kick_snare_filter:
        print("Filtering into kick/snare frequency range first")
        beat_times, _ = apply_single_beat_tracker(
            kick_snare_filter(x, 44100, args.filter_order), args.algorithm
        )
    else:
        beat_times, _ = apply_single_beat_tracker(x, args.algorithm)

    print("Overlaying clicks at beat locations")

    beat_clicks = librosa.clicks(beat_times, sr=44100, length=len(x))

    beat_waveform = (x + beat_clicks).astype(numpy.single)

    print("Writing outputs with clicks to {0}".format(args.beat_wav_out))
    write_wave_file(beat_waveform, args.beat_wav_out, sample_rate=44100)


if __name__ == "__main__":
    main()
