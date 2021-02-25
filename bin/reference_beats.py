#!/usr/bin/env python3

import argparse
import sys
import json
import numpy
import librosa
from madmom.io.audio import load_audio_file, write_wave_file
from headbang.util import load_wav, overlay_clicks
from headbang.beattrack import apply_single_beat_tracker, algo_names
import madmom


def main():
    parser = argparse.ArgumentParser(
        prog="reference_beats.py",
        description="Apply beat tracking - part of headbang.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--algorithm",
        type=int,
        default=1,
        help="which single algorithm to use (default=%(default)s)",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=2,
        help="butter filter order (default=%(default)s)",
    )
    parser.add_argument("wav_in", help="input wav file")
    parser.add_argument("beat_wav_out", help="output beat wav file")

    args = parser.parse_args()

    print("Loading file {0} with 44100 sampling rate".format(args.wav_in))
    x = load_wav(args.wav_in)

    print("Applying algorithm: {0}".format(algo_names[args.algorithm]))
    beat_times = apply_single_beat_tracker(x, args.algorithm)

    print("Overlaying clicks at beat locations")
    beat_waveform = overlay_clicks(x, beat_times)

    print("Writing outputs with clicks to {0}".format(args.beat_wav_out))
    write_wave_file(beat_waveform, args.beat_wav_out, sample_rate=44100)


if __name__ == "__main__":
    main()
