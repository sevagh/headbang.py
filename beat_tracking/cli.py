import argparse
import json
import multiprocessing
import numpy
import librosa
from madmom.io.audio import load_audio_file, write_wave_file
from .beat_tracking import apply_beat_tracker

INTRO = """
Beat tracking
"""


def load_wav(wav_in):
    x, fs = load_audio_file(wav_in, sample_rate=44100)

    # stereo to mono if necessary
    if len(x.shape) > 1 and x.shape[1] == 2:
        x = x.sum(axis=1) / 2

    # cast to float
    x = x.astype(numpy.single)

    # normalize between -1.0 and 1.0
    x /= numpy.max(numpy.abs(x))

    return x


def write_wav(wav_out, x):
    write_wave_file(x, wav_out, sample_rate=44100)


def main():
    parser = argparse.ArgumentParser(
        prog="beat-tracking",
        description=INTRO,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--algorithms", type=str, default='1', help="List of beat tracking algorithms to apply"
    )
    parser.add_argument(
        "--shape-transients", action="store_true", help="Apply transient enhancing"
    )
    parser.add_argument(
        "--fast-attack-ms", type=int, default=1, help="Fast attack (ms)"
    )
    parser.add_argument(
        "--slow-attack-ms", type=int, default=15, help="Slow attack (ms)"
    )
    parser.add_argument("--release-ms", type=int, default=20, help="Release (ms)")
    parser.add_argument(
        "--power-memory-ms", type=int, default=1, help="Power filter memory (ms)"
    )
    parser.add_argument(
        "--n-pool", type=int, default=multiprocessing.cpu_count()-1, help="How many threads to use in multiprocessing pool (default = thread count - 1)"
    )

    parser.add_argument(
        "--chunk-size-seconds", type=float, default=numpy.inf, help="Apply beat tracking to song split into chunks"
    )

    parser.add_argument("wav_in", help="input wav file")
    parser.add_argument("wav_out", help="output wav file")

    args = parser.parse_args()
    print(args)

    beat_tracking_algos = [int(x) for x in args.algorithms.split(',')]

    x = load_wav(args.wav_in)

    pool = multiprocessing.Pool(args.n_pool)
    beats = apply_beat_tracker(beat_tracking_algos, x, pool)

    clicks = librosa.clicks(beats, sr=44100, length=len(x))
    write_wav(args.wav_out, x+clicks)
