import argparse
import json
import multiprocessing
import numpy
import librosa
from madmom.io.audio import load_audio_file, write_wave_file
from .meta_algorithms import apply_meta_algorithm

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


class BeatTrackingCli:
    def __init__(self, args):
        # shared multiprocessing pool
        self.pool = multiprocessing.Pool(args.n_pool)

        self.x = load_wav(args.wav_in)

        # here's where i could parameter check
        # if i cared

        # consensus params
        self.beat_tracking_algorithms = [int(x) for x in args.algorithms.split(',')]
        self.meta_algorithm = args.meta_algorithm
        self.beat_near_threshold = args.beat_near_threshold
        self.consensus_ratio = args.consensus_ratio

        # hpss params
        self.percussive_frame = args.percussive_frame
        self.percussive_beta = args.percussive_beta
        self.harmonic_frame = args.harmonic_frame
        self.harmonic_beta = args.harmonic_beta

        # transient shaper params
        self.shape_transients = args.shape_transients
        self.fast_attack_ms = args.fast_attack_ms
        self.slow_attack_ms = args.slow_attack_ms
        self.release_ms = args.release_ms
        self.power_memory_ms = args.power_memory_ms

        # chunk params
        self.chunk_size = args.chunk_size_seconds


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
        "--meta-algorithm", type=int, default=1, help="Which meta algorithm to apply"
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
        "--harmonic-beta", type=float, default=2.0, help="Separation margin for HPSS harmonic iteration"
    )
    parser.add_argument(
        "--harmonic-frame", type=int, default=4096, help="T-F/frame size for HPSS harmonic iteration"
    )
    parser.add_argument(
        "--percussive-beta", type=float, default=2.0, help="Separation margin for HPSS percussive iteration"
    )
    parser.add_argument(
        "--percussive-frame", type=int, default=256, help="T-F/frame size for HPSS percussive iteration"
    )
    parser.add_argument(
        "--n-pool", type=int, default=multiprocessing.cpu_count()-1, help="How many threads to use in multiprocessing pool (default = thread count - 1)"
    )

    parser.add_argument(
        "--chunk-size-seconds", type=float, default=numpy.inf, help="Apply beat tracking to song split into chunks"
    )
    parser.add_argument(
        "--beat-near-threshold", type=float, default=0.05, help="How close beats should be in seconds to be considered the same beat (default 0.05, 50ms)"
    )
    parser.add_argument(
        "--consensus-ratio", type=float, default=0.5, help="How many (out of the maximum possible) beat locations should agree"
    )

    parser.add_argument("wav_in", help="input wav file")
    parser.add_argument("wav_out", help="output wav file")

    args = parser.parse_args()
    print(args)

    prog = BeatTrackingCli(args)
    beats = apply_meta_algorithm(prog)

    clicks = librosa.clicks(beats, sr=44100, length=len(prog.x))
    write_wav(args.wav_out, prog.x+clicks)
