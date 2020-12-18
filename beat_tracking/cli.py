import argparse
import json
import multiprocessing
import numpy
import librosa
from madmom.io.audio import load_audio_file, write_wave_file
from .algorithm import apply_meta_algorithm
from .effects import ihpss

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
        self.xp = None

        # here's where i could parameter check
        # if i cared

        # consensus params
        self.beat_tracking_algorithms = [int(x) for x in args.algorithms.split(",")]
        self.beat_near_threshold = args.beat_near_threshold
        self.consensus_ratio = args.consensus_ratio

        # hpss params
        self.percussive_frame = args.percussive_frame
        self.percussive_beta = args.percussive_beta
        self.harmonic_frame = args.harmonic_frame
        self.harmonic_beta = args.harmonic_beta

        # transient shaper params
        self.dont_shape_transients = args.dont_shape_transients
        self.fast_attack_ms = args.fast_attack_ms
        self.slow_attack_ms = args.slow_attack_ms
        self.release_ms = args.release_ms
        self.power_memory_ms = args.power_memory_ms

        # onset alignment params
        self.dont_onset_align = args.dont_align_onsets
        self.onset_silence_threshold = args.onset_silence_threshold


def main():
    parser = argparse.ArgumentParser(
        prog="beat-tracking",
        description=INTRO,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--algorithms",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="List of beat tracking algorithms to apply",
    )
    parser.add_argument(
        "--dont-shape-transients",
        action="store_true",
        help="Don't apply transient enhancing",
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
        "--harmonic-beta",
        type=float,
        default=3.0,
        help="Separation margin for HPSS harmonic iteration",
    )
    parser.add_argument(
        "--harmonic-frame",
        type=int,
        default=4096,
        help="T-F/frame size for HPSS harmonic iteration",
    )
    parser.add_argument(
        "--percussive-beta",
        type=float,
        default=3.0,
        help="Separation margin for HPSS percussive iteration",
    )
    parser.add_argument(
        "--percussive-frame",
        type=int,
        default=256,
        help="T-F/frame size for HPSS percussive iteration",
    )
    parser.add_argument(
        "--n-pool",
        type=int,
        default=multiprocessing.cpu_count() - 1,
        help="How many threads to use in multiprocessing pool",
    )
    parser.add_argument(
        "--beat-near-threshold",
        type=float,
        default=0.05,
        help="How close beats should be in seconds to be considered the same beat",
    )
    parser.add_argument(
        "--consensus-ratio",
        type=float,
        default=0.33,
        help="How many (out of the maximum possible) beat locations should agree",
    )
    parser.add_argument(
        "--dont-align-onsets", action="store_true", help="Don't align beats with onsets"
    )
    parser.add_argument(
        "--onset-silence-threshold", type=float, default=0.035, help="Silence threshold"
    )

    parser.add_argument("wav_in", help="input wav file")
    parser.add_argument("wav_out", help="output wav file")

    args = parser.parse_args()
    print(args)

    prog = BeatTrackingCli(args)

    beats = apply_meta_algorithm(prog)

    # if not prog.dont_onset_align:
    clicks = librosa.clicks(beats, sr=44100, length=len(prog.x))
    write_wav(args.wav_out, prog.x + clicks)
