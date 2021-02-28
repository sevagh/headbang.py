import matplotlib.pyplot as plt
import argparse
import multiprocessing
import sys
import json
import numpy
import librosa
from madmom.io.audio import write_wave_file

from headbang import HeadbangBeatTracker
from headbang.util import load_wav, overlay_clicks
from headbang.params import DEFAULTS


def main():
    parser = argparse.ArgumentParser(
        description="Accurate percussive beat tracking for metal songs",
    )

    beat_args = parser.add_argument_group("beat arguments")
    beat_args.add_argument(
        "--algorithms",
        type=str,
        default=DEFAULTS["algorithms"],
        help="List of beat tracking algorithms to apply (default=%(default)s)",
    )
    beat_args.add_argument(
        "--onset-align-threshold-s",
        type=float,
        default=DEFAULTS["onset_align_threshold_s"],
        help="How close beats should align with onsets (in seconds) (default=%(default)s)",
    )

    onset_args = parser.add_argument_group("onsets arguments")
    onset_args.add_argument(
        "--max-no-beats",
        type=float,
        default=DEFAULTS["max_no_beats"],
        help="Segments with missing beats to substitute onsets (default=%(default)s)",
    )
    onset_args.add_argument(
        "--onset-near-threshold-s",
        type=float,
        default=DEFAULTS["onset_near_threshold_s"],
        help="How close onsets should be (in seconds) when supplementing onset information (default=%(default)s)",
    )
    onset_args.add_argument(
        "--onset-silence-threshold",
        type=float,
        default=DEFAULTS["onset_silence_threshold"],
        help="Silence threshold",
    )

    parser.add_argument(
        "--n-pool",
        type=int,
        default=multiprocessing.cpu_count() - 1,
        help="How many threads to use in multiprocessing pool (default=%(default)s)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots of intermediate steps describing the algorithm using matplotlib",
    )
    parser.add_argument(
        "--disable-onsets",
        action="store_true",
        help="disable onset alignment, only output consensus beats",
    )
    parser.add_argument(
        "--disable-transient-shaper",
        action="store_true",
        help="disable transient shaping, only use percussive separation",
    )
    parser.add_argument(
        "--beats-out",
        type=str,
        default="",
        help="output beats txt file (default=%(default)s)",
    )

    hpss_args = parser.add_argument_group("hpss arguments")
    hpss_args.add_argument(
        "--harmonic-margin",
        type=float,
        default=DEFAULTS["harmonic_margin"],
        help="Separation margin for HPSS harmonic iteration (default=%(default)s)",
    )
    hpss_args.add_argument(
        "--harmonic-frame",
        type=int,
        default=DEFAULTS["harmonic_frame"],
        help="T-F/frame size for HPSS harmonic iteration (default=%(default)s)",
    )
    hpss_args.add_argument(
        "--percussive-margin",
        type=float,
        default=DEFAULTS["percussive_margin"],
        help="Separation margin for HPSS percussive iteration (default=%(default)s)",
    )
    hpss_args.add_argument(
        "--percussive-frame",
        type=int,
        default=DEFAULTS["percussive_frame"],
        help="T-F/frame size for HPSS percussive iteration (default=%(default)s)",
    )

    tshaper_args = parser.add_argument_group("multiband transient shaper arguments")
    tshaper_args.add_argument(
        "--fast-attack-ms",
        type=int,
        default=DEFAULTS["fast_attack_ms"],
        help="Fast attack (ms) (default=%(default)s)",
    )
    tshaper_args.add_argument(
        "--slow-attack-ms",
        type=int,
        default=DEFAULTS["slow_attack_ms"],
        help="Slow attack (ms) (default=%(default)s)",
    )
    tshaper_args.add_argument(
        "--release-ms",
        type=int,
        default=DEFAULTS["release_ms"],
        help="Release (ms) (default=%(default)s)",
    )
    tshaper_args.add_argument(
        "--power-memory-ms",
        type=int,
        default=DEFAULTS["power_memory_ms"],
        help="Power filter memory (ms) (default=%(default)s)",
    )
    tshaper_args.add_argument(
        "--filter-order",
        type=int,
        default=DEFAULTS["filter_order"],
        help="Bandpass (butter) filter order (default=%(default)s)",
    )

    parser.add_argument("wav_in", help="input wav file")
    parser.add_argument("wav_out", help="output wav file")

    args = parser.parse_args()

    print("Loading file {0} with 44100 sampling rate".format(args.wav_in))
    x = load_wav(args.wav_in)

    pool = multiprocessing.Pool(args.n_pool)

    hbt = HeadbangBeatTracker(
        pool,
        # consensus beat tracking params
        args.algorithms,
        args.onset_align_threshold_s,
        # perccussive onset alignment params
        args.disable_onsets,
        args.max_no_beats,
        args.onset_near_threshold_s,
        args.onset_silence_threshold,
        # hpss params
        args.harmonic_margin,
        args.harmonic_frame,
        args.percussive_margin,
        args.percussive_frame,
        # transient shaper params
        args.fast_attack_ms,
        args.slow_attack_ms,
        args.release_ms,
        args.power_memory_ms,
        args.filter_order,
        args.disable_transient_shaper,
    )

    beats = None
    print("Applying HeadbangBeatTracker algorithm")
    beats = hbt.beats(x)

    if args.beats_out:
        print("Writing beat locations to file {0}".format(args.beats_out))
        with open(args.beats_out, "w") as f:
            for b in beats:
                f.write(f"{b}\n")

    print("Overlaying clicks at beat locations")
    x_stereo = load_wav(args.wav_in, stereo=True)
    x_with_clicks = overlay_clicks(x_stereo, beats)

    print("Writing output with clicks to {0}".format(args.wav_out))
    write_wave_file(x_with_clicks, args.wav_out, sample_rate=44100)

    if args.show_plots:
        print("Displaying plots")
        generate_all_plots(
            x,
            hbt.cbt.beat_results,
            hbt.beat_consensus,
            hbt.onsets,
            hbt.xp,
            hbt.xp_hpss,
            hbt.aligned,
            hbt.to_concat,
        )


def generate_all_plots(
    x, beat_results, beat_consensus, onsets, xp, xp_hpss, aligned, to_concat
):
    timestamps = [i / 44100.0 for i in range(len(x))]

    plt.figure(1)
    plt.title("Input waveform")
    plt.plot(timestamps, x)
    plt.xlabel("time (seconds)")
    plt.ylabel("amplitude")
    plt.show()

    plt.figure(1)
    plt.title("Input waveform with all beats")
    plt.plot(timestamps, x)
    plt.xlabel("time (seconds)")
    plt.ylabel("amplitude")
    for i, beats in enumerate(beat_results):
        # offset each different algo
        plt.plot(
            beats,
            numpy.zeros(len(beats), dtype=numpy.float) + i * 0.12,
            marker="o",
            linestyle="None",
            markersize=10,
        )
    plt.ylim([-1, 1])
    plt.show()

    plt.figure(1)
    plt.title("Input waveform with beat consensus")
    plt.plot(timestamps, x)
    plt.xlabel("time (seconds)")
    plt.ylabel("amplitude")
    plt.plot(
        beat_consensus,
        numpy.zeros(len(beat_consensus)),
        marker="o",
        linestyle="None",
        color="red",
        markersize=10,
    )
    plt.legend(["waveform", "beats"])
    plt.show()

    plt.figure(1)
    plt.title("Percussive-attack-enhanced waveform with onsets")
    plt.plot(timestamps, xp)
    plt.xlabel("time (seconds)")
    plt.ylabel("amplitude")
    plt.plot(
        onsets,
        numpy.zeros(len(onsets)),
        marker="o",
        linestyle="None",
        color="orange",
        markersize=10,
    )
    plt.legend(["waveform", "onsets"])
    plt.show()

    plt.figure(1)
    plt.title("Waveform with onset-aligned beats")
    plt.plot(timestamps, x)
    plt.xlabel("time (seconds)")
    plt.ylabel("amplitude")
    plt.plot(
        aligned,
        numpy.zeros(len(aligned)),
        marker="o",
        linestyle="None",
        color="red",
        markersize=5,
    )
    plt.plot(
        onsets,
        numpy.zeros(len(onsets)),
        marker="x",
        linestyle="None",
        color="orange",
        markersize=10,
    )
    plt.legend(["waveform", "beats", "onsets"])
    plt.show()

    if to_concat.size > 0:
        plt.figure(1)
        plt.title("Waveform with aligned beats and supplemented onsets")
        plt.plot(timestamps, x)
        plt.xlabel("time (seconds)")
        plt.ylabel("amplitude")
        plt.plot(
            aligned,
            numpy.zeros(len(aligned)),
            marker="o",
            linestyle="None",
            color="red",
            markersize=10,
        )
        plt.plot(
            to_concat,
            numpy.zeros(len(to_concat)),
            marker="o",
            linestyle="None",
            color="orange",
            markersize=10,
        )
        plt.legend(["waveform", "beats", "onsets"])
        plt.show()

    plt.figure(1)
    plt.title("Percussive separation after iterative HPSS")
    plt.plot(timestamps, xp_hpss)
    plt.xlabel("time (seconds)")
    plt.ylabel("amplitude")
    plt.show()

    plt.figure(1)
    plt.title("Percussive separation with enhanced attacks")
    plt.plot(timestamps, xp)
    plt.xlabel("time (seconds)")
    plt.ylabel("amplitude")
    plt.show()
