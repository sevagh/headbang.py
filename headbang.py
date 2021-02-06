#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
import sys
import json
import multiprocessing
import numpy
import librosa
from madmom.io.audio import write_wave_file

from headbang import HeadbangBeatTracker
from headbang.util import load_wav
from headbang.percussive_transients import kick_snare_filter


def main():
    parser = argparse.ArgumentParser(
        prog="headbang.py",
        description="Accurate percussive beat tracking for metal songs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    beat_args = parser.add_argument_group("beat arguments")
    beat_args.add_argument(
        "--algorithms",
        type=str,
        default="1,2,3,4,5,6",
        help="List of beat tracking algorithms to apply",
    )
    beat_args.add_argument(
        "--beat-near-threshold",
        type=float,
        default=0.1,
        help="How close beats should be in seconds to be considered the same beat",
    )
    beat_args.add_argument(
        "--consensus-ratio",
        type=float,
        default=0.75,
        help="How many (out of the maximum possible) beat locations should agree",
    )

    onset_args = parser.add_argument_group("onsets arguments")
    onset_args.add_argument(
        "--max-no-beats",
        type=float,
        default=2.0,
        help="Segments with missing beats to substitute onsets",
    )
    onset_args.add_argument(
        "--onset-near-threshold",
        type=float,
        default=0.15,
        help="How close onsets should be in seconds when supplementing onset information",
    )
    onset_args.add_argument(
        "--onset-silence-threshold", type=float, default=0.035, help="Silence threshold"
    )

    parser.add_argument(
        "--n-pool",
        type=int,
        default=multiprocessing.cpu_count() - 1,
        help="How many threads to use in multiprocessing pool",
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
        "--pre-kick-snare-filter",
        action="store_true",
        help="filter into kick (0-150) and snare (200-500) frequency bands first",
    )
    parser.add_argument(
        "--beats-out", type=str, default="", help="output beats txt file"
    )

    hpss_args = parser.add_argument_group("hpss arguments")
    hpss_args.add_argument(
        "--harmonic-margin",
        type=float,
        default=2.3,
        help="Separation margin for HPSS harmonic iteration",
    )
    hpss_args.add_argument(
        "--harmonic-frame",
        type=int,
        default=16384,
        help="T-F/frame size for HPSS harmonic iteration",
    )
    hpss_args.add_argument(
        "--percussive-margin",
        type=float,
        default=2.3,
        help="Separation margin for HPSS percussive iteration",
    )
    hpss_args.add_argument(
        "--percussive-frame",
        type=int,
        default=128,
        help="T-F/frame size for HPSS percussive iteration",
    )

    tshaper_args = parser.add_argument_group("multiband transient shaper arguments")
    tshaper_args.add_argument(
        "--fast-attack-ms", type=int, default=1, help="Fast attack (ms)"
    )
    tshaper_args.add_argument(
        "--slow-attack-ms", type=int, default=15, help="Slow attack (ms)"
    )
    tshaper_args.add_argument("--release-ms", type=int, default=20, help="Release (ms)")
    tshaper_args.add_argument(
        "--power-memory-ms", type=int, default=1, help="Power filter memory (ms)"
    )
    tshaper_args.add_argument(
        "--filter-order", type=int, default=3, help="Bandpass (butter) filter order"
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
        args.beat_near_threshold,
        args.consensus_ratio,
        # perccussive onset alignment params
        args.disable_onsets,
        args.max_no_beats,
        args.onset_near_threshold,
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
    )

    beats = None
    print("Applying HeadbangBeatTracker algorithm")
    if args.pre_kick_snare_filter:
        beats = hbt.beats(kick_snare_filter(x, 44100, order=args.filter_order))
    else:
        beats = hbt.beats(x)

    if args.beats_out:
        print("Writing beat locations to file {0}".format(args.beats_out))
        with open(args.beats_out, "w") as f:
            for b in beats:
                f.write(f"{b}\n")

    print("Overlaying clicks at beat locations")
    clicks = librosa.clicks(beats, sr=44100, length=len(x))
    final_waveform = (x + clicks).astype(numpy.single)

    print("Writing output with clicks to {0}".format(args.wav_out))
    write_wave_file(final_waveform, args.wav_out, sample_rate=44100)

    if args.show_plots:
        print("Displaying plots")
        generate_all_plots(
            x,
            hbt.cbt.all_beats,
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
    for beats in beat_results:
        plt.plot(
            beats,
            numpy.zeros(len(beats)),
            marker="o",
            linestyle="None",
            markersize=10,
        )
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


if __name__ == "__main__":
    main()
