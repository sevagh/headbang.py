#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
import sys
import json
import multiprocessing
import numpy
import librosa
from madmom.io.audio import load_audio_file, write_wave_file
import itertools
import essentia
from essentia.standard import (
    OnsetDetection,
    Onsets,
    Windowing,
    FFT,
    CartesianToPolar,
    FrameGenerator,
    BeatTrackerMultiFeature,
    BeatTrackerDegara,
)
import madmom
from librosa.beat import beat_track
from scipy.signal import butter, lfilter
import multiprocessing
from librosa.decompose import hpss
from librosa.core import stft, istft
from librosa.util import fix_length


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
        default="1,2,3,4,5,6,7,8",
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
        default=0.5,
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
        default=0.1,
        help="How close onsets should be in seconds when supplementing onset information",
    )
    onset_args.add_argument(
        "--onset-silence-threshold", type=float, default=0.035, help="Silence threshold"
    )

    hpss_args = parser.add_argument_group("hpss arguments")
    hpss_args.add_argument(
        "--harmonic-margin",
        type=float,
        default=2.0,
        help="Separation margin for HPSS harmonic iteration",
    )
    hpss_args.add_argument(
        "--harmonic-frame",
        type=int,
        default=4096,
        help="T-F/frame size for HPSS harmonic iteration",
    )
    hpss_args.add_argument(
        "--percussive-margin",
        type=float,
        default=2.0,
        help="Separation margin for HPSS percussive iteration",
    )
    hpss_args.add_argument(
        "--percussive-frame",
        type=int,
        default=256,
        help="T-F/frame size for HPSS percussive iteration",
    )

    tshaper_args = parser.add_argument_group("transient shaper arguments")
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
        "--beats-out", type=str, default="", help="output beats txt file"
    )

    parser.add_argument("wav_in", help="input wav file")
    parser.add_argument("wav_out", help="output wav file")

    args = parser.parse_args()

    print("Loading file {0} with 44100 sampling rate".format(args.wav_in))
    x = load_wav(args.wav_in)

    print("Applying meta algorithm")
    beats = apply_meta_algorithm(x, args)

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


def apply_single_beat_tracker(x, beat_algo):
    beats = None
    act = madmom.features.beats.RNNBeatProcessor()(x)

    if beat_algo == 1:
        beats = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)(act)
    elif beat_algo == 2:
        beats = madmom.features.beats.BeatTrackingProcessor(fps=100)(act)
    elif beat_algo == 3:
        beats = madmom.features.beats.CRFBeatDetectionProcessor(fps=100)(act)
    elif beat_algo == 4:
        beats = madmom.features.beats.BeatDetectionProcessor(fps=100)(act)
    elif beat_algo == 5:
        beats, _ = BeatTrackerMultiFeature()(x)
    elif beat_algo == 6:
        beats = BeatTrackerDegara()(x)
    elif beat_algo == 7:
        _, beats = beat_track(x, sr=44100, units="time")
    elif beat_algo == 8:
        try:
            import btrack
        except Exception:
            print("you must install btrack yourself manually", file=sys.stderr)
            sys.exit(1)
        beats = btrack.trackBeats(x)

    return beats


ODF = ["hfc", "flux", "rms"]
ONSET_DETECTORS = [OnsetDetection(method=f) for f in ODF]

w = Windowing(type="hann")
fft = FFT()
c2p = CartesianToPolar()


def apply_single_odf(odf_idx, frame):
    (
        mag,
        phase,
    ) = c2p(fft(w(frame.astype(numpy.single))))
    return ONSET_DETECTORS[odf_idx](mag, phase)


class OnsetGenerator:
    def __init__(self, silence_threshold):
        self.pool = essentia.Pool()
        self.onsets = Onsets(silenceThreshold=silence_threshold)

        weights = numpy.ones(len(ODF))
        weights[0] = 3.0  # weight hfc stronger

        self.weights = weights.astype(numpy.single)

    def get_onsets(self, x, pool):
        # Computing onset detection functions.
        for frame in FrameGenerator(
            x.astype(numpy.single), frameSize=1024, hopSize=512
        ):
            onset_features = pool.starmap(
                apply_single_odf,
                zip(
                    range(len(ODF)),
                    itertools.repeat(frame),
                ),
            )
            for i, of in enumerate(onset_features):
                self.pool.add("features.{0}".format(ODF[i]), of)

        # convert pool into matrix
        matrix = [
            essentia.array(self.pool["features.{0}".format(ODF[i])])
            for i in range(len(ODF))
        ]

        return self.onsets(matrix, self.weights)


def get_consensus_beats(all_beats, beat_near_threshold, consensus):
    good_beats = numpy.sort(numpy.unique(all_beats))
    grouped_beats = numpy.split(
        good_beats,
        numpy.where(numpy.diff(good_beats) > beat_near_threshold)[0] + 1,
    )

    beats = [x[0] for x in grouped_beats]
    beats = [numpy.mean(x) for x in grouped_beats]

    tick_agreements = [len(x) for x in grouped_beats]

    final_beats = []
    for i, tick in enumerate(beats):
        # at least CONSENSUS beat trackers agree
        if tick_agreements[i] >= consensus:
            final_beats.append(tick)

    return final_beats


def align_beats_onsets(beats, onsets, thresh):
    i = 0
    j = 0

    aligned_beats = []
    time_since_last_beat = 0.0

    while i < len(onsets) and j < len(beats):
        curr_onset = onsets[i]
        curr_beat = beats[j]

        if numpy.abs(curr_onset - curr_beat) <= thresh:
            aligned_beats.append((curr_onset + curr_beat) / 2)
            i += 1
            j += 1
            continue

        if curr_beat < curr_onset:
            # increment beats
            j += 1
        elif curr_beat > curr_onset:
            i += 1

    return aligned_beats


def apply_meta_algorithm(x, args):
    pool = multiprocessing.Pool(args.n_pool)
    beat_tracking_algorithms = [int(x) for x in args.algorithms.split(",")]

    print("Applying all beat trackers in parallel: {0}".format(args.algorithms))
    # gather all the beats from all beat tracking algorithms
    beat_results = pool.starmap(
        apply_single_beat_tracker,
        zip(itertools.repeat(x), beat_tracking_algorithms),
    )

    # need a consensus across all algorithms
    all_beats = numpy.array([])

    for beats in beat_results:
        all_beats = numpy.concatenate((all_beats, beats))

    all_beats = numpy.sort(all_beats)

    total = len(beat_tracking_algorithms)
    consensus = int(numpy.ceil(args.consensus_ratio * total))

    print(
        "Creating percussive separation with enhanced transients for percussive onset detection"
    )
    # get a percussive separation for onset alignment, and the percussive spectrum
    xp, xp_hpss = ihpss(
        x,
        # hpss params
        (
            args.harmonic_frame,
            args.harmonic_margin,
            args.percussive_frame,
            args.percussive_margin,
        ),
        # transient shaper params
        (
            args.fast_attack_ms,
            args.slow_attack_ms,
            args.release_ms,
            args.power_memory_ms,
        ),
        pool,
    )

    print(
        "Applying beat consensus to get agreed beats: need {0} / {1} agreements".format(
            consensus, total
        )
    )
    beat_consensus = get_consensus_beats(
        all_beats,
        args.beat_near_threshold,
        consensus,
    )

    print("Detecting percussive onsets with methods {0}".format(ODF))
    onsets = OnsetGenerator(args.onset_silence_threshold).get_onsets(xp, pool)

    print("Aligning agreed beats with percussive onsets")
    aligned = align_beats_onsets(beat_consensus, onsets, args.beat_near_threshold)

    print("Trying to substitute percussive onsets in place of absent beats")
    # add a 0 in there in case no beats have been found until the first, very deep into the song
    # also concatenate the max length for that case too
    endofsong = (len(x) - 1) / 44100.0

    aligned_prime = numpy.concatenate(([0.0], aligned, [endofsong]))

    beat_jumps = numpy.where(numpy.diff(aligned_prime) > args.max_no_beats)[0]

    to_concat = numpy.array([])

    # collect extra beats by applying consensus beat tracking specifically to low-information segments
    for j in beat_jumps:
        try:
            print(
                "segment with no beats: {0}-{1}".format(
                    aligned_prime[j], aligned_prime[j + 1]
                )
            )

            segment_onsets = onsets[
                numpy.where(
                    numpy.logical_and(
                        onsets > aligned_prime[j] + 1.0,
                        onsets < aligned_prime[j + 1] - 1.0,
                    )
                )[0]
            ]

            sparse_onsets = numpy.split(
                segment_onsets,
                numpy.where(numpy.diff(segment_onsets) > args.onset_near_threshold)[0]
                + 1,
            )

            so = [s[0] for s in sparse_onsets if s.size > 0]

            if so:
                print(
                    "supplementing with percussive onsets from this region: {0}".format(
                        so
                    )
                )
                to_concat = numpy.concatenate((to_concat, so))
        except IndexError:
            break

    aligned = numpy.sort(numpy.concatenate((aligned, to_concat)))

    if args.show_plots:
        generate_all_plots(
            x, beat_results, beat_consensus, onsets, xp, xp_hpss, aligned, to_concat
        )

    return aligned


# bark frequency bands
FREQ_BANDS = [
    20,
    119,
    224,
    326,
    438,
    561,
    698,
    850,
    1021,
    1213,
    1433,
    1685,
    1978,
    2322,
    2731,
    3227,
    3841,
    4619,
    5638,
    6938,
    8492,
    10705,
    14105,
    20000,
]


def bandpass(lo, hi, x, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return lfilter(b, a, x)


def lowpass(hi, x, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, hi / nyq, btype="low")
    return lfilter(b, a, x)


def envelope(x, fs, params):
    fast_attack = params[0]
    slow_attack = params[1]
    release = params[2]
    power_mem = params[3]

    g_fast = numpy.exp(-1.0 / (fs * fast_attack / 1000.0))
    g_slow = numpy.exp(-1.0 / (fs * slow_attack / 1000.0))
    g_release = numpy.exp(-1.0 / (fs * release / 1000.0))
    g_power = numpy.exp(-1.0 / (fs * power_mem / 1000.0))

    fb_fast = 0
    fb_slow = 0
    fb_pow = 0

    N = len(x)

    fast_envelope = numpy.zeros(N)
    slow_envelope = numpy.zeros(N)
    attack_gain_curve = numpy.zeros(N)

    x_power = numpy.zeros(N)
    x_deriv_power = numpy.zeros(N)

    for n in range(N):
        x_power[n] = (1 - g_power) * x[n] * x[n] + g_power * fb_pow
        fb_pow = x_power[n]

    x_deriv_power[0] = x_power[0]

    # simple differentiator filter
    for n in range(1, N):
        x_deriv_power[n] = x_power[n] - x_power[n - 1]

    for n in range(N):
        if fb_fast > x_deriv_power[n]:
            fast_envelope[n] = (1 - g_release) * x_deriv_power[n] + g_release * fb_fast
        else:
            fast_envelope[n] = (1 - g_fast) * x_deriv_power[n] + g_fast * fb_fast
        fb_fast = fast_envelope[n]

        if fb_slow > x_deriv_power[n]:
            slow_envelope[n] = (1 - g_release) * x_deriv_power[n] + g_release * fb_slow
        else:
            slow_envelope[n] = (1 - g_slow) * x_deriv_power[n] + g_slow * fb_slow
        fb_slow = slow_envelope[n]

        attack_gain_curve[n] = fast_envelope[n] - slow_envelope[n]

    attack_gain_curve /= numpy.max(attack_gain_curve)

    # normalize to [0, 1.0]
    return x * attack_gain_curve


def single_band_transient_shaper(band, x, fs, shaper_params, order=2):
    lo = FREQ_BANDS[band]
    hi = FREQ_BANDS[band + 1]

    y = bandpass(lo, hi, x, fs)

    # per bark band, apply a differential envelope attack/transient enhancer
    y_shaped = envelope(y, fs, shaper_params)

    return y_shaped


def multiband_transient_shaper(x, fs, shaper_params, pool):
    if shaper_params[0] not in [0, 1]:
        raise ValueError("attack should be 0 (boost sustain) or 1 (boost attacks)")

    # bark band decomposition
    band_results = list(
        pool.starmap(
            single_band_transient_shaper,
            zip(
                range(0, len(FREQ_BANDS) - 1, 1),
                itertools.repeat(x),
                itertools.repeat(fs),
                itertools.repeat(shaper_params),
            ),
        )
    )

    y_t = numpy.zeros(len(x))
    for banded_attacks in band_results:
        y_t += banded_attacks

    return y_t


# iterative hpss
def ihpss(x, hpss_params, transient_shaper_params, pool):
    harmonic_frame = hpss_params[0]
    harmonic_beta = hpss_params[1]
    percussive_frame = hpss_params[2]
    percussive_beta = hpss_params[3]

    print(
        "Iteration 1 of hpss: frame = {0}, margin = {1}".format(
            harmonic_frame, harmonic_beta
        )
    )
    # big t-f resolution for harmonic
    S1 = stft(
        x,
        n_fft=2 * harmonic_frame,
        win_length=harmonic_frame,
        hop_length=int(harmonic_frame // 2),
    )
    S_h1, S_p1 = hpss(S1, margin=harmonic_beta, power=numpy.inf)  # hard mask
    S_r1 = S1 - (S_h1 + S_p1)

    yh = fix_length(istft(S_h1, dtype=x.dtype), len(x))
    yp1 = fix_length(istft(S_p1, dtype=x.dtype), len(x))
    yr1 = fix_length(istft(S_r1, dtype=x.dtype), len(x))

    print(
        "Iteration 2 of hpss: frame = {0}, margin = {1}".format(
            percussive_frame, percussive_beta
        )
    )
    # small t-f resolution for percussive
    S2 = stft(
        yp1 + yr1,
        n_fft=2 * percussive_frame,
        win_length=percussive_frame,
        hop_length=int(percussive_frame // 2),
    )
    _, S_p2 = hpss(S2, margin=percussive_beta, power=numpy.inf)  # hard mask

    yp = fix_length(istft(S_p2, dtype=x.dtype), len(x))

    print("Applying multiband transient shaper")
    yp_tshaped = multiband_transient_shaper(
        yp,
        44100,
        transient_shaper_params,
        pool,
    )

    return yp_tshaped, yp


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
