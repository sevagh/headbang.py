#!/usr/bin/env python3

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
        self.percussive_beta = args.percussive_margin
        self.harmonic_frame = args.harmonic_frame
        self.harmonic_beta = args.harmonic_margin

        # transient shaper params
        self.fast_attack_ms = args.fast_attack_ms
        self.slow_attack_ms = args.slow_attack_ms
        self.release_ms = args.release_ms
        self.power_memory_ms = args.power_memory_ms

        # onset alignment params
        self.onset_silence_threshold = args.onset_silence_threshold

        # segmented supplementary beats
        self.max_no_beats = args.max_no_beats
        self.onset_near_threshold = args.onset_near_threshold


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
        "--max-no-beats", type=float, default=2.0, help="Time without beats to tolerate"
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
        "--harmonic-margin",
        type=float,
        default=2.0,
        help="Separation margin for HPSS harmonic iteration",
    )
    parser.add_argument(
        "--harmonic-frame",
        type=int,
        default=4096,
        help="T-F/frame size for HPSS harmonic iteration",
    )
    parser.add_argument(
        "--percussive-margin",
        type=float,
        default=2.0,
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
        "--onset-near-threshold",
        type=float,
        default=0.1,
        help="How close onsets should be in seconds when supplementing onset information",
    )
    parser.add_argument(
        "--consensus-ratio",
        type=float,
        default=0.50,
        help="How many (out of the maximum possible) beat locations should agree",
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
    final_waveform = (prog.x + clicks).astype(numpy.single)

    write_wav(args.wav_out, final_waveform)


def madmom_1(x, act):
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    return proc(act)


def madmom_2(x, act):
    proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
    return proc(act)


def madmom_3(x, act):
    proc = madmom.features.beats.CRFBeatDetectionProcessor(fps=100)
    return proc(act)


def madmom_4(x, act):
    proc = madmom.features.beats.BeatDetectionProcessor(fps=100)
    return proc(act)


######################################
# activation unused below this point #
######################################


def essentia_mfbt(x, act):
    beats, _ = BeatTrackerMultiFeature()(x)
    return beats


def essentia_degara(x, act):
    return BeatTrackerDegara()(x)


def librosa_beats(x, act):
    _, beats = beat_track(x, sr=44100, units="time")
    return beats


def btrack(x, act):
    try:
        import btrack
    except Exception:
        print("you must install btrack yourself manually", file=sys.stderr)
        sys.exit(1)
    return btrack.trackBeats(x)


BEAT_TRACK_ALGOS = {
    1: madmom_1,
    2: madmom_2,
    3: madmom_3,
    4: madmom_4,
    5: essentia_mfbt,
    6: essentia_degara,
    7: librosa_beats,
    8: btrack,
}


def apply_single_beat_tracker(x, beat_algo, frame_offset=0):
    # global RNN activation for all madmom algorithms
    act = madmom.features.beats.RNNBeatProcessor()(x)
    beats_pre_offset = BEAT_TRACK_ALGOS[beat_algo](x, act)
    beats_post_offset = beats_pre_offset + frame_offset
    return beats_post_offset


ODF = ["hfc", "complex", "flux", "rms"]
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
        # cribbed straight from the essentia examples
        # https://essentia.upf.edu/essentia_python_examples.html
        self.pool = essentia.Pool()
        self.onsets = Onsets(silenceThreshold=silence_threshold)

        weights = numpy.ones(len(ODF))
        weights[0] = 2.0  # weight hfc onsets a bit stronger

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


def get_consensus_beats(all_beats, max_consensus, beat_near_threshold, consensus_ratio):
    # no point getting a consensus of a single algorithm
    if max_consensus == 1:
        return all_beats

    good_beats = numpy.sort(numpy.unique(all_beats))
    grouped_beats = numpy.split(
        good_beats,
        numpy.where(numpy.diff(good_beats) ** 2 > beat_near_threshold)[0] + 1,
    )

    beats = [x[0] for x in grouped_beats]
    beats = [numpy.mean(x) for x in grouped_beats]

    tick_agreements = [len(x) for x in grouped_beats]

    final_beats = []
    for i, tick in enumerate(beats):
        # at least CONSENSUS beat trackers agree
        if tick_agreements[i] > max_consensus * consensus_ratio:
            final_beats.append(tick)

    return final_beats


# apply beat tracking to the song within the bounds of an inconclusive segment
def segmented_beat_tracking(
    x, pool, beat_tracking_algorithms, segment_begin, segment_end
):
    chunk_begin = int(segment_begin * 44100.0)
    chunk_end = int(segment_end * 44100.0)

    # need a consensus across all algorithms
    all_beats = numpy.array([])

    print("applying segmented beat tracking with increment: {0}".format(segment_begin))

    beat_results = pool.starmap(
        apply_single_beat_tracker,
        zip(
            itertools.repeat(x[chunk_begin:chunk_end]),
            beat_tracking_algorithms,
            itertools.repeat(segment_begin),
        ),
    )

    for beats in beat_results:
        all_beats = numpy.concatenate((all_beats, beats))

    all_beats = numpy.sort(all_beats)
    return all_beats


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


def apply_meta_algorithm(prog):
    # gather all the beats from all beat tracking algorithms
    beat_results = prog.pool.starmap(
        apply_single_beat_tracker,
        zip(itertools.repeat(prog.x), prog.beat_tracking_algorithms),
    )

    # need a consensus across all algorithms
    all_beats = numpy.array([])

    for beats in beat_results:
        all_beats = numpy.concatenate((all_beats, beats))

    # get a percussive separation for onset alignment
    _, xp = ihpss(
        prog.x,
        # hpss params
        (
            prog.harmonic_frame,
            prog.harmonic_beta,
            prog.percussive_frame,
            prog.percussive_beta,
        ),
        # transient shaper params
        (
            prog.fast_attack_ms,
            prog.slow_attack_ms,
            prog.release_ms,
            prog.power_memory_ms,
        ),
        prog.pool,
    )

    all_beats = numpy.sort(all_beats)

    beat_consensus = get_consensus_beats(
        all_beats,
        len(prog.beat_tracking_algorithms),
        prog.beat_near_threshold,
        prog.consensus_ratio,
    )

    onsets = OnsetGenerator(prog.onset_silence_threshold).get_onsets(xp, prog.pool)
    aligned = align_beats_onsets(beat_consensus, onsets, prog.beat_near_threshold)

    # add a 0 in there in case no beats have been found until the first, very deep into the song
    # also concatenate the max length for that case too

    endofsong = (len(prog.x) - 1) / 44100.0

    beat_jumps = numpy.where(
        numpy.diff(numpy.concatenate(([0.0], aligned, [endofsong]))) > prog.max_no_beats
    )[0]
    to_concat = numpy.array([])

    # collect extra beats by applying consensus beat tracking specifically to low-information segments
    for j in beat_jumps:
        try:
            print("segment with no beats: {0}-{1}".format(aligned[j], aligned[j + 1]))

            segment_onsets = onsets[
                numpy.where(
                    numpy.logical_and(
                        onsets > aligned[j] + 1.0, onsets < aligned[j + 1] - 1.0
                    )
                )[0]
            ]

            spread_onsets = numpy.split(
                segment_onsets,
                numpy.where(
                    numpy.diff(segment_onsets) ** 2 > prog.onset_near_threshold
                )[0]
                + 1,
            )

            so = [s[0] for s in spread_onsets if s.size > 0]

            print(
                "supplementing with percussive onsets from this region: {0}".format(so)
            )
            to_concat = numpy.concatenate((to_concat, so))
        except IndexError:
            break

    aligned = numpy.sort(numpy.concatenate((aligned, to_concat)))

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

    # small t-f resolution for percussive
    S2 = stft(
        yp1 + yr1,
        n_fft=2 * percussive_frame,
        win_length=percussive_frame,
        hop_length=int(percussive_frame // 2),
    )
    _, S_p2 = hpss(S2, margin=percussive_beta, power=numpy.inf)  # hard mask

    yp = fix_length(istft(S_p2, dtype=x.dtype), len(x))

    yp = multiband_transient_shaper(
        yp,
        44100,
        transient_shaper_params,
        pool,
    )

    return yh, yp


if __name__ == "__main__":
    main()
