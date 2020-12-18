from .effects import ihpss, bandpass, lowpass
import numpy
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
import sys
from librosa.beat import beat_track


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

    endofsong = (len(prog.x)-1)/44100.0

    beat_jumps = numpy.where(numpy.diff(numpy.concatenate(([0.0], aligned, [endofsong]))) > prog.max_no_beats)[0]
    to_concat = numpy.array([])

    # collect extra beats by applying consensus beat tracking specifically to low-information segments
    for j in beat_jumps:
        try:
            print("segment with no beats: {0}-{1}".format(aligned[j], aligned[j + 1]))

            segment_onsets = onsets[
                numpy.where(
                    numpy.logical_and(onsets > aligned[j]+1.0, onsets < aligned[j + 1]-1.0)
                )[0]
            ]

            spread_onsets = numpy.split(
                segment_onsets,
                numpy.where(numpy.diff(segment_onsets) ** 2 > prog.onset_near_threshold)[0]
                + 1,
            )

            so = [s[0] for s in spread_onsets if s.size > 0]

            print("supplementing with percussive onsets from this region: {0}".format(so))
            to_concat = numpy.concatenate((to_concat, so))
        except IndexError:
            break

    aligned = numpy.sort(numpy.concatenate((aligned, to_concat)))

    return aligned
