from .beat_tracking import apply_single_beat_tracker
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
    _, beats = beat_track(x, sr=44100)
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
    return BEAT_TRACK_ALGOS[beat_algo](x, act) + frame_offset


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
        self.weights = numpy.ones(len(ODF))
        self.weights[0] = 5.0  # heavily weight hfc onsets

    def get_onsets(self, prog):
        # Computing onset detection functions.
        prog.xp = prog.xp.astype(numpy.single)
        for frame in FrameGenerator(prog.xp, frameSize=1024, hopSize=512):
            onset_features = prog.pool.starmap(
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


def get_consensus_beats(all_beats, max_consensus, prog):
    # no point getting a consensus of a single algorithm
    if max_consensus == 1:
        return all_beats

    good_beats = numpy.sort(numpy.unique(all_beats))
    grouped_beats = numpy.split(
        good_beats,
        numpy.where(numpy.diff(good_beats) ** 2 > prog.beat_near_threshold)[0] + 1,
    )

    beats = [x[0] for x in grouped_beats]
    beats = [numpy.mean(x) for x in grouped_beats]

    tick_agreements = [len(x) for x in grouped_beats]

    final_beats = []
    for i, tick in enumerate(beats):
        # at least CONSENSUS beat trackers agree
        if tick_agreements[i] > max_consensus * prog.consensus_ratio:
            final_beats.append(tick)

    return final_beats


# apply beat tracking to the song segmented into chunks
def chunked_algorithm(prog, chunk_seconds):
    chunk_samples = int(chunk_seconds * 44100.0)

    # need a consensus across all algorithms
    all_beats = numpy.array([])
    nframes = 0

    for frame in FrameGenerator(prog.x, frameSize=chunk_samples, hopSize=chunk_samples):
        beat_results = prog.pool.starmap(
            apply_single_beat_tracker,
            zip(
                itertools.repeat(prog.x),
                prog.beat_tracking_algorithms,
                itertools.repeat(nframes * chunk_seconds),
            ),
        )

        for beats in beat_results:
            all_beats = numpy.concatenate((all_beats, beats))

        nframes += 1

    all_beats = numpy.sort(all_beats)

    return get_consensus_beats(all_beats, len(prog.beat_tracking_algorithms), prog)


def apply_meta_algorithm(prog):
    #######################
    # pass 1 - whole song #
    #######################

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
    _, prog.xp = ihpss(prog.x, prog)

    onsets = OnsetGenerator(prog.onset_silence_threshold).get_onsets(prog)

    # add the onsets in the mix
    all_beats = numpy.concatenate((all_beats, onsets))

    all_beats = numpy.sort(all_beats)

    # get consensus with a little help from the percussive onsets
    beat_consensus = get_consensus_beats(
        all_beats, len(prog.beat_tracking_algorithms), prog
    )

    return beat_consensus
