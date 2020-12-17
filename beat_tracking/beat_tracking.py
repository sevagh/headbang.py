import madmom
import sys
import numpy
from librosa.beat import beat_track
import itertools
from essentia.standard import BeatTrackerMultiFeature, BeatTrackerDegara


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
