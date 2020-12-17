import madmom
import sys
import numpy
from librosa.beat import beat_track
import itertools
from essentia.standard import BeatTrackerMultiFeature


def madmom_beats(x):
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(x)
    return proc(act)


def essentia_mfbt(x):
    beats, _ = BeatTrackerMultiFeature()(x)
    return beats


def librosa_beats(x):
    _, beats = beat_track(x, sr=44100)
    return beats


def btrack(x):
    try:
        import btrack
    except Exception:
        print('you must install btrack yourself manually', file=sys.stderr)
        sys.exit(1)
    return btrack.trackBeats(x)


_ALGOS = {
        1: madmom_beats,
        2: essentia_mfbt,
        3: librosa_beats,
        4: btrack,
}


def apply_single_beat_tracker(x, beat_algo):
    return _ALGOS[beat_algo](x)
