import madmom
import sys
import numpy
from librosa.beat import beat_track
import itertools
from essentia.standard import BeatTrackerMultiFeature
from .consensus import get_consensus_beats


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


def apply_beat_tracker(beat_tracking_algorithms, x, pool, beat_near_threshold=0.05, consensus_ratio=0.5, tick_strategy='first'):
    if len(beat_tracking_algorithms) == 1:
        return _ALGOS[beat_tracking_algorithms[0]](x)
    else:
        # need a consensus across many
        all_beats = numpy.array([])

        # gather all the beats from all beat tracking algorithms
        beat_results = pool.starmap(
            apply_single_beat_tracker,
            zip(
                itertools.repeat(x),
                beat_tracking_algorithms
            ),
        )

        for beats in beat_results:
            all_beats = numpy.concatenate((all_beats, beats))

        all_beats = numpy.sort(all_beats)

        return get_consensus_beats(all_beats, len(beat_tracking_algorithms), beat_near_threshold, consensus_ratio, tick_strategy)
