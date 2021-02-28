import sys
import multiprocessing
import numpy
import librosa
import btrack
import itertools
from essentia.standard import (
    BeatTrackerMultiFeature,
    BeatTrackerDegara,
    TempoTapMaxAgreement,
)
import madmom
from librosa.beat import beat_track
from .params import DEFAULTS


algo_names = [
    "_",  # dummy at index 0 because in this code, beat trackers start at 1: "1,2,3...8"
    "madmom DBNBeatTrackingProcessor",
    "madmom BeatDetectionProcessor",
    "essentia BeatTrackerMultiFeature",
    "essentia BeatTrackerDegara",
    "librosa beat_track",
    "BTrack",
]


def apply_single_beat_tracker(x, beat_algo):
    beats = None

    act = madmom.features.beats.RNNBeatProcessor()(x)

    if beat_algo == 1:
        beats = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)(act)
    elif beat_algo == 2:
        beats = madmom.features.beats.BeatDetectionProcessor(fps=100)(act)
    elif beat_algo == 3:
        beats, _ = BeatTrackerMultiFeature()(x)
    elif beat_algo == 4:
        beats = BeatTrackerDegara()(x)
    elif beat_algo == 5:
        _, beats = beat_track(x, sr=44100, units="time")
    elif beat_algo == 6:
        beats = btrack.trackBeats(x)

    return beats


class ConsensusBeatTracker:
    def __init__(
        self,
        pool,
        algorithms=DEFAULTS["algorithms"],
    ):
        self.pool = pool
        self.algorithms = algorithms
        self.beat_tracking_algorithms = [int(x) for x in algorithms.split(",")]

        self.ttap = TempoTapMaxAgreement()
        self.beat_results = None

    def print_params(self):
        print(
            "Consensus beat tracker algos: {0}".format(
                ",\n\t\t".join(
                    [
                        algo_name
                        for i, algo_name in enumerate(algo_names)
                        if i in self.beat_tracking_algorithms
                    ]
                ),
            )
        )

    def beats(self, x):
        # gather all the beats from all beat tracking algorithms
        beat_results = self.pool.starmap(
            apply_single_beat_tracker,
            zip(itertools.repeat(x), self.beat_tracking_algorithms),
        )

        self.beat_results = [b.astype(numpy.single) for b in beat_results]

        beat_consensus = None
        if len(self.beat_results) > 1:
            beat_consensus, _ = self.ttap(self.beat_results)
        else:
            beat_consensus = self.beat_results[0]

        return beat_consensus
