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
    "_",
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
        beats, confidence = BeatTrackerMultiFeature()(x)
    elif beat_algo == 4:
        beats = BeatTrackerDegara()(x)
    elif beat_algo == 5:
        _, beats = beat_track(x, sr=44100, units="time")
    elif beat_algo == 6:
        beats = btrack.trackBeats(x)

    return beats


def get_consensus_beats(
    all_beats, beat_near_threshold_s, consensus, beat_pick="mean"
):
    final_beats = []
    if len(all_beats) == 0:
        return final_beats

    good_beats = numpy.sort(numpy.unique(all_beats))
    grouped_beats = numpy.split(
        good_beats,
        numpy.where(numpy.diff(good_beats) > beat_near_threshold_s)[0] + 1,
    )

    beats = None
    if beat_pick == "mean":
        beats = [numpy.mean(x) for x in grouped_beats]
    elif beat_pick == "first":
        beats = [x[0] for x in grouped_beats]
    else:
        raise ValueError("unrecognized beat_pick strategy {0}".format(beat_pick))

    tick_agreements = [len(x) for x in grouped_beats]

    for i, tick in enumerate(beats):
        # at least CONSENSUS beat trackers agree
        if tick_agreements[i] >= consensus:
            final_beats.append(tick)

    return final_beats


class ConsensusBeatTracker:
    def __init__(
        self,
        pool,
        algorithms=DEFAULTS["algorithms"],
    ):
        self.pool = pool
        self.beat_tracking_algorithms = [int(x) for x in algorithms.split(",")]

        self.ttap = TempoTapMaxAgreement()

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

        beat_results = [b.astype(numpy.single) for b in beat_results]
        beat_consensus, _ = self.ttap(beat_results)

        return beat_consensus
